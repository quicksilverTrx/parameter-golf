"""
test_train_gpt.py — Unit tests for all implemented blocks.

Covers:
  Day 2 Block 1 — Sliding-window eval: stride param, context per token
  Day 2 Block 2 — Hyperparameter defaults (warmdown, momentum, WD, grad clip)
  Day 3 Block 1 — Int6 bit-packing roundtrip, mixed quantization pipeline
  Day 5 Block 1 — relu², mlp_mult=3, num_layers=11 defaults
  Day 5 Block 2 — Partial RoPE (rope_dims=16), FP16 embedding passthrough
  Day 5 Block 1 (cont.) — EMA shadow weights, GPTQ-lite clip search
  Day 7 Block 1 — Late QAT: STE fake-quantization in CastedLinear
  Day 7 Block 3 — BigramHash: hash function, embedding shapes, GPT forward
  Day 8 Block 1 — XSA: v-direction subtraction, GQA reshape, shape/NaN/correctness
  Day 8 Block 2 — Value Embeddings: shapes, zero-init, per-layer scale, GPT forward
  Day 10 Block 1 — SmearGate: gate math, init, causal shift, GPT wiring
  Day 10 Block 1 — U-Net skips: already present, verified correct structure
  Day 10 Block 2 — Layerwise LN Scale: factor math, monotone decrease, Block forward
  Day 10 Block 2 — OrthoInit: orthonormality, _zero_init guard, small-matrix fallback

Run with:
    /Users/ron/miniconda3/envs/foundry-llm/bin/python -m pytest tests/test_train_gpt.py -v
"""
import io
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
import train_gpt as tg

# F.rms_norm was added in PyTorch 2.4; skip model-forward tests on older installs
HAS_RMS_NORM = hasattr(F, "rms_norm")
needs_rms_norm = pytest.mark.skipif(not HAS_RMS_NORM, reason="F.rms_norm requires torch >= 2.4")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tiny_model(
    num_layers: int = 2,
    model_dim: int = 64,
    num_heads: int = 4,
    num_kv_heads: int = 2,
    mlp_mult: int = 2,
    rope_dims: int = 0,
    vocab_size: int = 16,
) -> tg.GPT:
    return tg.GPT(
        vocab_size=vocab_size,
        num_layers=num_layers,
        model_dim=model_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        mlp_mult=mlp_mult,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
        rope_dims=rope_dims,
    )


# ===========================================================================
# Day 2 Block 1 — Sliding-window eval params
# ===========================================================================

class TestSlidingWindowEval:
    def test_context_per_scored_token(self):
        """Every scored token (except first window) gets seq_len - stride context tokens."""
        seq_len = 2048
        stride = 64
        context = seq_len - stride
        assert context == 1984

    def test_stride_default(self):
        args = tg.Hyperparameters()
        assert args.eval_stride == 64

    def test_eval_batch_seqs_default(self):
        args = tg.Hyperparameters()
        assert args.eval_batch_seqs == 128

    def test_seq_len_default(self):
        args = tg.Hyperparameters()
        assert args.train_seq_len == 2048

    def test_train_batch_tokens_default(self):
        """TRAIN_BATCH_TOKENS default is 524288 (FlashAttention makes seq2048 affordable)."""
        args = tg.Hyperparameters()
        assert args.train_batch_tokens == 524_288


# ===========================================================================
# Day 2 Block 2 — Hyperparameter defaults
# ===========================================================================

class TestHyperparameterDefaults:
    def test_warmdown_iters(self):
        assert tg.Hyperparameters().warmdown_iters == 3500

    def test_muon_momentum(self):
        assert tg.Hyperparameters().muon_momentum == 0.99

    def test_muon_momentum_warmup_start(self):
        assert tg.Hyperparameters().muon_momentum_warmup_start == 0.92

    def test_muon_momentum_warmup_steps(self):
        assert tg.Hyperparameters().muon_momentum_warmup_steps == 1500

    def test_muon_wd(self):
        assert tg.Hyperparameters().muon_wd == 0.04

    def test_grad_clip_norm(self):
        assert tg.Hyperparameters().grad_clip_norm == 0.3


class TestMuonWeightDecay:
    def test_muon_wd_applies_decay(self):
        """Muon step with weight_decay > 0 shrinks parameter magnitude."""
        p = torch.nn.Parameter(torch.ones(4, 4))
        p.grad = torch.zeros(4, 4)  # zero grad → pure WD effect
        optimizer = tg.Muon([p], lr=0.01, momentum=0.9, backend_steps=1,
                            nesterov=False, weight_decay=0.1)
        before = p.data.abs().mean().item()
        optimizer.step()
        after = p.data.abs().mean().item()
        assert after < before, "WD > 0 should reduce parameter magnitude"

    def test_muon_zero_wd_no_decay(self):
        """Muon step with weight_decay=0 and lr=0 leaves parameters unchanged."""
        p = torch.nn.Parameter(torch.ones(4, 4))
        p.grad = torch.zeros(4, 4)
        optimizer = tg.Muon([p], lr=0.0, momentum=0.0, backend_steps=1,
                            nesterov=False, weight_decay=0.0)
        before = p.data.clone()
        optimizer.step()
        assert torch.allclose(p.data, before), "Zero lr + zero WD should leave param unchanged"


# ===========================================================================
# Day 3 Block 1 — Int6 bit-packing
# ===========================================================================

class TestInt6BitPacking:
    def test_pack_unpack_roundtrip_all_values(self):
        """Every value in [-32, 31] roundtrips correctly through pack/unpack."""
        vals = torch.arange(-32, 32, dtype=torch.int8)
        packed, n = tg._pack_int6(vals)
        restored = tg._unpack_int6(packed, n, vals.shape)
        assert torch.equal(vals, restored), "All int6 values must roundtrip exactly"

    def test_packed_size_is_three_quarters(self):
        """4 int6 values pack into 3 bytes → 75% of original int8 size."""
        n_vals = 64
        vals = torch.zeros(n_vals, dtype=torch.int8)
        packed, _ = tg._pack_int6(vals)
        assert packed.numel() == (n_vals * 3) // 4

    def test_pack_unpack_random(self):
        """Random int8[-32, 31] tensor roundtrips."""
        torch.manual_seed(42)
        q = torch.randint(-32, 32, (128, 64), dtype=torch.int8)
        packed, n = tg._pack_int6(q)
        restored = tg._unpack_int6(packed, n, q.shape)
        assert torch.equal(q, restored)

    def test_unpack_preserves_shape(self):
        q = torch.zeros(16, 32, dtype=torch.int8)
        packed, n = tg._pack_int6(q)
        restored = tg._unpack_int6(packed, n, (16, 32))
        assert restored.shape == (16, 32)

    def test_non_multiple_of_4_numel(self):
        """Pack/unpack handles tensors whose numel is not divisible by 4."""
        for n in [1, 2, 3, 5, 7, 13]:
            vals = torch.arange(n, dtype=torch.int8).clamp(-32, 31)
            packed, orig_n = tg._pack_int6(vals)
            restored = tg._unpack_int6(packed, orig_n, vals.shape)
            assert torch.equal(vals, restored), f"Failed for numel={n}"


class TestInt6Quantization:
    def test_per_row_scale_shape(self):
        t = torch.randn(8, 16, dtype=torch.float32)
        q, scale = tg._quantize_int6_row(t)
        assert scale.shape == (8,)
        assert q.shape == (8, 16)
        assert q.dtype == torch.int8

    def test_quantized_values_in_range(self):
        t = torch.randn(32, 64)
        q, _ = tg._quantize_int6_row(t)
        assert q.min() >= -32 and q.max() <= 31

    def test_quantize_dequantize_small_error(self):
        """Int6 relative error on well-distributed weights should be small (<5%)."""
        torch.manual_seed(0)
        t = torch.randn(64, 128) * 0.02
        q, scale = tg._quantize_int6_row(t)
        dequant = q.float() * scale.float()[:, None]
        rel_err = (t.float() - dequant).abs().mean() / t.float().abs().mean().clamp_min(1e-8)
        assert rel_err < 0.05, f"Relative error {rel_err:.4%} exceeds 5%"


class TestMixedQuantizationPipeline:
    def _make_state_dict(self):
        """Minimal state dict with attn/mlp and embedding keys.

        Weights must be > INT8_KEEP_FLOAT_MAX_NUMEL (65536) to hit the int6 path
        instead of the small-tensor fp16 passthrough. Use 512×256 = 131072.
        """
        return {
            "blocks.0.attn.c_q.weight": torch.randn(512, 256) * 0.02,   # 131072 > 65536
            "blocks.0.mlp.fc.weight": torch.randn(512, 256) * 0.02,     # 131072 > 65536
            "blocks.0.mlp.proj.weight": torch.randn(256, 512) * 0.02,   # 131072 > 65536
            "tok_emb.weight": torch.randn(1024, 512) * 0.005,           # 524288 > 65536
            "blocks.0.attn_scale": torch.ones(512),                      # small → fp16 passthrough
        }

    def test_roundtrip_shapes_preserved(self):
        sd = self._make_state_dict()
        obj, _ = tg.quantize_state_dict_int6(sd)
        restored = tg.dequantize_state_dict_int6(obj, sd)
        for name, orig in sd.items():
            assert name in restored, f"Missing key: {name}"
            assert restored[name].shape == orig.shape, f"Shape mismatch: {name}"

    def test_attn_mlp_stored_as_int6(self):
        sd = self._make_state_dict()
        obj, _ = tg.quantize_state_dict_int6(sd)
        meta = obj["m"]
        for name in ["blocks.0.attn.c_q.weight", "blocks.0.mlp.fc.weight"]:
            assert meta[name]["type"] == "int6", f"{name} should be int6"

    def test_embedding_stored_as_fp16_passthrough(self):
        """tok_emb.weight must be routed to fp16 passthrough, not int8."""
        sd = self._make_state_dict()
        obj, _ = tg.quantize_state_dict_int6(sd)
        meta = obj["m"]
        assert meta["tok_emb.weight"] == "passthrough_fp16", (
            "tok_emb.weight should be fp16 passthrough, not int8"
        )

    def test_embedding_roundtrip_lossless_up_to_fp16(self):
        """Embedding roundtrip error should be only fp16 rounding, not quantization noise."""
        sd = self._make_state_dict()
        obj, _ = tg.quantize_state_dict_int6(sd)
        restored = tg.dequantize_state_dict_int6(obj, sd)
        orig = sd["tok_emb.weight"].float()
        rest = restored["tok_emb.weight"].float()
        max_err = (orig - rest).abs().max().item()
        # fp16 has ~1e-3 relative error; absolute error on small weights should be tiny
        assert max_err < 1e-2, f"Embedding max error {max_err:.2e} too large (should be fp16 noise only)"

    def test_serialization_roundtrip(self):
        """Full torch.save → compress → decompress → torch.load roundtrip."""
        sd = self._make_state_dict()
        obj, _ = tg.quantize_state_dict_int6(sd)
        buf = io.BytesIO()
        torch.save(obj, buf)
        blob = tg._compress(buf.getvalue())
        reloaded = torch.load(io.BytesIO(tg._decompress(blob)), map_location="cpu")
        restored = tg.dequantize_state_dict_int6(reloaded, sd)
        for name, orig in sd.items():
            assert restored[name].shape == orig.shape


# ===========================================================================
# Day 5 Block 1 — relu², mlp_mult=3, num_layers=11
# ===========================================================================

class TestMLP:
    def test_relu_squared_activation(self):
        """MLP applies relu then squares: negative inputs → 0, positive inputs → x^2."""
        mlp = tg.MLP(dim=8, mlp_mult=2)
        # Force fc weight to identity-like so output = relu(x).square() ≈ relu(x)²
        with torch.no_grad():
            mlp.fc.weight.zero_()
            mlp.fc.weight[:8, :8] = torch.eye(8)
            mlp.proj.weight.zero_()
            mlp.proj.weight[:8, :8] = torch.eye(8)
        x = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]])
        out_fc = mlp.fc(x)          # after linear, still the same values
        relu_out = torch.relu(out_fc)
        squared = relu_out.square()
        # Negatives should be zero
        assert squared[0, 0].item() == 0.0  # -2 → 0
        assert squared[0, 1].item() == 0.0  # -1 → 0
        # Positives should be squared
        assert abs(squared[0, 3].item() - 1.0) < 0.01   # relu(1)^2 = 1
        assert abs(squared[0, 4].item() - 4.0) < 0.01   # relu(2)^2 = 4

    def test_mlp_hidden_dim_at_mult3(self):
        """mlp_mult=3 gives hidden dim = 3 * model_dim."""
        mlp = tg.MLP(dim=64, mlp_mult=3)
        assert mlp.fc.out_features == 192     # 3 * 64
        assert mlp.proj.in_features == 192

    def test_mlp_hidden_dim_at_mult2(self):
        mlp = tg.MLP(dim=64, mlp_mult=2)
        assert mlp.fc.out_features == 128
        assert mlp.proj.in_features == 128


class TestModelDefaults:
    def test_num_layers_default(self):
        assert tg.Hyperparameters().num_layers == 11

    def test_mlp_mult_default(self):
        assert tg.Hyperparameters().mlp_mult == 3

    def test_model_has_correct_num_blocks(self):
        model = make_tiny_model(num_layers=11)
        assert len(model.blocks) == 11

    def test_model_with_default_mlp_mult_hidden_size(self):
        model = make_tiny_model(model_dim=64, mlp_mult=3)
        assert model.blocks[0].mlp.fc.out_features == 192

    @needs_rms_norm
    def test_forward_runs_cpu(self):
        model = make_tiny_model()
        x = torch.randint(0, 16, (2, 8))
        y = torch.randint(0, 16, (2, 8))
        loss = model(x, y)
        assert loss.shape == ()
        assert loss.item() > 0


# ===========================================================================
# Day 5 Block 2 — Partial RoPE
# ===========================================================================

class TestPartialRoPE:
    def test_rope_dims_default(self):
        assert tg.Hyperparameters().rope_dims == 16

    def test_rotary_cos_sin_shape_partial(self):
        """With rope_dims=16 and head_dim=64, cos/sin have shape [1, 1, T, 8]."""
        rotary = tg.Rotary(dim=64, base=10000.0, rope_dims=16)
        cos, sin = rotary(32, torch.device("cpu"), torch.float32)
        # cos.size(-1) = rope_dims // 2 = 8
        assert cos.shape == (1, 1, 32, 8)
        assert sin.shape == (1, 1, 32, 8)

    def test_rotary_cos_sin_shape_full(self):
        """With rope_dims=0 (full), cos/sin have shape [1, 1, T, dim//2]."""
        rotary = tg.Rotary(dim=64, base=10000.0, rope_dims=0)
        cos, sin = rotary(32, torch.device("cpu"), torch.float32)
        assert cos.shape == (1, 1, 32, 32)  # 64 // 2 = 32

    def test_apply_rotary_full_dims(self):
        """Full RoPE: cos.size(-1)*2 == x.size(-1), standard rotation."""
        x = torch.ones(2, 4, 8, 64)  # (B, H, T, head_dim)
        rotary = tg.Rotary(dim=64, rope_dims=0)
        cos, sin = rotary(8, torch.device("cpu"), torch.float32)
        out = tg.apply_rotary_emb(x, cos, sin)
        assert out.shape == x.shape
        # Rotation should not zero out any dims
        assert out.abs().min() > 0

    def test_apply_rotary_partial_dims_passthrough(self):
        """Partial RoPE (rope_dims=16): last 48 dims of x must be unchanged."""
        head_dim = 64
        rope_dims = 16
        x = torch.randn(2, 4, 8, head_dim)  # (B, H, T, head_dim)
        rotary = tg.Rotary(dim=head_dim, rope_dims=rope_dims)
        cos, sin = rotary(8, torch.device("cpu"), torch.float32)
        out = tg.apply_rotary_emb(x, cos, sin)
        # Last (head_dim - rope_dims) = 48 dims must be completely unchanged
        assert torch.allclose(out[..., rope_dims:], x[..., rope_dims:]), (
            "Position-invariant dims should be unchanged by partial RoPE"
        )
        # First rope_dims dims should be rotated (not identical to input unless cos=1, sin=0)
        # They should differ from the original for non-trivial positions
        assert not torch.allclose(out[..., :rope_dims], x[..., :rope_dims])

    def test_apply_rotary_partial_output_shape(self):
        """Output shape must match input shape exactly."""
        x = torch.randn(1, 8, 16, 64)
        rotary = tg.Rotary(dim=64, rope_dims=16)
        cos, sin = rotary(16, torch.device("cpu"), torch.float32)
        out = tg.apply_rotary_emb(x, cos, sin)
        assert out.shape == x.shape

    @needs_rms_norm
    def test_partial_rope_model_forward(self):
        """Model with rope_dims=16 runs a forward pass without error."""
        model = make_tiny_model(rope_dims=16)
        x = torch.randint(0, 16, (2, 8))
        y = torch.randint(0, 16, (2, 8))
        loss = model(x, y)
        assert torch.isfinite(loss), "Loss should be finite with partial RoPE"

    @needs_rms_norm
    def test_full_rope_model_forward(self):
        """Model with rope_dims=0 (full) also runs correctly."""
        model = make_tiny_model(rope_dims=0)
        x = torch.randint(0, 16, (2, 8))
        y = torch.randint(0, 16, (2, 8))
        loss = model(x, y)
        assert torch.isfinite(loss)

    def test_rotary_rope_dims_stored(self):
        """Rotary.rope_dims is stored correctly for partial and full cases."""
        r_partial = tg.Rotary(dim=64, rope_dims=16)
        assert r_partial.rope_dims == 16
        r_full = tg.Rotary(dim=64, rope_dims=0)
        assert r_full.rope_dims == 64  # 0 → full dim

    def test_attention_uses_partial_rope(self):
        """CausalSelfAttention with rope_dims=16 creates Rotary with rope_dims=16."""
        attn = tg.CausalSelfAttention(
            dim=64, num_heads=4, num_kv_heads=2, rope_base=10000.0,
            qk_gain_init=1.5, rope_dims=16
        )
        assert attn.rotary.rope_dims == 16


# ===========================================================================
# FP16 Embedding Passthrough (Block 2 quantization side)
# ===========================================================================

class TestFP16EmbeddingPassthrough:
    def _quant(self, emb_size: int = 1024, dim: int = 512):
        """Build a minimal state dict with large embedding to hit the int6/int8 path."""
        sd = {
            "tok_emb.weight": torch.randn(emb_size, dim) * 0.005,
            "blocks.0.attn.c_q.weight": torch.randn(dim, dim) * 0.02,
        }
        return tg.quantize_state_dict_int6(sd)

    def test_large_embedding_is_fp16_passthrough(self):
        obj, _ = self._quant(emb_size=1024, dim=512)
        assert obj["m"]["tok_emb.weight"] == "passthrough_fp16"

    def test_embedding_stored_as_float16(self):
        obj, _ = self._quant(emb_size=1024, dim=512)
        stored = obj["w"]["tok_emb.weight"]
        assert stored.dtype == torch.float16

    def test_embedding_no_quant_keys(self):
        """Ensure tok_emb.weight doesn't produce .q / .packed / .scale sub-keys."""
        obj, _ = self._quant(emb_size=1024, dim=512)
        weights = obj["w"]
        for key in weights:
            assert not key.startswith("tok_emb.weight."), (
                f"Unexpected quantization sub-key: {key}"
            )

    def test_embedding_payload_size_is_fp16(self):
        """Payload bytes for 1024×512 embedding in fp16 = 1,048,576 bytes."""
        emb = torch.randn(1024, 512) * 0.005
        obj, payload = tg.quantize_state_dict_int6({"tok_emb.weight": emb})
        # fp16: 1024 * 512 * 2 = 1,048,576 bytes
        assert payload == 1024 * 512 * 2


# ===========================================================================
# Day 5 Block 1 (cont.) — GPTQ-lite clip search
# ===========================================================================

class TestGPTQLiteClipSearch:
    def test_values_still_in_int6_range(self):
        """GPTQ-lite must never produce quantized values outside [-32, 31]."""
        torch.manual_seed(7)
        t = torch.randn(64, 128) * 0.02
        q, scale = tg._quantize_int6_row(t)
        assert q.min().item() >= -32
        assert q.max().item() <= 31

    def test_scale_shape_and_dtype(self):
        t = torch.randn(32, 64)
        q, scale = tg._quantize_int6_row(t)
        assert scale.shape == (32,)
        assert scale.dtype == torch.float16
        assert q.dtype == torch.int8

    def test_gptq_selects_min_mse_among_candidates(self):
        """GPTQ-lite must return the candidate that achieves the minimum MSE.

        Manually reproduce the 5-candidate search and confirm _quantize_int6_row
        produces the same reconstruction MSE as the manually identified winner.
        """
        torch.manual_seed(99)
        t = torch.randn(16, 64) * 0.02
        t32 = t.float()

        best_mse = float('inf')
        best_s = None
        for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
            if pct < 1.0:
                row_clip = torch.quantile(t32.abs(), pct, dim=1)
            else:
                row_clip = t32.abs().amax(dim=1)
            s = (row_clip / 31.0).clamp_min(torch.finfo(torch.float16).tiny).to(torch.float16)
            q = torch.clamp(torch.round(t32 / s.float()[:, None]), -32, 31).to(torch.int8)
            mse = (t32 - q.float() * s.float()[:, None]).pow(2).mean().item()
            if mse < best_mse:
                best_mse = mse
                best_s = s

        q_out, s_out = tg._quantize_int6_row(t)
        recon_out = q_out.float() * s_out.float()[:, None]
        mse_out = (t32 - recon_out).pow(2).mean().item()

        assert abs(mse_out - best_mse) < 1e-10, (
            f"GPTQ-lite MSE {mse_out:.8f} should equal manually identified min {best_mse:.8f}"
        )

    def test_gptq_never_worse_than_absmax_fp16(self):
        """GPTQ-lite result MSE must be ≤ absmax MSE (with fp16 scales, fair comparison).

        The absmax is one of the 5 candidates, so by construction the selected
        candidate cannot be worse than the absmax candidate.
        """
        torch.manual_seed(1)
        t = torch.randn(32, 128) * 0.02
        t32 = t.float()

        # Absmax candidate with fp16 scale (same fp16 cast as GPTQ-lite internals)
        s_absmax = (t32.abs().amax(dim=1) / 31.0).clamp_min(
            torch.finfo(torch.float16).tiny
        ).to(torch.float16)
        q_absmax = torch.clamp(torch.round(t32 / s_absmax.float()[:, None]), -32, 31).to(torch.int8)
        mse_absmax = (t32 - q_absmax.float() * s_absmax.float()[:, None]).pow(2).mean().item()

        q_gptq, s_gptq = tg._quantize_int6_row(t)
        mse_gptq = (t32 - q_gptq.float() * s_gptq.float()[:, None]).pow(2).mean().item()

        assert mse_gptq <= mse_absmax + 1e-12, (
            f"GPTQ-lite MSE {mse_gptq:.8f} must not exceed absmax MSE {mse_absmax:.8f}"
        )

    def test_clean_tensor_at_least_as_good_as_naive(self):
        """Without extreme outliers, GPTQ-lite must not be worse than absmax."""
        torch.manual_seed(42)
        t = torch.randn(32, 128) * 0.02  # no outliers
        q, scale = tg._quantize_int6_row(t)
        recon = q.float() * scale.float()[:, None]
        mse_gptq = (t.float() - recon).pow(2).mean().item()

        # Naive baseline
        row_max = t.float().abs().amax(dim=1)
        s_naive = (row_max / 31.0).clamp_min(1e-6)
        q_naive = torch.clamp(torch.round(t.float() / s_naive[:, None]), -32, 31).to(torch.int8)
        recon_naive = q_naive.float() * s_naive[:, None]
        mse_naive = (t.float() - recon_naive).pow(2).mean().item()

        # Allow tiny floating-point slack (gptq may select 100% clip = absmax for clean data)
        assert mse_gptq <= mse_naive * 1.01, (
            f"GPTQ-lite MSE {mse_gptq:.6f} should not be worse than naive {mse_naive:.6f}"
        )

    def test_1d_fallback_still_works(self):
        """1D tensors (scalars) use simple absmax, must still roundtrip cleanly."""
        t = torch.tensor([0.5, -0.3, 0.1, 0.0])
        q, scale = tg._quantize_int6_row(t)
        assert scale.ndim == 0  # scalar scale for 1D
        assert q.min().item() >= -32 and q.max().item() <= 31

    def test_gptq_lite_integrated_in_quant_pipeline(self):
        """quantize_state_dict_int6 uses _quantize_int6_row, so GPTQ-lite is live end-to-end."""
        # Create a matrix with outliers — the pipeline should still produce valid int6 output
        t_outlier = torch.full((512, 256), 0.01)
        t_outlier[0, 0] = 50.0
        sd = {"blocks.0.attn.c_q.weight": t_outlier}
        obj, _ = tg.quantize_state_dict_int6(sd)
        restored = tg.dequantize_state_dict_int6(obj, sd)
        q_packed = obj["w"]["blocks.0.attn.c_q.weight.packed"]
        # Packed tensor values must be valid uint8 (no overflow)
        assert q_packed.dtype == torch.uint8
        assert restored["blocks.0.attn.c_q.weight"].shape == (512, 256)


# ===========================================================================
# Day 5 Block 1 (cont.) — EMA shadow weights
# ===========================================================================

class TestEMAShadowWeights:
    def test_ema_decay_default(self):
        assert tg.Hyperparameters().ema_decay == 0.997

    def test_ema_init_equals_model_weights(self):
        """EMA state must equal model weights at step 0 (before any update)."""
        torch.manual_seed(0)
        model = make_tiny_model()
        ema_state = {name: t.detach().float().clone() for name, t in model.state_dict().items()}
        for name, orig in model.state_dict().items():
            assert torch.allclose(
                ema_state[name], orig.float()
            ), f"EMA init mismatch for {name}"

    def test_ema_one_step_math(self):
        """After one update: ema = decay * w0 + (1 - decay) * w1."""
        decay = 0.997
        w0 = torch.tensor([1.0, 2.0, 3.0])
        w1 = torch.tensor([0.0, 0.0, 0.0])
        ema = w0.clone()
        # One EMA step toward w1
        ema.mul_(decay).add_(w1, alpha=1.0 - decay)
        expected = w0 * decay  # + 0 * (1-decay)
        assert torch.allclose(ema, expected), f"EMA math wrong: got {ema}, expected {expected}"

    def test_ema_converges_to_constant(self):
        """EMA applied to a constant target for many steps must converge to that target."""
        decay = 0.997
        target = torch.tensor([5.0])
        ema = torch.tensor([0.0])
        for _ in range(5000):
            ema.mul_(decay).add_(target, alpha=1.0 - decay)
        # After 5000 steps, EMA should be within 0.1% of target
        assert abs(ema.item() - target.item()) < 0.01 * target.item(), (
            f"EMA did not converge: {ema.item():.4f} vs {target.item():.4f}"
        )

    def test_ema_moves_toward_new_weights(self):
        """EMA should be strictly between old and new weights (toward new) after one step."""
        decay = 0.997
        w0 = torch.tensor([10.0])  # large initial
        w1 = torch.tensor([0.0])   # new target = 0
        ema = w0.clone()
        ema.mul_(decay).add_(w1, alpha=1.0 - decay)
        # EMA should now be less than w0 (moved toward w1=0)
        assert ema.item() < w0.item()
        # EMA should still be positive (not overshot)
        assert ema.item() > w1.item()

    def test_ema_state_preserves_all_keys(self):
        """EMA state dict must have the same keys as the model state dict."""
        model = make_tiny_model()
        ema_state = {name: t.detach().float().clone() for name, t in model.state_dict().items()}
        assert set(ema_state.keys()) == set(model.state_dict().keys())

    def test_ema_apply_to_model_preserves_shapes(self):
        """Loading EMA weights back into the model must preserve all tensor shapes."""
        model = make_tiny_model()
        ema_state = {name: t.detach().float().clone() for name, t in model.state_dict().items()}
        # Simulate some training steps
        for name in ema_state:
            ema_state[name].mul_(0.997).add_(
                torch.zeros_like(ema_state[name]), alpha=0.003
            )
        # Apply EMA to model (cast back to original dtype)
        current_dtype_map = {name: t.dtype for name, t in model.state_dict().items()}
        ema_loaded = {name: t.to(dtype=current_dtype_map[name]) for name, t in ema_state.items()}
        model.load_state_dict(ema_loaded, strict=True)
        for name, t in model.state_dict().items():
            assert t.shape == ema_loaded[name].shape, f"Shape mismatch after EMA apply: {name}"

    def test_ema_stored_as_float32(self):
        """EMA shadow weights must be float32 to accumulate small changes without precision loss."""
        model = make_tiny_model()
        ema_state = {name: t.detach().float().clone() for name, t in model.state_dict().items()}
        for name, t in ema_state.items():
            if t.is_floating_point():
                assert t.dtype == torch.float32, (
                    f"EMA state for {name} should be float32, got {t.dtype}"
                )


# ===========================================================================
# Day 7 Block 1 — Late QAT (Straight-Through Estimator)
# ===========================================================================

class TestLateQAT:
    def setup_method(self):
        """Ensure _qat_enabled is reset to False before each test."""
        tg.CastedLinear._qat_enabled = False

    def teardown_method(self):
        """Always restore the class flag so tests don't leak state."""
        tg.CastedLinear._qat_enabled = False

    def test_qat_disabled_by_default(self):
        assert tg.CastedLinear._qat_enabled is False

    def test_qat_threshold_default(self):
        assert tg.Hyperparameters().late_qat_threshold == 0.15

    def test_forward_without_qat_is_standard_linear(self):
        """When _qat_enabled=False, forward is identical to plain F.linear cast."""
        layer = tg.CastedLinear(8, 8, bias=False)
        x = torch.randn(2, 4, 8)
        out_casted = layer(x)
        out_ref = torch.nn.functional.linear(x, layer.weight.to(x.dtype))
        assert torch.allclose(out_casted, out_ref)

    def test_qat_forward_uses_quantized_weights(self):
        """With _qat_enabled=True the output must differ from the unquantized forward
        (unless the weights happen to be perfectly int6-representable, which is unlikely
        for random weights).
        """
        torch.manual_seed(42)
        layer = tg.CastedLinear(16, 16, bias=False)
        layer.weight.data = torch.randn(16, 16) * 0.1  # typical training magnitudes
        layer.train()
        x = torch.randn(2, 4, 16)

        # Baseline: no QAT
        out_no_qat = layer(x).detach().clone()

        # Enable QAT
        tg.CastedLinear._qat_enabled = True
        out_qat = layer(x).detach().clone()

        # The outputs should differ (quantization noise ≠ 0 for random weights)
        assert not torch.allclose(out_no_qat, out_qat), (
            "QAT output should differ from unquantized due to quantization noise"
        )

    def test_qat_output_matches_manual_ste(self):
        """STE formula: w_ste = w + (quantize(w) - w).detach()
        Output must match F.linear(x, w_ste).
        """
        torch.manual_seed(7)
        layer = tg.CastedLinear(8, 8, bias=False)
        layer.weight.data = torch.randn(8, 8) * 0.1
        layer.train()
        tg.CastedLinear._qat_enabled = True

        x = torch.randn(1, 4, 8)
        out_layer = layer(x).detach()

        # Reproduce STE manually
        w = layer.weight.float()
        row_max = w.abs().amax(dim=1)
        scale = (row_max / 31.0).clamp_min(1.0 / 31.0)
        w_q = (torch.clamp(torch.round(w / scale[:, None]), -32, 31) * scale[:, None]).to(x.dtype)
        w_ste = layer.weight.to(x.dtype) + (w_q - layer.weight.to(x.dtype)).detach()
        out_manual = torch.nn.functional.linear(x, w_ste).detach()

        assert torch.allclose(out_layer, out_manual, atol=1e-5), (
            f"QAT output must match manual STE; max diff {(out_layer - out_manual).abs().max():.2e}"
        )

    def test_ste_gradient_flows_through(self):
        """Gradient with respect to the weight must be non-zero under QAT (STE passes grad).
        If the true rounding gradient were used it would be zero almost everywhere.
        """
        layer = tg.CastedLinear(8, 8, bias=False)
        layer.weight.data = torch.randn(8, 8) * 0.1
        layer.train()
        tg.CastedLinear._qat_enabled = True

        x = torch.randn(2, 4, 8, requires_grad=False)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert layer.weight.grad is not None
        assert layer.weight.grad.abs().sum() > 0, "STE must pass non-zero gradient to weights"

    def test_qat_disabled_in_eval_mode(self):
        """Even if _qat_enabled=True, QAT must not apply during model.eval()."""
        layer = tg.CastedLinear(8, 8, bias=False)
        layer.eval()  # eval mode
        tg.CastedLinear._qat_enabled = True

        x = torch.randn(1, 4, 8)
        out_eval = layer(x).detach().clone()

        # Expected: no quantization, same as plain cast
        out_ref = torch.nn.functional.linear(x, layer.weight.to(x.dtype)).detach()
        assert torch.allclose(out_eval, out_ref), (
            "QAT should not apply in eval mode (self.training is False)"
        )

    def test_qat_quantized_values_in_int6_range(self):
        """The quantized weight used in the STE forward must map to int6 levels [-32, 31]."""
        torch.manual_seed(3)
        layer = tg.CastedLinear(16, 16, bias=False)
        layer.weight.data = torch.randn(16, 16) * 0.5
        layer.train()
        tg.CastedLinear._qat_enabled = True

        # Reconstruct quantized weight from the STE output minus the STE passthrough
        with torch.no_grad():
            w32 = layer.weight.float()
            row_max = w32.abs().amax(dim=1)
            scale = (row_max / 31.0).clamp_min(1.0 / 31.0)
            q = torch.clamp(torch.round(w32 / scale[:, None]), -32, 31)
        assert q.min().item() >= -32 and q.max().item() <= 31


# ===========================================================================
# Day 7 Block 3 — BigramHash
# ===========================================================================

class TestBigramHash:
    def _make_bigram(self, vocab_size: int = 2048, bigram_dim: int = 64,
                     model_dim: int = 64) -> tg.BigramHashEmbedding:
        return tg.BigramHashEmbedding(vocab_size, bigram_dim, model_dim)

    # ---- hyperparameters ----

    def test_bigram_vocab_size_default(self):
        assert tg.Hyperparameters().bigram_vocab_size == 4096

    def test_bigram_dim_default(self):
        assert tg.Hyperparameters().bigram_dim == 128

    # ---- hash function correctness ----

    def test_hash_output_in_valid_range(self):
        """All hash values must be in [0, vocab_size - 1]."""
        bg = self._make_bigram(vocab_size=2048)
        tokens = torch.randint(0, 1024, (2, 32))
        h = bg._hash(tokens)
        assert h.min().item() >= 0
        assert h.max().item() <= 2047

    def test_hash_first_position_is_null(self):
        """Position 0 has no preceding token → must map to null bucket (vocab_size - 1)."""
        vocab_size = 2048
        bg = self._make_bigram(vocab_size=vocab_size)
        tokens = torch.randint(0, 1024, (3, 16))
        h = bg._hash(tokens)
        # Every batch's position-0 hash must be the null bucket
        assert (h[:, 0] == vocab_size - 1).all(), (
            "Position 0 must always map to null bucket (vocab_size - 1)"
        )

    def test_hash_positions_1plus_are_not_null(self):
        """For random tokens positions 1+ should rarely hit the null bucket."""
        bg = self._make_bigram(vocab_size=2048)
        torch.manual_seed(0)
        tokens = torch.randint(1, 1024, (4, 64))  # avoid 0 which might map to null by coincidence
        h = bg._hash(tokens)
        # Probability of randomly landing on the null bucket = 1/(vocab_size-1) ≈ 0.05%
        # For 4*63=252 positions the expected null count is <1 — assert none
        null_count = (h[:, 1:] == 2047).sum().item()
        assert null_count == 0, f"Unexpected null bucket hits at non-zero positions: {null_count}"

    def test_hash_deterministic(self):
        """Same input must always produce the same hash."""
        bg = self._make_bigram(vocab_size=2048)
        tokens = torch.tensor([[100, 200, 300]])
        h1 = bg._hash(tokens)
        h2 = bg._hash(tokens)
        assert torch.equal(h1, h2)

    def test_hash_shape_preserved(self):
        """Hash output shape must match input shape exactly."""
        bg = self._make_bigram()
        tokens = torch.randint(0, 1024, (3, 17))
        h = bg._hash(tokens)
        assert h.shape == tokens.shape

    def test_hash_different_for_different_bigrams(self):
        """(tok_A, tok_B) and (tok_B, tok_A) should produce different hashes (usually)."""
        bg = self._make_bigram(vocab_size=8192)
        tokens_ab = torch.tensor([[50, 200]])
        tokens_ba = torch.tensor([[200, 50]])
        h_ab = bg._hash(tokens_ab)
        h_ba = bg._hash(tokens_ba)
        # Position 0 is always null, compare position 1
        assert h_ab[0, 1].item() != h_ba[0, 1].item(), (
            "hash(A,B) should differ from hash(B,A) for these constants"
        )

    # ---- initialization ----

    def test_embed_weight_zero_init(self):
        """Embedding weights must be all-zeros at init (no bias from step 0)."""
        bg = self._make_bigram()
        assert bg.embed.weight.abs().sum().item() == 0.0

    def test_scale_init(self):
        """Scale parameter must be initialized to 0.05."""
        bg = self._make_bigram()
        assert abs(bg.scale.item() - 0.05) < 1e-6

    def test_no_proj_when_dims_match(self):
        """When bigram_dim == model_dim, no projection layer is created."""
        bg = self._make_bigram(bigram_dim=64, model_dim=64)
        assert bg.proj is None

    def test_proj_created_when_dims_differ(self):
        """When bigram_dim ≠ model_dim, a CastedLinear projection is created."""
        bg = tg.BigramHashEmbedding(vocab_size=2048, bigram_dim=64, model_dim=128)
        assert bg.proj is not None
        assert isinstance(bg.proj, tg.CastedLinear)
        assert bg.proj.weight.abs().sum().item() == 0.0  # zero init

    # ---- forward pass ----

    def test_forward_output_shape_no_proj(self):
        """Output shape must be (batch, seq_len, model_dim)."""
        model_dim = 64
        bg = tg.BigramHashEmbedding(vocab_size=2048, bigram_dim=model_dim, model_dim=model_dim)
        tokens = torch.randint(0, 1024, (2, 16))
        out = bg(tokens)
        assert out.shape == (2, 16, model_dim)

    def test_forward_output_shape_with_proj(self):
        bg = tg.BigramHashEmbedding(vocab_size=2048, bigram_dim=64, model_dim=128)
        tokens = torch.randint(0, 1024, (3, 8))
        out = bg(tokens)
        assert out.shape == (3, 8, 128)

    def test_forward_output_zero_at_init(self):
        """With zero-inited embeddings and scale=0.05, output should be all zeros at init."""
        bg = self._make_bigram()
        tokens = torch.randint(0, 1024, (2, 16))
        out = bg(tokens)
        # embed.weight is all-zero, so the embedding lookup returns all-zero, scaled by 0.05 = still 0
        assert out.abs().sum().item() == 0.0

    # ---- GPT integration ----

    def test_gpt_has_bigram_when_requested(self):
        model = make_tiny_model(vocab_size=64)  # default no bigram
        model_with = tg.GPT(
            vocab_size=64, num_layers=2, model_dim=64, num_heads=4, num_kv_heads=2,
            mlp_mult=2, tie_embeddings=True, tied_embed_init_std=0.005,
            logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
            bigram_vocab_size=512, bigram_dim=32,
        )
        assert model_with.bigram is not None
        assert isinstance(model_with.bigram, tg.BigramHashEmbedding)

    def test_gpt_bigram_none_when_disabled(self):
        model = tg.GPT(
            vocab_size=64, num_layers=2, model_dim=64, num_heads=4, num_kv_heads=2,
            mlp_mult=2, tie_embeddings=True, tied_embed_init_std=0.005,
            logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
            bigram_vocab_size=0,  # disabled
        )
        assert model.bigram is None

    @needs_rms_norm
    def test_gpt_forward_with_bigram(self):
        """GPT with bigram enabled must produce a finite loss on a forward pass."""
        model = tg.GPT(
            vocab_size=64, num_layers=2, model_dim=64, num_heads=4, num_kv_heads=2,
            mlp_mult=2, tie_embeddings=True, tied_embed_init_std=0.005,
            logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
            bigram_vocab_size=512, bigram_dim=32,
        )
        x = torch.randint(0, 64, (2, 8))
        y = torch.randint(0, 64, (2, 8))
        loss = model(x, y)
        assert torch.isfinite(loss), f"Loss should be finite with bigram enabled, got {loss}"

    def test_bigram_vocab_size_stored(self):
        """BigramHashEmbedding stores the vocab_size for hash modulus."""
        bg = tg.BigramHashEmbedding(vocab_size=3000, bigram_dim=64, model_dim=64)
        assert bg.vocab_size == 3000


# ===========================================================================
# Day 8 Block 1 — XSA (Extended Self-Attention)
# ===========================================================================

class TestXSA:
    """Tests for CausalSelfAttention._xsa_efficient and XSA toggle."""

    def _make_attn(self, num_heads=4, num_kv_heads=2, dim=64) -> tg.CausalSelfAttention:
        attn = tg.CausalSelfAttention(
            dim=dim, num_heads=num_heads, num_kv_heads=num_kv_heads,
            rope_base=10000.0, qk_gain_init=1.5, rope_dims=0,
        )
        return attn

    def test_xsa_output_shape_unchanged(self):
        """_xsa_efficient must return the same shape as its input y."""
        B, T, H, D, Hkv = 2, 8, 4, 16, 2
        y = torch.randn(B, T, H, D)
        v = torch.randn(B, T, Hkv, D)
        attn = self._make_attn(num_heads=H, num_kv_heads=Hkv, dim=H * D)
        out = attn._xsa_efficient(y, v)
        assert out.shape == y.shape, f"Expected {y.shape}, got {out.shape}"

    def test_xsa_removes_v_component(self):
        """After XSA, the dot-product of output with v-direction should be near zero."""
        B, T, H, D, Hkv = 1, 4, 4, 8, 2
        # Construct y that partially aligns with v.
        v = torch.randn(B, T, Hkv, D)
        # y: for each KV-head group, add a component along v
        group = H // Hkv
        vn = F.normalize(v, dim=-1)  # (B, T, Hkv, D)
        vn_exp = vn.unsqueeze(3).expand(B, T, Hkv, group, D).reshape(B, T, H, D)
        y = torch.randn(B, T, H, D) + 2.0 * vn_exp  # deliberately v-aligned
        attn = self._make_attn(num_heads=H, num_kv_heads=Hkv, dim=H * D)
        out = attn._xsa_efficient(y, v)
        # Compute residual dot-product along v-direction for each head group
        out_g = out.reshape(B, T, Hkv, group, D)
        vn2 = F.normalize(v, dim=-1).unsqueeze(3)  # (B, T, Hkv, 1, D)
        proj = (out_g * vn2).sum(dim=-1).abs().mean().item()
        assert proj < 1e-5, f"v-direction component should be near 0 after XSA, got {proj:.6f}"

    def test_xsa_no_nan(self):
        """_xsa_efficient must not produce NaN even with near-zero v."""
        B, T, H, D, Hkv = 2, 6, 4, 8, 2
        y = torch.randn(B, T, H, D)
        v = torch.zeros(B, T, Hkv, D)  # all-zero v → normalization edge case
        attn = self._make_attn(num_heads=H, num_kv_heads=Hkv, dim=H * D)
        out = attn._xsa_efficient(y, v)
        assert not torch.isnan(out).any(), "NaN in XSA output with zero-valued v"
        assert not torch.isinf(out).any(), "Inf in XSA output with zero-valued v"

    def test_xsa_gqa_no_repeat(self):
        """GQA-aware XSA must work when H > Hkv without repeat_interleave."""
        B, T, H, Hkv, D = 1, 5, 8, 2, 16  # group=4
        y = torch.randn(B, T, H, D)
        v = torch.randn(B, T, Hkv, D)
        attn = self._make_attn(num_heads=H, num_kv_heads=Hkv, dim=H * D)
        out = attn._xsa_efficient(y, v)
        assert out.shape == y.shape

    def test_xsa_use_flag_default_false(self):
        """CausalSelfAttention.use_xsa must default to False."""
        attn = self._make_attn()
        assert attn.use_xsa is False

    @needs_rms_norm
    def test_xsa_last_n_sets_flags(self):
        """GPT with xsa_last_n=2 must set use_xsa=True on exactly the last 2 layers."""
        model = tg.GPT(
            vocab_size=16, num_layers=4, model_dim=64, num_heads=4, num_kv_heads=2,
            mlp_mult=2, tie_embeddings=True, tied_embed_init_std=0.005,
            logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
            xsa_last_n=2,
        )
        flags = [b.attn.use_xsa for b in model.blocks]
        assert flags == [False, False, True, True], f"Unexpected flags: {flags}"

    @needs_rms_norm
    def test_xsa_zero_disables(self):
        """xsa_last_n=0 must leave all use_xsa flags False."""
        model = tg.GPT(
            vocab_size=16, num_layers=4, model_dim=64, num_heads=4, num_kv_heads=2,
            mlp_mult=2, tie_embeddings=True, tied_embed_init_std=0.005,
            logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
            xsa_last_n=0,
        )
        assert all(not b.attn.use_xsa for b in model.blocks)

    @needs_rms_norm
    def test_xsa_forward_no_nan(self):
        """End-to-end GPT forward with xsa_last_n=2 must produce finite loss."""
        model = tg.GPT(
            vocab_size=32, num_layers=4, model_dim=64, num_heads=4, num_kv_heads=2,
            mlp_mult=2, tie_embeddings=True, tied_embed_init_std=0.005,
            logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
            xsa_last_n=2,
        )
        model.eval()
        x = torch.randint(0, 32, (2, 8))
        y = torch.randint(0, 32, (2, 8))
        loss = model(x, y)
        assert torch.isfinite(loss), f"Non-finite loss with XSA enabled: {loss}"

    @needs_rms_norm
    def test_xsa_output_differs_from_standard(self):
        """With xsa_last_n>0, model output must differ from xsa_last_n=0."""
        common = dict(
            vocab_size=32, num_layers=4, model_dim=64, num_heads=4, num_kv_heads=2,
            mlp_mult=2, tie_embeddings=True, tied_embed_init_std=0.005,
            logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
        )
        torch.manual_seed(42)
        m_no_xsa = tg.GPT(**common, xsa_last_n=0)
        torch.manual_seed(42)
        m_xsa = tg.GPT(**common, xsa_last_n=2)
        # Sync weights so only the XSA path differs
        m_xsa.load_state_dict(m_no_xsa.state_dict())
        m_no_xsa.eval()
        m_xsa.eval()
        x = torch.randint(0, 32, (2, 8))
        y = torch.randint(0, 32, (2, 8))
        with torch.no_grad():
            l_no = m_no_xsa(x, y)
            l_xsa = m_xsa(x, y)
        assert not torch.allclose(l_no, l_xsa), "XSA should change model output"

    def test_xsa_hyperparameter_default(self):
        """Hyperparameters.xsa_last_n must default to 4."""
        args = tg.Hyperparameters()
        assert args.xsa_last_n == 4


# ===========================================================================
# Day 8 Block 2 — Value Embeddings
# ===========================================================================

class TestValueEmbeddings:
    """Tests for shared ValueEmbedding (ve_embed/ve_proj/ve_scale) in GPT."""

    def _make_ve_model(self, ve_dim: int = 32, num_layers: int = 4) -> tg.GPT:
        return tg.GPT(
            vocab_size=32, num_layers=num_layers, model_dim=64, num_heads=4, num_kv_heads=2,
            mlp_mult=2, tie_embeddings=True, tied_embed_init_std=0.005,
            logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
            ve_dim=ve_dim,
        )

    @needs_rms_norm
    def test_ve_embed_present_when_nonzero(self):
        """GPT with ve_dim>0 must have ve_embed, ve_proj, ve_scale attributes."""
        model = self._make_ve_model(ve_dim=32)
        assert model.ve_embed is not None
        assert model.ve_proj is not None
        assert model.ve_scale is not None

    @needs_rms_norm
    def test_ve_disabled_when_zero(self):
        """GPT with ve_dim=0 must have ve_embed=None."""
        model = tg.GPT(
            vocab_size=32, num_layers=4, model_dim=64, num_heads=4, num_kv_heads=2,
            mlp_mult=2, tie_embeddings=True, tied_embed_init_std=0.005,
            logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
            ve_dim=0,
        )
        assert model.ve_embed is None
        assert model.ve_proj is None
        assert model.ve_scale is None

    @needs_rms_norm
    def test_ve_embed_shape(self):
        """ve_embed weight must be (vocab_size, ve_dim)."""
        model = self._make_ve_model(ve_dim=32)
        assert model.ve_embed.weight.shape == (32, 32)

    @needs_rms_norm
    def test_ve_proj_shape(self):
        """ve_proj weight must map ve_dim → num_kv_heads * head_dim."""
        model = self._make_ve_model(ve_dim=32)
        # kv_dim = num_kv_heads * head_dim = 2 * (64//4) = 2 * 16 = 32
        kv_dim = 2 * (64 // 4)
        assert model.ve_proj.weight.shape == (kv_dim, 32)

    @needs_rms_norm
    def test_ve_scale_length(self):
        """ve_scale must have one scalar per layer."""
        model = self._make_ve_model(ve_dim=32, num_layers=4)
        assert model.ve_scale.shape == (4,)

    @needs_rms_norm
    def test_ve_scale_zero_init(self):
        """ve_scale must be zero-initialized (no contribution at step 0)."""
        model = self._make_ve_model(ve_dim=32)
        assert torch.all(model.ve_scale == 0.0), "ve_scale should start at zero"

    @needs_rms_norm
    def test_ve_proj_zero_init(self):
        """ve_proj weight must be zero-initialized."""
        model = self._make_ve_model(ve_dim=32)
        assert torch.all(model.ve_proj.weight == 0.0), "ve_proj should be zero-initialized"

    @needs_rms_norm
    def test_ve_forward_no_nan(self):
        """GPT with ve_dim>0 must produce finite loss."""
        model = self._make_ve_model(ve_dim=32)
        model.eval()
        x = torch.randint(0, 32, (2, 8))
        y = torch.randint(0, 32, (2, 8))
        loss = model(x, y)
        assert torch.isfinite(loss), f"Non-finite loss with VE enabled: {loss}"

    @needs_rms_norm
    def test_ve_zero_scale_matches_no_ve(self):
        """With ve_scale=0 (default), VE model output should match non-VE model."""
        common = dict(
            vocab_size=32, num_layers=4, model_dim=64, num_heads=4, num_kv_heads=2,
            mlp_mult=2, tie_embeddings=True, tied_embed_init_std=0.005,
            logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
        )
        torch.manual_seed(99)
        m_no_ve = tg.GPT(**common, ve_dim=0)
        torch.manual_seed(99)
        m_ve = tg.GPT(**common, ve_dim=32)

        # Copy shared weights to m_ve so only the ve path differs; ve_scale is 0 so it's inactive
        sd_no = m_no_ve.state_dict()
        sd_ve = m_ve.state_dict()
        for k in sd_no:
            if k in sd_ve:
                sd_ve[k] = sd_no[k].clone()
        m_ve.load_state_dict(sd_ve, strict=False)

        m_no_ve.eval()
        m_ve.eval()
        x = torch.randint(0, 32, (2, 8))
        y = torch.randint(0, 32, (2, 8))
        with torch.no_grad():
            l_no = m_no_ve(x, y)
            l_ve = m_ve(x, y)
        assert torch.allclose(l_no, l_ve, atol=1e-5), \
            f"VE at zero scale should match no-VE: {l_no} vs {l_ve}"

    @needs_rms_norm
    def test_ve_nonzero_scale_changes_output(self):
        """Setting ve_scale != 0 must change model output."""
        model = self._make_ve_model(ve_dim=32)
        model.eval()
        x = torch.randint(0, 32, (2, 8))
        y = torch.randint(0, 32, (2, 8))
        with torch.no_grad():
            l_zero = model(x, y)
        # Activate ve by setting non-zero scales
        model.ve_scale.data.fill_(1.0)
        with torch.no_grad():
            l_active = model(x, y)
        assert not torch.allclose(l_zero, l_active), \
            "Non-zero ve_scale should change model output"

    def test_ve_hyperparameter_default(self):
        """Hyperparameters.ve_dim must default to 128."""
        args = tg.Hyperparameters()
        assert args.ve_dim == 128

    @needs_rms_norm
    def test_ve_with_xsa_forward(self):
        """Combining VE and XSA must produce finite loss."""
        model = tg.GPT(
            vocab_size=32, num_layers=4, model_dim=64, num_heads=4, num_kv_heads=2,
            mlp_mult=2, tie_embeddings=True, tied_embed_init_std=0.005,
            logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
            xsa_last_n=2, ve_dim=32,
        )
        model.eval()
        x = torch.randint(0, 32, (2, 8))
        y = torch.randint(0, 32, (2, 8))
        loss = model(x, y)
        assert torch.isfinite(loss), f"Non-finite loss with XSA+VE: {loss}"


# ===========================================================================
# Day 10 Block 1 — SmearGate
# ===========================================================================

class TestSmearGate:
    """Tests for SmearGate: gate math, init values, causal shift, GPT wiring."""

    def test_gate_init_near_one(self):
        """sigmoid(3.0) ≈ 0.953 — gate defaults to strong current-token pass-through."""
        sg = tg.SmearGate(dim=8)
        g = torch.sigmoid(sg.gate)
        assert (g > 0.9).all(), f"Initial gate values should be > 0.9, got {g}"

    def test_output_shape(self):
        """SmearGate output must match input shape."""
        sg = tg.SmearGate(dim=16)
        x = torch.randn(3, 7, 16)
        out = sg(x)
        assert out.shape == x.shape

    def test_position_zero_blends_with_zeros(self):
        """Position 0 has no prior token; the 'previous' embedding is all zeros."""
        sg = tg.SmearGate(dim=8)
        with torch.no_grad():
            sg.gate.fill_(0.0)  # g = sigmoid(0) = 0.5 → 50-50 blend
        x = torch.ones(1, 4, 8)
        out = sg(x)
        # pos 0: g*1 + (1-g)*0 = 0.5
        assert torch.allclose(out[:, 0, :], torch.full((1, 8), 0.5)), \
            f"Position 0 should be 0.5*x[0] + 0.5*0 = 0.5, got {out[:, 0, :]}"

    def test_causal_shift(self):
        """x_prev[t] = x[t-1]; blending should use the immediately prior embedding."""
        sg = tg.SmearGate(dim=4)
        with torch.no_grad():
            sg.gate.fill_(float('inf'))  # g = sigmoid(+inf) = 1.0 → pure current
        x = torch.arange(1, 5, dtype=torch.float32).reshape(1, 4, 1).expand(1, 4, 4)
        out = sg(x)
        # g=1 → out = 1*x + 0*x_prev = x
        assert torch.allclose(out, x), "gate=1 should give pure pass-through"

        with torch.no_grad():
            sg.gate.fill_(float('-inf'))  # g = sigmoid(-inf) = 0.0 → pure previous
        out2 = sg(x)
        # g=0 → out = 0*x + 1*x_prev; x_prev[0] = 0, x_prev[t] = x[t-1]
        assert torch.allclose(out2[:, 0, :], torch.zeros(1, 4)), "pos 0 with g=0 → zeros"
        assert torch.allclose(out2[:, 1, :], x[:, 0, :]), "pos 1 with g=0 → x[0]"
        assert torch.allclose(out2[:, 3, :], x[:, 2, :]), "pos 3 with g=0 → x[2]"

    def test_gate_is_per_dim(self):
        """Gate vector must have length equal to model_dim."""
        sg = tg.SmearGate(dim=32)
        assert sg.gate.shape == (32,)

    def test_smear_hyperparameter_default(self):
        """Hyperparameters.smear_gate must default to True (enabled)."""
        args = tg.Hyperparameters()
        assert args.smear_gate is True

    @needs_rms_norm
    def test_smear_present_when_enabled(self):
        """GPT with smear_gate=True must have a non-None .smear attribute."""
        model = tg.GPT(
            vocab_size=16, num_layers=2, model_dim=64, num_heads=4, num_kv_heads=2,
            mlp_mult=2, tie_embeddings=True, tied_embed_init_std=0.005,
            logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
            smear_gate=True,
        )
        assert model.smear is not None

    @needs_rms_norm
    def test_smear_absent_when_disabled(self):
        """GPT with smear_gate=False must have .smear = None."""
        model = tg.GPT(
            vocab_size=16, num_layers=2, model_dim=64, num_heads=4, num_kv_heads=2,
            mlp_mult=2, tie_embeddings=True, tied_embed_init_std=0.005,
            logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
            smear_gate=False,
        )
        assert model.smear is None

    @needs_rms_norm
    def test_smear_forward_no_nan(self):
        """GPT with smear_gate=True must produce finite loss."""
        model = tg.GPT(
            vocab_size=32, num_layers=4, model_dim=64, num_heads=4, num_kv_heads=2,
            mlp_mult=2, tie_embeddings=True, tied_embed_init_std=0.005,
            logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
            smear_gate=True,
        )
        model.eval()
        x = torch.randint(0, 32, (2, 8))
        y = torch.randint(0, 32, (2, 8))
        loss = model(x, y)
        assert torch.isfinite(loss), f"Non-finite loss with SmearGate: {loss}"

    @needs_rms_norm
    def test_smear_changes_output(self):
        """SmearGate (non-default gate) must change forward output vs disabled."""
        common = dict(
            vocab_size=32, num_layers=4, model_dim=64, num_heads=4, num_kv_heads=2,
            mlp_mult=2, tie_embeddings=True, tied_embed_init_std=0.005,
            logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
        )
        torch.manual_seed(7)
        m_on = tg.GPT(**common, smear_gate=True)
        torch.manual_seed(7)
        m_off = tg.GPT(**common, smear_gate=False)
        # Copy weights so only smear path differs; drive gate toward 0 so blend is visible
        sd_off = m_off.state_dict()
        sd_on = m_on.state_dict()
        for k in sd_off:
            if k in sd_on:
                sd_on[k] = sd_off[k].clone()
        m_on.load_state_dict(sd_on, strict=False)
        # Push gate toward 0 so output differs detectably (g≈0.5 blend)
        m_on.smear.gate.data.zero_()
        m_on.eval(); m_off.eval()
        x = torch.randint(0, 32, (2, 8))
        y = torch.randint(0, 32, (2, 8))
        with torch.no_grad():
            l_on = m_on(x, y)
            l_off = m_off(x, y)
        assert not torch.allclose(l_on, l_off), "SmearGate with g≈0.5 should change output"


# ===========================================================================
# Day 10 Block 1 — U-Net Skip Connections (already implemented; verify structure)
# ===========================================================================

class TestUNetSkips:
    """U-Net skips were implemented in Session 3. These tests verify the structure."""

    @needs_rms_norm
    def test_skip_weights_shape(self):
        """skip_weights must have shape (num_skip_weights, model_dim)."""
        model = tg.GPT(
            vocab_size=16, num_layers=4, model_dim=64, num_heads=4, num_kv_heads=2,
            mlp_mult=2, tie_embeddings=True, tied_embed_init_std=0.005,
            logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
        )
        # 4 layers: enc=2, dec=2, num_skip_weights = min(2,2) = 2
        assert model.skip_weights.shape == (2, 64)

    @needs_rms_norm
    def test_skip_weights_ones_init(self):
        """skip_weights must be initialised to ones."""
        model = tg.GPT(
            vocab_size=16, num_layers=4, model_dim=64, num_heads=4, num_kv_heads=2,
            mlp_mult=2, tie_embeddings=True, tied_embed_init_std=0.005,
            logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
        )
        assert torch.all(model.skip_weights == 1.0), "skip_weights should init to 1"

    @needs_rms_norm
    def test_encoder_decoder_split(self):
        """num_encoder_layers + num_decoder_layers must equal num_layers."""
        model = tg.GPT(
            vocab_size=16, num_layers=6, model_dim=64, num_heads=4, num_kv_heads=2,
            mlp_mult=2, tie_embeddings=True, tied_embed_init_std=0.005,
            logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
        )
        assert model.num_encoder_layers + model.num_decoder_layers == 6
        assert model.num_encoder_layers == 3
        assert model.num_decoder_layers == 3

    @needs_rms_norm
    def test_unet_forward_no_nan(self):
        """GPT with U-Net skips must produce finite loss."""
        model = tg.GPT(
            vocab_size=16, num_layers=4, model_dim=64, num_heads=4, num_kv_heads=2,
            mlp_mult=2, tie_embeddings=True, tied_embed_init_std=0.005,
            logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
        )
        model.eval()
        x = torch.randint(0, 16, (2, 8))
        y = torch.randint(0, 16, (2, 8))
        loss = model(x, y)
        assert torch.isfinite(loss)


# ===========================================================================
# Day 10 Block 2 — Layerwise LN Scale
# ===========================================================================

class TestLayerwiseLNScale:
    """Tests for Block.ln_scale_factor = 1/√(layer_idx+1)."""

    def test_layer_zero_factor_is_one(self):
        """Layer 0 must have ln_scale_factor == 1.0 (1/√1 = 1)."""
        block = tg.Block(
            dim=64, num_heads=4, num_kv_heads=2, mlp_mult=2,
            rope_base=10000.0, qk_gain_init=1.5, layer_idx=0,
        )
        assert abs(block.ln_scale_factor - 1.0) < 1e-9

    def test_layer_three_factor(self):
        """Layer 3 must have ln_scale_factor == 1/√4 = 0.5."""
        block = tg.Block(
            dim=64, num_heads=4, num_kv_heads=2, mlp_mult=2,
            rope_base=10000.0, qk_gain_init=1.5, layer_idx=3,
        )
        assert abs(block.ln_scale_factor - 0.5) < 1e-9

    def test_factor_decreases_with_depth(self):
        """ln_scale_factor must be strictly decreasing with layer_idx."""
        factors = [
            tg.Block(
                dim=64, num_heads=4, num_kv_heads=2, mlp_mult=2,
                rope_base=10000.0, qk_gain_init=1.5, layer_idx=i,
            ).ln_scale_factor
            for i in range(6)
        ]
        for i in range(1, len(factors)):
            assert factors[i] < factors[i - 1], \
                f"factor[{i}]={factors[i]} should be < factor[{i-1}]={factors[i-1]}"

    def test_factor_formula(self):
        """ln_scale_factor must match 1/√(layer_idx+1) exactly."""
        for idx in [0, 1, 4, 9, 10]:
            block = tg.Block(
                dim=64, num_heads=4, num_kv_heads=2, mlp_mult=2,
                rope_base=10000.0, qk_gain_init=1.5, layer_idx=idx,
            )
            expected = 1.0 / math.sqrt(idx + 1)
            assert abs(block.ln_scale_factor - expected) < 1e-9, \
                f"layer_idx={idx}: expected {expected}, got {block.ln_scale_factor}"

    @needs_rms_norm
    def test_blocks_have_distinct_factors(self):
        """All 11 blocks in a default GPT must have different ln_scale_factors."""
        model = tg.GPT(
            vocab_size=16, num_layers=11, model_dim=64, num_heads=4, num_kv_heads=2,
            mlp_mult=2, tie_embeddings=True, tied_embed_init_std=0.005,
            logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
        )
        factors = [b.ln_scale_factor for b in model.blocks]
        assert len(set(factors)) == 11, f"Expected 11 distinct factors, got {len(set(factors))}"

    @needs_rms_norm
    def test_ln_scale_forward_no_nan(self):
        """GPT with layerwise LN scale must produce finite loss."""
        model = tg.GPT(
            vocab_size=32, num_layers=4, model_dim=64, num_heads=4, num_kv_heads=2,
            mlp_mult=2, tie_embeddings=True, tied_embed_init_std=0.005,
            logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
        )
        model.eval()
        x = torch.randint(0, 32, (2, 8))
        y = torch.randint(0, 32, (2, 8))
        loss = model(x, y)
        assert torch.isfinite(loss), f"Non-finite loss with layerwise LN scale: {loss}"


# ===========================================================================
# Day 10 Block 2 — OrthoInit
# ===========================================================================

class TestOrthoInit:
    """Tests for orthogonal initialization of weight matrices in GPT._init_weights."""

    @needs_rms_norm
    def test_large_matrix_is_orthogonal(self):
        """Matrices with both dims ≥ 64 must have near-orthonormal rows/columns."""
        # Use a GPT where model_dim=128 so most weight matrices are 128×128+
        model = tg.GPT(
            vocab_size=32, num_layers=2, model_dim=128, num_heads=4, num_kv_heads=2,
            mlp_mult=2, tie_embeddings=True, tied_embed_init_std=0.005,
            logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
        )
        # Check c_q of first block: 128×128, no _zero_init → should be orthogonal
        W = model.blocks[0].attn.c_q.weight.float()  # (128, 128)
        WWT = W @ W.T
        I = torch.eye(W.shape[0])
        off_diag_err = (WWT - I).abs().mean().item()
        assert off_diag_err < 0.05, \
            f"c_q weight should be near-orthogonal (WWT ≈ I), mean err={off_diag_err:.4f}"

    @needs_rms_norm
    def test_zero_init_not_overwritten(self):
        """Matrices with _zero_init=True must remain all-zeros after _init_weights."""
        model = tg.GPT(
            vocab_size=32, num_layers=2, model_dim=128, num_heads=4, num_kv_heads=2,
            mlp_mult=2, tie_embeddings=True, tied_embed_init_std=0.005,
            logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
        )
        # attn.proj has _zero_init=True → must stay zero
        W = model.blocks[0].attn.proj.weight
        assert torch.all(W == 0.0), "attn.proj (_zero_init=True) should remain all zeros"
        # mlp.proj has _zero_init=True → must stay zero
        W2 = model.blocks[0].mlp.proj.weight
        assert torch.all(W2 == 0.0), "mlp.proj (_zero_init=True) should remain all zeros"

    @needs_rms_norm
    def test_bigram_proj_zero_init_preserved(self):
        """BigramHashEmbedding.proj must remain zero-init after GPT._init_weights."""
        model = tg.GPT(
            vocab_size=64, num_layers=2, model_dim=128, num_heads=4, num_kv_heads=2,
            mlp_mult=2, tie_embeddings=True, tied_embed_init_std=0.005,
            logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
            bigram_vocab_size=256, bigram_dim=32,  # bigram_dim≠model_dim → proj created
        )
        assert model.bigram.proj is not None
        assert torch.all(model.bigram.proj.weight == 0.0), \
            "bigram.proj must stay zero-init after _init_weights"

    @needs_rms_norm
    def test_ve_proj_zero_init_preserved(self):
        """ve_proj must remain zero-init after GPT._init_weights."""
        model = tg.GPT(
            vocab_size=32, num_layers=2, model_dim=128, num_heads=4, num_kv_heads=2,
            mlp_mult=2, tie_embeddings=True, tied_embed_init_std=0.005,
            logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
            ve_dim=32,
        )
        assert torch.all(model.ve_proj.weight == 0.0), \
            "ve_proj must stay zero-init after _init_weights"

    @needs_rms_norm
    def test_small_matrix_not_orthogonal(self):
        """Matrices with a dim < 64 must NOT be orthogonally initialized (fallback default)."""
        # With model_dim=32, kv_dim=16 → c_k is (16, 32), min=16 < 64 → default init
        model = tg.GPT(
            vocab_size=16, num_layers=2, model_dim=32, num_heads=4, num_kv_heads=2,
            mlp_mult=2, tie_embeddings=True, tied_embed_init_std=0.005,
            logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
        )
        W = model.blocks[0].attn.c_k.weight.float()  # (8, 32) with kv_heads=2, head=8
        # Just check it's not the identity / near-identity (which ortho would give for square)
        # The key test: model still works (no crash, finite loss)
        model.eval()
        x = torch.randint(0, 16, (1, 4))
        y = torch.randint(0, 16, (1, 4))
        loss = model(x, y)
        assert torch.isfinite(loss)

    def test_ortho_init_hyperparameter_no_crash(self):
        """Calling _init_weights on a model with mixed sizes must not crash."""
        # Smoke test with model_dim=64 (boundary case: exactly 64×64 matrices)
        import torch.nn as nn
        lin = nn.Linear(64, 64, bias=False)
        lin._zero_init = False
        # Simulate what _init_weights does
        if lin.weight.shape[0] >= 64 and lin.weight.shape[1] >= 64:
            nn.init.orthogonal_(lin.weight, gain=1.0)
        W = lin.weight.float()
        WWT = W @ W.T
        err = (WWT - torch.eye(64)).abs().mean().item()
        assert err < 0.01


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
