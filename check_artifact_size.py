"""
check_artifact_size.py — Verify int6+zstd artifact size WITHOUT running training.

Instantiates the real model with real default hyperparameters, fills weights with
normally-distributed synthetic values (same dtype/shape as a trained model), runs
the full quantize → compress → decompress → load roundtrip, and reports sizes.

Usage:
    python check_artifact_size.py           # default 9L-512d config
    NUM_LAYERS=11 MODEL_DIM=512 MLP_MULT=3 python check_artifact_size.py
"""
import io
import os
import sys
import zlib
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Import the competition module directly so we reuse the exact same
# quantize/dequantize/pack/unpack code that will run at submission time.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
import train_gpt as tg

BUDGET_BYTES = 16 * 1024 * 1024  # 16 MB competition limit

def make_model() -> torch.nn.Module:
    args = tg.Hyperparameters()
    model = tg.GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        rope_base=args.rope_base,
        logit_softcap=args.logit_softcap,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        qk_gain_init=args.qk_gain_init,
        rope_dims=args.rope_dims,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n,
        ve_dim=args.ve_dim,
        smear_gate=args.smear_gate,
        recur_n=args.recur_n,
        recur_blocks=args.recur_blocks,
    )
    # Fill with realistic synthetic weights: N(0, 0.02) in bfloat16,
    # matching the dtype the training loop produces.
    with torch.no_grad():
        for p in model.parameters():
            p.data = torch.randn_like(p.data, dtype=torch.bfloat16) * 0.02
    return model

def fmt(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / 1024 / 1024:.3f} MB ({n:,} bytes)"
    return f"{n / 1024:.1f} KB ({n:,} bytes)"

def main():
    print("=" * 60)
    print("Artifact size verification")
    print("=" * 60)

    # --- Model ---
    model = make_model()
    n_params = sum(p.numel() for p in model.parameters())
    n_param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"\nModel: {n_params:,} params  ({fmt(n_param_bytes)} bf16)")

    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # --- Int6 quantization ---
    quant_obj, payload_bytes = tg.quantize_state_dict_int6(sd)
    print(f"\nInt6 payload (pre-torch-overhead): {fmt(payload_bytes)}")

    # Break down by type
    int6_bytes = int8_bytes = pass_bytes = 0
    for name, info in quant_obj["m"].items():
        if isinstance(info, dict) and info["type"] == "int6":
            int6_bytes += quant_obj["w"][name + ".packed"].numel()
            int6_bytes += quant_obj["w"][name + ".scale"].numel() * 2  # fp16
        elif isinstance(info, dict) and info["type"] == "int8":
            int8_bytes += tg.tensor_nbytes(quant_obj["w"][name + ".q"])
            int8_bytes += tg.tensor_nbytes(quant_obj["w"][name + ".scale"])
        else:
            t = quant_obj["w"].get(name)
            if t is not None:
                pass_bytes += tg.tensor_nbytes(t)
    print(f"  int6 bit-packed (attn+mlp):  {fmt(int6_bytes)}")
    print(f"  int8 (embeddings + other):   {fmt(int8_bytes)}")
    print(f"  passthrough (small/control): {fmt(pass_bytes)}")

    # --- Serialise ---
    buf = io.BytesIO()
    torch.save(quant_obj, buf)
    raw_bytes = len(buf.getvalue())
    print(f"\nTorch-serialised (pre-compress): {fmt(raw_bytes)}")

    # --- Compress ---
    blob_zstd = tg._compress(buf.getvalue())
    blob_zlib = zlib.compress(buf.getvalue(), level=9)
    print(f"\nCompressed ({tg._CODEC}):  {fmt(len(blob_zstd))}")
    print(f"Compressed (zlib-9):    {fmt(len(blob_zlib))}")
    better = "zstd" if len(blob_zstd) <= len(blob_zlib) else "zlib"
    saving = abs(len(blob_zstd) - len(blob_zlib))
    print(f"  → {better} is smaller by {fmt(saving)}")

    # --- Code size ---
    code_bytes = Path(__file__).parent.joinpath("train_gpt.py").stat().st_size
    print(f"\nCode (train_gpt.py): {fmt(code_bytes)}")

    # --- Total ---
    artifact_bytes = len(blob_zstd)
    total = artifact_bytes + code_bytes
    headroom = BUDGET_BYTES - total
    pct = total / BUDGET_BYTES * 100
    print(f"\n{'='*60}")
    print(f"Total submission: {fmt(total)}  ({pct:.1f}% of 16MB budget)")
    if headroom >= 0:
        print(f"Headroom:         {fmt(headroom)}  ✅  UNDER budget")
    else:
        print(f"OVER BUDGET by:   {fmt(-headroom)}  ❌")
    print(f"{'='*60}")

    # --- Roundtrip check ---
    print("\nRoundtrip validation ... ", end="", flush=True)
    reloaded = torch.load(io.BytesIO(tg._decompress(blob_zstd)), map_location="cpu")
    restored = tg.dequantize_state_dict_int6(reloaded, sd)
    for name, orig in sd.items():
        if name not in restored:
            print(f"MISSING: {name}")
            sys.exit(1)
        if restored[name].shape != orig.shape:
            print(f"SHAPE MISMATCH: {name} {restored[name].shape} vs {orig.shape}")
            sys.exit(1)
    # Check max reconstruction error on a sample int6 tensor
    sample_errs = []
    for name, info in quant_obj["m"].items():
        if isinstance(info, dict) and info["type"] == "int6":
            orig_t = sd[name].float()
            rest_t = restored[name].float()
            rel_err = (orig_t - rest_t).abs().mean() / orig_t.abs().mean().clamp_min(1e-8)
            sample_errs.append(rel_err.item())
            if len(sample_errs) >= 5:
                break
    mean_rel = sum(sample_errs) / len(sample_errs) if sample_errs else 0.0
    print(f"OK  (mean relative recon error on int6 tensors: {mean_rel:.4%})")

if __name__ == "__main__":
    main()
