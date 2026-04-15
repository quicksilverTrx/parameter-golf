"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: To keep readable for newcomers, let's make sure `train_gpt.py` and `train_gpt_mlx.py` never are longer than 1500 lines.
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
try:
    import zstandard as _zstd
    def _compress(data: bytes) -> bytes: return _zstd.ZstdCompressor(level=22).compress(data)
    def _decompress(data: bytes) -> bytes: return _zstd.ZstdDecompressor().decompress(data)
    _CODEC = "zstd-22"
except ImportError:
    def _compress(data: bytes) -> bytes: return zlib.compress(data, level=9)
    def _decompress(data: bytes) -> bytes: return zlib.decompress(data)
    _CODEC = "zlib-9"
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------
# HYPERPARAMETERS
# -----------------------------
# Default config (Day 5):
# - 11 transformer blocks at width 512
# - 8 attention heads with 4 KV heads (GQA) and 3x MLP expansion (relu²)
# - partial RoPE: 16 of 64 head dims rotated, 48 dims position-invariant
# - vocab size 1024, sequence length 2048, tied embeddings (stored fp16 in artifact)

class Hyperparameters:
    # Data paths are shard globs produced by the existing preprocessing pipeline.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    # Validation cadence and batch size. Validation always uses the full fineweb_val split.
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3500))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))  # sliding-window eval stride; set to train_seq_len for non-overlapping
    eval_batch_seqs = int(os.environ.get("EVAL_BATCH_SEQS", 128))     # windows per forward pass during sliding eval
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 3))
    rope_dims = int(os.environ.get("ROPE_DIMS", 16))  # partial RoPE: dims rotated per head (0 = full)
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # Optimizer hyperparameters.
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_wd = float(os.environ.get("MUON_WD", 0.04))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.997))
    # Late QAT: enable STE fake-quantization when the LR scale drops below this threshold.
    # 0.15 → activates during the last ~15% of the LR schedule (warmdown phase).
    # Set 0 to disable QAT entirely.
    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 4096))  # hash buckets (0=off)
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 128))
    smear_gate = bool(int(os.environ.get("SMEAR_GATE", "1")))  # 0=disabled
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 4))          # 0=disabled
    ve_dim = int(os.environ.get("VE_DIM", 128))                 # 0=disabled
    # Depth recurrence: loop the bottleneck blocks an extra (RECUR_N-1) times.
    # RECUR_BLOCKS is a comma-separated list of block indices (default: last encoder + first decoder).
    recur_n = int(os.environ.get("RECUR_N", 2))
    recur_blocks = os.environ.get("RECUR_BLOCKS", "4,5")  # block indices to recur

# -----------------------------
# MUON OPTIMIZER
# -----------------------------

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov, weight_decay=weight_decay),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    # Scale correction from Muon reference implementations.
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            wd = group.get("weight_decay", 0.0)
            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                if wd > 0:
                    p.data.mul_(1.0 - lr * wd)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION SETUP
# -----------------------------

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    # The export pipeline writes the fixed first-50k-doc validation set to fineweb_val_*.
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


def eval_val_sliding(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int,
    batch_seqs: int = 128,
) -> tuple[float, float]:
    # Slide windows of seq_len by `stride`; score only last `stride` tokens per window.
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1  # -1 because last token has no target

    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]
    total_windows = len(window_starts)

    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi : bi + batch_seqs]
            bsz = len(batch_ws)

            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens: list[int] = []

            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws : end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = model.forward_logits(x_batch)  # (bsz, seq_len, vocab)

            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len)

            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                # First window scores all its tokens; subsequent windows score only the last `stride`.
                score_start = 0 if ws == 0 else max(wlen - stride, 0)
                scored_nll = nll[i, score_start:wlen].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(wlen - score_start)
                tgt = y_batch[i, score_start:wlen]
                prev = x_batch[i, score_start:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    val_loss = loss_sum / token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)

# -----------------------------
# POST-TRAINING QUANTIZATION
# -----------------------------

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,smear",
    ).split(",")
    if pattern
)
SMALL_TENSOR_MAX_NUMEL = 65_536
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0
# SDClip: std-based clip threshold for int6 quantization (Track 1, #1394 technique).
# clip = k * row_std; k=2.5 is the default from #1394 which optimises compression entropy.
_SDCLIP_K: float = float(os.environ.get("SDCLIP_K", "2.5"))

def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())

def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    # Vectors / scalars use a simpler per-tensor scale.
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale

# -----------------------------
# INT6 QUANTIZATION (per-row absmax, bit-packed) + ZSTD-22
# -----------------------------

_INT6_PATTERNS = ("attn.", ".mlp.")  # matrices trained with STE fake-quant → int6

def _pack_int6(q: Tensor) -> tuple[Tensor, int]:
    """Pack int8[-32..31] tensor → 4 values per 3 bytes. Returns (uint8 tensor, orig numel)."""
    flat = q.reshape(-1)
    n = flat.numel()
    pad = (-n) % 4
    u = flat.to(torch.int32).add_(32)          # shift to unsigned [0, 63]
    if pad:
        u = torch.cat([u, u.new_zeros(pad)])
    v = u.to(torch.uint8).numpy().reshape(-1, 4)
    out = np.empty((len(v), 3), dtype=np.uint8)
    out[:, 0] = v[:, 0]         | ((v[:, 1] & 0x03) << 6)
    out[:, 1] = (v[:, 1] >> 2)  | ((v[:, 2] & 0x0F) << 4)
    out[:, 2] = (v[:, 2] >> 4)  |  (v[:, 3] << 2)
    return torch.from_numpy(out.reshape(-1).copy()), n

def _unpack_int6(packed: Tensor, n: int, shape: tuple) -> Tensor:
    """Unpack bit-packed uint8 tensor → int8[-32..31] with given shape."""
    b = packed.numpy().reshape(-1, 3)
    u = np.empty((len(b), 4), dtype=np.uint8)
    u[:, 0] =  b[:, 0] & 0x3F
    u[:, 1] = ((b[:, 0] >> 6) & 0x03) | ((b[:, 1] & 0x0F) << 2)
    u[:, 2] = ((b[:, 1] >> 4) & 0x0F) | ((b[:, 2] & 0x03) << 4)
    u[:, 3] =  (b[:, 2] >> 2) & 0x3F
    q = u.reshape(-1)[:n].astype(np.int8)
    q -= 32
    return torch.from_numpy(q.copy()).reshape(shape)

def _quantize_int6_row(t: Tensor) -> tuple[Tensor, Tensor]:
    # Per-row int6: SDClip (clip = _SDCLIP_K * row_std); optimises compression entropy
    # rather than reconstruction MSE. Replaces old GPTQ-lite percentile search. 1D → absmax.
    t32 = t.float()
    if t32.ndim == 2:
        row_clip = (_SDCLIP_K * t32.std(dim=1)).clamp_min(1e-12)
        s = (row_clip / 31.0).clamp_min(torch.finfo(torch.float16).tiny).to(torch.float16)
        q = torch.clamp(torch.round(t32 / s.float()[:, None]), -32, 31).to(torch.int8)
        return q, s
    # 1D fallback: simple absmax (biases, scalars)
    amax = float(t32.abs().max().item())
    scale = torch.tensor(max(amax / 31.0, 1e-12), dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / float(scale.item())), -32, 31).to(torch.int8)
    return q, scale

def quantize_state_dict_int6(state_dict: dict[str, Tensor]) -> tuple[dict, int]:
    # int6 bit-packed for attn/mlp matrices; int8 fallback; fp16/fp32 passthrough for small/control tensors.
    weights: dict[str, object] = {}
    meta: dict[str, object] = {}
    payload_bytes = 0

    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()

        if not t.is_floating_point():
            weights[name] = t
            meta[name] = "passthrough"
            payload_bytes += tensor_nbytes(t)
            continue

        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            weights[name] = t.float()
            meta[name] = "passthrough_ctrl"
            payload_bytes += tensor_nbytes(t.float())
            continue

        if t.numel() <= SMALL_TENSOR_MAX_NUMEL:
            w = t.to(torch.float16)
            weights[name] = w
            meta[name] = "passthrough_fp16"
            payload_bytes += tensor_nbytes(w)
            continue

        # Token embedding: tied with LM head, never STE-trained → keep fp16 for quality.
        # At vocab=1024, dim=512 this is only 1 MB, cheaper than the int8 quality penalty.
        if name == "tok_emb.weight":
            w = t.to(torch.float16)
            weights[name] = w
            meta[name] = "passthrough_fp16"
            payload_bytes += tensor_nbytes(w)
            continue

        orig_dtype = str(t.dtype).removeprefix("torch.")
        use_int6 = t.ndim == 2 and any(p in name for p in _INT6_PATTERNS)
        if use_int6:
            q, scale = _quantize_int6_row(t)
            packed, n = _pack_int6(q)
            weights[name + ".packed"] = packed
            weights[name + ".n"]      = torch.tensor(n, dtype=torch.int64)
            weights[name + ".scale"]  = scale
            weights[name + ".shape"]  = list(t.shape)
            meta[name] = {"type": "int6", "dtype": orig_dtype}
            payload_bytes += packed.numel() + tensor_nbytes(scale)
        else:
            q, scale = quantize_float_tensor(t)
            weights[name + ".q"]     = q
            weights[name + ".scale"] = scale
            meta[name] = {"type": "int8", "dtype": orig_dtype}
            payload_bytes += tensor_nbytes(q) + tensor_nbytes(scale)

    return {"w": weights, "m": meta}, payload_bytes

def dequantize_state_dict_int6(obj: dict, template_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    """Dequantize mixed int6/int8 artifact back to original dtypes."""
    weights, meta = obj["w"], obj["m"]
    out: dict[str, Tensor] = {}
    for name, orig in template_sd.items():
        info = meta[name]
        dtype = orig.dtype
        if isinstance(info, str):  # passthrough variants
            t = weights[name]
            if t.dtype == torch.float16 and dtype in (torch.float32, torch.bfloat16):
                t = t.to(dtype)
            out[name] = t
            continue
        saved_dtype = getattr(torch, info["dtype"])
        if info["type"] == "int6":
            packed = weights[name + ".packed"]
            n      = int(weights[name + ".n"].item())
            scale  = weights[name + ".scale"]
            shape  = tuple(weights[name + ".shape"])
            q = _unpack_int6(packed, n, shape)
            s = scale.float().view(shape[0], *([1] * (len(shape) - 1)))
            out[name] = (q.float() * s).to(saved_dtype)
        else:
            q, scale = weights[name + ".q"], weights[name + ".scale"]
            s = scale.float()
            if s.ndim > 0:
                s = s.view(q.shape[0], *([1] * (q.ndim - 1)))
                out[name] = (q.float() * s).to(saved_dtype)
            else:
                out[name] = (q.float() * float(s.item())).to(saved_dtype)
    return out


# -----------------------------
# DATA LOADING 
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    # SHARD HEADER INTS & SHARD_MAGIC
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    # Reads shards sequentially, wrapping forever.
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    # Distributes disjoint token spans across ranks; "+1" for (x, y) shift.
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    # fp32 weights cast to x.dtype at matmul time; when _qat_enabled=True applies STE
    # fake-quant (int6 forward, unquantized backward) during the final ~15% of training.
    _qat_enabled: bool = False  # class-level flag; set True once late in training

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight
        if CastedLinear._qat_enabled and self.training and w.ndim == 2:
            # STE: compute quantized weights in no_grad, then re-attach via straight-through trick.
            with torch.no_grad():
                w32 = w.float()
                # Match SDClip post-training: row_clip = k*std (same clamp_min as quantizer).
                # scale.clamp_min(1/31) is a QAT safety floor for gradient stability only.
                row_clip = (_SDCLIP_K * w32.std(dim=1)).clamp_min(1e-12)
                scale = (row_clip / 31.0).clamp_min(1.0 / 31.0)
                w_q = (torch.clamp(torch.round(w32 / scale[:, None]), -32, 31) * scale[:, None]).to(x.dtype)
            # w_ste has quantized values in forward, but grad of (w_q - w).detach() is 0
            # so the full gradient passes straight through to w as if no rounding occurred.
            w = w.to(x.dtype) + (w_q - w.to(x.dtype)).detach()
        else:
            w = w.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    # Keep small/control parameters in fp32 even when the model body runs in bf16.
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    # Cached cos/sin tables; rope_dims < dim enables partial RoPE (rotated dims + passthrough).
    def __init__(self, dim: int, base: float = 10000.0, rope_dims: int = 0):
        super().__init__()
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        rd = self.rope_dims
        inv_freq = 1.0 / (base ** (torch.arange(0, rd, 2, dtype=torch.float32) / rd))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    rd = cos.size(-1) * 2  # number of dims that are actually rotated
    if rd < x.size(-1):
        # Partial RoPE: rotate first rd dims, leave the rest position-invariant
        x_rope, x_pass = x[..., :rd], x[..., rd:]
        half = rd // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rot = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
        return torch.cat((x_rot, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        rope_dims: int = 0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base, rope_dims=rope_dims)
        # XSA flag — toggled from GPT.__init__ for the last xsa_last_n layers.
        self.use_xsa: bool = False

    def _xsa_efficient(self, y: Tensor, v: Tensor) -> Tensor:
        """XSA: subtract the v-direction component from the attention output.

        WHY: Standard attention can "cheat" by attending strongly to a single
        token and returning its value directly.  XSA removes the component of
        the output that lies along the current-query value direction, forcing
        upper layers to extract information *orthogonal* to plain value copying
        and encouraging richer cross-token mixing.

        Implementation is GQA-aware: instead of expanding v to match all H heads
        (expensive), we reshape y to expose the KV-head grouping so each group
        shares the same v unit vector.  No repeat_interleave needed.

        Args:
            y: attention output, shape (B, T, H, D)
            v: value vectors,    shape (B, T, Hkv, D)
        Returns:
            y with v-projection subtracted, shape (B, T, H, D)
        """
        B, T, H, D = y.shape
        Hkv = v.size(2)
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)          # expose GQA groups
        vn = F.normalize(v, dim=-1).unsqueeze(-2)      # (B, T, Hkv, 1, D)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn  # project onto v̂
        return (y_g - proj).reshape(B, T, H, D)

    def forward(self, x: Tensor, v_extra: "Tensor | None" = None) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # Inject value embedding: per-token identity signal pre-computed in GPT.forward_logits.
        # v_extra shape: (B, T, Hkv, D) → transpose to match v's (B, Hkv, T, D).
        if v_extra is not None:
            v = v + v_extra.to(dtype=v.dtype).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        # y: (B, H, T, D) → (B, T, H, D)
        y = y.transpose(1, 2)
        if self.use_xsa:
            # v is currently (B, Hkv, T, D); XSA wants (B, T, Hkv, D).
            y = self._xsa_efficient(y, v.transpose(1, 2))
        y = y.contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    # relu^2 MLP from the original modded-nanogpt setup
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = F.leaky_relu(self.fc(x), negative_slope=0.5)
        return self.proj(x.square())


class BigramHashEmbedding(nn.Module):
    # Hashes token pairs (t[i-1], t[i]) into a learned table; adds co-occurrence signal before attention.
    def __init__(self, vocab_size: int, bigram_dim: int, model_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)  # zero init: no bigram signal at step 0
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
            self.proj._zero_init = True
        # Gating scalar: starts at 0.05 so the bigram contribution is small early.
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def _hash(self, tokens: Tensor) -> Tensor:
        # XOR hash of bigram pairs → bucket in [0, vocab_size); position 0 → null bucket.
        t = tokens.to(torch.int32)
        mod = self.vocab_size - 1                          # leave bucket (vocab_size-1) as null
        out = torch.empty_like(t)
        out[..., 0] = mod                                  # no preceding token → null
        out[..., 1:] = torch.bitwise_xor(
            36313 * t[..., 1:], 27191 * t[..., :-1]
        ) % mod
        return out.long()

    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(self._hash(token_ids))   # (B, T, bigram_dim)
        if self.proj is not None:
            h = self.proj(h)                    # (B, T, model_dim)
        return h * self.scale.to(dtype=h.dtype)


class SmearGate(nn.Module):
    # Blends each raw embedding with the previous position's via a learned per-dim gate.
    def __init__(self, dim: int):
        super().__init__()
        # gate=3.0 → sigmoid(3.0)≈0.953: mostly current token at init.
        self.gate = nn.Parameter(torch.full((dim,), 3.0, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate).to(dtype=x.dtype)
        # Causal shift: x_prev[t] = x[t-1]; position 0 gets zeros (no prior token).
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return g * x + (1.0 - g) * x_prev


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        rope_dims: int = 0,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, rope_dims=rope_dims)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        # ln_scale_factor = 1/√(layer_idx+1): dampens deeper layers for gradient stability.
        self.ln_scale_factor: float = 1.0 / math.sqrt(layer_idx + 1)

    def forward(self, x: Tensor, x0: Tensor, v_extra: "Tensor | None" = None) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        # ln_scale_factor shrinks the normed input for deeper layers, dampening their
        # contribution and improving gradient flow stability across depth.
        attn_out = self.attn(self.attn_norm(x) * self.ln_scale_factor, v_extra=v_extra)
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x) * self.ln_scale_factor)
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        rope_dims: int = 0,
        bigram_vocab_size: int = 0,
        bigram_dim: int = 128,
        xsa_last_n: int = 0,
        ve_dim: int = 0,
        smear_gate: bool = True,
        recur_n: int = 1,
        recur_blocks: str = "",
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        # Depth recurrence: run _recur_block_indices blocks extra (_recur_n - 1) times
        # between the encoder and decoder passes. Adds effective depth at zero parameter cost.
        self._recur_n = max(1, recur_n)
        self._recur_block_indices: list[int] = (
            [int(b.strip()) for b in recur_blocks.split(",") if b.strip()]
            if recur_blocks else []
        )
        # Validate block indices eagerly — IndexError during forward is harder to debug.
        for bi in self._recur_block_indices:
            if bi < 0 or bi >= num_layers:
                raise ValueError(
                    f"RECUR_BLOCKS index {bi} is out of range for num_layers={num_layers}. "
                    f"Valid indices: 0..{num_layers - 1}."
                )
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        # BigramHash: additive to the token embedding when vocab_size > 0.
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim) if bigram_vocab_size > 0 else None
        # SmearGate: blends raw embeddings with the previous position's before RMSNorm.
        self.smear = SmearGate(model_dim) if smear_gate else None
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                    rope_dims=rope_dims,
                    layer_idx=i,   # layerwise LN scale: 1/√(i+1) applied to norm'd inputs
                )
                for i in range(num_layers)
            ]
        )
        # XSA: enable on last xsa_last_n layers.  Subtracts the value-direction from
        # attention output, discouraging trivial value-copy behaviour in upper layers.
        if xsa_last_n > 0:
            for i in range(max(0, num_layers - xsa_last_n), num_layers):
                self.blocks[i].attn.use_xsa = True
        # Value embeddings: shared embedding + projection, per-layer scale gates.
        # ve_embed: vocab_size × ve_dim (shared across all layers — tiny footprint).
        # ve_proj:  ve_dim → kv_dim  (also shared; scale per-layer starts at 0).
        # _zero_init flag ensures OrthoInit (in _init_weights) doesn't overwrite the zero init.
        self._ve_num_kv_heads = num_kv_heads
        self._ve_head_dim = model_dim // num_heads
        if ve_dim > 0:
            kv_dim = num_kv_heads * self._ve_head_dim
            self.ve_embed = nn.Embedding(vocab_size, ve_dim)
            nn.init.normal_(self.ve_embed.weight, std=0.02)
            self.ve_proj = CastedLinear(ve_dim, kv_dim, bias=False)
            nn.init.zeros_(self.ve_proj.weight)
            self.ve_proj._zero_init = True  # guard against OrthoInit overwrite
            # Per-layer scalar gates: zero-init so VE contributes nothing at step 0.
            self.ve_scale = nn.Parameter(torch.zeros(num_layers, dtype=torch.float32))
        else:
            self.ve_embed = None
            self.ve_proj = None
            self.ve_scale = None
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        num_layers = len(self.blocks)
        for name, module in self.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)
            elif module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                # OrthoInit (all singular values = 1): better depth-aware init than Xavier.
                nn.init.orthogonal_(module.weight, gain=1.0)

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        """Return logits (batch, seq_len, vocab_size). Used directly by sliding-window eval;
        forward() delegates here then flattens for cross-entropy."""
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        if self.smear is not None:
            # SmearGate blends raw embeddings before normalization (matches record pattern).
            x = self.smear(x)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        # Pre-compute value embeddings once (shared lookup + projection).
        # Each layer then scales by its own ve_scale[i] before passing to attention.
        ve_base: "Tensor | None" = None
        if self.ve_embed is not None:
            ve_raw = self.ve_embed(input_ids).to(dtype=x.dtype)  # (B, T, ve_dim)
            ve_proj = self.ve_proj(ve_raw)                        # (B, T, kv_dim)
            B, T, _ = ve_proj.shape
            ve_base = ve_proj.reshape(B, T, self._ve_num_kv_heads, self._ve_head_dim)
        skips: list[Tensor] = []
        for i in range(self.num_encoder_layers):
            v_e = ve_base * self.ve_scale[i].to(ve_base.dtype) if ve_base is not None else None
            x = self.blocks[i](x, x0, v_extra=v_e)
            skips.append(x)
        # Depth recurrence: run bottleneck blocks extra (recur_n - 1) times between
        # encoder and decoder. Same weights, different inputs each pass — free effective depth.
        if self._recur_n > 1 and self._recur_block_indices:
            for _ in range(self._recur_n - 1):
                for bi in self._recur_block_indices:
                    v_e = ve_base * self.ve_scale[bi].to(ve_base.dtype) if ve_base is not None else None
                    x = self.blocks[bi](x, x0, v_extra=v_e)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            li = self.num_encoder_layers + i
            v_e = ve_base * self.ve_scale[li].to(ve_base.dtype) if ve_base is not None else None
            x = self.blocks[li](x, x0, v_extra=v_e)
        x = self.final_norm(x)
        logits = F.linear(x, self.tok_emb.weight) if self.tie_embeddings else self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits / self.logit_softcap)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        logits = self.forward_logits(input_ids)  # (batch, seq_len, vocab)
        return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                               target_ids.reshape(-1), reduction="mean")


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # -----------------------------
    # DISTRIBUTED + CUDA SETUP
    # -----------------------------

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    # Fast math knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    # -----------------------------
    # TOKENIZER + VALIDATION METRIC SETUP
    # -----------------------------

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------

    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        rope_dims=args.rope_dims,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n,
        ve_dim=args.ve_dim,
        smear_gate=args.smear_gate,
        recur_n=args.recur_n,
        recur_blocks=args.recur_blocks,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    # tok_emb → Adam/EMBED_LR; lm_head → Adam/HEAD_LR; matrices → Muon/MATRIX_LR; scalars → Adam/SCALAR_LR
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    if base_model.smear is not None:
        # SmearGate.gate lives outside blocks; it's a 1D param → scalar_lr via Adam.
        scalar_params.append(base_model.smear.gate)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
        weight_decay=args.muon_wd,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.bigram is not None:
        # BigramHash params live outside blocks; use Adam at embed LR (same as tok_emb).
        bigram_params = list(base_model.bigram.parameters())
        optimizer_bigram = torch.optim.Adam(
            [{"params": bigram_params, "lr": token_lr, "base_lr": token_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.append(optimizer_bigram)
    if base_model.ve_embed is not None:
        ve_params = list(base_model.ve_embed.parameters()) + list(base_model.ve_proj.parameters()) + [base_model.ve_scale]
        optimizer_ve = torch.optim.Adam(
            [{"params": ve_params, "lr": token_lr, "base_lr": token_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.append(optimizer_ve)
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")
    log0(
        f"arch:layers={args.num_layers} dim={args.model_dim} heads={args.num_heads} "
        f"mlp_mult={args.mlp_mult} rope_dims={args.rope_dims}"
    )
    log0(
        f"features:xsa_last_n={args.xsa_last_n} ve_dim={args.ve_dim} "
        f"bigram={args.bigram_vocab_size} smear_gate={int(args.smear_gate)}"
    )
    log0(
        f"track1:sdclip_k={_SDCLIP_K} recur_n={args.recur_n} "
        f"recur_blocks={args.recur_blocks!r}"
    )

    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    # Warmup primes the compiled forward/backward/optimizer paths, then we restore the
    # initial weights/optimizer state so measured training starts from the true init.
    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------

    # EMA shadow weights: float32 copies, updated each step; applied before quantization.
    ema_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)

        # Late QAT: arm STE fake-quantization once the LR scale drops into the warmdown tail.
        # This lets the model experience int6 quantization noise while gradients can still adapt,
        # producing weights that are "comfortable" at int6 precision before the final export.
        if args.late_qat_threshold > 0 and scale < args.late_qat_threshold and not CastedLinear._qat_enabled:
            CastedLinear._qat_enabled = True
            log0(f"late_qat:enabled step:{step} lr_scale:{scale:.4f}")

        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        # EMA update: lerp shadow weights toward current weights each step.
        # Cost: one mul_ + one add_ per parameter tensor — ~0.1% of step time.
        with torch.no_grad():
            for name, t in base_model.state_dict().items():
                ema_state[name].mul_(args.ema_decay).add_(t.detach().float(), alpha=1.0 - args.ema_decay)

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        # Needed to sync whether we've reached the wallclock cap.
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    # -----------------------------
    # EMA APPLY + SERIALIZATION + ROUNDTRIP VALIDATION
    # -----------------------------
    log0("ema:applying EMA weights for quantization")
    current_dtype_map = {name: t.dtype for name, t in base_model.state_dict().items()}
    ema_loaded = {name: t.to(dtype=current_dtype_map[name]) for name, t in ema_state.items()}
    base_model.load_state_dict(ema_loaded, strict=True)

    # Quantize to mixed int6/int8 (GPTQ-lite clip search), compress with zstd-22
    # (falls back to zlib-9), write artifact, then reload and eval for submission score.
    sd_cpu = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}
    quant_obj, payload_bytes = quantize_state_dict_int6(sd_cpu)
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = _compress(quant_raw)
    if master_process:
        with open("final_model.int6.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int6.ptz")
        code_bytes = len(code.encode("utf-8"))
        log0(
            f"Serialized model int6+{_CODEC}: {quant_file_bytes} bytes "
            f"(payload_raw:{payload_bytes} torch_overhead:{len(quant_raw) - payload_bytes})"
        )
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {quant_file_bytes + code_bytes} bytes")
        if quant_file_bytes + code_bytes > 16 * 1024 * 1024:
            log0("WARNING: artifact exceeds 16MB budget!")

    if distributed:
        dist.barrier()
    with open("final_model.int6.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(_decompress(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int6(quant_state, sd_cpu), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    if args.eval_stride < args.train_seq_len:
        q_val_loss, q_val_bpb = eval_val_sliding(
            args,
            base_model,
            rank,
            world_size,
            device,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            stride=args.eval_stride,
            batch_seqs=args.eval_batch_seqs,
        )
        log0(f"final_eval_mode:sliding_window stride:{args.eval_stride} batch_seqs:{args.eval_batch_seqs}")
    else:
        q_val_loss, q_val_bpb = eval_val(
            args,
            base_model,
            rank,
            world_size,
            device,
            grad_accum_steps,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
        )
        log0("final_eval_mode:non_overlapping")
    torch.cuda.synchronize()
    log0(
        f"final_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
