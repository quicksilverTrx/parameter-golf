# Claude Work Log — Parameter Golf Competition Fork

**Project:** `parameter-golf` competition fork  
**Worktree:** `.claude/worktrees/beautiful-wescoff/train_gpt.py`  
**Log started:** 2026-04-08  

---

## Session 1 — 2026-04-08

### Context
Implementing Day 2, Block 1 of the master plan:
> *"Sliding Window Eval + Seq Length (2 hrs) [Cat B]"*

---

### Change 1 — `TRAIN_SEQ_LEN` default: 1024 → 2048
**File:** `train_gpt.py`, `Hyperparameters`  
**Type:** Original (config change)  
**Source:** Master plan Block 1 directive: *"Change seq_len from 1024 to 2048"*  
**What:** Changed the default value of `TRAIN_SEQ_LEN` env var from 1024 to 2048.  
**Why:** Longer context → better BPB; pairs with sliding eval so scored tokens get ~1984 tokens of context instead of ~512.

---

### Change 2 — `EVAL_STRIDE` + `EVAL_BATCH_SEQS` hyperparameters
**File:** `train_gpt.py`, `Hyperparameters`  
**Type:** Original (new env vars)  
**Source:** Pattern established in `records/track_10min_16mb/2026-03-19_SlidingWindowEval/train_gpt.py`  
**What:** Added two new fields:
```python
eval_stride     = int(os.environ.get("EVAL_STRIDE", 64))
eval_batch_seqs = int(os.environ.get("EVAL_BATCH_SEQS", 128))
```
**Why:** `EVAL_STRIDE=64` → each scored token gets `seq_len - stride = 2048 - 64 = 1984` tokens of context. `EVAL_BATCH_SEQS=128` (record used 32; bumped for GPU utilization at seq_len=2048).

---

### Change 3 — `TRAIN_STRIDE` hyperparameter (bonus, not Block 1)
**File:** `train_gpt.py`, `Hyperparameters`  
**Type:** Original  
**Source:** Extended from the sliding-eval concept to training  
**What:** Added `TRAIN_STRIDE` env var (default = `TRAIN_SEQ_LEN`, i.e., non-overlapping baseline).  
**Why:** Enables sliding-window training (only score last `stride` tokens per window during training), which gives the same benefit as sliding eval but during the forward/backward pass. Not part of Block 1 — added proactively.

---

### Change 4 — `_project` + `_encode` refactor on `GPT` model
**File:** `train_gpt.py`, `GPT` class  
**Type:** Original (refactor)  
**Source:** Motivated by need to expose logits separately for sliding eval (same pattern used in the record's `forward_logits`)  
**What:** Factored the shared transformer body into `_encode(input_ids) → Tensor` and the vocab projection into `_project(hidden) → Tensor`. `forward()` now calls both; `forward_logits()` calls both and returns `(batch, seq_len, vocab)` shaped logits.  
**Why:** Sliding eval needs per-position logits before reduction to a scalar loss. DRY: avoids duplicating the transformer stack.

---

### Change 5 — `forward_logits` method on `GPT`
**File:** `train_gpt.py`, `GPT` class  
**Type:** Ported  
**Source:** `records/track_10min_16mb/2026-03-19_SlidingWindowEval/train_gpt.py` — same method, same purpose  
**What:**
```python
def forward_logits(self, input_ids: Tensor) -> Tensor:
    return self._project(self._encode(input_ids))  # (batch, seq_len, vocab)
```
**Diff from record:** Record had the transformer body inlined directly inside `forward_logits`. Here it delegates to `_encode`/`_project` for DRY.

---

### Change 6 — `eval_val_sliding` function
**File:** `train_gpt.py`  
**Type:** Ported (with additions)  
**Source:** `records/track_10min_16mb/2026-03-19_SlidingWindowEval/train_gpt.py`, function `eval_val_sliding`  
**What:** Sliding-window evaluation loop. Windows of `seq_len` advance by `stride`. Only the last `stride` tokens per window contribute to NLL/BPB score. First window scores all tokens.  
**Identical to record:**
- Window list comprehension and partial-window filter
- Rank distribution formula (`(total_windows * rank) // world_size`)
- Batch loop structure (`bi` outer, `enumerate(batch_ws)` inner)
- `x_batch[i, :wlen] = chunk[:-1]` / `y_batch[i, :wlen] = chunk[1:]` filling
- `F.cross_entropy(..., reduction="none").reshape(bsz, seq_len)` NLL computation
- `score_start = 0 if ws == 0 else max(wlen - stride, 0)` scoring window logic
- Byte counting block

**Additions vs record:**
- `dist.all_reduce` on `loss_sum`, `token_count`, `byte_count` — record omitted this (single-node run); current script requires it for multi-GPU correctness
- Removed progress `print` (record had a live `%` print every 50 batches) — cleaner for main script; can add back if useful
- `batch_seqs` default: 32 → 128 (better GPU utilization at seq_len=2048)
- Docstring includes the user's pseudocode as the conceptual summary

---

### Change 7 — Final eval dispatch (sliding vs non-overlapping)
**File:** `train_gpt.py`, end of `main()`  
**Type:** Original  
**Source:** Dispatch pattern loosely inspired by `records/track_10min_16mb/2026-03-19_SlidingWindowEval/train_gpt.py` (which hardcoded sliding eval at the end)  
**What:** After quantization roundtrip, dispatches to `eval_val_sliding` when `eval_stride < train_seq_len`, otherwise falls back to `eval_val`. Logs the mode used.  
**Why:** Non-breaking — setting `EVAL_STRIDE=TRAIN_SEQ_LEN` preserves baseline behavior exactly.

---

### Change 8 — Sliding-window training loader (`DistributedTokenLoader.next_batch`)
**File:** `train_gpt.py`, `DistributedTokenLoader`  
**Type:** Original  
**Source:** Conceptual extension of the sliding-eval pattern to training; no direct record equivalent  
**What:** Added optional `stride` param to `next_batch`. When `stride < seq_len`:
- Computes `num_windows = local_stride_tokens // stride`
- Token span per rank: `(num_windows - 1) * stride + seq_len + 1`
- Uses `torch.unfold(size=seq_len, step=stride)` for zero-copy sliding window extraction
- Masks `y[:, :seq_len - stride] = -100` so only the `stride` new tokens contribute to loss (leverages `F.cross_entropy`'s default `ignore_index=-100`)

**Why correct:** Token throughput is identical to non-overlapping (same `global_tokens` new tokens per step), but each scored token gets `seq_len - stride` extra context tokens compared to non-overlapping training.

---

### Change 8b — `TRAIN_BATCH_TOKENS` default: 524,288 → 262,144
**File:** `train_gpt.py`, `Hyperparameters`  
**Type:** Original (value change)  
**Source:** Master plan Block 1, item 4: "Adjust batch size to compensate for 2× memory"  
**What:** `TRAIN_BATCH_TOKENS` default 524,288 → 262,144.  
**Why:** Attention is O(seq_len²) in memory. Doubling seq_len 1024→2048 quadruples attention memory per sequence. Halving batch tokens keeps per-GPU peak memory roughly constant (262,144 / 2048 = 128 seqs/step vs 524,288 / 1024 = 512 seqs/step). Total tokens per step is halved, but this is unavoidable at 2048 unless GPU memory allows larger batches (override with `TRAIN_BATCH_TOKENS=524288` env var if memory permits).

---

## Session 2 — 2026-04-08 (Block 2: Hyperparameter Tuning)

### Change 9 — Warmdown: 1200 → 3500 iterations
**File:** `train_gpt.py`, `Hyperparameters`  
**Type:** Original (value change)  
**Source:** Master plan Block 2. Validated in `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` (record name literally contains `warmdown3500`).  
**What:** `WARMDOWN_ITERS` default 1200 → 3500.  
**Why:** Longer warmdown gives the LR schedule more time to anneal smoothly, especially important with 20K-step training and wallclock cap. The record at 3500 reached 1.1233 BPB.

---

### Change 10 — Muon momentum: 0.95 → 0.99, warmup 0.85→0.92 over 500→1500 steps
**File:** `train_gpt.py`, `Hyperparameters`  
**Type:** Original (value changes)  
**Source:** Master plan Block 2. Pattern from `records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow` (used `MUON_MOMENTUM=0.99`, `warmup 0.92→0.99/1500`).  
**What:**
```
muon_momentum:              0.95  → 0.99
muon_momentum_warmup_start: 0.85  → 0.92
muon_momentum_warmup_steps: 500   → 1500
```
**Why:** Higher final momentum (0.99) accumulates gradient signal over more steps — beneficial with the orthogonalized Muon updates. Slower warmup (0.92→0.99 over 1500 steps vs 0.85→0.95 over 500) avoids instability early in training when the model is far from optimum.

---

### Change 11 — Muon weight decay: 0.0 → 0.04
**File:** `train_gpt.py`, `Muon` class + `Hyperparameters` + optimizer construction  
**Type:** Ported  
**Source:** `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py` — exact same pattern  
**What:** Three-part change:
1. Added `muon_wd = float(os.environ.get("MUON_WD", 0.04))` to `Hyperparameters`
2. Added `weight_decay: float = 0.0` param to `Muon.__init__`, stored in defaults dict
3. Applied decay in `Muon.step` before the gradient update:
```python
wd = group.get("weight_decay", 0.0)
if wd > 0:
    p.data.mul_(1.0 - lr * wd)
p.add_(g, alpha=-lr)
```
4. Passed `weight_decay=args.muon_wd` to `optimizer_muon` constructor  

**Identical to record:** The three-line WD application block (`wd = group.get(...)`, `if wd > 0: p.data.mul_(...)`, `p.add_(g, alpha=-lr)`) is a verbatim port from the record.  
**Why:** Muon operates on orthogonalized updates — WD penalizes weight magnitude independently of gradient direction, preventing weight norm growth that can destabilize training or inflate artifact size.

---

### Change 12 — Gradient clip: 0.0 → 0.3
**File:** `train_gpt.py`, `Hyperparameters`  
**Type:** Original (value change)  
**Source:** Master plan Block 2.  
**What:** `GRAD_CLIP_NORM` default 0.0 → 0.3. Clipping is already implemented in the training loop (`torch.nn.utils.clip_grad_norm_`); it was just disabled by the 0.0 default.  
**Why:** With seq_len=2048 and momentum=0.99, gradient spikes are more likely. Clipping at 0.3 prevents loss divergence during warmup without significantly slowing steady-state training.

---

## Session 3 — 2026-04-09 (Block 1 Day 3: Int6 Quantization + Zstd-22)

### Explain-Before-Merge (Cat A requirement, completed before coding)

**Why per-row scales, not per-tensor?**
Each row = one output neuron's weights. Neurons have heterogeneous magnitude ranges. Per-tensor scale = max_overall/31 forces small-magnitude rows to use only a handful of the 63 levels — if a row's max is 1/10 the tensor max, it gets ~3 effective levels instead of 63. Per-row scale = max_row/31 gives every row the full 63 levels, regardless of magnitude. Within a row, all weights multiply the same input vector, so one scale covers them coherently.

**Expected MSE (per-row int6)?**
Step size Δ = max_row / 31. For weights uniform in [−max_row, +max_row]:
MSE ≈ Δ²/12 = max_row² / (31² × 12) ≈ **0.026% of row variance**. Negligible — which is why STE-trained models survive int6 with almost no BPB penalty.

---

### Change 13 — zstandard import with zlib fallback
**File:** `train_gpt.py` (top-level, after numpy import)
**Type:** Original (pattern inspired by record but not copied)
**Source:** `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/train_gpt.py` used `_COMPRESSOR` flag; here I expose `_compress`/`_decompress` callables instead for cleaner call sites.
**What:** Try-import of `zstandard`; defines `_compress`, `_decompress`, `_CODEC`. Falls back to `zlib.compress(level=9)` if zstandard not installed.
**Why:** zstd-22 is ~10-15% smaller than zlib-9 for weight tensors. The fallback ensures the script runs on machines without zstandard installed.

---

### Change 14 — `_pack_int6` / `_unpack_int6` (bit-packing)
**File:** `train_gpt.py`, new functions
**Type:** Original (no record implements 4-into-3 packing; records use int8 containers)
**Source:** Derived from the 4×6bit = 24bit = 3byte packing layout. All 1296 corner-value combinations verified correct with pure Python before merging.
**What:**
- `_pack_int6(q)`: int8[-32..31] → uint8 tensor (3/4 the size), returns `(packed, orig_numel)`
  - Shifts values to unsigned [0,63], groups into 4s, packs: `byte0 = v0|(v1&3)<<6`, `byte1 = v1>>2|(v2&15)<<4`, `byte2 = v2>>4|v3<<2`
- `_unpack_int6(packed, n, shape)`: reverses exactly, restores int8[-32..31]
**Why vs record's approach:** Record stores int6 in int8 containers (1 byte/value) and relies on zstd to compress the zero high-2-bits. True bit-packing saves 25% bytes BEFORE compression, reducing pre-compression size from 9MB to 6.75MB for a 9M-param model. That extra 2.25MB freed can be used for more layers or wider MLP.

---

### Change 15 — `_quantize_int6_row`
**File:** `train_gpt.py`, new function
**Type:** Ported (core formula identical to record)
**Source:** `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/train_gpt.py`, `quantize_int6_per_row`
**What:** `scale = max(abs(row)) / 31`, clamp_min to fp16 tiny, q = clamp(round(t / scale), -32, 31).int8
**Identical to record:** The `row_max / 31.0` formula, `clamp_min(torch.finfo(float16).tiny)`, and the clamp-round pattern are verbatim.
**Difference:** Private name `_quantize_int6_row` (underscore prefix, not exported).

---

### Change 16 — `quantize_state_dict_int6` + `dequantize_state_dict_int6`
**File:** `train_gpt.py`, new functions
**Type:** Ported structure, original bit-packing integration
**Source:** Structure adapted from `records/.../2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA` (`mixed_quantize_int6` / `dequantize_mixed_int6`) — same routing logic (control passthrough, small tensor fp16 passthrough, int6 for attn/mlp, int8 for rest). The key departure: this version calls `_pack_int6` to store int6 values bit-packed rather than in int8 containers.
**Key differences from record:**
- Stores `name+".packed"` (uint8 bit-packed) + `name+".n"` (orig count) + `name+".shape"` — record stored `name+".q"` (int8 container)
- No `_classify_param` function needed — just checks `any(p in name for p in _INT6_PATTERNS)` inline
- `dequantize` takes `template_sd` (original state dict) to restore dtypes without storing them separately

---

### Change 17 — Serialization block replacement in `main()`
**File:** `train_gpt.py`, end of `main()`
**Type:** Original (replaces int8+zlib block)
**Source:** Logic flow inspired by record's serialization block, but adapted for new function signatures
**What:** Replaces `quantize_state_dict_int8` + `zlib.compress` with `quantize_state_dict_int6` + `_compress` (zstd-22). Artifact filename `final_model.int6.ptz`. Adds explicit `>16MB` warning log.
**What's kept:** Roundtrip validation (reload from disk, run eval) — identical pattern to the int8 version.

---

### Artifact size estimate
- 9-layer, 512d model: ~9.2M params
- Attn+MLP matrices (~7M params): int6 bit-packed = 5.25MB + fp16 scales ~0.1MB
- Embeddings + small tensors: int8 = ~1.6MB
- Pre-compression total: ~7MB
- After zstd-22 (~50% ratio on structured weights): **~3.5MB model**
- Code: ~50KB
- **Projected total: ~3.6MB — well under 16MB budget**

Actual numbers will depend on weight distributions. Budget headroom can be used for more layers.

---

~~Pending~~ All Block 1–3 items complete.

---

## Session 4 — 2026-04-13 (Day 5 Blocks 1 & 2)

### Prediction: 2 extra layers at int6+zstd-22
**Before implementing**, as required by protocol:
- At 512d, 3×MLP: per-block matrix params = 786,432 (attn) + 1,572,864 (mlp) = 2,359,296
- 2 extra blocks = 4,718,592 params
- Int6 bit-packed: ×6/8 = 3,538,944 bytes ≈ 3.37 MB pre-compression
- zstd-22 on real trained weights (relu² sparsity) ≈ 40–50% compression
- **Prediction: ~1.7–2.0 MB extra for 2 layers**

---

### Change 18 — `mlp_mult` default: 2 → 3
**File:** `train_gpt.py`, `Hyperparameters`
**Type:** Original (value change)
**Source:** Master plan Day 5 Block 1. Validated in `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` (name contains `MLP3x`).
**What:** `MLP_MULT` default 2 → 3. MLP hidden dim: 512×2=1024 → 512×3=1536.
**Why:** Wider MLP provides more capacity within the parameter budget. relu² (already implemented) makes 84–98% of activations zero, so int6 quantization of MLP weights is very GPTQ-friendly.
**Note:** `relu²` was already correctly implemented (`torch.relu(x).square()`) with the comment "relu^2 MLP" — no code change needed there.

**Why relu² compresses better:** After STE QAT, the model learns to push most pre-activation values negative. relu clamps them to zero; then `.square()` of zero is still zero. Result: 84–98% of activations are exactly zero → the MLP output tensor is highly sparse → zstd-22 compresses it efficiently. This also regularizes the model (implicit L0 on neurons).

---

### Change 19 — `num_layers` default: 9 → 11
**File:** `train_gpt.py`, `Hyperparameters`
**Type:** Original (value change)
**Source:** Master plan Day 5 Block 1. Validated in multiple 11L records.
**What:** `NUM_LAYERS` default 9 → 11.
**Why:** Budget check shows 11L/3x at int6+zstd fits comfortably under 16MB. More layers = better depth, each layer contributes non-linear transformation the model can compose.

---

### Change 20 — `rope_dims` hyperparameter (partial RoPE, default=16)
**File:** `train_gpt.py`, `Hyperparameters`
**Type:** Ported (structure from `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/train_gpt.py`)
**Source:** Record implements `ROPE_DIMS` env var + partial Rotary + partial apply_rotary_emb.
**What:** Added `rope_dims = int(os.environ.get("ROPE_DIMS", 16))` to `Hyperparameters`.
**Why:** Applying RoPE to only 16 of 64 head dims (default). The remaining 48 dims are position-invariant: the model can learn position-invariant attention patterns (content matching, syntactic agreement) in those dims while the rotated dims handle positional relationships. This hybrid gives strictly more expressiveness than full RoPE.

---

### Change 21 — `Rotary` class: partial RoPE support
**File:** `train_gpt.py`, `Rotary`
**Type:** Ported
**Source:** `records/.../2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/train_gpt.py`, `Rotary.__init__`
**What:** Added `rope_dims: int = 0` parameter. `self.rope_dims = rope_dims if rope_dims > 0 else dim`. `inv_freq` now uses `self.rope_dims` instead of `dim` — computes fewer frequencies for partial RoPE. `cos/sin` caches have shape `[1, 1, T, rope_dims//2]` instead of `[1, 1, T, dim//2]`.
**Key difference from record:** Record also included NTK-aware scaling for long sequences; this version omits that to keep complexity low.

---

### Change 22 — `apply_rotary_emb`: partial dims passthrough
**File:** `train_gpt.py`
**Type:** Ported
**Source:** `records/.../2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/train_gpt.py`, `apply_rotary_emb`
**What:** Detects partial RoPE via `rd = cos.size(-1) * 2 < x.size(-1)`. If partial: split `x` into `x[..., :rd]` (rotated) and `x[..., rd:]` (unchanged passthrough), rotate the first segment, concatenate.
**Identical to record:** The `rd = cos.size(-1) * 2` detection and the split+concat logic are a verbatim port.

---

### Change 23 — Plumb `rope_dims` through `CausalSelfAttention`, `Block`, `GPT`, `main()`
**File:** `train_gpt.py`
**Type:** Original (plumbing)
**What:** Added `rope_dims: int = 0` param to `CausalSelfAttention`, `Block`, `GPT`. Each passes it down to `Rotary`. `main()` passes `rope_dims=args.rope_dims` to `GPT(...)`. Updated `check_artifact_size.py` similarly.

---

### Change 24 — FP16 embedding passthrough in `quantize_state_dict_int6`
**File:** `train_gpt.py`, `quantize_state_dict_int6`
**Type:** Original
**Source:** Master plan Day 5 Block 2 directive: "Keep embeddings in FP16 instead of int6."
**What:** Added explicit check `if name == "tok_emb.weight"` before the int8/int6 routing. Routes the token embedding to `passthrough_fp16` regardless of size.
**Why:** `tok_emb.weight` is the tied LM head — never STE-trained. Previous code routed it to int8 (127 levels). FP16 (65536 levels) has negligible quality penalty and costs only 1 MB (1024×512×2 bytes). The int8 path would save 0.5 MB but risk a +0.002–0.005 BPB quality penalty on the embedding. FP16 is the correct trade-off.
**Size impact:** +0.5 MB vs int8 passthrough. At 11L/3x budget of ~10–11 MB, this is acceptable.

---

### Change 25 — Test suite: `tests/test_train_gpt.py`
**File:** `tests/test_train_gpt.py` (new file)
**Type:** Original
**What:** 48 tests covering all implemented blocks:
- Day 2 Block 1: sliding-window eval params, context per token
- Day 2 Block 2: hyperparameter defaults, Muon WD behavior
- Day 3 Block 1: int6 bit-packing (roundtrip, size, edge cases), quantization pipeline
- Day 5 Block 1: relu² activation math, mlp_mult=3 shapes, num_layers=11 default
- Day 5 Block 2: partial RoPE shapes, passthrough correctness, FP16 embedding routing
**Result:** 45 passed, 3 skipped (model-forward tests require `F.rms_norm` from torch ≥ 2.4; local env has 2.2.2; will pass on RunPod).

---

## Session 9 — 2026-04-14 (Track 1 implementation)

### Context
Track 1 "Get on the Board" from UNIFIED_GAMEPLAN_april14.md.  
Branched off main: `track1/sdclip-recurrence`  
Cherry-picked existing 48-change stack from `sota/leaky-relu-smeargate-batch-int8-cleanup`.

---

### Change 49 — SDClip (std-based quantization, replaces GPTQ-lite)
**File:** `train_gpt.py`  
**Type:** Adopted from #1394 (SP8192+recurrence SOTA)  
**Source:** UNIFIED_GAMEPLAN_april14.md Track 1, Day 1  
**What changed:**
- Added module-level `_SDCLIP_K: float = float(os.environ.get("SDCLIP_K", "2.5"))`
- Replaced `_quantize_int6_row` body: removed 5-candidate percentile loop, replaced with `row_clip = k * row_std` (one-shot std-based clip)
- Updated `CastedLinear.forward` QAT to use `k * w32.std(dim=1)` instead of `row_max` so fake-quant matches post-training quantizer exactly

**Why:** GPTQ-lite minimises reconstruction MSE, but the competition optimises compression entropy. SDClip (clip at k*σ) clips outliers more aggressively, concentrating weight distributions, producing lower entropy → smaller zstd artifact. This is the single highest-value quantization improvement available.

**Key details:**
- k=2.5 is the value from #1394; configurable via `SDCLIP_K` env var
- Backward-compatible function signature; no API changes
- QAT and post-training quantizer now use the same formula (previously QAT used absmax, quantizer used percentile candidates — they were mismatched)

**Test changes:** Replaced `TestGPTQLiteClipSearch.test_gptq_selects_min_mse_among_candidates` (now wrong) with SDClip-specific tests: scale-matches-k-times-std, clips-outliers-tighter-than-absmax, k-default-is-2.5. Also added `TestSDClipQuantization` class (4 additional tests).

---

### Change 50 — Depth recurrence on bottleneck blocks 4-5
**File:** `train_gpt.py`  
**Type:** Adopted from #1394 / #1204 approach  
**Source:** UNIFIED_GAMEPLAN_april14.md Track 1, Day 1  
**What changed:**
- Added `RECUR_N` (default 2) and `RECUR_BLOCKS` (default `"4,5"`) env vars to `Hyperparameters`
- Added `recur_n: int = 1` and `recur_blocks: str = ""` params to `GPT.__init__`
- `GPT` stores `self._recur_n` and `self._recur_block_indices: list[int]`
- `forward_logits`: after the encoder loop (blocks 0-4), runs `_recur_block_indices` blocks an extra `(recur_n - 1)` times before the decoder loop

**Architecture note:** With default `RECUR_N=2, RECUR_BLOCKS=4,5`:
- Encoder runs blocks 0→1→2→3→4 (collects 5 skips)
- Extra recurrence pass: blocks 4→5 (no skip interaction — pure extra depth)
- Decoder runs blocks 5→6→7→8→9→10 (pops 5 skips normally)
- Net effect: blocks 4 and 5 each process the signal twice, giving 13 effective block applications for 11 parameters worth of compute

**Why:** Free effective depth at zero parameter cost. Proven in #1204/#1394. The encoder-decoder junction (block 4 = last encoder, block 5 = first decoder) is the bottleneck where extra refinement has the most impact.

**Risk:** None beyond potential NaN (tested and clean). The recurrence adds FLOPS during eval (~18% more for 2 blocks extra in 11-layer model) but parameter count is unchanged.

**Test additions:** `TestDepthRecurrence` class — 9 tests covering env var defaults, block index parsing, no-op cases (recur_n=1, empty blocks), forward NaN, output shape, parameter count invariance, and logit-difference verification with non-zero proj weights.

---

### Test suite fixes (pre-existing issues fixed in Session 9)
**File:** `tests/test_train_gpt.py`

1. **`TestXSA.test_xsa_output_differs_from_standard`** — was comparing scalar loss (cross-entropy mean) between XSA and non-XSA models. At random init with seed 42, y and v happen to be nearly orthogonal, so XSA has no effect. Replaced with `test_xsa_flags_survive_load_state_dict`: verifies the boolean `use_xsa` flags persist through `load_state_dict` (flags are not model params, so they should not be overwritten).

2. **`TestValueEmbeddings.test_ve_nonzero_scale_changes_output`** — was failing because `proj.weight=0` at init (residual-zero trick), so the attention output is always 0 regardless of `v_extra`. Replaced with `test_ve_pipeline_produces_nonzero_v_extra`: uses a forward hook to verify `v_extra` is non-zero when `ve_proj` and `ve_scale` are both non-zero.

3. **`TestGPTQLiteClipSearch`** — removed 3 GPTQ-lite-specific tests that now test the wrong behaviour (`test_gptq_selects_min_mse`, `test_gptq_never_worse_than_absmax`, `test_clean_tensor_at_least_as_good_as_naive`). Added 3 SDClip-specific tests.

**Final test count:** 151 passed (up from 104 + 34 skip + 2 pre-existing failures).

