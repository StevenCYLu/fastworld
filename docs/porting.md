# Porting guide: an exact static-shape CUDA-graph runtime for a new world model

This is the recipe used for the four models in this repo. It is deliberately conservative: every change must keep the
output bitwise identical to the official code, so that the result is an acceleration of the released model and not a
different model. Expect a day or two per model; most of the time goes into finding the hidden host syncs and
reproducing the official RNG order.

## 0. Decide whether it is worth it

Profile one steady-state chunk of the official loop (`fastworld.common.gpu_profile`): wall time vs CUDA-kernel time,
kernel launches and host syncs. The GPU-busy fraction (kernel / wall) bounds what graphs can give:

| GPU busy | what to expect |
|---|---|
| < 70 % (small model, small resolution, many launches/syncs: Matrix-Game-2) | 1.3-1.7x from graphs alone |
| 90-98 % (1.3B at 480p+, 5B at 720p) | 1.05-1.2x, almost all of it from redundant work the rewrite removes |

Removing redundant work is often the bigger win on compute-bound models: whole-cache clones per forward, cross-attention
K/V or text/camera/time embeddings recomputed every denoising step, fp64 RoPE tables rebuilt per call, outputs computed
and discarded. All of these are exact to remove.

## 1. Wrap the official loop one chunk at a time

Write a `Model` class that loads the official pipeline unchanged and exposes `gen_chunk(path, k)` (all denoising steps
of chunk k + the cache-update forward), `prefill(latent, path, k)` (cache-update forward only), `reset()`. Check it is
bitwise identical to the official entry point (`inference.py` / `pipe.generate()`), and that eager is deterministic
(eager vs eager). This is the reference for everything after.

## 2. Make the shapes static

Attention windows in these models are bounded (rolling window, sink + window, or a fixed memory count), so from some
chunk on every tensor shape is constant. Replace rolling caches whose indices live in device tensors (read back with
`.item()`) by fixed per-layer buffers plus host-side (Python int) bookkeeping that replays the official index logic.
Warm-up positions with different shapes get their own graph (MG2: chunk 0 vs steady; LongLive: one per window size).
Shape-changing events that are rare (prompt switch re-cache, memory retrieval) can run eagerly outside the graph.

The order of keys in the attention window must match the official code exactly, otherwise flash-attention's output
differs in the last bits.

## 3. Everything chunk-dependent goes through static input buffers

RoPE multiplier tables, timestep embeddings, noise, conditions/actions, camera features: computed on the host or in
small eager ops, then `copy_` into buffers the captured graph reads. Precompute what is constant per denoising step
(timestep sinusoid, scheduler sigmas, UniPC coefficients) with the same ops the official code uses.

Noise must be drawn in the official RNG order from a generator seeded the official way, or the outputs differ. Watch
for warped/float timesteps: MG2's `add_noise` takes a float step that the integer forward timestep does not reveal.

## 4. Remove hidden host<->device traffic inside the forward

Common offenders: `flash_attention` wrappers that build `cu_seqlens` on the CPU every call, `torch.arange` on CPU +
copy for sinusoidal embeddings, `torch.tensor([...], device="cuda")` in scheduler steps, `linalg.solve` on tiny
matrices (a sync), `nonzero().item()` for sigma lookup, RoPE tables kept on the CPU. Replace with device constants or
direct calls to the underlying kernel (`flash_attn_varlen_func` with device `cu_seqlens`).

## 5. Capture

`fastworld.graph.capture({key: fn})` warms each function on a side stream, records one `torch.cuda.CUDAGraph` per key
in a shared pool, and restores the session state afterwards (warm-up mutates it). A chunk = one graph: all denoising
forwards + scheduler steps + the cache-update forward. Replay is `graph.replay()` after filling the input buffers.

## 6. Validate in layers (all on the same GPU, same process)

1. prepared inputs equal the official ones (RoPE tables, action windows, index plans);
2. one forward, chunk 0, bitwise;
3. multi-chunk rollout without graphs, bitwise per chunk (and per denoising step if cheap);
4. the same with graphs, two rollouts;
5. save state, run a different chunk, restore, continue: bitwise with the uninterrupted rollout.

Report median warm chunk time for eager vs static (with and without graphs), GPU-busy, state bytes, capture time, and a
`torch.compile` baseline with the model's recommended mode (warm time, compile time, bitwise or not). See
`scripts/mg2/test_static.py` and `scripts/mg2/compile_baselines.py` for the shape of these scripts, and
`docs/models/*.md` for what the numbers looked like on the models done so far.
