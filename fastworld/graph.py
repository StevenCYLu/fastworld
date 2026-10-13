"""CUDA-graph capture for a static-shape runtime.

A runtime exposes a few callables that read and write only static buffers (inputs, outputs, session state) and
launch no host syncs. `capture` warms each one up on a side stream (cuBLAS/flash-attn workspaces, lazy inits),
then records each as one torch.cuda.CUDAGraph. All graphs share one memory pool: only one of them runs at a time.
Warm-up and capture execute the callables, which mutates the session state, so the state is saved before and
restored after; `replay` is then bitwise identical to running the callable eagerly.
"""
import torch


@torch.no_grad()
def capture(fns, warmup=2, save=None, load=None):
    """fns: {key: callable}. Returns {key: CUDAGraph}. save()/load(state) snapshot and restore the session state."""
    saved = save() if save else None
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for fn in fns.values():
            for _ in range(warmup):
                fn()
    torch.cuda.current_stream().wait_stream(s)
    graphs, pool = {}, None
    for key, fn in fns.items():
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=pool):
            fn()
        pool = g.pool()
        graphs[key] = g
    if load:
        load(saved)
    torch.cuda.synchronize()
    return graphs


def state_bytes(tensors):
    return sum(t.numel() * t.element_size() for t in tensors if torch.is_tensor(t))
