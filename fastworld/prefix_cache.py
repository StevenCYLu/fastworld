"""Exact cross-request prefix sharing for autoregressive world models.

A request is a path of chunks, each chunk a hashable conditioning key (for Matrix-Game-2: (image, (key, mouse))).
Chunk k of a path depends only on chunks 0..k (the noise is seeded by the path prefix), so two requests sharing a
prefix produce bitwise-identical chunks for it. The cache is a radix tree over chunk keys; a node holds its chunk's
latent, its decoded uint8 frames and, every `every` chunks, a full session checkpoint (runtime state + decoder state).

Serving a path:
  1. walk the tree; the matched chunks' frames come from the cache;
  2. if the path continues past the match, restore the deepest checkpoint on the matched chain and replay the
     matched chunks after it (one context forward + one decoder step each, both from the stored latents);
  3. generate and decode the rest, inserting nodes.
Branching from any served prefix therefore costs a checkpoint restore (<1 ms) plus at most `every - 1` replays.

Duck-typed interfaces (see fastworld/models/mg2 for an implementation):
  runtime:  gen_chunk(path, k) -> latent;  prefill(latent, path, k);  state() -> obj;  load(obj);  [reset()]
  decoder:  decode(latent) -> frames [T, 3, H, W] float in [-1, 1];  reset();  state() -> obj;  load(obj)
"""
import time

import torch


def to_u8(frames):
    return ((frames.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).cpu()


class Node:
    __slots__ = ("children", "lat", "frames", "ckpt", "depth")

    def __init__(self, depth):
        self.children, self.lat, self.frames, self.ckpt, self.depth = {}, None, None, None, depth


class PrefixCache:
    def __init__(self, runtime, decoder, every=2):
        """every: checkpoint period in chunks; None disables sharing (every request generated from scratch)."""
        self.rt, self.dec, self.every = runtime, decoder, every
        self.root, self.n_ckpt = Node(-1), 0

    def _fresh(self):
        if hasattr(self.rt, "reset"):
            self.rt.reset()
        self.dec.reset()

    @torch.no_grad()
    def serve(self, path):
        """Returns (list of uint8 frame tensors, one per chunk; stats dict with per-chunk completion times)."""
        torch.cuda.synchronize()
        t0 = time.time()
        C = len(path)
        node, chain = self.root, []
        if self.every:
            for c in path:
                if c not in node.children:
                    break
                node = node.children[c]
                chain.append(node)
        m = len(chain)
        st = dict(hit=m, restored=0, replayed=0, generated=C - m, done=[])
        frames = [n.frames for n in chain]
        if m < C:
            ck = max([i for i, n in enumerate(chain) if n.ckpt is not None], default=-1)
            if ck >= 0:
                rt_state, dec_state = chain[ck].ckpt
                self.rt.load(rt_state)
                self.dec.load(dec_state)
                st["restored"] = 1
            else:
                self._fresh()
            for k in range(ck + 1, m):  # replay matched chunks after the checkpoint
                self.rt.prefill(chain[k].lat, path, k)
                self.dec.decode(chain[k].lat)
                st["replayed"] += 1
        torch.cuda.synchronize()
        st["done"] = [time.time() - t0] * m
        node = chain[-1] if chain else self.root
        for k in range(m, C):
            lat = self.rt.gen_chunk(path, k)
            fr = to_u8(self.dec.decode(lat))
            frames.append(fr)
            if self.every:
                n = Node(k)
                n.lat, n.frames = lat.clone(), fr
                if (k + 1) % self.every == 0 and k < C - 1:
                    n.ckpt = (self.rt.state(), self.dec.state())
                    self.n_ckpt += 1
                node.children[path[k]] = n
                node = n
            st["done"].append(time.time() - t0)
        st["time"] = time.time() - t0
        return frames, st
