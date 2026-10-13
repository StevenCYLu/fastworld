"""Decoders for streaming MG2 latents, with forkable per-session state.

StaticVAE : the original Wan2.1 causal VAE decoder (fp16), chunk 0 eager (its cache starts empty), every later chunk one
            CUDA-graph replay with the causal feature cache (32 tensors, ~1 GB at 352x640) kept in static buffers and
            updated inside the graph. Exact: same kernels as eager.
TinyVAE   : TAEHV (taew2_1) streaming decoder, lossy, tiny state.
"""
import sys
import torch
from fastworld import graph
from fastworld.models.mg2.model import CKPT, MG
from fastworld.paths import TAEHV_DIR

TAEHV_DIR = str(TAEHV_DIR)


class StaticVAE:
    def __init__(self, dev="cuda"):
        sys.path.insert(0, MG)
        from demo_utils.vae_block3 import VAEDecoderWrapper
        from demo_utils.constant import ZERO_VAE_CACHE
        self.n, self.dev = len(ZERO_VAE_CACHE), dev
        d = VAEDecoderWrapper()
        sd = torch.load(f"{CKPT}/Wan2.1_VAE.pth", map_location="cpu")
        d.load_state_dict({k: v for k, v in sd.items() if "decoder." in k or "conv2" in k})
        self.d = d.to(dev, torch.float16).eval().requires_grad_(False)
        self.d.mean, self.d.std = self.d.mean.to(dev), self.d.std.to(dev)   # float32 on device; cast per call as before
        self.cache = [None] * self.n
        self.graph = None
        self.use_graph = True      # set False to force the eager decoder (same kernels; used as the baseline)

    @torch.no_grad()
    def _eager(self, lat):
        out, cache = self.d(lat.transpose(1, 2).half(), *self.cache)
        return out, list(cache)

    def _step(self):
        out, new = self.d(self.z, *self.cin)
        for c, n in zip(self.cin, new):
            c.copy_(n)
        self.out.copy_(out)

    @torch.no_grad()
    def capture(self, warmup=2):
        """Call after the first chunk of some session (needs the steady cache shapes)."""
        assert all(torch.is_tensor(c) for c in self.cache)
        self.cin = [c.clone() for c in self.cache]
        self.z = torch.zeros(1, 3, 16, 44, 80, device=self.dev, dtype=torch.float16)
        self.out = torch.zeros(1, 12, 3, 352, 640, device=self.dev, dtype=torch.float32)
        cin = self.cin
        self.graph = graph.capture({"dec": self._step}, warmup=warmup, save=lambda: [c.clone() for c in cin],
                                   load=lambda st: [c.copy_(v) for c, v in zip(cin, st)])["dec"]
        self.cache = self.cin
        torch.cuda.synchronize()

    @torch.no_grad()
    def decode(self, lat, graph=None):
        """lat: [1,16,3,44,80] (MG2 latent chunk). Returns frames [T,3,352,640] float32 in [-1,1] on GPU."""
        graph = self.use_graph if graph is None else graph
        if any(c is None or not torch.is_tensor(c) for c in self.cache) or not graph or self.graph is None:
            out, cache = self._eager(lat)
            if self.graph is not None and all(torch.is_tensor(c) for c in cache):
                for c, n in zip(self.cin, cache):
                    c.copy_(n)
                self.cache = self.cin
            else:
                self.cache = cache
            return out[0]
        self.z.copy_(lat.transpose(1, 2))
        self.graph.replay()
        return self.out[0].clone()

    def reset(self):
        self.cache = [None] * self.n

    def state(self):
        return [c.clone() if torch.is_tensor(c) else c for c in self.cache]

    def load(self, st):
        if all(torch.is_tensor(c) for c in st) and self.graph is not None:
            for c, v in zip(self.cin, st):
                c.copy_(v, non_blocking=True)
            self.cache = self.cin
        else:
            self.cache = [c.clone() if torch.is_tensor(c) else c for c in st]

    def state_bytes(self):
        return graph.state_bytes(self.cache)


class TinyVAE:
    def __init__(self, dev="cuda", ckpt="taew2_1.pth"):
        sys.path.insert(0, TAEHV_DIR)
        from taehv import TAEHV, StreamingTAEHV
        self.t = TAEHV(f"{TAEHV_DIR}/{ckpt}").to(dev, torch.float16).eval().requires_grad_(False)
        self.s = StreamingTAEHV(self.t)
        self.dev = dev

    @torch.no_grad()
    def decode(self, lat):
        """lat: [1,16,3,44,80]. Returns frames [T,3,352,640] float32 in [-1,1] on GPU."""
        fr = [self.s.decode(lat.transpose(1, 2).half())]
        while True:
            f = self.s.decode()
            if f is None:
                break
            fr.append(f)
        fr = [f for f in fr if f is not None]
        return torch.cat(fr, 1)[0].float() * 2 - 1

    def reset(self):
        self.s.reset()

    def state(self):
        s = self.s
        return ([m.clone() if torch.is_tensor(m) else m for m in s.decoder_memory], list(s.decoder_work_queue), s.n_frames_decoded)

    def load(self, st):
        mem, q, n = st
        self.s.decoder_memory = [m.clone() if torch.is_tensor(m) else m for m in mem]
        self.s.decoder_work_queue = list(q); self.s.n_frames_decoded = n

    def state_bytes(self):
        return graph.state_bytes(self.s.decoder_memory)
