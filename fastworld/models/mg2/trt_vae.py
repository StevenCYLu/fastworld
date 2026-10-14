"""TensorRT FP16 engine for the Wan2.1 VAE decoder step (approximate; ~3x faster than the exact decoder on Jetson Thor).

TRTVAE has the StaticVAE interface (decode / reset / state / load / state_bytes), so it plugs into PrefixCache. Chunk 0
runs the eager decoder (its feature cache starts empty, a different graph); every later chunk is one engine call that
reads the 32-tensor causal cache and writes the next one. The cache lives in two static buffer sets used alternately,
so no copies are needed. The engine has static shapes for MG2's 352x640, 3-latent-frame chunks.

  python -m fastworld.models.mg2.trt_vae        # export ONNX + build the engine into build/trt/ (~15 min on Thor)
"""
import os, time
import torch
from fastworld.models.mg2.vae import StaticVAE
from fastworld.paths import ROOT

TRT_DIR = os.environ.get("FASTWORLD_TRT_DIR", f"{ROOT}/build/trt")
ONNX, PLAN = f"{TRT_DIR}/vae_dec.onnx", f"{TRT_DIR}/vae_dec_fp16.plan"


def _steady_cache(V):
    g = torch.Generator(device="cuda").manual_seed(0)
    V.reset()
    for _ in range(2):
        V.decode(torch.randn(1, 16, 3, 44, 80, device="cuda", generator=g, dtype=torch.bfloat16), graph=False)
    return [c.clone() for c in V.cache]


@torch.no_grad()
def build(onnx=ONNX, plan=PLAN, norm_fp32=False):
    import tensorrt as trt
    os.makedirs(os.path.dirname(plan), exist_ok=True)
    V = StaticVAE()
    cache = _steady_cache(V)
    n = len(cache)
    if os.path.exists(onnx):
        cache = None

    class Step(torch.nn.Module):
        def __init__(s, d):
            super().__init__(); s.d = d

        def forward(s, z, *c):
            out, nc = s.d(z, *c)
            return (out, *nc)

    z = torch.zeros(1, 3, 16, 44, 80, device="cuda", dtype=torch.float16)
    if cache is not None:
        torch.onnx.export(Step(V.d), (z, *cache), onnx, dynamo=False, opset_version=17,
                          input_names=["z"] + [f"c{i}" for i in range(n)], output_names=["out"] + [f"nc{i}" for i in range(n)])
    log = trt.Logger(trt.Logger.WARNING)
    b = trt.Builder(log); net = b.create_network(0); p = trt.OnnxParser(net, log)
    assert p.parse_from_file(onnx), [p.get_error(i) for i in range(p.num_errors)]
    cfg = b.create_builder_config(); cfg.set_flag(trt.BuilderFlag.FP16)
    if norm_fp32:
        # optional: RMS norms (F.normalize -> ReduceL2 ... Div ... Mul) in fp32. Measured on Thor: +2 % time, output
        # changes only at the 78 dB level vs the fp16 engine, so it is off by default.
        scopes = {net.get_layer(i).name.rsplit("/", 1)[0] + "/" for i in range(net.num_layers)
                  if net.get_layer(i).type == trt.LayerType.REDUCE}
        for i in range(net.num_layers):
            L = net.get_layer(i)
            if any(L.name.startswith(sc) for sc in scopes) and L.type not in (trt.LayerType.CONSTANT, trt.LayerType.SHAPE):
                L.precision = trt.float32
                for j in range(L.num_outputs):
                    if L.get_output(j).dtype in (trt.float16, trt.float32):
                        L.set_output_type(j, trt.float32)
        cfg.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    t = time.time()
    ser = b.build_serialized_network(net, cfg)
    assert ser is not None, "TensorRT build failed"
    open(plan, "wb").write(ser)
    print(f"built {plan} in {time.time() - t:.0f} s")


class TRTVAE:
    def __init__(self, plan=PLAN, dev="cuda"):
        import tensorrt as trt
        self.V = StaticVAE(dev)                       # chunk 0 (empty cache) runs eager
        self.rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.eng = self.rt.deserialize_cuda_engine(open(plan, "rb").read())
        self.ctx = self.eng.create_execution_context()
        shapes = _steady_cache(self.V)
        # contiguous: TensorRT reads/writes dense NCDHW bytes, but three eager cache tensors (after each spatial
        # upsample) come out with permuted strides; empty_like would keep those and scramble the eager->engine handover
        self.bufs = [[torch.empty(c.shape, dtype=c.dtype, device=c.device) for c in shapes] for _ in range(2)]
        self.cur, self.warm = 0, False                # warm: bufs[cur] holds a steady cache
        self.z = torch.zeros(1, 3, 16, 44, 80, device=dev, dtype=torch.float16)
        self.out = torch.zeros(1, 12, 3, 352, 640, device=dev, dtype=torch.float32)
        self.names = [self.eng.get_tensor_name(i) for i in range(self.eng.num_io_tensors)]
        for nm in self.names:
            assert tuple(self.eng.get_tensor_shape(nm)) == tuple(self._t(nm).shape), nm
        self.V.reset()

    def _t(self, nm):
        if nm == "z":
            return self.z
        if nm == "out":
            return self.out
        i = int(nm.lstrip("nc"))
        return self.bufs[self.cur][i] if nm.startswith("c") else self.bufs[1 - self.cur][i]

    @torch.no_grad()
    def decode(self, lat, graph=None):
        if not self.warm:
            out = self.V.decode(lat, graph=False)
            if all(torch.is_tensor(c) and c.shape == d.shape for c, d in zip(self.V.cache, self.bufs[self.cur])):
                for d, c in zip(self.bufs[self.cur], self.V.cache):
                    d.copy_(c)
                self.warm = True
            return out
        self.z.copy_(lat.transpose(1, 2))
        for nm in self.names:
            self.ctx.set_tensor_address(nm, self._t(nm).data_ptr())
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        self.cur = 1 - self.cur
        return self.out[0].clone()

    def reset(self):
        self.V.reset(); self.warm = False

    def state(self):
        return ("trt", [c.clone() for c in self.bufs[self.cur]]) if self.warm else ("eager", self.V.state())

    def load(self, st):
        kind, s = st
        if kind == "trt":
            for d, c in zip(self.bufs[self.cur], s):
                d.copy_(c, non_blocking=True)
            self.warm = True
        else:
            self.V.load(s); self.warm = False

    def state_bytes(self):
        return sum(c.numel() * c.element_size() for c in self.bufs[0])


if __name__ == "__main__":
    build()
