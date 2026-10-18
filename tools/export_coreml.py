# /// script
# requires-python = ">=3.12"
# dependencies = ["coremltools==9.0", "numpy"]
# ///
"""Compile AIFS Single v2's processor into Core ML models for the Neural Engine.

The processor is 16 transformer layers over 40,320 mesh nodes, 60% of a forecast step.
Its dense parts -- layer norms, the q/k/v and output projections, and the 1024 -> 4096 ->
1024 MLPs -- run on the iPad's Neural Engine at ~6 TFLOPS versus ~2.2 on its GPU
(measured on an M1 iPad Pro). The sliding-window attention between them stays on the GPU
in MLX, so each layer is split around it, and layer l's second half is fused with layer
l+1's first half to halve the hand-offs:

    proc_in    x                -> q, k, v                      (layer 0, first half)
    proc_NN    x, att(NN-1)     -> x, q, k, v                   (layer NN-1 second half,
                                                                  layer NN first half)
    proc_out   x, att(15)       -> x                            (layer 15, second half)

and the decoder's dense second half, everything after its graph attention, per grid point:

    dec_post   a = att + lin_self(x), dst  -> y (120 outputs, normalised)

Tensors are (1, C, 1, S) fp16, the Neural Engine's channels-first layout, with S tokens
per call (the 40,320 nodes in 10 chunks). Linear layers are 1x1 convolutions.

    uv run tools/export_coreml.py aifs-mps/exports/aifs-single-v2

writes <bundle>/coreml/*.mlmodelc and records it in <bundle>/coreml/coreml.json.
"""

import json
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

CHUNK = 4032  # tokens per call; 40320 = 10 x 4032, and full Neural Engine speed from ~4k
DEC_CHUNK = 8470  # grid points per decoder call; 542080 = 64 x 8470


def load_safetensors(path: Path) -> dict[str, np.ndarray]:
    dtypes = {"F16": np.float16, "F32": np.float32, "I32": np.int32}
    raw = path.read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + n])
    header.pop("__metadata__", None)
    out = {}
    for name, info in header.items():
        a, b = info["data_offsets"]
        out[name] = np.frombuffer(raw[8 + n + a : 8 + n + b], dtype=dtypes[info["dtype"]]).reshape(info["shape"])
    return out


def main(bundle: Path) -> None:
    manifest = json.loads((bundle / "manifest.json").read_text())
    layers, c = manifest["processor"]["layers"], manifest["num_channels"]
    tokens = manifest["num_hidden_nodes"]
    if tokens % CHUNK:
        raise SystemExit(f"{tokens} mesh nodes do not split into chunks of {CHUNK}")
    W = load_safetensors(bundle / "weights.safetensors")

    def w16(name):
        return W[name].astype(np.float16)

    def conv(x, name, bias=True):
        weight = w16(f"{name}.weight")[:, :, None, None]
        if bias:
            return mb.conv(x=x, weight=weight, bias=w16(f"{name}.bias"))
        return mb.conv(x=x, weight=weight)

    def norm(x, name):
        return mb.layer_norm(x=x, axes=[1], gamma=w16(f"{name}.weight"), beta=w16(f"{name}.bias"),
                             epsilon=np.float16(1e-5))

    def qkv(x, layer, names=("q", "k", "v")):
        p = f"processor.blocks.{layer}"
        h = norm(x, f"{p}.layer_norm_attention")
        return tuple(mb.identity(x=conv(h, f"{p}.attention.lin_{n}", bias=False), name=n) for n in names)

    def second_half(x, att, layer, name=None):
        p = f"processor.blocks.{layer}"
        x = mb.add(x=x, y=conv(att, f"{p}.attention.projection"))
        h = mb.gelu(x=conv(norm(x, f"{p}.layer_norm_mlp"), f"{p}.mlp.0"), mode="EXACT")
        return mb.add(x=x, y=conv(h, f"{p}.mlp.2"), name=name) if name else mb.add(x=x, y=conv(h, f"{p}.mlp.2"))

    def spec():  # one object per input: MIL names inputs after the bound argument
        return mb.TensorSpec(shape=(1, c, 1, CHUNK), dtype=types.fp16)

    def program(kind, layer):
        if kind == "in":
            @mb.program(input_specs=[spec()], opset_version=ct.target.iOS18)
            def prog(x):
                return qkv(x, 0)
        elif kind == "mid":
            @mb.program(input_specs=[spec(), spec()], opset_version=ct.target.iOS18)
            def prog(x, att):
                y = second_half(x, att, layer - 1, name="x_out")
                return (y, *qkv(y, layer))
        else:
            @mb.program(input_specs=[spec(), spec()], opset_version=ct.target.iOS18)
            def prog(x, att):
                return second_half(x, att, layers - 1, name="x_out")
        return prog

    out = bundle / "coreml"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    names = ["proc_in", *[f"proc_{i:02d}" for i in range(1, layers)], "proc_out"]
    with tempfile.TemporaryDirectory() as tmp:
        for i, name in enumerate(names):
            kind = "in" if i == 0 else "out" if i == len(names) - 1 else "mid"
            model = ct.convert(
                program(kind, i), compute_precision=ct.precision.FLOAT16,
                minimum_deployment_target=ct.target.iOS18, compute_units=ct.ComputeUnit.CPU_AND_NE)
            package = Path(tmp) / f"{name}.mlpackage"
            model.save(str(package))
            subprocess.run(["xcrun", "coremlcompiler", "compile", str(package), str(out)],
                           check=True, capture_output=True)
            print(f"  {name}.mlmodelc", flush=True)
        # Decoder: projection + residual, MLP + residual, then the output head.
        n_out = len(manifest["output_vars"])
        grid = manifest["num_data_nodes"]
        if grid % DEC_CHUNK:
            raise SystemExit(f"{grid} grid points do not split into chunks of {DEC_CHUNK}")

        def dspec():
            return mb.TensorSpec(shape=(1, c, 1, DEC_CHUNK), dtype=types.fp16)

        @mb.program(input_specs=[dspec(), dspec()], opset_version=ct.target.iOS18)
        def dec_post(a, dst):
            p = "decoder.proc"
            out = mb.add(x=dst, y=conv(a, f"{p}.projection"))
            h = mb.gelu(x=conv(norm(out, f"{p}.layer_norm_mlp_dst"), f"{p}.node_dst_mlp.0"), mode="EXACT")
            out = mb.add(x=out, y=conv(h, f"{p}.node_dst_mlp.2"))
            return conv(norm(out, "decoder.node_data_extractor.0"), "decoder.node_data_extractor.1")

        model = ct.convert(
            dec_post, compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.iOS18, compute_units=ct.ComputeUnit.CPU_AND_NE)
        package = Path(tmp) / "dec_post.mlpackage"
        model.save(str(package))
        subprocess.run(["xcrun", "coremlcompiler", "compile", str(package), str(out)],
                       check=True, capture_output=True)
        print("  dec_post.mlmodelc", flush=True)
        output_name = model.get_spec().description.output[0].name

    (out / "coreml.json").write_text(json.dumps(
        {"chunk": CHUNK, "chunks": tokens // CHUNK, "models": names,
         "decoder": {"model": "dec_post", "chunk": DEC_CHUNK, "outputs": n_out, "output": output_name}},
        indent=1) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
