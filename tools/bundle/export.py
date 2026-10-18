"""Export AIFS Single v2 for on-device inference (MLX / iPad).

PyTorch does not run on iPadOS, so the iPad app re-implements the forward pass in MLX Swift
and reads everything it needs from a *bundle* written here:

``manifest.json``
    Shapes, variable order, normalisation, imputation, bounding -- every constant the
    PyTorch pipeline applies around the network, resolved to plain indices and numbers.
``weights.safetensors``
    The network parameters, renamed onto a flat layout (``processor.blocks.<i>``).
``graph.safetensors``
    The encoder/decoder graphs, pre-arranged for gathers: edges are already sorted by
    destination, so the encoder becomes CSR (``enc_ptr``) and the decoder, which has
    exactly three incoming edges per grid point, a dense ``(N, 3)`` table. Edge features
    and node attributes have their trainable parts concatenated in, since both are
    constants at inference time.

and initial conditions as ``ic_<date>.safetensors``: one ``(2, N)`` float32 array per
field the model reads from the state (the computed forcings are recomputed on device).

Runs in the *single* runtime -- loading the checkpoint needs its anemoi-models -- and, like
the CLI, must install the patches before ``torch.load``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import safetensors_io as safetensors  # beside this file; not the PyPI package
from aifs_mps.models import get_model
from aifs_mps.paths import input_state_path, weights_dir

LOG = logging.getLogger("aifs.export")

BUNDLE_FORMAT = "aifs-mlx-bundle"
BUNDLE_VERSION = 1

# Forcings derived from the date and grid, recomputed on device every step. Everything
# else the model reads comes from the initial state.
COMPUTED_FORCINGS = (
    "cos_latitude",
    "cos_longitude",
    "sin_latitude",
    "sin_longitude",
    "cos_julian_day",
    "cos_local_time",
    "sin_julian_day",
    "sin_local_time",
    "insolation",
)


def export_dir() -> Path:
    """Where bundles and exported ICs go (``AIFS_MPS_EXPORT_DIR``)."""
    # This repo's exports/ (gitignored), not the aifs-mps submodule's: the submodule stays
    # exactly as published.
    return Path(os.environ.get("AIFS_MPS_EXPORT_DIR", Path(__file__).resolve().parents[2] / "exports"))


def _np(t, dtype=None) -> np.ndarray:
    a = t.detach().cpu().numpy()
    return a.astype(dtype) if dtype is not None else a


def rename_parameter(name: str) -> str | None:
    """Map a ``model.model`` parameter name onto the bundle layout.

    The processor is stored by anemoi as ``proc.<chunk>.blocks.<block>``; the chunking is
    a training-time memory device with no numerical meaning, so the bundle flattens it to
    ``processor.blocks.<layer>``. Trainable node/edge tensors are not weights here -- they
    are folded into the graph -- so they map to None.
    """
    if "trainable" in name:
        return None
    parts = name.split(".")
    if parts[:2] == ["processor", "proc"]:
        chunk, block = int(parts[2]), int(parts[4])
        return ".".join(["processor", "blocks", f"{chunk}:{block}", *parts[5:]])
    return name


def _flatten_processor_names(weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Turn ``processor.blocks.<chunk>:<block>`` into a flat layer index."""
    keys = sorted(
        {k.split(".")[2] for k in weights if k.startswith("processor.blocks.")},
        key=lambda s: tuple(int(x) for x in s.split(":")),
    )
    layer_of = {k: i for i, k in enumerate(keys)}
    out = {}
    for k, v in weights.items():
        if k.startswith("processor.blocks."):
            parts = k.split(".")
            parts[2] = str(layer_of[parts[2]])
            k = ".".join(parts)
        out[k] = v
    return out


def csr_from_sorted_dst(dst: np.ndarray, num_dst: int) -> np.ndarray:
    """CSR row pointer for edges sorted by destination."""
    if np.any(np.diff(dst) < 0):
        raise ValueError("edges must be sorted by destination")
    ptr = np.zeros(num_dst + 1, dtype=np.int64)
    np.cumsum(np.bincount(dst, minlength=num_dst), out=ptr[1:])
    return ptr


def build_graph(inner, dtype: np.dtype) -> tuple[dict[str, np.ndarray], dict]:
    """Graph tensors for the bundle, plus the shape facts that go in the manifest."""
    na = inner.node_attributes
    data_attr = np.concatenate(
        [_np(na.latlons_data), _np(na.trainable_tensors["data"].trainable)], axis=1
    )
    hidden_attr = np.concatenate(
        [_np(na.latlons_hidden), _np(na.trainable_tensors["hidden"].trainable)], axis=1
    )
    num_data, num_hidden = data_attr.shape[0], hidden_attr.shape[0]

    enc, dec = inner.encoder, inner.decoder
    enc_index = _np(enc.edge_index_base).astype(np.int64)
    enc_attr = np.concatenate([_np(enc.edge_attr), _np(enc.trainable.trainable)], axis=1)
    enc_ptr = csr_from_sorted_dst(enc_index[1], num_hidden)

    dec_index = _np(dec.edge_index_base).astype(np.int64)
    dec_attr = np.concatenate([_np(dec.edge_attr), _np(dec.trainable.trainable)], axis=1)
    dec_ptr = csr_from_sorted_dst(dec_index[1], num_data)
    dec_degree = np.diff(dec_ptr)
    if not np.all(dec_degree == dec_degree[0]):
        raise ValueError("decoder graph is not fixed-degree; the dense layout needs it")
    k = int(dec_degree[0])

    tensors = {
        "data_attr": data_attr.astype(dtype),
        "hidden_attr": hidden_attr.astype(dtype),
        "enc_src": enc_index[0].astype(np.int32),
        "enc_ptr": enc_ptr.astype(np.int32),
        "enc_attr": enc_attr.astype(dtype),
        "dec_src": dec_index[0].reshape(num_data, k).astype(np.int32),
        "dec_attr": dec_attr.reshape(num_data, k, -1).astype(dtype),
    }
    facts = {
        "num_data_nodes": num_data,
        "num_hidden_nodes": num_hidden,
        "num_encoder_edges": int(enc_index.shape[1]),
        "encoder_max_in_degree": int(np.diff(enc_ptr).max()),
        "decoder_in_degree": k,
        "edge_dim": int(enc_attr.shape[1]),
        "node_attr_dim": int(data_attr.shape[1]),
    }
    return tensors, facts


def build_manifest(model, checkpoint, facts: dict, dtype: np.dtype) -> dict:
    """Everything the PyTorch pipeline applies around the network, as plain data."""
    inner = model.model
    input_vars = [None] * len(checkpoint.variable_to_input_tensor_index)
    for name, i in checkpoint.variable_to_input_tensor_index.items():
        input_vars[i] = name
    output_vars = [
        checkpoint.output_tensor_index_to_variable[i]
        for i in range(len(checkpoint.output_tensor_index_to_variable))
    ]

    norm = model.pre_processors.processors["normalizer"]
    mul, add = _np(norm._norm_mul, np.float64), _np(norm._norm_add, np.float64)
    in_idx, out_idx = _np(norm._input_idx), _np(norm._output_idx)

    imputer = model.pre_processors.processors["const_imputer"]
    imputation = [
        {"input": int(i), "output": None if o is None else int(o), "value": float(v)}
        for i, o, v in zip(
            imputer.index_inference_input,
            imputer.index_inference_output,
            imputer.replacement,
            strict=True,
        )
        if i is not None
    ]

    # The pickled module carries no config; the checkpoint metadata does.
    config = checkpoint._metadata._config
    cond_cfg = config["data"]["processors"]["conditional_nan_postprocessor"]["config"]
    conditional_nan = {
        "mask": output_vars.index(cond_cfg["remap"]),
        "targets": [output_vars.index(v) for v in cond_cfg["nan"]],
    }

    boundings = []
    for b in inner.boundings:
        kind = type(b).__name__
        entry = {"indices": [int(i) for i in np.atleast_1d(_np(b.data_index))]}
        if kind == "ReluBounding":
            entry["type"] = "relu"
        elif kind == "HardtanhBounding":
            entry.update(type="hardtanh", min=float(b.min_val), max=float(b.max_val))
        elif kind == "FractionBounding":
            entry.update(
                type="fraction",
                min=float(b.min_val),
                max=float(b.max_val),
                total=int(np.atleast_1d(_np(b.total_variable))[0]),
            )
        else:
            raise NotImplementedError(f"bounding {kind} has no on-device equivalent")
        boundings.append(entry)

    if inner.A_down is not None or inner.A_up is not None:
        raise NotImplementedError("checkpoints with spectral truncation are not supported")

    proc_cfg = config["model"]["processor"]
    enc_cfg = config["model"]["encoder"]
    first_block = inner.processor.proc[0].blocks[0]

    computed = [v for v in input_vars if v in COMPUTED_FORCINGS]
    state_vars = [v for v in input_vars if v not in COMPUTED_FORCINGS]

    return {
        "format": BUNDLE_FORMAT,
        "version": BUNDLE_VERSION,
        "model": "single",
        "description": "AIFS Single v2 (aifs-single-mse-2.0)",
        "dtype": np.dtype(dtype).name,
        "exported": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "multi_step": int(checkpoint.multi_step_input),
        "timestep_hours": int(checkpoint.timestep.total_seconds() // 3600),
        "num_channels": int(config["model"]["num_channels"]),
        **facts,
        "mapper_heads": int(enc_cfg["num_heads"]),
        "processor": {
            "layers": len(inner.processor.proc) * len(inner.processor.proc[0].blocks),
            "heads": int(first_block.attention.num_heads),
            "window": int(proc_cfg["window_size"]),
        },
        "input_vars": input_vars,
        "output_vars": output_vars,
        "state_vars": state_vars,
        "computed_forcings": computed,
        "prognostic_input": [int(i) for i in checkpoint.prognostic_input_mask],
        "prognostic_output": [int(i) for i in checkpoint.prognostic_output_mask],
        "input_norm_mul": mul[in_idx].tolist(),
        "input_norm_add": add[in_idx].tolist(),
        "output_norm_mul": mul[out_idx].tolist(),
        "output_norm_add": add[out_idx].tolist(),
        "imputation": imputation,
        "conditional_nan": conditional_nan,
        "boundings": boundings,
        "accumulations": list(checkpoint.accumulations),
    }


def export_bundle(checkpoint_path: Path, out: Path, dtype: np.dtype) -> Path:
    os.environ.setdefault("ANEMOI_INFERENCE_NUM_CHUNKS", "32")
    from aifs_mps import patches

    patches.apply_all()
    import torch
    from aifs_mps.models import check_runtime
    from anemoi.inference.checkpoint import Checkpoint

    check_runtime(get_model("single"))

    LOG.info("Loading %s", checkpoint_path)
    model = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint = Checkpoint(str(checkpoint_path))
    inner = model.model

    out.mkdir(parents=True, exist_ok=True)

    graph, facts = build_graph(inner, dtype)
    safetensors.save(out / "graph.safetensors", graph, {"format": BUNDLE_FORMAT})
    LOG.info("graph: %s", {k: v.shape for k, v in graph.items()})

    weights = {}
    for name, p in inner.named_parameters():
        new = rename_parameter(name)
        if new is not None:
            weights[new] = _np(p, dtype)
    weights = _flatten_processor_names(weights)
    safetensors.save(out / "weights.safetensors", weights, {"format": BUNDLE_FORMAT})
    LOG.info("weights: %d tensors, %.0f MB", len(weights), sum(w.nbytes for w in weights.values()) / 1e6)

    # float32: MLX's safetensors reader has no F64, and 1e-6 degrees is far below anything
    # the forcings (the only consumer) can resolve.
    grid = {
        "latitudes": np.asarray(checkpoint.latitudes, dtype=np.float32),
        "longitudes": np.asarray(checkpoint.longitudes, dtype=np.float32),
    }
    safetensors.save(out / "grid.safetensors", grid, {"format": BUNDLE_FORMAT})

    # The same N320 -> 0.25 deg operator the Zarr writer uses, for drawing maps on device.
    # Rows run 90N -> 90S, columns 0E -> 359.75E.
    # What the device needs to build initial conditions from ECMWF open data itself:
    # the 0.25 deg -> N320 operator `opendata.build_input_state` regrids with, and the
    # N320 sea mask it NaNs the land-only fields with. Columns of the operator are the
    # open-data grid *after* the 180-degree longitude roll (0E first), rows 90N first.
    import earthkit.data as ekd
    from aifs_mps.paths import lsm_path
    from aifs_mps.regrid import LATLON_SHAPE, latlon_to_n320_matrix, n320_to_latlon_matrix

    to_n320 = latlon_to_n320_matrix()
    lsm = ekd.from_source("file", str(lsm_path()))[0].to_numpy(flatten=True)
    if lsm.shape[0] != facts["num_data_nodes"]:
        raise ValueError(f"land-sea mask has {lsm.shape[0]} points, model has {facts['num_data_nodes']}")
    safetensors.save(
        out / "opendata.safetensors",
        {
            "indptr": to_n320.indptr.astype(np.int32),
            "indices": to_n320.indices.astype(np.int32),
            "weights": to_n320.data.astype(np.float32),
            "sea": np.equal(lsm, 0).astype(np.uint8),
        },
        {"format": BUNDLE_FORMAT, "rows": LATLON_SHAPE[0], "cols": LATLON_SHAPE[1]},
    )

    matrix = n320_to_latlon_matrix()
    safetensors.save(
        out / "display.safetensors",
        {
            "indptr": matrix.indptr.astype(np.int32),
            "indices": matrix.indices.astype(np.int32),
            "weights": matrix.data.astype(np.float32),
        },
        {"format": BUNDLE_FORMAT, "rows": LATLON_SHAPE[0], "cols": LATLON_SHAPE[1]},
    )

    manifest = build_manifest(model, checkpoint, facts, dtype)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    LOG.info("Bundle written to %s", out)
    return out


def export_ic(state: dict, manifest: dict, out: Path) -> Path:
    """Write the fields of `state` the bundle's model reads, in manifest order."""
    fields = state["fields"]
    missing = [v for v in manifest["state_vars"] if v not in fields]
    if missing:
        raise SystemExit(f"initial state lacks {', '.join(missing)}")
    tensors = {v: np.asarray(fields[v], dtype=np.float32) for v in manifest["state_vars"]}
    shape = (manifest["multi_step"], manifest["num_data_nodes"])
    bad = {v: t.shape for v, t in tensors.items() if t.shape != shape}
    if bad:
        raise SystemExit(f"fields with unexpected shape (want {shape}): {bad}")
    date = state["date"]
    path = out / f"ic_{date.strftime('%Y%m%dT%H')}.safetensors"
    out.mkdir(parents=True, exist_ok=True)
    safetensors.save(path, tensors, {"date": date.isoformat(), "model": manifest["model"]})
    LOG.info("IC for %s: %d fields -> %s", date, len(tensors), path)
    return path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="tools/bundle/aifs-export", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bundle", help="convert the AIFS Single v2 checkpoint to an MLX bundle")
    b.add_argument("--checkpoint", type=Path, default=weights_dir() / get_model("single").checkpoint)
    b.add_argument("--out", type=Path, default=export_dir() / "aifs-single-v2")
    b.add_argument("--dtype", choices=["float16", "float32"], default="float16")

    ic = sub.add_parser("ic", help="export a cached initial state for the device")
    src = ic.add_mutually_exclusive_group(required=True)
    src.add_argument("--date", help="init time of a cached state, e.g. 20260924T00")
    src.add_argument("--input-state", type=Path, help="a pickled state")
    ic.add_argument("--bundle", type=Path, default=export_dir() / "aifs-single-v2")
    ic.add_argument("--out", type=Path, default=export_dir() / "ic")

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")

    if args.cmd == "bundle":
        export_bundle(args.checkpoint, args.out, np.dtype(args.dtype))
        return

    manifest = json.loads((args.bundle / "manifest.json").read_text())
    path = args.input_state or input_state_path(datetime.datetime.strptime(args.date, "%Y%m%dT%H"))
    if not path.exists():
        raise SystemExit(f"{path} not found; run a forecast for that date first to cache it")
    with path.open("rb") as f:
        state = pickle.load(f)
    export_ic(state, manifest, args.out)


if __name__ == "__main__":
    main(sys.argv[1:])
