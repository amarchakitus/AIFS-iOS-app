"""Minimal safetensors writer/reader, so the export path needs no extra dependency.

The format is an 8-byte little-endian header length, a JSON header naming each tensor's
dtype, shape and byte range, then the raw little-endian data, contiguous and without gaps
(the spec forbids holes). MLX and MLX Swift load it natively, which is the point: it is the
hand-off format to the iPad.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

__all__ = ["load", "save"]

_DTYPES = {
    np.dtype("float16"): "F16",
    np.dtype("float32"): "F32",
    np.dtype("float64"): "F64",
    np.dtype("int32"): "I32",
    np.dtype("int64"): "I64",
    np.dtype("uint8"): "U8",
    np.dtype("bool"): "BOOL",
}
_NUMPY = {v: k for k, v in _DTYPES.items()}


def save(path: Path, tensors: dict[str, np.ndarray], metadata: dict[str, str] | None = None) -> None:
    """Write `tensors` to `path`, staged through a ``.partial`` and renamed on success."""
    path = Path(path)
    header: dict[str, dict] = {}
    if metadata:
        header["__metadata__"] = {k: str(v) for k, v in metadata.items()}
    arrays = {}
    offset = 0
    for name, array in tensors.items():
        array = np.ascontiguousarray(array)
        if array.dtype not in _DTYPES:
            raise TypeError(f"{name}: unsupported dtype {array.dtype}")
        array = array.astype(array.dtype.newbyteorder("<"), copy=False)
        header[name] = {
            "dtype": _DTYPES[array.dtype],
            "shape": list(array.shape),
            "data_offsets": [offset, offset + array.nbytes],
        }
        offset += array.nbytes
        arrays[name] = array

    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)  # spaces are valid JSON padding; keeps data 8-aligned

    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for array in arrays.values():
            f.write(array.tobytes())
    partial.replace(path)


def load(path: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Read every tensor in `path` (memory-mapped) and the metadata dict."""
    path = Path(path)
    with path.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    metadata = header.pop("__metadata__", {})
    raw = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + n)
    tensors = {}
    for name, info in header.items():
        start, end = info["data_offsets"]
        tensors[name] = raw[start:end].view(_NUMPY[info["dtype"]]).reshape(info["shape"])
    return tensors, metadata
