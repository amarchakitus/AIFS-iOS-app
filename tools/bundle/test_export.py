"""The export path to the iPad: everything here fails silently on device if wrong --
weights land in the wrong layer or at the wrong offset and the forecast is merely bad."""

import itertools
import json
import struct

import numpy as np
import pytest
import safetensors_io as safetensors
from export import _flatten_processor_names, csr_from_sorted_dst, rename_parameter


def test_safetensors_round_trip_keeps_every_tensor_distinct(tmp_path):
    rng = np.random.default_rng(0)
    tensors = {
        "a": rng.standard_normal((3, 5)).astype(np.float16),
        "b": rng.standard_normal(7).astype(np.float32),
        "c": rng.integers(-5, 5, (2, 2, 3)).astype(np.int32),
        "d": np.array([1, 0, 1], dtype=np.uint8),
    }
    path = tmp_path / "t.safetensors"
    safetensors.save(path, tensors, {"date": "2026-10-17T00:00:00"})
    loaded, meta = safetensors.load(path)
    assert meta == {"date": "2026-10-17T00:00:00"}
    assert loaded.keys() == tensors.keys()
    for name, t in tensors.items():
        assert loaded[name].dtype == t.dtype
        np.testing.assert_array_equal(loaded[name], t)


def test_safetensors_layout_is_contiguous_and_aligned(tmp_path):
    """The spec forbids holes, and readers (MLX) check offsets against the file size."""
    path = tmp_path / "t.safetensors"
    safetensors.save(path, {"x": np.zeros(3, np.float16), "y": np.ones(5, np.float32)})
    raw = path.read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    assert (8 + n) % 8 == 0
    header = json.loads(raw[8 : 8 + n])
    spans = sorted(v["data_offsets"] for v in header.values())
    assert spans[0][0] == 0
    assert all(a[1] == b[0] for a, b in itertools.pairwise(spans))
    assert spans[-1][1] == len(raw) - 8 - n


def test_processor_layers_flatten_in_numeric_order():
    # Chunk-major, block-minor, compared as numbers: "0:10" must not sort before "0:2".
    names = {}
    for chunk in range(2):
        for block in range(12):
            raw = f"processor.proc.{chunk}.blocks.{block}.mlp.0.weight"
            names[rename_parameter(raw)] = np.full(1, chunk * 100 + block)
    flat = _flatten_processor_names(names)
    order = [int(flat[f"processor.blocks.{i}.mlp.0.weight"][0]) for i in range(24)]
    assert order == [c * 100 + b for c in range(2) for b in range(12)]


def test_trainable_tensors_are_not_exported_as_weights():
    assert rename_parameter("encoder.trainable.trainable") is None
    assert rename_parameter("node_attributes.trainable_tensors.data.trainable") is None
    assert rename_parameter("encoder.proc.lin_key.weight") == "encoder.proc.lin_key.weight"


def test_csr_pointer_counts_edges_per_destination_including_empty_rows():
    ptr = csr_from_sorted_dst(np.array([0, 0, 2, 2, 2, 4]), num_dst=6)
    np.testing.assert_array_equal(ptr, [0, 2, 2, 5, 5, 6, 6])


def test_csr_pointer_rejects_unsorted_edges():
    with pytest.raises(ValueError, match="sorted"):
        csr_from_sorted_dst(np.array([1, 0]), num_dst=2)
