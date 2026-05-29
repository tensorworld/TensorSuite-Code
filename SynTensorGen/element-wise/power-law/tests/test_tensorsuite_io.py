from pathlib import Path
import json

import numpy as np
import pytest

from kronweave.io.tensorsuite import TNS_MAGIC, BlockTensorData, TensorData, parse_tns_header, read_tns, write_tns


def test_write_and_read_tns(tmp_path: Path):
    path = tmp_path / "x.tns"
    coords = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.uint64)
    values = np.asarray([1.0, 2.5], dtype=np.float64)
    write_tns(path, "x", [4, 5, 6], coords, values)

    text = path.read_text(encoding="utf-8").splitlines()
    assert text[0] == TNS_MAGIC
    assert text[3] == "3 4 5 6 2"

    header = parse_tns_header(path)
    assert header.name == "x"
    assert header.dimensions == (4, 5, 6)
    assert header.nnz == 2

    data = read_tns(path)
    assert isinstance(data, TensorData)
    np.testing.assert_array_equal(data.coords, coords)
    np.testing.assert_allclose(data.values, values)


def test_invalid_tns_magic_raises(tmp_path: Path):
    path = tmp_path / "bad.tns"
    path.write_text("bad\n% version: 0.1\n% name: x\n2 2 2 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid TNS magic"):
        parse_tns_header(path)


def test_read_tns_dispatches_to_block_loader_with_metadata(tmp_path: Path):
    bundle = tmp_path / "block_x"
    bundle.mkdir()
    tns = bundle / "block_x.tns"
    tns.write_text(
        "\n".join(
            [
                "%%TensorSuite-TNS",
                "% version: 0.1",
                "% name: block_x",
                "3 4 4 4 32",
                "2 2 2 4",
                "2 2",
                "2 2",
                "2 2",
                "0 0 0",
                "0 1 1",
                "1 0 1",
                "1 1 0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    metadata = {
        "version": "0.1",
        "name": "block_x",
        "group": "test",
        "id": "0001",
        "time": "2026-05-06",
        "source_type": "synthetic",
        "source": "unit test",
        "source_url": "",
        "value_type": "float64",
        "value_domain": "general",
        "values_provided": False,
        "index_type": "uint64",
        "endianness": "none",
        "index_base": 0,
        "sorted": "lexicographic",
        "sort_order": [1, 2, 3],
        "duplicates": "disallowed",
        "explicit_zeros": "disallowed",
        "pattern_symmetry": "no",
        "numerical_symmetry": "no",
        "sparsity_type": "block",
        "dense_modes": [],
        "order": 3,
        "dimensions": [4, 4, 4],
        "nnz": 32,
        "block_partitions": [2, 2, 2],
        "nnz_block": 4,
        "files": {"text": "block_x.tns", "binary": None, "readme": "README.md"},
    }
    (bundle / "block_x_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    data = read_tns(tns)

    assert isinstance(data, BlockTensorData)
    assert data.header.block_partitions == (2, 2, 2)
    assert data.header.nnz_block == 4
    np.testing.assert_array_equal(data.block_coords, np.asarray([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]))
    assert [arr.tolist() for arr in data.partition_sizes] == [[2, 2], [2, 2], [2, 2]]
