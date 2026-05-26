from pathlib import Path

import numpy as np
import pytest

from kronweave.io.tensorsuite import TNS_MAGIC, parse_tns_header, read_tns, write_tns


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
    np.testing.assert_array_equal(data.coords, coords)
    np.testing.assert_allclose(data.values, values)


def test_invalid_tns_magic_raises(tmp_path: Path):
    path = tmp_path / "bad.tns"
    path.write_text("bad\n% version: 1.0\n% name: x\n2 2 2 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid TNS magic"):
        parse_tns_header(path)
