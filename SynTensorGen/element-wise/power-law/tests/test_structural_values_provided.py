from __future__ import annotations

from pathlib import Path

import yaml

from kronweave.api import generate_from_config, validate_bundle
from kronweave.io.tensorsuite import read_tns


def test_values_provided_false_writes_text_pattern_only_bundle(tmp_path: Path):
    cfg = {
        "name": "structural_only",
        "output_dir": str(tmp_path / "out"),
        "values_provided": False,
        "generator": {"type": "zipf", "random_seed": 23},
        "tensor": {"sizes": [4, 4], "index_base": 0, "nnz": 8},
        "format": {
            "storage": "coordinate",
            "index_type": "uint64",
            "endianness": "little",
            "sorted": "lexicographic",
            "sorted_order": [0, 1],
            "explicit_zeros": "disallowed",
            "pattern_symmetry": "no",
            "numerical_symmetry": "no",
            "sparsity_type": "element",
            "dense_modes": [],
        },
        "zipf": {"alpha": [1.3, 1.4]},
        "value": {"type": "float64", "mode": "constant", "constant": 1.0},
        "duplicates": {"policy": "keep"},
        "output": {
            "write_tns": True,
            "write_tnsb": False,
            "write_metadata": True,
            "write_readme": True,
            "copy_config": False,
        },
    }
    path = tmp_path / "structural.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    result = generate_from_config(path)
    metadata = validate_bundle(result["bundle_dir"])["metadata"]
    lines = result["tns"].read_text(encoding="utf-8").splitlines()

    assert metadata["values_provided"] is False
    assert metadata["files"]["binary"] is None
    assert metadata["files"]["config"] is None
    assert not result["tnsb"].exists()
    assert all(len(line.split()) == metadata["order"] for line in lines[4:])
    data = read_tns(result["tns"])
    assert data.coords.shape[1] == metadata["order"]


def test_provide_value_alias_normalizes_to_values_provided(tmp_path: Path):
    cfg = {
        "name": "provide_value_alias",
        "output_dir": str(tmp_path / "out"),
        "provide_value": True,
        "generator": {"type": "zipf"},
        "tensor": {"sizes": [3, 3], "index_base": 0, "nnz": 4},
        "format": {
            "storage": "coordinate",
            "index_type": "uint64",
            "endianness": "little",
            "sorted": "none",
            "sorted_order": [],
            "explicit_zeros": "allowed",
            "pattern_symmetry": "no",
            "numerical_symmetry": "no",
            "sparsity_type": "element",
            "dense_modes": [],
        },
        "zipf": {"alpha": 1.3},
        "value": {"type": "float64", "mode": "constant", "constant": 1.0},
        "duplicates": {"policy": "allow"},
        "output": {"write_tns": True, "write_tnsb": False, "write_metadata": True},
    }
    path = tmp_path / "alias.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    result = generate_from_config(path)
    metadata = validate_bundle(result["bundle_dir"])["metadata"]

    assert metadata["values_provided"] is True
