from __future__ import annotations

import json
from pathlib import Path

import yaml

from kronweave.api import generate_from_config, validate_bundle


def _config(tmp_path: Path) -> dict:
    return {
        "name": "metadata_schema",
        "output_dir": str(tmp_path / "out"),
        "generator": {"type": "zipf", "random_seed": 17},
        "tensor": {"sizes": [4, 4, 4], "index_base": 0, "nnz": 16},
        "format": {
            "storage": "coordinate",
            "index_type": "uint64",
            "endianness": "little",
            "sorted": "lexicographic",
            "sorted_order": [0, 1, 2],
            "explicit_zeros": "disallowed",
            "pattern_symmetry": "no",
            "numerical_symmetry": "no",
            "sparsity_type": "element",
            "dense_modes": [],
        },
        "zipf": {"alpha": [1.4]},
        "value": {"type": "float64", "mode": "constant", "constant": 1.0},
        "duplicates": {"policy": "keep"},
        "output": {
            "write_tns": True,
            "write_tnsb": True,
            "write_metadata": True,
            "write_readme": True,
            "copy_config": True,
        },
    }


def test_metadata_matches_tensorsuite_element_expectations(tmp_path: Path):
    config_path = tmp_path / "metadata.yaml"
    config_path.write_text(yaml.safe_dump(_config(tmp_path), sort_keys=False), encoding="utf-8")

    result = generate_from_config(config_path)
    metadata = validate_bundle(result["bundle_dir"])["metadata"]
    raw = json.loads(result["metadata"].read_text(encoding="utf-8"))

    assert metadata["version"] == "1.0"
    assert metadata["name"] == "metadata_schema"
    assert isinstance(metadata["id"], int) and metadata["id"] > 1000
    assert metadata["time"] == "2025-06-05"
    assert metadata["source_type"] == "synthetic"
    assert metadata["source"] == "KronWeave powerLawGenerator"
    assert metadata["source_url"] == ""
    assert metadata["values_provided"] is True
    assert metadata["endianness"] == "little"
    assert metadata["sort_order"] == [0, 1, 2]
    assert "sorted_order" not in raw
    assert metadata["sparsity_type"] == "element"
    assert metadata["dense_modes"] == []
    assert metadata["block_partitions"] is None
    assert metadata["nnz_block"] is None
    assert metadata["files"] == {
        "binary": "metadata_schema.tnsb",
        "text": "metadata_schema.tns",
        "readme": "README.md",
        "config": "config.yaml",
    }
