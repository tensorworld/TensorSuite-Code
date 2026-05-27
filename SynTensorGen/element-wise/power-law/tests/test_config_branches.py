from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from kronweave.api import generate_from_config, validate_bundle
from kronweave.config import load_and_validate_config


def _base_config(tmp_path: Path) -> dict:
    return {
        "name": "zipf_branch",
        "output_dir": str(tmp_path / "out"),
        "generator": {"type": "zipf", "random_seed": 11},
        "tensor": {"sizes": [4, 5], "index_base": 0, "nnz": 12},
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
            "write_tnsb": True,
            "write_metadata": True,
            "write_readme": True,
            "copy_config": True,
        },
    }


def _write_yaml(tmp_path: Path, cfg: dict, name: str = "config.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


@pytest.mark.parametrize("index_base", [0, 1])
@pytest.mark.parametrize("index_type", ["uint32", "uint64"])
@pytest.mark.parametrize("sorted_state,order", [("lexicographic", [1, 0]), ("none", [])])
@pytest.mark.parametrize("duplicates", ["keep", "sum", "allow"])
def test_zipf_generation_format_branches(tmp_path: Path, index_base, index_type, sorted_state, order, duplicates):
    cfg = _base_config(tmp_path)
    cfg["name"] = f"zipf_{index_base}_{index_type}_{sorted_state}_{duplicates}"
    cfg["tensor"]["index_base"] = index_base
    cfg["format"]["index_type"] = index_type
    cfg["format"]["sorted"] = sorted_state
    cfg["format"]["sorted_order"] = order
    cfg["duplicates"]["policy"] = duplicates

    result = generate_from_config(_write_yaml(tmp_path, cfg))
    metadata = validate_bundle(result["bundle_dir"])["metadata"]

    assert metadata["index_base"] == index_base
    assert metadata["index_type"] == index_type
    assert metadata["sorted"] == sorted_state
    assert metadata["sort_order"] == order


@pytest.mark.parametrize(
    "value_cfg,expected_domain",
    [
        ({"type": "float32", "mode": "constant", "constant": 1.0}, "binary"),
        ({"type": "float64", "mode": "constant", "constant": 2.0}, "nonnegative"),
        ({"type": "int32", "mode": "uniform", "low": 0, "high": 3}, "nonnegative"),
        ({"type": "int64", "mode": "normal", "mean": 0, "std": 1, "clip": 2}, "general"),
    ],
)
def test_value_type_and_mode_branches(tmp_path: Path, value_cfg, expected_domain):
    cfg = _base_config(tmp_path)
    cfg["name"] = f"value_{value_cfg['type']}_{value_cfg['mode']}"
    cfg["value"] = value_cfg
    cfg["format"]["value_domain"] = expected_domain

    result = generate_from_config(_write_yaml(tmp_path, cfg))
    metadata = validate_bundle(result["bundle_dir"])["metadata"]

    assert metadata["value_type"] == value_cfg["type"]
    assert metadata["value_domain"] == expected_domain


def test_legacy_experiment_name_and_sort_order_alias(tmp_path: Path):
    cfg = _base_config(tmp_path)
    cfg.pop("name")
    cfg["experiment"] = {"name": "legacy_name"}
    cfg["format"]["sort_order"] = cfg["format"].pop("sorted_order")

    loaded = load_and_validate_config(_write_yaml(tmp_path, cfg))

    assert loaded["experiment"]["name"] == "legacy_name"
    assert loaded["format"]["sorted_order"] == [0, 1]


def test_fastskg_input_seed_relative_to_config_path_executes(tmp_path: Path):
    seed = tmp_path / "seed.tns"
    seed.write_text(
        "\n".join(
            [
                "%%TensorSuite-TNS",
                "% version: 0.1",
                "% name: seed",
                "2 2 2 4",
                "0 0 0.8",
                "0 1 0.2",
                "1 0 0.2",
                "1 1 0.05",
                "",
            ]
        ),
        encoding="utf-8",
    )
    cfg = _base_config(tmp_path)
    cfg["name"] = "fastskg_input"
    cfg["generator"] = {"type": "fastskg", "random_seed": 3}
    cfg["tensor"] = {"sizes": [4, 4], "index_base": 0, "nnz": 8}
    cfg.pop("zipf")
    cfg["seed"] = {"source": "input_tns", "input_tns": "seed.tns", "sizes": [2, 2]}
    cfg["fastskg"] = {"iter": 2}

    result = generate_from_config(_write_yaml(tmp_path, cfg))

    assert validate_bundle(result["bundle_dir"])["metadata"]["dimensions"] == [4, 4]


def test_cp_fastskg_cpd_config_executes(tmp_path: Path):
    cfg = _base_config(tmp_path)
    cfg["name"] = "cp_fastskg_branch"
    cfg["generator"] = {"type": "cp_fastskg", "random_seed": 5}
    cfg["tensor"] = {"sizes": [4, 4], "index_base": 0, "nnz": 8}
    cfg.pop("zipf")
    cfg["seed"] = {
        "source": "cpd",
        "cpd": {
            "sizes": [2, 2],
            "rank": 1,
            "alpha": 1.3,
            "noise": 0.0,
            "lambda_controller": "none",
        },
    }
    cfg["fastskg"] = {"iter": 2}

    result = generate_from_config(_write_yaml(tmp_path, cfg))

    assert (result["bundle_dir"] / "cp_seed").exists()
    assert validate_bundle(result["bundle_dir"])["metadata"]["dimensions"] == [4, 4]
