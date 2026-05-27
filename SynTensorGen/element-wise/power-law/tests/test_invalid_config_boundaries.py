from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from kronweave.config import load_and_validate_config


def _valid_zipf(tmp_path: Path) -> dict:
    return {
        "name": "invalid_boundary",
        "output_dir": str(tmp_path / "out"),
        "generator": {"type": "zipf", "random_seed": 1},
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
            "write_tnsb": True,
            "write_metadata": True,
            "write_readme": True,
            "copy_config": True,
        },
    }


def _write(tmp_path: Path, cfg) -> Path:
    path = tmp_path / "bad.yaml"
    if isinstance(cfg, str):
        path.write_text(cfg, encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda c: c.pop("name"), "Config must contain `name`"),
        (lambda c: c.update({"unexpected": True}), "Unknown top-level key"),
        (lambda c: c["generator"].update({"type": "unknown"}), "generator.type"),
        (lambda c: c.pop("generator"), "generator"),
        (lambda c: c["tensor"].update({"sizes": []}), "tensor.sizes"),
        (lambda c: c["tensor"].update({"sizes": [4, 1.5]}), "values must be integers"),
        (lambda c: c["tensor"].update({"sizes": [4, 0]}), "values must be positive"),
        (lambda c: c["tensor"].update({"index_base": 2}), "index_base"),
        (lambda c: c["tensor"].update({"nnz": 0}), "tensor.nnz"),
        (lambda c: c["format"].update({"storage": "csr"}), "format.storage"),
        (lambda c: c["format"].update({"endianness": "big"}), "endianness"),
        (lambda c: c["format"].update({"sorted_order": [0, 0]}), "permutation"),
        (lambda c: c["format"].update({"sorted": "none", "sorted_order": [0, 1]}), "sorted_order"),
        (lambda c: c["format"].update({"pattern_symmetry": "symmetric"}), "pattern_symmetry"),
        (lambda c: c["format"].update({"numerical_symmetry": "symmetric"}), "numerical_symmetry"),
        (lambda c: c["format"].update({"sparsity_type": "block"}), "sparsity_type"),
        (lambda c: c["format"].update({"dense_modes": [0]}), "dense_modes"),
        (lambda c: c["format"].update({"value_domain": "binary"}) or c["value"].update({"constant": 2.0}), "value_domain"),
        (lambda c: c["zipf"].update({"alpha": 1.0}), "zipf.alpha"),
        (lambda c: c["zipf"].update({"alpha": [1.3, 1.4, 1.5]}), "zipf.alpha length"),
        (lambda c: c["value"].update({"type": "float16"}), "value.type"),
        (lambda c: c["value"].update({"mode": "unknown"}), "value.mode"),
        (lambda c: c["value"].update({"mode": "uniform", "low": 2, "high": 2}), "low < high"),
        (lambda c: c["value"].update({"mode": "normal", "std": 0, "clip": 1}), "std > 0"),
        (lambda c: c["value"].update({"mode": "normal", "std": 1, "clip": 0}), "clip > 0"),
        (lambda c: c["duplicates"].update({"policy": "merge"}), "duplicates.policy"),
        (lambda c: c["output"].update({"unknown": True}), "Unknown output key"),
        (lambda c: c["output"].update({"write_metadata": False}), "write_metadata"),
        (lambda c: c["output"].update({"write_tns": False, "write_tnsb": True}), "write_tns=false"),
        (lambda c: c.update({"values_provided": False}) or c["output"].update({"write_tnsb": True}), "values_provided=false"),
    ],
)
def test_invalid_zipf_boundaries(tmp_path: Path, mutate, match):
    cfg = deepcopy(_valid_zipf(tmp_path))
    mutate(cfg)

    with pytest.raises(ValueError, match=match):
        load_and_validate_config(_write(tmp_path, cfg))


@pytest.mark.parametrize("yaml_text", ["- not\n- mapping\n", "name: [unterminated\n"])
def test_malformed_or_non_mapping_yaml_rejected(tmp_path: Path, yaml_text: str):
    with pytest.raises((ValueError, yaml.YAMLError)):
        load_and_validate_config(_write(tmp_path, yaml_text))


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda c: c["seed"].update({"source": "cpd"}), "requires seed.source=input_tns"),
        (lambda c: c["seed"].pop("input_tns"), "seed.input_tns is required"),
        (lambda c: c["seed"].update({"input_tns": "missing.tns"}), "does not exist"),
        (lambda c: c["seed"].update({"sizes": [3, 3]}), "do not match seed.sizes"),
        (lambda c: c["fastskg"].pop("iter"), "fastskg.iter is required"),
        (lambda c: c["fastskg"].update({"iter": 0}), "fastskg.iter"),
        (lambda c: c["fastskg"].update({"nnz": 0}), "fastskg.nnz"),
        (lambda c: c["tensor"].update({"sizes": [8, 8]}), "expected \\[4, 4\\]"),
    ],
)
def test_invalid_fastskg_boundaries(tmp_path: Path, mutate, match):
    seed = tmp_path / "seed.tns"
    seed.write_text(
        "%%TensorSuite-TNS\n% version: 0.1\n% name: seed\n2 2 2 1\n0 0 1.0\n",
        encoding="utf-8",
    )
    cfg = _valid_zipf(tmp_path)
    cfg["generator"] = {"type": "fastskg"}
    cfg.pop("zipf")
    cfg["seed"] = {"source": "input_tns", "input_tns": "seed.tns", "sizes": [2, 2]}
    cfg["fastskg"] = {"iter": 2}
    cfg["tensor"]["sizes"] = [4, 4]
    mutate(cfg)

    with pytest.raises(ValueError, match=match):
        load_and_validate_config(_write(tmp_path, cfg))


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda c: c["seed"].update({"source": "input_tns", "input_tns": "seed.tns"}), "requires seed.source=cpd"),
        (lambda c: c["seed"].pop("cpd"), "seed.cpd is required"),
        (lambda c: c["seed"]["cpd"].update({"rank": 0}), "rank"),
        (lambda c: c["seed"]["cpd"].update({"alpha": [1.2, 1.3, 1.4]}), "alpha length"),
        (lambda c: c["seed"]["cpd"].update({"noise": -0.1}), "noise"),
        (lambda c: c["seed"]["cpd"].update({"lambda_controller": "bad"}), "lambda_controller"),
    ],
)
def test_invalid_cp_fastskg_boundaries(tmp_path: Path, mutate, match):
    (tmp_path / "seed.tns").write_text(
        "%%TensorSuite-TNS\n% version: 0.1\n% name: seed\n2 2 2 1\n0 0 1.0\n",
        encoding="utf-8",
    )
    cfg = _valid_zipf(tmp_path)
    cfg["generator"] = {"type": "cp_fastskg"}
    cfg.pop("zipf")
    cfg["seed"] = {
        "source": "cpd",
        "cpd": {"sizes": [2, 2], "rank": 1, "alpha": [1.3], "noise": 0.0, "lambda_controller": "softmax"},
    }
    cfg["fastskg"] = {"iter": 2}
    cfg["tensor"]["sizes"] = [4, 4]
    mutate(cfg)

    with pytest.raises(ValueError, match=match):
        load_and_validate_config(_write(tmp_path, cfg))


def test_kp_from_meta_requires_expand_iter(tmp_path: Path):
    seed = tmp_path / "seed.tns"
    seed.write_text(
        "%%TensorSuite-TNS\n% version: 0.1\n% name: seed\n2 2 2 1\n0 0 1.0\n",
        encoding="utf-8",
    )
    cfg = _valid_zipf(tmp_path)
    cfg["generator"] = {"type": "fastskg"}
    cfg.pop("zipf")
    cfg["seed"] = {"source": "kp_from_meta", "input_tns": "seed.tns"}
    cfg["fastskg"] = {"iter": 2}

    with pytest.raises(ValueError, match="expand_iter"):
        load_and_validate_config(_write(tmp_path, cfg))
