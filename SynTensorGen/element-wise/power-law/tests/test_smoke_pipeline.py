from pathlib import Path

import yaml

from kronweave.api import generate_from_config, validate_bundle


def _write_config(tmp_path: Path, generator_type: str) -> Path:
    cfg = {
        "name": f"{generator_type}_tiny",
        "output_dir": str(tmp_path / "outputs"),
        "generator": {"type": generator_type, "random_seed": 7},
        "tensor": {"sizes": [16, 16, 16], "index_base": 0, "nnz": 100},
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
    if generator_type == "cp_fastskg":
        cfg["seed"] = {
            "source": "cpd",
            "cpd": {
                "sizes": [4, 4, 4],
                "rank": 2,
                "alpha": [1.4, 1.4, 1.4],
                "noise": 0.0,
                "lambda_controller": "softmax",
            },
        }
        cfg["fastskg"] = {"iter": 2}
    else:
        cfg["zipf"] = {"alpha": [1.4, 1.4, 1.4]}
    path = tmp_path / f"{generator_type}.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_zipf_smoke_pipeline(tmp_path: Path):
    result = generate_from_config(_write_config(tmp_path, "zipf"))
    assert result["tns"].exists()
    assert result["tnsb"].exists()
    assert result["metadata"].exists()
    assert result["readme"].exists()
    assert result["config"].exists()
    validate_bundle(result["bundle_dir"])


def test_cp_fastskg_smoke_pipeline(tmp_path: Path):
    result = generate_from_config(_write_config(tmp_path, "cp_fastskg"))
    assert (result["bundle_dir"] / "cp_seed").exists()
    validate_bundle(result["bundle_dir"])
