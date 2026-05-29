from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tensorsuiteIO.metadata import build_metadata, read_metadata, write_metadata  # noqa: E402
from tensorsuiteIO.readme_writer import write_bundle_readme  # noqa: E402
from tensorsuiteIO.tensorsuite import (  # noqa: E402
    INDEX_FORMATS,
    cross_check_header_metadata,
    cross_check_tns_tnsb,
    parse_tns_header,
    parse_tnsb_header,
    write_tns,
    write_tnsb_from_tns,
)

VALUE_DTYPES = {"float32": np.float32, "float64": np.float64, "int32": np.int32, "int64": np.int64}


def _load_yaml(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("YAML config must be a mapping")
    return data


def _positive_int_list(value: Any, key: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a non-empty list")
    if any(isinstance(x, bool) or not isinstance(x, int) for x in value):
        raise ValueError(f"{key} values must be integers")
    out = [int(x) for x in value]
    if any(x <= 0 for x in out):
        raise ValueError(f"{key} values must be positive")
    return out


def _normalize_values_provided(config: dict[str, Any]) -> None:
    has_provide = "provide_value" in config
    has_values = "values_provided" in config
    for key in ("provide_value", "values_provided"):
        if key in config and not isinstance(config[key], bool):
            raise ValueError(f"{key} must be true or false")
    if has_provide and has_values and config["provide_value"] != config["values_provided"]:
        raise ValueError("provide_value and values_provided disagree")
    if has_provide:
        config["values_provided"] = bool(config.pop("provide_value"))
    elif has_values:
        config["values_provided"] = bool(config["values_provided"])
    else:
        config["values_provided"] = True


def infer_value_domain(value_cfg: dict[str, Any]) -> str:
    mode = value_cfg.get("mode", "constant")
    if mode == "constant":
        c = float(value_cfg.get("constant", 1.0))
        if c == 1.0:
            return "binary"
        return "nonnegative" if c >= 0 else "general"
    if mode == "uniform":
        low = float(value_cfg.get("low", 0.0))
        high = float(value_cfg.get("high", 1.0))
        if low >= high:
            raise ValueError("value.low must be < value.high")
        return "nonnegative" if low >= 0 else "general"
    if mode == "normal":
        if float(value_cfg.get("std", 1.0)) <= 0:
            raise ValueError("value.std must be > 0")
        if float(value_cfg.get("clip", 3.0)) <= 0:
            raise ValueError("value.clip must be > 0")
        return "general"
    raise ValueError("value.mode must be constant, uniform, or normal")


def validate_config(config: dict[str, Any], expected_type: str) -> dict[str, Any]:
    _normalize_values_provided(config)
    if "name" not in config:
        raise ValueError("Config must set name")
    config.setdefault("output_dir", "generated")
    generator = config.get("generator")
    if not isinstance(generator, dict):
        raise ValueError("Missing generator section")
    if generator.get("type") != expected_type:
        raise ValueError(f"generator.type must be {expected_type}")
    tensor = config.get("tensor")
    if not isinstance(tensor, dict):
        raise ValueError("Missing tensor section")
    tensor["sizes"] = _positive_int_list(tensor.get("sizes"), "tensor.sizes")
    tensor["nnz"] = int(tensor.get("nnz", 0))
    if tensor["nnz"] <= 0:
        raise ValueError("tensor.nnz must be > 0")
    tensor["index_base"] = int(tensor.get("index_base", 0))
    if tensor["index_base"] not in (0, 1):
        raise ValueError("tensor.index_base must be 0 or 1")

    fmt = config.setdefault("format", {})
    fmt.setdefault("storage", "coordinate")
    fmt.setdefault("index_type", "uint64")
    fmt.setdefault("endianness", "little")
    fmt.setdefault("sorted", "lexicographic")
    fmt.setdefault("sorted_order", list(range(len(tensor["sizes"]))))
    fmt.setdefault("explicit_zeros", "disallowed")
    fmt.setdefault("pattern_symmetry", "no")
    fmt.setdefault("numerical_symmetry", "no")
    fmt.setdefault("sparsity_type", "element")
    fmt.setdefault("dense_modes", [])
    if fmt["storage"] != "coordinate":
        raise ValueError("format.storage must be coordinate")
    if fmt["index_type"] not in INDEX_FORMATS:
        raise ValueError("format.index_type must be uint32 or uint64")
    if fmt["endianness"] != "little":
        raise ValueError("format.endianness must be little")
    if fmt["sorted"] not in {"none", "lexicographic"}:
        raise ValueError("format.sorted must be none or lexicographic")
    if fmt["sorted"] == "lexicographic" and sorted(fmt["sorted_order"]) != list(range(len(tensor["sizes"]))):
        raise ValueError("format.sorted_order must be a 0-based permutation")
    if fmt["sorted"] == "none":
        fmt["sorted_order"] = []
    if fmt["explicit_zeros"] not in {"allowed", "disallowed"}:
        raise ValueError("format.explicit_zeros must be allowed or disallowed")
    if fmt["pattern_symmetry"] is False:
        fmt["pattern_symmetry"] = "no"
    if fmt["numerical_symmetry"] is False:
        fmt["numerical_symmetry"] = "no"
    if fmt["pattern_symmetry"] != "no" or fmt["numerical_symmetry"] != "no":
        raise ValueError("symmetry must be no")
    if fmt["sparsity_type"] != "element" or fmt["dense_modes"] != []:
        raise ValueError("only element sparse tensors with dense_modes=[] are supported")

    value = config.setdefault("value", {"type": "float64", "mode": "constant", "constant": 1.0})
    value.setdefault("type", "float64")
    if value["type"] not in VALUE_DTYPES:
        raise ValueError("value.type must be float32, float64, int32, or int64")
    fmt["value_domain"] = infer_value_domain(value)

    duplicates = config.setdefault("duplicates", {"policy": "allow"})
    if duplicates.get("policy", "allow") not in {"allow", "keep", "sum"}:
        raise ValueError("duplicates.policy must be allow, keep, or sum")
    output = config.setdefault("output", {})
    for key in ("write_tns", "write_tnsb", "write_metadata", "write_readme", "copy_config"):
        output.setdefault(key, True)
    if not output["write_tns"] and output["write_tnsb"]:
        raise ValueError("output.write_tns=false with write_tnsb=true is not supported")
    if not output["write_tns"] and not output["write_tnsb"]:
        raise ValueError("at least one tensor data file must be written")
    if not config["values_provided"] and output["write_tnsb"]:
        raise ValueError("values_provided=false requires output.write_tnsb=false")
    if expected_type == "gaussian":
        config.setdefault("gaussian", {})
    else:
        config.setdefault("random", {})
    return config


def generate_values(n: int, value_cfg: dict[str, Any], seed: int | None) -> np.ndarray:
    mode = value_cfg.get("mode", "constant")
    dtype = VALUE_DTYPES[value_cfg.get("type", "float64")]
    if mode == "constant":
        return np.full(n, float(value_cfg.get("constant", 1.0)), dtype=dtype)
    rng = np.random.default_rng(seed)
    if mode == "uniform":
        values = rng.uniform(float(value_cfg.get("low", 0.0)), float(value_cfg.get("high", 1.0)), size=n)
    elif mode == "normal":
        mean = float(value_cfg.get("mean", 0.0))
        std = float(value_cfg.get("std", 1.0))
        clip = float(value_cfg.get("clip", 3.0))
        values = np.clip(rng.normal(mean, std, size=n), mean - clip * std, mean + clip * std)
    else:
        raise ValueError("value.mode must be constant, uniform, or normal")
    return values.astype(dtype)


def _deduplicate(coords: np.ndarray, values: np.ndarray, policy: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if policy == "allow":
        return coords, values, {"raw_samples": int(len(coords)), "unique_nnz": None, "duplicate_count": None}
    seen: dict[tuple[int, ...], int] = {}
    out_coords: list[np.ndarray] = []
    out_values: list[float] = []
    duplicate_count = 0
    for coord, value in zip(coords, values):
        key = tuple(int(x) for x in coord)
        if key in seen:
            duplicate_count += 1
            if policy == "sum":
                out_values[seen[key]] += float(value)
        else:
            seen[key] = len(out_coords)
            out_coords.append(coord.copy())
            out_values.append(float(value))
    return (
        np.asarray(out_coords, dtype=np.uint64).reshape((len(out_coords), coords.shape[1])),
        np.asarray(out_values, dtype=values.dtype),
        {"raw_samples": int(len(coords)), "unique_nnz": len(out_coords), "duplicate_count": duplicate_count},
    )


def _sort_entries(config: dict[str, Any], coords: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fmt = config["format"]
    if fmt["sorted"] != "lexicographic" or len(coords) == 0:
        return coords, values
    keys = tuple(coords[:, d] for d in reversed(fmt["sorted_order"]))
    order = np.lexsort(keys)
    return coords[order], values[order]


def _metadata(config: dict[str, Any], name: str, kind: str, coords: np.ndarray) -> dict[str, Any]:
    fmt = config["format"]
    output = config["output"]
    return build_metadata(
        name=name,
        dims=config["tensor"]["sizes"],
        nnz=len(coords),
        index_base=int(config["tensor"].get("index_base", 0)),
        value_domain=fmt["value_domain"],
        duplicate_policy=config["duplicates"].get("policy", "allow"),
        value_type=config["value"].get("type", "float64"),
        index_type=fmt["index_type"],
        endianness=fmt["endianness"],
        sorted_state=fmt["sorted"],
        sorted_order=fmt["sorted_order"],
        explicit_zeros=fmt["explicit_zeros"],
        pattern_symmetry=fmt["pattern_symmetry"],
        numerical_symmetry=fmt["numerical_symmetry"],
        sparsity_type=fmt["sparsity_type"],
        values_provided=bool(config["values_provided"]),
        source=f"{kind.capitalize()} structural TensorSuite generator",
        files={
            "text": f"{name}.tns" if output.get("write_tns", True) else None,
            "binary": f"{name}.tnsb" if output.get("write_tnsb", True) else None,
            "readme": "README.md" if output.get("write_readme", True) else None,
            "config": "config.yaml" if output.get("copy_config", True) else None,
        },
    )


def generate_from_config(
    config_path: str | Path,
    expected_type: str,
    coord_generator: Callable[[dict[str, Any]], np.ndarray],
) -> dict[str, Path]:
    config_path = Path(config_path).resolve()
    config = validate_config(_load_yaml(config_path), expected_type)
    name = str(config["name"])
    bundle_dir = Path(config["output_dir"])
    if not bundle_dir.is_absolute():
        bundle_dir = Path.cwd() / bundle_dir
    bundle_dir = bundle_dir / name
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    coords = coord_generator(config)
    values = generate_values(len(coords), config["value"], seed=(config["generator"].get("random_seed") or 0) + 1000)
    coords, values, stats = _deduplicate(coords, values, config["duplicates"].get("policy", "allow"))
    if config["format"]["explicit_zeros"] == "disallowed" and config["values_provided"]:
        keep = values != 0
        coords = coords[keep]
        values = values[keep]
    coords, values = _sort_entries(config, coords, values)

    tns = bundle_dir / f"{name}.tns"
    tnsb = bundle_dir / f"{name}.tnsb"
    metadata_path = bundle_dir / f"{name}_metadata.json"
    readme_path = bundle_dir / "README.md"
    if config["output"].get("copy_config", True):
        (bundle_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    if config["output"].get("write_tns", True):
        write_tns(
            tns,
            name,
            config["tensor"]["sizes"],
            coords,
            values,
            index_base=config["tensor"]["index_base"],
            values_provided=bool(config["values_provided"]),
        )
    if config["output"].get("write_tnsb", True):
        write_tnsb_from_tns(
            tns,
            tnsb,
            index_type=config["format"]["index_type"],
            value_type=config["value"].get("type", "float64"),
            endianness=config["format"]["endianness"],
        )
    metadata = _metadata(config, name, expected_type, coords)
    if config["output"].get("write_metadata", True):
        write_metadata(metadata_path, metadata)
    if config["output"].get("write_readme", True):
        write_bundle_readme(
            readme_path,
            name,
            metadata,
            config,
            config_path=str(config_path),
            config_snapshot_path="config.yaml" if config["output"].get("copy_config", True) else None,
            duplicate_stats=stats,
            generation_info={"role": expected_type, "meta_seed": "Not applicable."},
        )
    validate_bundle(bundle_dir)
    return {"bundle_dir": bundle_dir, "tns": tns, "tnsb": tnsb, "metadata": metadata_path, "readme": readme_path}


def validate_bundle(bundle_dir: str | Path) -> dict[str, Any]:
    bundle = Path(bundle_dir)
    name = bundle.name
    metadata_path = bundle / f"{name}_metadata.json"
    metadata = read_metadata(metadata_path)
    files = metadata["files"]
    tns = bundle / files["text"] if files.get("text") else None
    tnsb = bundle / files["binary"] if files.get("binary") else None
    if files.get("text"):
        cross_check_header_metadata(parse_tns_header(tns), metadata)
        expected_fields = metadata["order"] + (1 if metadata["values_provided"] else 0)
        rows = [line.split() for line in tns.read_text(encoding="utf-8").splitlines()[4:] if line.strip()]
        if len(rows) != metadata["nnz"] or any(len(row) != expected_fields for row in rows):
            raise ValueError("TNS payload mismatch")
    if files.get("binary"):
        cross_check_header_metadata(parse_tnsb_header(tnsb), metadata)
    if tns and tnsb:
        cross_check_tns_tnsb(tns, tnsb, metadata=metadata)
    if not files.get("text") and not files.get("binary"):
        raise ValueError("bundle has no tensor data file")
    return {"bundle_dir": bundle, "metadata": metadata}


def run_cli(argv: list[str] | None, expected_type: str, coord_generator: Callable[[dict[str, Any]], np.ndarray]) -> int:
    parser = argparse.ArgumentParser(description=f"{expected_type} structural TensorSuite generator")
    sub = parser.add_subparsers(dest="command", required=True)
    p_validate = sub.add_parser("validate-config")
    p_validate.add_argument("--config", required=True)
    p_generate = sub.add_parser("generate")
    p_generate.add_argument("--config", required=True)
    p_bundle = sub.add_parser("validate-bundle")
    p_bundle.add_argument("--bundle", required=True)
    args = parser.parse_args(argv)
    if args.command == "validate-config":
        validate_config(_load_yaml(args.config), expected_type)
        print(f"OK: {args.config}")
    elif args.command == "generate":
        result = generate_from_config(args.config, expected_type, coord_generator)
        print(f"Generated: {result['bundle_dir']}")
    elif args.command == "validate-bundle":
        validate_bundle(args.bundle)
        print(f"OK: {args.bundle}")
    return 0
