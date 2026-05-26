from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from kronweave.io.tensorsuite import parse_tns_header
from kronweave.postprocess.values import infer_value_domain

MAX_UINT32 = 2**32 - 1
ALLOWED_GENERATORS = {"cp_fastskg", "fastskg", "zipf"}
ALLOWED_SEED_SOURCES = {"cpd", "input_tns", "kp_from_meta"}
ALLOWED_SORTED = {"none", "lexicographic"}
ALLOWED_ENDIANNESS = {"little"}
ALLOWED_INDEX_TYPES = {"uint32", "uint64"}
ALLOWED_VALUE_TYPES = {"float32", "float64", "int32", "int64"}
ALLOWED_EXPLICIT_ZEROS = {"allowed", "disallowed"}
ALLOWED_STORAGE = {"coordinate"}
ALLOWED_SYMMETRY = {"no"}
ALLOWED_SPARSITY_TYPE = {"element"}
ALLOWED_VALUE_DOMAINS = {"binary", "nonnegative", "general"}
TOP_LEVEL_KEYS = {
    "name",
    "experiment",
    "output_dir",
    "generator",
    "seed",
    "meta_seed",
    "fastskg",
    "tensor",
    "format",
    "zipf",
    "value",
    "duplicates",
    "output",
    "provide_value",
    "values_provided",
}
EXPERIMENT_KEYS = {"name"}
FORMAT_KEYS = {
    "storage",
    "index_type",
    "endianness",
    "sorted",
    "sorted_order",
    "sort_order",
    "explicit_zeros",
    "pattern_symmetry",
    "numerical_symmetry",
    "sparsity_type",
    "dense_modes",
    "value_domain",
}
GENERATOR_KEYS = {"type", "random_seed"}
TENSOR_KEYS = {"sizes", "nnz", "index_base"}
SEED_KEYS = {"source", "input_tns", "sizes", "cpd", "expand_iter"}
CPD_KEYS = {"sizes", "rank", "alpha", "noise", "lambda_controller"}
FASTSKG_KEYS = {"iter", "nnz"}
VALUE_KEYS = {"type", "mode", "constant", "low", "high", "mean", "std", "clip"}
DUPLICATES_KEYS = {"policy"}
OUTPUT_KEYS = {"write_tns", "write_tnsb", "write_metadata", "write_readme", "copy_config"}
ZIPF_KEYS = {"alpha"}


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("YAML config must be a mapping")
    return data


def _require_mapping(config: dict, key: str) -> dict:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid `{key}` section")
    return value


def _reject_unknown_keys(mapping: dict, allowed: set[str], path: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"Unknown {path} key: {unknown[0]}")


def _normalize_yaml_no(value):
    return "no" if value is False else value


def _positive_int_list(value, key: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"`{key}` must be a non-empty list")
    if any(isinstance(x, bool) or not isinstance(x, int) for x in value):
        raise ValueError(f"`{key}` values must be integers")
    out = [int(x) for x in value]
    if any(x <= 0 for x in out):
        raise ValueError(f"`{key}` values must be positive")
    return out


def _normalize_sort_order_alias(fmt: dict) -> None:
    if "sorted_order" in fmt and "sort_order" in fmt:
        if fmt["sorted_order"] != fmt["sort_order"]:
            raise ValueError("format.sorted_order and format.sort_order aliases disagree")
        fmt.pop("sort_order")
    elif "sort_order" in fmt:
        fmt["sorted_order"] = fmt.pop("sort_order")


def _normalize_values_provided_aliases(config: dict[str, Any]) -> None:
    has_provide_value = "provide_value" in config
    has_values_provided = "values_provided" in config
    for key in ("provide_value", "values_provided"):
        if key in config and not isinstance(config[key], bool):
            raise ValueError(f"{key} must be boolean")
    if has_provide_value and has_values_provided and bool(config["provide_value"]) != bool(config["values_provided"]):
        raise ValueError("provide_value and values_provided aliases disagree")
    if has_provide_value:
        config["values_provided"] = bool(config.pop("provide_value"))
    elif has_values_provided:
        config["values_provided"] = bool(config["values_provided"])
    else:
        config["values_provided"] = True


def _resolve_path(path_value: str | Path, base_dir: Path | None) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    if base_dir is not None and (base_dir / path).exists():
        return base_dir / path
    return Path.cwd() / path


def _validate_cpd_config(cpd_cfg: dict, key: str, order: int | None = None) -> dict:
    _reject_unknown_keys(cpd_cfg, CPD_KEYS, key)
    cpd_cfg["sizes"] = _positive_int_list(cpd_cfg.get("sizes"), f"{key}.sizes")
    cpd_cfg["rank"] = int(cpd_cfg.get("rank", 1))
    if cpd_cfg["rank"] <= 0:
        raise ValueError(f"{key}.rank must be > 0")
    cpd_cfg.setdefault("alpha", [1.2])
    if not isinstance(cpd_cfg["alpha"], list):
        cpd_cfg["alpha"] = [float(cpd_cfg["alpha"])]
    cpd_cfg["alpha"] = [float(x) for x in cpd_cfg["alpha"]]
    cpd_cfg["noise"] = float(cpd_cfg.get("noise", 0.0))
    if cpd_cfg["noise"] < 0:
        raise ValueError(f"{key}.noise must be >= 0")
    cpd_cfg.setdefault("lambda_controller", "softmax")
    if cpd_cfg["lambda_controller"] not in {"softmax", "none"}:
        raise ValueError(f"{key}.lambda_controller must be softmax or none")
    if order is not None and len(cpd_cfg["sizes"]) != order:
        raise ValueError(f"{key}.sizes order must match source seed order")
    if len(cpd_cfg["alpha"]) not in {1, len(cpd_cfg["sizes"])}:
        raise ValueError(f"{key}.alpha length must be 1 or match {key}.sizes")
    return cpd_cfg


def _validate_common_format(config: dict[str, Any], order: int) -> None:
    fmt = _require_mapping(config, "format")
    _reject_unknown_keys(fmt, FORMAT_KEYS, "format")
    _normalize_sort_order_alias(fmt)
    if "index_type" not in fmt:
        raise ValueError("YAML must explicitly set format.index_type")
    if fmt["index_type"] not in ALLOWED_INDEX_TYPES:
        raise ValueError(f"Invalid format.index_type: {fmt['index_type']}. Allowed values: uint32, uint64.")
    if "endianness" not in fmt:
        raise ValueError("YAML must explicitly set format.endianness")
    if fmt["endianness"] not in ALLOWED_ENDIANNESS:
        raise ValueError("format.endianness must be little for TensorSuite branch tests")

    value = config.setdefault("value", {"mode": "constant", "constant": 1.0})
    _reject_unknown_keys(value, VALUE_KEYS, "value")
    if "type" not in value:
        raise ValueError("YAML value section must explicitly set value.type")
    if value["type"] not in ALLOWED_VALUE_TYPES:
        raise ValueError(f"Invalid value.type: {value['type']}. Allowed values: float32, float64, int32, int64.")
    inferred_value_domain = infer_value_domain(value)
    if "value_domain" in fmt:
        if fmt["value_domain"] not in ALLOWED_VALUE_DOMAINS:
            raise ValueError("format.value_domain must be binary, nonnegative, or general")
        if fmt["value_domain"] != inferred_value_domain:
            raise ValueError(
                f"format.value_domain={fmt['value_domain']} does not match inferred value domain "
                f"{inferred_value_domain}"
            )
    fmt["value_domain"] = inferred_value_domain

    if "sorted" not in fmt:
        raise ValueError("YAML must explicitly set format.sorted=none or format.sorted=lexicographic")
    if fmt["sorted"] not in ALLOWED_SORTED:
        raise ValueError(f"Invalid format.sorted: {fmt['sorted']}. Allowed values: none, lexicographic.")
    if fmt["sorted"] == "lexicographic":
        if "sorted_order" not in fmt:
            raise ValueError("format.sorted_order is required when format.sorted=lexicographic")
        if not isinstance(fmt["sorted_order"], list):
            raise ValueError("format.sorted_order must be a list of 0-based mode ids")
        fmt["sorted_order"] = [int(x) for x in fmt["sorted_order"]]
        if sorted(fmt["sorted_order"]) != list(range(order)):
            raise ValueError("format.sorted_order must be a 0-based permutation of tensor modes")
    else:
        if fmt.get("sorted_order") not in (None, []):
            raise ValueError("format.sorted_order must be omitted, null, or [] when format.sorted=none")
        fmt["sorted_order"] = []

    fmt.setdefault("storage", "coordinate")
    if fmt["storage"] not in ALLOWED_STORAGE:
        raise ValueError(f"Invalid format.storage: {fmt['storage']}. Allowed values: coordinate.")
    fmt.setdefault("explicit_zeros", "disallowed")
    if fmt["explicit_zeros"] not in ALLOWED_EXPLICIT_ZEROS:
        raise ValueError("format.explicit_zeros must be allowed or disallowed")
    fmt.setdefault("pattern_symmetry", "no")
    fmt.setdefault("numerical_symmetry", "no")
    fmt["pattern_symmetry"] = _normalize_yaml_no(fmt["pattern_symmetry"])
    fmt["numerical_symmetry"] = _normalize_yaml_no(fmt["numerical_symmetry"])
    if fmt["pattern_symmetry"] not in ALLOWED_SYMMETRY:
        raise ValueError("format.pattern_symmetry must be no in v0.1")
    if fmt["numerical_symmetry"] not in ALLOWED_SYMMETRY:
        raise ValueError("format.numerical_symmetry must be no in v0.1")
    fmt.setdefault("sparsity_type", "element")
    if fmt["sparsity_type"] not in ALLOWED_SPARSITY_TYPE:
        raise ValueError("format.sparsity_type must be element in v0.1")
    fmt.setdefault("dense_modes", [])
    if not isinstance(fmt["dense_modes"], list):
        raise ValueError("format.dense_modes must be a list")
    if fmt["dense_modes"]:
        raise ValueError("format.dense_modes must be [] in v0.1")


def _check_index_capacity(config: dict[str, Any], dims: list[int]) -> None:
    if config.get("format", {}).get("index_type") == "uint32" and any(int(d) > MAX_UINT32 for d in dims):
        raise ValueError("index_type=uint32 requires all dimensions <= 2^32 - 1")


def validate_config(config: dict[str, Any], base_dir: str | Path | None = None) -> dict[str, Any]:
    base_path = Path(base_dir) if base_dir is not None else None
    _reject_unknown_keys(config, TOP_LEVEL_KEYS, "top-level")
    _normalize_values_provided_aliases(config)
    if "name" not in config and "experiment" not in config:
        raise ValueError("Config must contain `name` for single-run generation")
    if "experiment" in config:
        if not isinstance(config["experiment"], dict):
            raise ValueError("experiment must be a mapping")
        _reject_unknown_keys(config["experiment"], EXPERIMENT_KEYS, "experiment")
        if "name" not in config and not config["experiment"].get("name"):
            raise ValueError("experiment.name is required when top-level name is omitted")
    if "meta_seed" in config:
        if "seed" in config:
            raise ValueError("Use either seed or meta_seed, not both")
        config["seed"] = config.pop("meta_seed")
    config.setdefault("output_dir", "outputs")
    generator = _require_mapping(config, "generator")
    _reject_unknown_keys(generator, GENERATOR_KEYS, "generator")
    gtype = generator.get("type")
    if gtype not in ALLOWED_GENERATORS:
        raise ValueError("generator.type must be cp_fastskg, fastskg, or zipf")
    tensor = _require_mapping(config, "tensor")
    _reject_unknown_keys(tensor, TENSOR_KEYS, "tensor")
    tensor["index_base"] = int(tensor.get("index_base", 0))
    if tensor["index_base"] not in (0, 1):
        raise ValueError("tensor.index_base must be 0 or 1")
    if gtype in {"cp_fastskg", "fastskg"}:
        if "sizes" in tensor:
            tensor["sizes"] = _positive_int_list(tensor.get("sizes"), "tensor.sizes")
        if "nnz" in tensor and tensor["nnz"] is not None:
            tensor["nnz"] = int(tensor["nnz"])
            if tensor["nnz"] <= 0:
                raise ValueError("tensor.nnz must be > 0 when provided")
        seed = _require_mapping(config, "seed")
        _reject_unknown_keys(seed, SEED_KEYS, "seed")
        source = seed.get("source")
        if source not in ALLOWED_SEED_SOURCES:
            raise ValueError("seed.source must be cpd, input_tns, or kp_from_meta")
        if gtype == "cp_fastskg" and source != "cpd":
            raise ValueError("generator.type=cp_fastskg requires seed.source=cpd")
        if gtype == "fastskg" and source == "cpd":
            raise ValueError("generator.type=fastskg requires seed.source=input_tns or kp_from_meta")
        input_header = None
        if source in {"input_tns", "kp_from_meta"} and "input_tns" in seed:
            seed_path = _resolve_path(seed["input_tns"], base_path)
            if not seed_path.exists():
                raise ValueError(f"seed.input_tns does not exist: {seed['input_tns']}")
            input_header = parse_tns_header(seed_path)
            if "sizes" in seed:
                expected = tuple(_positive_int_list(seed.get("sizes"), "seed.sizes"))
                if input_header.dimensions != expected:
                    raise ValueError(
                        f"seed.input_tns dimensions {input_header.dimensions} do not match seed.sizes {expected}"
                    )

        if source == "cpd":
            cpd_cfg = seed.get("cpd")
            if not isinstance(cpd_cfg, dict):
                raise ValueError("seed.cpd is required when seed.source=cpd")
            _validate_cpd_config(cpd_cfg, "seed.cpd", order=None)
            cp_seed_sizes = cpd_cfg["sizes"]
        elif source == "input_tns":
            if input_header is None:
                raise ValueError("seed.input_tns is required when seed.source=input_tns")
            if "cpd" in seed:
                raise ValueError("seed.cpd is not allowed when seed.source=input_tns")
            cp_seed_sizes = list(input_header.dimensions)
        else:
            if input_header is None:
                raise ValueError("seed.input_tns is required when seed.source=kp_from_meta")
            if "cpd" in seed:
                raise ValueError("seed.cpd is not allowed when seed.source=kp_from_meta")
            if "expand_iter" not in seed:
                raise ValueError("seed.expand_iter is required when seed.source=kp_from_meta")
            seed["expand_iter"] = int(seed["expand_iter"])
            if seed["expand_iter"] <= 0:
                raise ValueError("seed.expand_iter must be > 0")
            cp_seed_sizes = [int(s) ** int(seed["expand_iter"]) for s in input_header.dimensions]
        fastskg = _require_mapping(config, "fastskg")
        _reject_unknown_keys(fastskg, FASTSKG_KEYS, "fastskg")
        if "iter" not in fastskg:
            raise ValueError("fastskg.iter is required")
        fastskg["iter"] = int(fastskg["iter"])
        if fastskg["iter"] <= 0:
            raise ValueError("fastskg.iter must be > 0")
        if "nnz" in fastskg and fastskg["nnz"] is not None:
            fastskg["nnz"] = int(fastskg["nnz"])
            if fastskg["nnz"] <= 0:
                raise ValueError("fastskg.nnz must be > 0 when provided")
        derived_sizes = [int(s) ** int(fastskg["iter"]) for s in cp_seed_sizes]
        if "sizes" in tensor and len(cp_seed_sizes) != len(tensor["sizes"]):
            raise ValueError("seed dimensions and tensor.sizes must have the same order")
        if "sizes" in tensor and tensor["sizes"] != derived_sizes:
            raise ValueError(
                "For fastSKG, tensor.sizes is derived from seed dimensions ** fastskg.iter; "
                f"got {tensor['sizes']}, expected {derived_sizes}"
            )
        tensor["sizes"] = derived_sizes
    else:
        if "seed" in config:
            raise ValueError("generator.type=zipf does not allow seed")
        tensor["sizes"] = _positive_int_list(tensor.get("sizes"), "tensor.sizes")
        tensor["nnz"] = int(tensor.get("nnz", 0))
        if tensor["nnz"] <= 0:
            raise ValueError("tensor.nnz must be > 0")
        zipf = _require_mapping(config, "zipf")
        _reject_unknown_keys(zipf, ZIPF_KEYS, "zipf")
        zipf.setdefault("alpha", [1.2])
        if not isinstance(zipf["alpha"], list):
            zipf["alpha"] = [float(zipf["alpha"])]
        zipf["alpha"] = [float(x) for x in zipf["alpha"]]
        if any(x <= 1.0 for x in zipf["alpha"]):
            raise ValueError("zipf.alpha values must be > 1.0")
        if len(zipf["alpha"]) not in {1, len(tensor["sizes"])}:
            raise ValueError("zipf.alpha length must be 1 or match tensor.sizes")
    _validate_common_format(config, order=len(tensor["sizes"]))
    _check_index_capacity(config, tensor["sizes"])
    duplicates = config.setdefault("duplicates", {"policy": "keep"})
    _reject_unknown_keys(duplicates, DUPLICATES_KEYS, "duplicates")
    if duplicates.get("policy", "keep") not in {"keep", "sum", "allow"}:
        raise ValueError("duplicates.policy must be keep, sum, or allow")
    output = config.setdefault("output", {})
    _reject_unknown_keys(output, OUTPUT_KEYS, "output")
    for key in ["write_tns", "write_tnsb", "write_metadata", "write_readme", "copy_config"]:
        output.setdefault(key, True)
    if not output["write_metadata"]:
        raise ValueError("output.write_metadata must be true for TensorSuite bundles")
    if not output["write_tns"] and output["write_tnsb"]:
        raise ValueError("output.write_tns=false with write_tnsb=true is not supported")
    if not output["write_tns"] and not output["write_tnsb"]:
        raise ValueError("At least one tensor data file must be written")
    if not config["values_provided"] and output["write_tnsb"]:
        raise ValueError("values_provided=false requires output.write_tnsb=false")
    return config


def load_and_validate_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    return validate_config(load_config(path), base_dir=path.resolve().parent)
