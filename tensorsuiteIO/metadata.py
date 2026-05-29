from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Literal, Sequence

ValidationProfile = Literal["tensorsuite", "synthetic_element_compact"] | None

TENSORSUITE_REQUIRED_KEYS = {
    "version",
    "name",
    "group",
    "id",
    "time",
    "source_type",
    "source",
    "source_url",
    "value_type",
    "value_domain",
    "values_provided",
    "index_type",
    "endianness",
    "index_base",
    "sorted",
    "sort_order",
    "duplicates",
    "explicit_zeros",
    "pattern_symmetry",
    "numerical_symmetry",
    "sparsity_type",
    "dense_modes",
    "order",
    "dimensions",
    "nnz",
    "block_partitions",
    "nnz_block",
    "files",
}

TENSORSUITE_ALLOWED_METADATA_VALUES = {
    "value_type": {"float32", "float64", "int32", "int64"},
    "value_domain": {"binary", "nonnegative", "general"},
    "source_type": {"synthetic", "real"},
    "index_type": {"uint32", "uint64"},
    "endianness": {"none", "little", "big"},
    "sorted": {"none", "lexicographic"},
    "duplicates": {"disallowed", "allowed"},
    "explicit_zeros": {"allowed", "disallowed"},
    "pattern_symmetry": {"full", "partial", "no"},
    "numerical_symmetry": {"full", "partial", "no"},
    "sparsity_type": {"no", "semi", "element", "block"},
}


SYNTHETIC_ELEMENT_COMPACT_ALLOWED_METADATA_VALUES = {
    **TENSORSUITE_ALLOWED_METADATA_VALUES,
    "source_type": {"synthetic"},
    "endianness": {"little"},
    "pattern_symmetry": {"no"},
    "numerical_symmetry": {"no"},
    "sparsity_type": {"element"},
}


SYNTHETIC_ELEMENT_COMPACT_FORBIDDEN_METADATA_KEYS = {
    "storage",
    "source_url",
    "dense_modes",
    "block_partitions",
    "nnz_block",
    "collision_rate",
    "duplicate_rate",
    "raw_sample_count",
    "raw_samples",
    "unique_nnz",
    "power_law_alpha",
    "alpha_fit",
    "xmin",
    "ks_distance",
    "low_rank_stats",
    "mode_correlation",
    "blocking_analysis",
    "coupling_analysis",
    "mttkrp_imbalance",
    "plot_path",
    "evaluation_result",
}


def duplicate_metadata_value(policy: str) -> str:
    mapping = {"keep": "disallowed", "sum": "disallowed", "allow": "allowed"}
    try:
        return mapping[policy]
    except KeyError as exc:
        raise ValueError("duplicate policy must be keep, sum, or allow") from exc


def build_metadata(
    *,
    name: str,
    dims: Sequence[int],
    nnz: int,
    index_base: int,
    value_domain: str,
    duplicate_policy: str,
    value_type: str = "float64",
    index_type: str = "uint64",
    endianness: str = "little",
    sorted_state: str = "lexicographic",
    sorted_order: Sequence[int] | None = None,
    explicit_zeros: str = "disallowed",
    pattern_symmetry: str = "no",
    numerical_symmetry: str = "no",
    sparsity_type: str = "element",
    values_provided: bool = True,
    files: dict | None = None,
    source: str = "KronWeave powerLawGenerator",
    source_url: str = "",
    source_type: str = "synthetic",
    group: str = "Synthetic",
    time: str = "2025-06-05",
    version: str = "0.1",
    tensor_id: str | int | None = None,
    dense_modes: Sequence[int] | None = None,
    block_partitions: Sequence[int] | None = None,
    nnz_block: int | None = None,
) -> dict:
    dims_list = [int(d) for d in dims]
    order = len(dims_list)
    if sorted_order is None:
        sorted_order = list(range(order)) if sorted_state == "lexicographic" else []
    file_map = files or {
        "binary": f"{name}.tnsb",
        "text": f"{name}.tns",
        "readme": "README.md",
        "config": "config.yaml",
    }
    metadata_sort_order = [int(x) + 1 for x in sorted_order] if sorted_state == "lexicographic" else []
    metadata = {
        "version": version,
        "name": name,
        "group": group,
        "id": _format_tensor_id(tensor_id),
        "time": time,
        "source_type": source_type,
        "source": source,
        "source_url": source_url,
        "value_type": value_type,
        "value_domain": value_domain,
        "values_provided": bool(values_provided),
        "index_type": index_type,
        "endianness": "none" if not file_map.get("binary") else endianness,
        "index_base": int(index_base),
        "sorted": sorted_state,
        "sort_order": metadata_sort_order,
        "duplicates": duplicate_metadata_value(duplicate_policy),
        "explicit_zeros": explicit_zeros,
        "pattern_symmetry": pattern_symmetry,
        "numerical_symmetry": numerical_symmetry,
        "sparsity_type": sparsity_type,
        "dense_modes": [int(x) for x in (dense_modes or [])],
        "order": order,
        "dimensions": dims_list,
        "nnz": int(nnz),
        "block_partitions": None if block_partitions is None else [int(x) for x in block_partitions],
        "nnz_block": None if nnz_block is None else int(nnz_block),
        "files": file_map,
    }
    validate_tensorsuite_metadata(metadata)
    return metadata


def _format_tensor_id(tensor_id: str | int | None) -> str:
    if tensor_id is None:
        return "1001"
    if isinstance(tensor_id, int):
        if tensor_id < 0 or tensor_id > 9999:
            raise ValueError("metadata.id must fit four digits")
        return f"{tensor_id:04d}"
    return str(tensor_id)


def _require_keys(metadata: dict, required: set[str]) -> None:
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"metadata is missing required fields: {missing}")


def _validate_allowed_values(metadata: dict, allowed_values: dict[str, set[str]]) -> None:
    for key, allowed in allowed_values.items():
        value = metadata.get(key)
        if value not in allowed:
            raise ValueError(f"metadata.{key} must be one of {sorted(allowed)}, got {value!r}")


def _validate_common_metadata(metadata: dict, allowed_values: dict[str, set[str]]) -> None:
    _validate_allowed_values(metadata, allowed_values)
    if metadata.get("index_base") not in {0, 1}:
        raise ValueError("metadata.index_base must be 0 or 1")
    if not isinstance(metadata.get("values_provided"), bool):
        raise ValueError("metadata.values_provided must be boolean")
    if not isinstance(metadata.get("id"), str) or not re.fullmatch(r"\d{4}", metadata["id"]):
        raise ValueError("metadata.id must be a four-digit string")
    if not isinstance(metadata.get("time"), str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", metadata["time"]):
        raise ValueError("metadata.time must be an ISO date string YYYY-MM-DD")
    if not isinstance(metadata.get("name"), str) or not metadata["name"]:
        raise ValueError("metadata.name must be a non-empty string")
    if not isinstance(metadata.get("group"), str):
        raise ValueError("metadata.group must be a string")
    if not isinstance(metadata.get("source"), str):
        raise ValueError("metadata.source must be a string")
    order = int(metadata.get("order", -1))
    dimensions = metadata.get("dimensions")
    if order <= 0:
        raise ValueError("metadata.order must be positive")
    if not isinstance(dimensions, list) or len(dimensions) != order or any(not isinstance(x, int) or x <= 0 for x in dimensions):
        raise ValueError("metadata.dimensions must be a positive integer list matching metadata.order")
    if not isinstance(metadata.get("nnz"), int) or metadata["nnz"] < 0:
        raise ValueError("metadata.nnz must be a nonnegative integer")
    sorted_order = metadata.get("sort_order")
    if metadata.get("sorted") == "lexicographic":
        if sorted(sorted_order) != list(range(1, order + 1)):
            raise ValueError("metadata.sort_order must be a 1-based permutation when sorted=lexicographic")
    elif sorted_order != []:
        raise ValueError("metadata.sort_order must be [] when sorted=none")
    files = metadata.get("files")
    if not isinstance(files, dict):
        raise ValueError("metadata.files must be a mapping")
    if not files.get("text") and not files.get("binary"):
        raise ValueError("metadata.files must include at least one tensor data file")
    if files.get("binary"):
        if metadata.get("endianness") not in {"little", "big"}:
            raise ValueError("metadata.endianness must be little or big when a binary tensor file is present")
    elif metadata.get("endianness") != "none":
        raise ValueError("metadata.endianness must be none when no binary tensor file is present")


def validate_tensorsuite_metadata(metadata: dict) -> None:
    _require_keys(metadata, TENSORSUITE_REQUIRED_KEYS)
    _validate_common_metadata(metadata, TENSORSUITE_ALLOWED_METADATA_VALUES)
    if not isinstance(metadata.get("source_url"), str):
        raise ValueError("metadata.source_url must be a string")
    dense_modes = metadata.get("dense_modes")
    order = int(metadata["order"])
    if not isinstance(dense_modes, list) or any(not isinstance(x, int) or x < 1 or x > order for x in dense_modes):
        raise ValueError("metadata.dense_modes must be a list of 1-based mode ids")
    if metadata["sparsity_type"] == "block":
        block_partitions = metadata.get("block_partitions")
        if (
            not isinstance(block_partitions, list)
            or len(block_partitions) != order
            or any(not isinstance(x, int) or x <= 0 for x in block_partitions)
        ):
            raise ValueError("metadata.block_partitions must be a positive integer list for block tensors")
        if not isinstance(metadata.get("nnz_block"), int) or metadata["nnz_block"] < 0:
            raise ValueError("metadata.nnz_block must be a nonnegative integer for block tensors")
    else:
        if metadata.get("block_partitions") is not None:
            raise ValueError("metadata.block_partitions must be null for non-block tensors")
        if metadata.get("nnz_block") is not None:
            raise ValueError("metadata.nnz_block must be null for non-block tensors")


def validate_synthetic_element_compact_metadata(metadata: dict) -> None:
    _validate_common_metadata(metadata, SYNTHETIC_ELEMENT_COMPACT_ALLOWED_METADATA_VALUES)
    if metadata.get("source_type") != "synthetic":
        raise ValueError("metadata.source_type must be synthetic")
    ensure_no_synthetic_element_compact_forbidden_metadata(metadata)


def validate_synthetic_element_metadata(metadata: dict) -> None:
    validate_synthetic_element_compact_metadata(metadata)


def validate_canonical_metadata(metadata: dict) -> None:
    validate_tensorsuite_metadata(metadata)


def ensure_no_synthetic_element_compact_forbidden_metadata(metadata: dict) -> None:
    lowered = {str(k).lower() for k in metadata.keys()}
    forbidden = lowered & SYNTHETIC_ELEMENT_COMPACT_FORBIDDEN_METADATA_KEYS
    if forbidden:
        raise ValueError(f"Synthetic element compact metadata contains forbidden fields: {sorted(forbidden)}")


def ensure_no_forbidden_metadata(metadata: dict) -> None:
    ensure_no_synthetic_element_compact_forbidden_metadata(metadata)


def validate_metadata(metadata: dict, profile: ValidationProfile = "tensorsuite") -> None:
    if profile == "tensorsuite":
        validate_tensorsuite_metadata(metadata)
    elif profile == "synthetic_element_compact":
        validate_synthetic_element_compact_metadata(metadata)
    elif profile is None:
        return
    else:
        raise ValueError(f"Unknown metadata validation profile: {profile!r}")


def write_metadata(path: str | Path, metadata: dict, profile: ValidationProfile = "tensorsuite") -> None:
    validate_metadata(metadata, profile)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_metadata(path: str | Path, profile: ValidationProfile = "tensorsuite") -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_metadata(data, profile)
    return data
