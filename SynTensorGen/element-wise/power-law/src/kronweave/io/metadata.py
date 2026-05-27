from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

ALLOWED_METADATA_VALUES = {
    "value_type": {"float32", "float64", "int32", "int64"},
    "value_domain": {"binary", "nonnegative", "general"},
    "index_type": {"uint32", "uint64"},
    "endianness": {"little"},
    "sorted": {"none", "lexicographic"},
    "duplicates": {"disallowed", "summed", "allowed"},
    "explicit_zeros": {"allowed", "disallowed"},
    "pattern_symmetry": {"no"},
    "numerical_symmetry": {"no"},
    "sparsity_type": {"element"},
}


FORBIDDEN_METADATA_KEYS = {
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
    mapping = {"keep": "disallowed", "sum": "summed", "allow": "allowed"}
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
    version: str = "0.1",
    tensor_id: int | None = None,
) -> dict:
    dims_list = [int(d) for d in dims]
    order = len(dims_list)
    if sorted_order is None:
        sorted_order = list(range(order)) if sorted_state == "lexicographic" else []
    metadata = {
        "version": version,
        "name": name,
        "group": "Synthetic",
        "id": int(tensor_id or 1001),
        "time": "2025-06-05",
        "source_type": "synthetic",
        "source": source,
        "value_type": value_type,
        "value_domain": value_domain,
        "values_provided": bool(values_provided),
        "index_type": index_type,
        "endianness": endianness,
        "index_base": int(index_base),
        "sorted": sorted_state,
        "sort_order": [int(x) for x in sorted_order],
        "duplicates": duplicate_metadata_value(duplicate_policy),
        "explicit_zeros": explicit_zeros,
        "pattern_symmetry": pattern_symmetry,
        "numerical_symmetry": numerical_symmetry,
        "sparsity_type": sparsity_type,
        "order": order,
        "dimensions": dims_list,
        "nnz": int(nnz),
        "files": files or {
            "binary": f"{name}.tnsb",
            "text": f"{name}.tns",
            "readme": "README.md",
            "config": "config.yaml",
        },
    }
    validate_canonical_metadata(metadata)
    ensure_no_forbidden_metadata(metadata)
    return metadata


def validate_canonical_metadata(metadata: dict) -> None:
    for key, allowed in ALLOWED_METADATA_VALUES.items():
        value = metadata.get(key)
        if value not in allowed:
            raise ValueError(f"metadata.{key} must be one of {sorted(allowed)}, got {value!r}")
    if metadata.get("index_base") not in {0, 1}:
        raise ValueError("metadata.index_base must be 0 or 1")
    if not isinstance(metadata.get("values_provided"), bool):
        raise ValueError("metadata.values_provided must be boolean")
    if not isinstance(metadata.get("id"), int) or metadata["id"] <= 1000:
        raise ValueError("metadata.id must be an integer greater than 1000")
    if metadata.get("time") != "2025-06-05":
        raise ValueError("metadata.time must be 2025-06-05")
    if metadata.get("source_type") != "synthetic":
        raise ValueError("metadata.source_type must be synthetic")
    order = int(metadata.get("order", -1))
    sorted_order = metadata.get("sort_order")
    if metadata.get("sorted") == "lexicographic":
        if sorted(sorted_order) != list(range(order)):
            raise ValueError("metadata.sort_order must be a 0-based permutation when sorted=lexicographic")
    elif sorted_order != []:
        raise ValueError("metadata.sort_order must be [] when sorted=none")


def ensure_no_forbidden_metadata(metadata: dict) -> None:
    lowered = {str(k).lower() for k in metadata.keys()}
    forbidden = lowered & FORBIDDEN_METADATA_KEYS
    if forbidden:
        raise ValueError(f"Metadata contains forbidden fields: {sorted(forbidden)}")


def write_metadata(path: str | Path, metadata: dict) -> None:
    validate_canonical_metadata(metadata)
    ensure_no_forbidden_metadata(metadata)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_metadata(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_canonical_metadata(data)
    ensure_no_forbidden_metadata(data)
    return data
