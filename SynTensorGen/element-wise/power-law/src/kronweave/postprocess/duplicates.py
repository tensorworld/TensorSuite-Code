from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class DuplicateResult:
    coords: np.ndarray
    values: np.ndarray
    stats: dict


def _validate(coords: Sequence[Sequence[int]] | np.ndarray, values: Sequence[float] | np.ndarray | None):
    coord_arr = np.asarray(coords, dtype=np.int64)
    if coord_arr.size == 0:
        coord_arr = coord_arr.reshape((0, 0 if coord_arr.ndim == 1 else coord_arr.shape[-1]))
    if coord_arr.ndim != 2:
        raise ValueError("coords must be two-dimensional")
    if values is None:
        value_arr = np.ones(coord_arr.shape[0], dtype=np.float64)
    else:
        value_arr = np.asarray(values, dtype=np.float64)
        if value_arr.ndim != 1 or value_arr.shape[0] != coord_arr.shape[0]:
            raise ValueError("values must be one-dimensional and match coords length")
    return coord_arr, value_arr


def _lex_order(coords: np.ndarray) -> np.ndarray:
    if coords.shape[0] == 0:
        return np.asarray([], dtype=np.int64)
    keys = tuple(coords[:, d] for d in reversed(range(coords.shape[1])))
    return np.lexsort(keys)


def _stable_group(coord_arr: np.ndarray, value_arr: np.ndarray, policy: str) -> DuplicateResult:
    entries: dict[tuple[int, ...], list] = {}
    for coord, value in zip(coord_arr, value_arr):
        key = tuple(int(x) for x in coord)
        if key not in entries:
            entries[key] = [coord.copy(), float(value)]
        elif policy == "sum":
            entries[key][1] += float(value)
    result_coords = np.asarray([item[0] for item in entries.values()], dtype=np.uint64).reshape(
        (len(entries), coord_arr.shape[1])
    )
    result_values = np.asarray([item[1] for item in entries.values()], dtype=np.float64)
    return DuplicateResult(result_coords, result_values, compute_duplicate_stats(coord_arr))


def handle_duplicates_python(coords, values=None, policy: str = "keep", preserve_order: bool = False) -> DuplicateResult:
    coord_arr, value_arr = _validate(coords, values)
    if policy not in {"keep", "sum", "allow"}:
        raise ValueError("duplicate policy must be keep, sum, or allow")
    if policy == "allow":
        return DuplicateResult(
            coord_arr.astype(np.uint64, copy=False),
            value_arr.astype(np.float64, copy=False),
            allow_duplicate_stats(coord_arr),
        )
    if preserve_order:
        return _stable_group(coord_arr, value_arr, policy)
    order = _lex_order(coord_arr)
    sorted_coords = coord_arr[order]
    sorted_values = value_arr[order]
    if sorted_coords.shape[0] == 0:
        stats = compute_duplicate_stats(sorted_coords)
        return DuplicateResult(sorted_coords.astype(np.uint64), sorted_values.astype(np.float64), stats)

    out_coords: list[np.ndarray] = []
    out_values: list[float] = []
    i = 0
    while i < sorted_coords.shape[0]:
        j = i + 1
        while j < sorted_coords.shape[0] and np.array_equal(sorted_coords[j], sorted_coords[i]):
            j += 1
        out_coords.append(sorted_coords[i].copy())
        if policy == "sum":
            out_values.append(float(np.sum(sorted_values[i:j])))
        else:
            out_values.append(float(sorted_values[i]))
        i = j
    result_coords = np.asarray(out_coords, dtype=np.uint64).reshape((len(out_coords), coord_arr.shape[1]))
    result_values = np.asarray(out_values, dtype=np.float64)
    stats = compute_duplicate_stats(coord_arr)
    return DuplicateResult(result_coords, result_values, stats)


def handle_duplicates_torch(coords, values=None, policy: str = "keep", preserve_order: bool = False) -> DuplicateResult:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("torch is not available") from exc
    coord_arr, value_arr = _validate(coords, values)
    if policy not in {"keep", "sum", "allow"}:
        raise ValueError("duplicate policy must be keep, sum, or allow")
    if policy == "allow":
        return DuplicateResult(
            coord_arr.astype(np.uint64, copy=False),
            value_arr.astype(np.float64, copy=False),
            allow_duplicate_stats(coord_arr),
        )
    if preserve_order:
        return _stable_group(coord_arr, value_arr, policy)
    if coord_arr.shape[0] == 0:
        order = _lex_order(coord_arr)
        sorted_coords = coord_arr[order].astype(np.uint64)
        sorted_values = value_arr[order].astype(np.float64)
        return DuplicateResult(sorted_coords, sorted_values, compute_duplicate_stats(sorted_coords))

    coords_t = torch.as_tensor(coord_arr, dtype=torch.long)
    values_t = torch.as_tensor(value_arr, dtype=torch.float64)
    unique_t, inverse_t = torch.unique(coords_t, dim=0, sorted=True, return_inverse=True)
    if policy == "sum":
        out_values_t = torch.zeros(unique_t.shape[0], dtype=torch.float64)
        out_values_t.index_add_(0, inverse_t, values_t)
    else:
        out_values_t = torch.empty(unique_t.shape[0], dtype=torch.float64)
        seen = set()
        inverse_list = inverse_t.tolist()
        value_list = values_t.tolist()
        for raw_idx, unique_idx in enumerate(inverse_list):
            if unique_idx not in seen:
                out_values_t[unique_idx] = value_list[raw_idx]
                seen.add(unique_idx)
    out_coords = unique_t.cpu().numpy().astype(np.int64)
    out_values = out_values_t.cpu().numpy().astype(np.float64)
    order = _lex_order(out_coords)
    return DuplicateResult(
        out_coords[order].astype(np.uint64),
        out_values[order].astype(np.float64),
        compute_duplicate_stats(coord_arr),
    )


def handle_duplicates(coords, values=None, policy: str = "keep", preserve_order: bool = False) -> DuplicateResult:
    try:
        return handle_duplicates_torch(coords, values, policy, preserve_order=preserve_order)
    except RuntimeError:
        return handle_duplicates_python(coords, values, policy, preserve_order=preserve_order)


def compute_duplicate_stats(coords) -> dict:
    coord_arr = np.asarray(coords, dtype=np.int64)
    if coord_arr.size == 0:
        return {
            "raw_samples": 0,
            "unique_nnz": 0,
            "duplicate_count": 0,
            "duplicate_rate": 0.0,
            "max_multiplicity": 0,
        }
    if coord_arr.ndim != 2:
        raise ValueError("coords must be two-dimensional")
    counts: dict[tuple[int, ...], int] = {}
    for row in coord_arr:
        key = tuple(int(x) for x in row)
        counts[key] = counts.get(key, 0) + 1
    raw = int(coord_arr.shape[0])
    unique = len(counts)
    duplicate_count = raw - unique
    return {
        "raw_samples": raw,
        "unique_nnz": unique,
        "duplicate_count": duplicate_count,
        "duplicate_rate": float(duplicate_count / raw) if raw else 0.0,
        "max_multiplicity": max(counts.values()) if counts else 0,
    }


def allow_duplicate_stats(coords) -> dict:
    coord_arr = np.asarray(coords)
    raw = int(coord_arr.shape[0]) if coord_arr.ndim >= 1 else 0
    return {
        "raw_samples": raw,
        "unique_nnz": None,
        "duplicate_count": None,
        "duplicate_rate": None,
        "max_multiplicity": None,
        "note": "duplicates.policy=allow preserves fastSKG raw samples; uniqueness stats were not computed",
    }
