from __future__ import annotations

from pathlib import Path

import numpy as np

from kronweave.io.tensorsuite import read_tns


def dense_from_tns(path: str | Path) -> tuple[np.ndarray, tuple[int, ...]]:
    data = read_tns(path)
    dense = np.zeros(data.header.dimensions, dtype=np.float64)
    for coord, value in zip(data.coords, data.values):
        dense[tuple(int(x) for x in coord)] = float(value)
    return dense, data.header.dimensions


def generate_kp_seed_from_meta(
    meta_seed_path: str | Path,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Generate a dense KP input seed from a small meta seed.

    This mirrors the historical `kronecker/roofline/seed_generator.cpp`
    `seedGenerator` logic: enumerate output seed coordinates, map each
    coordinate digit back to a meta-seed entry for every iteration, and
    multiply the corresponding meta-seed values.
    """
    if iterations <= 0:
        raise ValueError("seed.expand_iter must be > 0")
    meta, meta_dims = dense_from_tns(meta_seed_path)
    order = len(meta_dims)
    side = [int(dim) ** int(iterations) for dim in meta_dims]
    coords: list[tuple[int, ...]] = []
    values: list[float] = []
    for coord in np.ndindex(*side):
        prob = 1.0
        base = [1] * order
        for _it in range(iterations):
            digit = []
            for d in range(order):
                sd = (coord[d] // base[d]) % meta_dims[d]
                digit.append(sd)
                base[d] *= meta_dims[d]
            prob *= float(meta[tuple(digit)])
        coords.append(tuple(int(x) for x in coord))
        values.append(prob)
    return np.asarray(coords, dtype=np.uint64), np.asarray(values, dtype=np.float64), side
