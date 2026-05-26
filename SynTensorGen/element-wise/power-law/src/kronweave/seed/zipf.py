from __future__ import annotations

from typing import Sequence

import numpy as np


def expand_list(x: Sequence[float], D: int) -> list[float]:
    values = list(x)
    if len(values) == 1:
        return values * D
    if len(values) != D:
        raise ValueError("alpha length must be 1 or D")
    return values


def sample_trunc_zipf(rng: np.random.Generator, alpha: float, N: int, m: int) -> np.ndarray:
    out = np.empty(m, dtype=np.int64)
    filled = 0
    while filled < m:
        need = m - filled
        draw = rng.zipf(alpha, size=int(need * 1.25) + 16) - 1
        draw = draw[(draw >= 0) & (draw < N)]
        take = min(draw.size, need)
        if take > 0:
            out[filled : filled + take] = draw[:take]
            filled += take
    return out


def generate_zipf_coords(
    sizes: Sequence[int],
    nnz: int,
    alpha: Sequence[float],
    random_seed: int | None = 42,
) -> np.ndarray:
    sizes = [int(s) for s in sizes]
    if not sizes or any(s <= 0 for s in sizes):
        raise ValueError("tensor sizes must be positive")
    if nnz <= 0:
        raise ValueError("nnz must be > 0")
    alphas = expand_list(alpha, len(sizes))
    for a in alphas:
        if a <= 1.0:
            raise ValueError("Zipf alpha must be > 1.0")
    rng = np.random.default_rng(random_seed)
    columns = [sample_trunc_zipf(rng, alphas[d], sizes[d], nnz) for d in range(len(sizes))]
    return np.stack(columns, axis=1).astype(np.uint64)
