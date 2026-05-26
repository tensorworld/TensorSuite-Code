from __future__ import annotations

from itertools import product
import math
from pathlib import Path
import random
from typing import Sequence

import numpy as np


def expand_list(x: Sequence[float], D: int) -> list[float]:
    values = list(x)
    if len(values) == 1:
        return values * D
    if len(values) != D:
        raise ValueError("alpha length must be 1 or D")
    return values


def power_law_vec(n: int, alpha: float) -> list[float]:
    v = [(i + 1) ** (-alpha) for i in range(n)]
    total = sum(v)
    return [x / total for x in v]


def generate_cp_seed(
    sizes: Sequence[int],
    rank: int,
    alpha: Sequence[float],
    noise: float = 0.0,
    random_seed: int | None = None,
    lambda_controller: str = "softmax",
) -> tuple[np.ndarray, np.ndarray]:
    # Historical behavior: default fixed seed 123, overridden by user seed.
    rng = random.Random(123 if random_seed is None else random_seed)
    sizes = [int(s) for s in sizes]
    if not sizes or any(s <= 0 for s in sizes):
        raise ValueError("seed sizes must be positive")
    D = len(sizes)
    if rank <= 0:
        raise ValueError("rank must be >= 1")
    alphas = expand_list(alpha, D)
    factors: list[list[list[float]]] = []
    for _r in range(rank):
        per_rank = []
        for d in range(D):
            vec = power_law_vec(sizes[d], alphas[d])
            perm = list(range(sizes[d]))
            rng.shuffle(perm)
            per_rank.append([vec[i] for i in perm])
        factors.append(per_rank)

    if lambda_controller == "softmax":
        lambdas = [math.exp(-r) for r in range(rank)]
        lambda_sum = sum(lambdas)
        lambdas = [x / lambda_sum for x in lambdas]
    elif lambda_controller == "none":
        lambdas = [1.0] * rank
    else:
        raise ValueError("lambda_controller must be softmax or none")

    entries: list[tuple[tuple[int, ...], float]] = []
    max_val = 0.0
    for idx in product(*[range(s) for s in sizes]):
        val = 0.0
        for r in range(rank):
            prod_val = lambdas[r]
            for d in range(D):
                prod_val *= factors[r][d][idx[d]]
            val += prod_val
        if noise > 0:
            val *= 1.0 + noise * (rng.random() - 0.5)
        entries.append((idx, val))
        max_val = max(max_val, val)
    if max_val <= 0:
        raise RuntimeError("CP seed construction failed: max value <= 0")

    eps = 1e-12
    coords = np.asarray([idx for idx, _ in entries], dtype=np.uint64)
    values = np.asarray([min(max(val / max_val, eps), 1.0 - eps) for _, val in entries], dtype=np.float64)
    return coords, values


def generate_cp_seed_from_meta_seed(
    meta_seed_path: str | Path,
    *,
    sizes: Sequence[int],
    rank: int,
    alpha: Sequence[float],
    noise: float = 0.0,
    random_seed: int | None = None,
    lambda_controller: str = "softmax",
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Backward-compatible wrapper for older callers.

    CP seed generation follows the repository-level ``seed_generator_cpd.py``
    logic directly: CP factors are generated from sizes/rank/alpha/noise and
    the resulting full CP seed is consumed by fastSKG. ``meta_seed_path`` is
    intentionally ignored; older configs may still pass it, but CP generation
    does not read a meta-seed tensor or depend on its file format.
    """
    _ = meta_seed_path
    target_sizes = [int(x) for x in sizes]
    coords, values = generate_cp_seed(
        sizes=target_sizes,
        rank=int(rank),
        alpha=alpha,
        noise=float(noise),
        random_seed=random_seed,
        lambda_controller=lambda_controller,
    )
    return coords, values, target_sizes
