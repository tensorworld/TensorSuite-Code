from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from distribution_generator import generate_from_config as _generate_from_config
from distribution_generator import run_cli


def _expand_float(value: Any, order: int, default: float, key: str) -> list[float]:
    if value is None:
        values = [default]
    elif isinstance(value, list):
        values = [float(x) for x in value]
    else:
        values = [float(value)]
    if len(values) == 1:
        return values * order
    if len(values) != order:
        raise ValueError(f"{key} length must be 1 or tensor order")
    return values


def _rebalance_counts(counts: np.ndarray, total: int, rng: np.random.Generator) -> np.ndarray:
    out = np.maximum(np.asarray(counts, dtype=np.int64), 0)
    current = int(out.sum())
    if current <= 0:
        out[:] = 1
        current = int(out.sum())

    diff = int(total - current)
    if diff > 0:
        additions = rng.multinomial(diff, np.full(len(out), 1.0 / len(out)))
        out += additions.astype(np.int64)
    elif diff < 0:
        remaining = -diff
        while remaining > 0:
            positive = np.flatnonzero(out > 0)
            if len(positive) == 0:
                break
            take = min(remaining, len(positive))
            selected = rng.choice(positive, size=take, replace=False)
            out[selected] -= 1
            remaining -= take
    return out


def _column_from_counts(counts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    column = np.repeat(np.arange(len(counts), dtype=np.uint64), counts.astype(np.int64))
    rng.shuffle(column)
    return column


def generate_random_coords(config: dict[str, Any]) -> np.ndarray:
    sizes = [int(x) for x in config["tensor"]["sizes"]]
    nnz = int(config["tensor"]["nnz"])
    rng = np.random.default_rng(config.get("generator", {}).get("random_seed"))

    random_cfg = config.get("random", {})
    target_means = [nnz / float(size) for size in sizes]
    lows = _expand_float(random_cfg.get("low"), len(sizes), 0.0, "random.low")
    highs = (
        _expand_float(random_cfg["high"], len(sizes), 2.0 * target_means[0], "random.high")
        if "high" in random_cfg
        else [2.0 * mean for mean in target_means]
    )

    columns = []
    for size, low, high in zip(sizes, lows, highs):
        low_i = int(np.floor(low))
        high_i = int(np.ceil(high))
        if low_i < 0:
            raise ValueError("random.low must be >= 0")
        if high_i < low_i:
            raise ValueError("random.high must be >= random.low")
        degree = rng.integers(low_i, high_i + 1, size=size, dtype=np.int64)
        counts = _rebalance_counts(degree, nnz, rng)
        columns.append(_column_from_counts(counts, rng))
    return np.stack(columns, axis=1)


def generate_from_config(config_path: str | Path) -> dict[str, Path]:
    return _generate_from_config(config_path, "random", generate_random_coords)


def main(argv: list[str] | None = None) -> int:
    return run_cli(argv, "random", generate_random_coords)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
