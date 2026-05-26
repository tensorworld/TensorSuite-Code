from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distribution_generator import generate_from_config as _generate_from_config
from distribution_generator import run_cli


def _rescale_counts(counts: np.ndarray, total: int) -> np.ndarray:
    counts = np.maximum(np.asarray(counts, dtype=np.float64), 0.0)
    if counts.sum() <= 0:
        counts[:] = 1.0
    scaled = counts * (float(total) / float(counts.sum()))
    out = np.floor(scaled).astype(np.int64)
    deficit = int(total - out.sum())
    if deficit > 0:
        for idx in np.argsort(-(scaled - out))[:deficit]:
            out[idx] += 1
    return out


def _column_from_counts(counts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    column = np.repeat(np.arange(len(counts), dtype=np.uint64), counts.astype(np.int64))
    rng.shuffle(column)
    return column


def generate_random_coords(config: dict[str, Any]) -> np.ndarray:
    sizes = [int(x) for x in config["tensor"]["sizes"]]
    nnz = int(config["tensor"]["nnz"])
    rng = np.random.default_rng(config.get("generator", {}).get("random_seed"))

    columns = []
    for size in sizes:
        degree = rng.random(size)
        counts = _rescale_counts(degree, nnz)
        columns.append(_column_from_counts(counts, rng))
    return np.stack(columns, axis=1)


def generate_from_config(config_path: str | Path) -> dict[str, Path]:
    return _generate_from_config(config_path, "random", generate_random_coords)


def main(argv: list[str] | None = None) -> int:
    return run_cli(argv, "random", generate_random_coords)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
