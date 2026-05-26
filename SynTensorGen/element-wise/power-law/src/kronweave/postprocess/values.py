from __future__ import annotations

from typing import Any

import numpy as np


def normalize_value_config(value_config: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(value_config or {})
    cfg.setdefault("mode", "constant")
    if cfg["mode"] == "constant":
        cfg.setdefault("constant", 1.0)
    elif cfg["mode"] == "uniform":
        cfg.setdefault("low", 0.0)
        cfg.setdefault("high", 1.0)
    elif cfg["mode"] == "normal":
        cfg.setdefault("mean", 0.0)
        cfg.setdefault("std", 1.0)
        cfg.setdefault("clip", 3.0)
    return cfg


def infer_value_domain(value_config: dict[str, Any] | None) -> str:
    cfg = normalize_value_config(value_config)
    mode = cfg["mode"]
    if mode == "constant":
        c = float(cfg.get("constant", 1.0))
        if c == 1.0:
            return "binary"
        return "nonnegative" if c >= 0 else "general"
    if mode == "uniform":
        low = float(cfg["low"])
        high = float(cfg["high"])
        if low >= high:
            raise ValueError("uniform value config requires low < high")
        return "nonnegative" if low >= 0 else "general"
    if mode == "normal":
        std = float(cfg["std"])
        clip = float(cfg["clip"])
        if std <= 0:
            raise ValueError("normal value config requires std > 0")
        if clip <= 0:
            raise ValueError("normal value config requires clip > 0")
        return "general"
    raise ValueError("value.mode must be constant, uniform, or normal")


def generate_values(num_entries: int, value_config: dict[str, Any] | None, seed: int | None = None) -> np.ndarray:
    if num_entries < 0:
        raise ValueError("num_entries must be nonnegative")
    cfg = normalize_value_config(value_config)
    mode = cfg["mode"]
    if mode == "constant":
        return np.full(num_entries, float(cfg.get("constant", 1.0)), dtype=np.float64)
    rng = np.random.default_rng(seed)
    if mode == "uniform":
        low = float(cfg["low"])
        high = float(cfg["high"])
        if low >= high:
            raise ValueError("uniform value config requires low < high")
        return rng.uniform(low, high, size=num_entries).astype(np.float64)
    if mode == "normal":
        mean = float(cfg["mean"])
        std = float(cfg["std"])
        clip = float(cfg["clip"])
        if std <= 0:
            raise ValueError("normal value config requires std > 0")
        if clip <= 0:
            raise ValueError("normal value config requires clip > 0")
        values = rng.normal(mean, std, size=num_entries)
        return np.clip(values, mean - clip * std, mean + clip * std).astype(np.float64)
    raise ValueError("value.mode must be constant, uniform, or normal")
