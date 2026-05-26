from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kronweave.io.tensorsuite import read_tns


def row_degree_samples(coords: np.ndarray, mode: int) -> np.ndarray:
    rows = coords[:, mode]
    _, degree = np.unique(rows, return_counts=True)
    return degree.astype(np.int64)


def analyze_tns(path: str | Path, verbose: bool = False) -> list[dict]:
    data = read_tns(path)
    try:
        import powerlaw
    except ImportError as exc:
        raise RuntimeError("Optional dependency `powerlaw` is required. Install with `pip install powerlaw`.") from exc
    results = []
    for mode in range(data.header.num_modes):
        degrees = row_degree_samples(data.coords, mode)
        fit = powerlaw.Fit(degrees, discrete=True)
        result = {
            "mode": mode + 1,
            "observed_rows": int(len(degrees)),
            "degree_min": int(degrees.min()) if len(degrees) else 0,
            "degree_max": int(degrees.max()) if len(degrees) else 0,
            "alpha": float(fit.power_law.alpha),
            "xmin": float(fit.power_law.xmin),
        }
        if verbose:
            result["ks_distance"] = float(getattr(fit.power_law, "D", 0.0))
        results.append(result)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tns", required=True)
    parser.add_argument("--out", help="Optional JSON report path")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    try:
        results = analyze_tns(args.tns, verbose=args.verbose)
        if args.out:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"tns": args.tns, "results": results}, indent=2) + "\n", encoding="utf-8")
            print(f"[Saved] {out}")
        for result in results:
            print(result)
    except RuntimeError as exc:
        print(exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
