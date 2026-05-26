from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from kronweave.io.tensorsuite import read_tns


def load_tensorsuite_tensor(path: str | Path):
    data = read_tns(path)
    print(f"Loading tensor file: {path}")
    print(f"Loaded {len(data.values):,} nonzeros with order={data.header.num_modes}")
    return data.coords.astype(np.int64), data.values, list(data.header.dimensions), data.header.name


def tensor_unfold_nd(coords: np.ndarray, dims: list[int], mode: int):
    mode -= 1
    N = coords.shape[1]
    rows = coords[:, mode]
    others = [i for i in range(N) if i != mode]
    multiplier = 1
    cols = np.zeros(len(rows), dtype=np.int64)
    for d in reversed(others):
        cols += coords[:, d] * multiplier
        multiplier *= dims[d]
    return rows, cols


def tensor_slice_nd(coords: np.ndarray, vals: np.ndarray, dims: list[int], mode: int, index: int):
    mode -= 1
    mask = coords[:, mode] == index
    coords_slice = coords[mask]
    vals_slice = vals[mask]
    new_coords = np.delete(coords_slice, mode, axis=1)
    new_dims = dims[:mode] + dims[mode + 1 :]
    return new_coords, vals_slice, new_dims


def plot_degree_distribution_from_rows(rows: np.ndarray, save_path: str | Path, title: str) -> None:
    _unique_rows, degree = np.unique(rows, return_counts=True)
    unique_d, Nd = np.unique(degree, return_counts=True)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.loglog(unique_d, Nd, "o", markersize=4)
    plt.xlabel("Degree d")
    plt.ylabel("Number of nodes Nd")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Saved] {save_path}")


def visualize_tns(tns_path: str | Path, out_dir: str | Path | None = None) -> list[Path]:
    coords, vals, dims, tensor_name = load_tensorsuite_tensor(tns_path)
    out_dir = Path(out_dir) if out_dir is not None else Path("plots") / tensor_name
    outputs: list[Path] = []
    order = coords.shape[1]
    for mode in range(1, order + 1):
        print(f"\n=== Processing mode {mode} ===")
        rows, _cols = tensor_unfold_nd(coords, dims, mode)
        save_path = out_dir / f"mode{mode}_unfold_powerlaw.png"
        title = f"{tensor_name} - Mode-{mode} Unfold Nd~d"
        plot_degree_distribution_from_rows(rows, save_path, title)
        outputs.append(save_path)
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tns", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    visualize_tns(args.tns, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
