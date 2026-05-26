from __future__ import annotations

from pathlib import Path
import subprocess

from kronweave.io.tensorsuite import read_tns


def cpp_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "cpp"


def ensure_fastskg_binary() -> Path:
    directory = cpp_dir()
    binary = directory / "run_fastSKG_omp"
    if binary.exists() and _binary_can_launch(binary, directory):
        return binary
    if binary.exists():
        binary.unlink()
    subprocess.run(["make"], cwd=directory, check=True)
    if not binary.exists():
        raise RuntimeError("fastSKG binary was not produced by make")
    return binary


def _binary_can_launch(binary: Path, cwd: Path) -> bool:
    try:
        subprocess.run([str(binary)], cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        return False
    return True


def run_fastskg(
    *,
    seed_tns: str | Path,
    raw_output_tns: str | Path,
    iterations: int,
    nnz: int | None,
    index_base: int,
    name: str,
) -> tuple:
    binary = ensure_fastskg_binary()
    cmd = [
        str(binary),
        "--seed",
        str(seed_tns),
        "--out",
        str(raw_output_tns),
        "--iter",
        str(iterations),
        "--nnz",
        str(-1 if nnz is None else nnz),
        "--index-base",
        str(index_base),
        "--name",
        name,
    ]
    subprocess.run(cmd, cwd=cpp_dir(), check=True)
    data = read_tns(raw_output_tns)
    coords = data.coords.astype("int64") - int(index_base)
    return coords, data.values
