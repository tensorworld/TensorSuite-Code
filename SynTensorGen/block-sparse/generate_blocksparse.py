#!/usr/bin/env python3
"""Generate block-sparse tensor pairs in TensorSuite bundle format.

Two modes:
  * count-driven (mirrors fuzz_pipeline): pick label counts, random labels a-z
  * einsum-driven: --einsum "abc,bcd->ad" pins labels and tensor layout

Always emits two sibling bundles <name>_A and <name>_B. Block pattern only
(values_provided=false). Defaults: float64, uint64, index_base=0.

Vocabulary matches the .tns block file layout:
  * dim_npartitions : number of partitions per dim  (== nblocks_i)
  * partition_size  : size of one partition along a dim
  * total dim size  : sum of partition_size over that dim (line 2 of .tns)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple


# ---------------------------------------------------------------------------
# Generator (port of include/general_sparse/random_contraction.hpp)
# ---------------------------------------------------------------------------

class Distribution(str, Enum):
    UNIFORM = "uniform"
    RANDOM = "random"
    MIXED = "mixed"
    POWER_LAW = "power_law"


@dataclass
class Params:
    # Rank composition (count-driven mode)
    num_free_A: int = 1
    num_free_B: int = 1
    num_contracted: int = 1
    num_batch: int = 0

    # Per-dim partition count
    min_dim_npartitions: int = 1
    max_dim_npartitions: int = 5

    # Per-partition sizes
    min_partition_size: int = 1
    max_partition_size: int = 64
    uniform_partition_sizes: bool = False
    distribution: Distribution = Distribution.RANDOM

    # Sparsity
    density: float = 0.3
    min_nnz: int = 1

    # Layout
    shuffle_labels: bool = True

    # Per-role partition-size overrides (0 = use global)
    small_free_A_lo: int = 0
    small_free_A_hi: int = 0
    small_free_B_lo: int = 0
    small_free_B_hi: int = 0
    contracted_lo: int = 0
    contracted_hi: int = 0

    # Variance (±fraction around midpoint), applied like fuzz_pipeline
    partition_size_variance: float = 0.0

    seed: int = 42

    # Einsum mode (None = count-driven). When set, all label-role/order
    # decisions come from the string and per-label overrides below.
    einsum: Optional[str] = None
    label_partition_size: Dict[str, int] = field(default_factory=dict)
    label_npartitions: Dict[str, int] = field(default_factory=dict)


def _apply_partition_size_variance(p: Params) -> None:
    """Mirror fuzz_pipeline.cu: expand min/max around midpoint by ±variance."""
    if p.partition_size_variance <= 0:
        return
    mid = (p.min_partition_size + p.max_partition_size) // 2
    delta = max(1, round(mid * p.partition_size_variance))
    p.min_partition_size = max(1, mid - delta)
    p.max_partition_size = mid + delta
    p.uniform_partition_sizes = False
    if p.distribution == Distribution.UNIFORM:
        p.distribution = Distribution.RANDOM


def _gen_partition_sizes(rng: random.Random, n: int, lo: int, hi: int,
                         distribution: Distribution, uniform: bool) -> List[int]:
    """Per-label partition sizes, matching gen_sections() in the C++ generator."""
    mode = Distribution.UNIFORM if uniform else distribution
    if mode == Distribution.UNIFORM:
        s = rng.randint(lo, hi)
        return [s] * n
    if mode == Distribution.RANDOM:
        return [rng.randint(lo, hi) for _ in range(n)]
    if mode == Distribution.MIXED:
        rng_span = hi - lo
        lo_hi = min(hi, lo + max(1, rng_span // 3))
        hi_lo = max(lo, hi - max(1, rng_span // 3))
        out = []
        for i in range(n):
            if i % 2 == 0:
                out.append(rng.randint(lo, lo_hi))
            else:
                out.append(rng.randint(hi_lo, hi))
        return out
    if mode == Distribution.POWER_LAW:
        span = hi - lo
        return [lo + round(span * (rng.random() ** 3)) for _ in range(n)]
    raise ValueError(f"unknown distribution: {distribution}")


@dataclass
class Tensor:
    name: str            # "A" or "B"
    labels: List[str]    # storage label order
    num_blocks: List[int]                 # per-axis partition count
    partition_sizes: Dict[str, List[int]] # per-label partition sizes
    nnz_blocks: List[Tuple[int, ...]]     # sorted lexicographically


@dataclass
class Problem:
    A: Tensor
    B: Tensor
    output_labels: List[str]
    partition_sizes_map: Dict[str, List[int]]
    npartitions_map: Dict[str, int]
    params: Params

    @property
    def einsum(self) -> str:
        return (
            "".join(self.A.labels) + ","
            + "".join(self.B.labels) + "->"
            + "".join(self.output_labels)
        )


def _parse_einsum(s: str) -> Tuple[List[str], List[str], List[str]]:
    s = s.replace(" ", "")
    if "->" not in s or "," not in s:
        raise ValueError(f"einsum must be 'A,B->O' (got {s!r})")
    lhs, out = s.split("->", 1)
    parts = lhs.split(",")
    if len(parts) != 2:
        raise ValueError("einsum must have exactly two input operands")
    A, B, O = list(parts[0]), list(parts[1]), list(out)
    for grp in (A, B, O):
        if len(set(grp)) != len(grp):
            raise ValueError(f"repeated labels within an operand are not supported: {grp}")
    for c in O:
        if c not in A and c not in B:
            raise ValueError(f"output label '{c}' missing from both inputs")
    return A, B, O


def _label_roles(A: Sequence[str], B: Sequence[str], O: Sequence[str]):
    setA, setB, setO = set(A), set(B), set(O)
    # Reject labels in only one operand and not in output (this generator does
    # not support "sum out an operand-only label").
    bad = sorted([c for c in (setA ^ setB) if c not in setO])
    if bad:
        raise ValueError(
            f"einsum labels {bad} appear in only one operand and not in the output; "
            "this generator only supports batch, contracted, free_A, and free_B "
            "roles (free labels must appear in the output)."
        )
    batch = [c for c in O if c in setA and c in setB]
    contracted = sorted((setA & setB) - setO)
    free_A = [c for c in O if c in setA and c not in setB]
    free_B = [c for c in O if c in setB and c not in setA]
    return batch, free_A, free_B, contracted


def _select_nnz_blocks(rng: random.Random, dims: List[int],
                       density: float, min_nnz: int) -> List[Tuple[int, ...]]:
    total = 1
    for d in dims:
        total *= d
    if total == 0:
        return []
    nnz = max(min_nnz, int(-(-int(density * total) // 1)))  # ceil
    # ceil with floats robust:
    nnz = max(min_nnz, int(density * total + 0.9999999))
    nnz = min(nnz, total)

    # Avoid materializing the full Cartesian product for huge grids.
    # rng.sample over an integer range is O(nnz) memory, then decode.
    if total <= 1_000_000:
        # Faithful to the C++ version (shuffle all, take first nnz).
        coords = []
        # iterative Cartesian product in C++ order: first axis slowest
        # (matches build_tensor's loop d=0..ndim-1)
        coords = [()]
        for d in dims:
            coords = [c + (s,) for c in coords for s in range(d)]
        rng.shuffle(coords)
        chosen = coords[:nnz]
    else:
        flat = rng.sample(range(total), nnz)
        chosen = []
        for idx in flat:
            coord = [0] * len(dims)
            for axis in range(len(dims) - 1, -1, -1):
                coord[axis] = idx % dims[axis]
                idx //= dims[axis]
            chosen.append(tuple(coord))
    chosen.sort()
    return chosen


def generate(p: Params) -> Problem:
    rng = random.Random(p.seed)
    p_local = Params(**{k: getattr(p, k) for k in p.__dataclass_fields__})
    _apply_partition_size_variance(p_local)

    # ---- determine labels + roles --------------------------------------
    if p_local.einsum is not None:
        A_labels, B_labels, output_labels = _parse_einsum(p_local.einsum)
        batch, free_A, free_B, contracted = _label_roles(A_labels, B_labels, output_labels)
    else:
        pool = "abcdefghijklmnopqrstuvwxyz"
        total = (p_local.num_batch + p_local.num_free_A
                 + p_local.num_free_B + p_local.num_contracted)
        if total > 26:
            raise ValueError("too many labels (max 26)")
        i = 0
        batch = list(pool[i:i + p_local.num_batch]); i += p_local.num_batch
        free_A = list(pool[i:i + p_local.num_free_A]); i += p_local.num_free_A
        free_B = list(pool[i:i + p_local.num_free_B]); i += p_local.num_free_B
        contracted = list(pool[i:i + p_local.num_contracted]); i += p_local.num_contracted
        A_labels = batch + free_A + contracted
        B_labels = batch + contracted + free_B
        output_labels = batch + free_A + free_B

    # ---- partition layout per label ------------------------------------
    npartitions_map: Dict[str, int] = {}
    partition_sizes_map: Dict[str, List[int]] = {}

    def _np_for(label: str) -> int:
        if label in p_local.label_npartitions:
            return p_local.label_npartitions[label]
        return rng.randint(p_local.min_dim_npartitions, p_local.max_dim_npartitions)

    def _sizes_for(label: str, lo_override: int, hi_override: int) -> List[int]:
        n = npartitions_map[label]
        if label in p_local.label_partition_size:
            return [p_local.label_partition_size[label]] * n
        lo = lo_override if lo_override > 0 else p_local.min_partition_size
        hi = hi_override if hi_override > 0 else p_local.max_partition_size
        if lo > hi:
            lo, hi = hi, lo
        return _gen_partition_sizes(rng, n, lo, hi,
                                    p_local.distribution,
                                    p_local.uniform_partition_sizes)

    for c in batch:
        npartitions_map[c] = _np_for(c)
        partition_sizes_map[c] = _sizes_for(c, 0, 0)
    for c in contracted:
        npartitions_map[c] = _np_for(c)
        partition_sizes_map[c] = _sizes_for(c, p_local.contracted_lo, p_local.contracted_hi)
    for c in free_A:
        npartitions_map[c] = _np_for(c)
        partition_sizes_map[c] = _sizes_for(c, p_local.small_free_A_lo, p_local.small_free_A_hi)
    for c in free_B:
        npartitions_map[c] = _np_for(c)
        partition_sizes_map[c] = _sizes_for(c, p_local.small_free_B_lo, p_local.small_free_B_hi)

    # ---- label storage order -------------------------------------------
    if p_local.einsum is None and p_local.shuffle_labels:
        rng.shuffle(A_labels)
        rng.shuffle(B_labels)

    # ---- nnz block selection (separate sub-seeds per operand) ----------
    seed_A = rng.randint(0, 2**63 - 1)
    seed_B = rng.randint(0, 2**63 - 1)

    def _build(name: str, labels: List[str], sub_seed: int) -> Tensor:
        dims = [npartitions_map[c] for c in labels]
        sub_rng = random.Random(sub_seed)
        nnz = _select_nnz_blocks(sub_rng, dims, p_local.density, p_local.min_nnz)
        return Tensor(
            name=name,
            labels=labels,
            num_blocks=dims,
            partition_sizes={c: partition_sizes_map[c] for c in labels},
            nnz_blocks=nnz,
        )

    A = _build("A", A_labels, seed_A)
    B = _build("B", B_labels, seed_B)

    return Problem(A=A, B=B, output_labels=output_labels,
                   partition_sizes_map=partition_sizes_map,
                   npartitions_map=npartitions_map,
                   params=p_local)


# ---------------------------------------------------------------------------
# TensorSuite bundle emission
# ---------------------------------------------------------------------------

def _block_volume(t: Tensor, coord: Tuple[int, ...]) -> int:
    vol = 1
    for axis, sec_idx in enumerate(coord):
        vol *= t.partition_sizes[t.labels[axis]][sec_idx]
    return vol


def _tensor_stats(t: Tensor) -> Tuple[List[int], int, int]:
    dims = [sum(t.partition_sizes[c]) for c in t.labels]
    nnz_block = len(t.nnz_blocks)
    nnz = sum(_block_volume(t, coord) for coord in t.nnz_blocks)
    return dims, nnz_block, nnz


def _metadata(t: Tensor, bundle_name: str, problem_id: str, group: str,
              source: str, source_url: str, date: str,
              dims: List[int], nnz: int, nnz_block: int) -> dict:
    return {
        "version": "0.1",
        "name": bundle_name,
        "group": group,
        "id": problem_id,
        "time": date,
        "source_type": "synthetic",
        "source": source,
        "source_url": source_url,
        "value_type": "float64",
        "value_domain": "general",
        "values_provided": False,
        "index_type": "uint64",
        "endianness": "none",
        "index_base": 0,
        "sorted": "lexicographic",
        "sort_order": list(range(1, len(t.labels) + 1)),
        "duplicates": "disallowed",
        "explicit_zeros": "disallowed",
        "pattern_symmetry": "no",
        "numerical_symmetry": "no",
        "sparsity_type": "block",
        "dense_modes": [],
        "order": len(t.labels),
        "dimensions": dims,
        "nnz": nnz,
        "block_partitions": list(t.num_blocks),
        "nnz_block": nnz_block,
        "files": {
            "text": f"{bundle_name}.tns",
            "readme": "README.md",
        },
    }


def _write_tns(path: Path, t: Tensor, bundle_name: str,
               dims: List[int], nnz: int, nnz_block: int) -> None:
    with path.open("w") as f:
        f.write("%%TensorSuite-TNS\n")
        f.write("% version: 0.1\n")
        f.write(f"% name: {bundle_name}\n")
        # size line: <order> <dim1> ... <nnz>
        f.write(f"{len(t.labels)} " + " ".join(str(d) for d in dims) + f" {nnz}\n")
        # partition-count line: <nblocks1> ... <nnz_block>
        f.write(" ".join(str(n) for n in t.num_blocks) + f" {nnz_block}\n")
        # partition-size arrays (one per mode, in storage label order)
        for c in t.labels:
            f.write(" ".join(str(s) for s in t.partition_sizes[c]) + "\n")
        # nonzero block coords (already sorted lex, 0-based)
        for coord in t.nnz_blocks:
            f.write(" ".join(str(x) for x in coord) + "\n")


def _write_readme(path: Path, t: Tensor, companion: str, problem: Problem,
                  dims: List[int], nnz: int, nnz_block: int,
                  cli_argv: List[str]) -> None:
    labels_str = "".join(t.labels)
    p = problem.params
    common = {
        "seed": p.seed,
        "mode": "einsum" if p.einsum is not None else "count",
        "min_dim_npartitions": p.min_dim_npartitions,
        "max_dim_npartitions": p.max_dim_npartitions,
        "min_partition_size": p.min_partition_size,
        "max_partition_size": p.max_partition_size,
        "uniform_partition_sizes": p.uniform_partition_sizes,
        "distribution": p.distribution.value,
        "density": p.density,
        "min_nnz": p.min_nnz,
        "partition_size_variance": p.partition_size_variance,
    }
    if p.einsum is not None:
        params_dict = {
            **common,
            "einsum": p.einsum,
            "label_partition_size": p.label_partition_size,
            "label_npartitions": p.label_npartitions,
        }
    else:
        params_dict = {
            **common,
            "num_free_A": p.num_free_A,
            "num_free_B": p.num_free_B,
            "num_contracted": p.num_contracted,
            "num_batch": p.num_batch,
            "shuffle_labels": p.shuffle_labels,
            "free_A_partition_size": p.small_free_A_lo,
            "free_B_partition_size": p.small_free_B_lo,
            "contracted_partition_size": p.contracted_lo,
        }
    with path.open("w") as f:
        f.write(f"# {path.parent.name}\n\n")
        f.write("Synthetic block-sparse tensor generated by "
                "`scripts/generate_blocksparse.py` "
                "(port of `include/general_sparse/random_contraction.hpp`).\n\n")
        f.write("## Role in the contraction\n\n")
        f.write(f"- Einsum: `{problem.einsum}`\n")
        f.write(f"- This bundle is operand **{t.name}** (labels `{labels_str}`).\n")
        f.write(f"- Companion operand: **{companion}**\n\n")
        f.write("## Tensor\n\n")
        f.write(f"- Order: {len(t.labels)}\n")
        f.write(f"- Dimensions: {dims}\n")
        f.write(f"- Partitions per dim: {list(t.num_blocks)}\n")
        f.write(f"- Non-zero blocks: {nnz_block}\n")
        f.write(f"- Non-zero elements: {nnz}\n\n")
        f.write("## Values\n\n")
        f.write("Values are **not** stored (`values_provided: false`). Each non-zero "
                "block is dense; users fill values under the metadata constraints "
                "(`value_type: float64`, `value_domain: general`). Block "
                "coordinates are 0-based and sorted lexicographically.\n\n")
        f.write("## Generation parameters\n\n")
        f.write("```json\n")
        f.write(json.dumps(params_dict, indent=2))
        f.write("\n```\n\n")
        f.write("## Reproduce\n\n")
        f.write("```\n")
        f.write("python3 " + " ".join(cli_argv) + "\n")
        f.write("```\n")


def emit_bundle(out_root: Path, bundle_name: str, t: Tensor, companion: str,
                problem: Problem, problem_id: str, group: str,
                source: str, source_url: str, date: str,
                cli_argv: List[str]) -> None:
    bundle_dir = out_root / bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=True)
    dims, nnz_block, nnz = _tensor_stats(t)

    md = _metadata(t, bundle_name, problem_id, group, source, source_url,
                   date, dims, nnz, nnz_block)
    with (bundle_dir / f"{bundle_name}_metadata.json").open("w") as f:
        json.dump(md, f, indent=2)
        f.write("\n")

    _write_tns(bundle_dir / f"{bundle_name}.tns",
               t, bundle_name, dims, nnz, nnz_block)

    _write_readme(bundle_dir / "README.md",
                  t, companion, problem, dims, nnz, nnz_block, cli_argv)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_label_map(spec: Optional[str]) -> Dict[str, int]:
    """Parse 'a=16,b=8,c=4' → {'a':16,'b':8,'c':4}."""
    if not spec:
        return {}
    out: Dict[str, int] = {}
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise ValueError(f"bad label spec '{piece}'; expected LABEL=N")
        k, v = piece.split("=", 1)
        k = k.strip()
        if len(k) != 1:
            raise ValueError(f"label must be a single character, got '{k}'")
        out[k] = int(v)
    return out


def _today() -> str:
    import datetime as _dt
    return _dt.date.today().isoformat()


def build_argparser() -> argparse.ArgumentParser:
    P = argparse.ArgumentParser(
        prog="generate_blocksparse.py",
        description="Generate block-sparse tensor pairs in TensorSuite format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode
    g = P.add_argument_group("mode (mutually exclusive)")
    g.add_argument("--einsum", type=str, default=None,
                   help="Einsum-driven: 'abc,bcd->ad' (skips count-based label assignment)")
    g.add_argument("--label-partition-size", type=str, default=None,
                   help="Per-label partition size, einsum mode (e.g. 'a=16,b=8')")
    g.add_argument("--label-npartitions", type=str, default=None,
                   help="Per-label number of partitions, einsum mode (e.g. 'a=10,b=20')")

    # Count-based rank composition (mirrors fuzz_pipeline)
    g = P.add_argument_group("rank composition (count-driven mode)")
    g.add_argument("--num-free-a", type=int, default=1)
    g.add_argument("--num-free-b", type=int, default=1)
    g.add_argument("--num-contracted", type=int, default=1)
    g.add_argument("--num-batch", type=int, default=0)

    # Partitions per dim (== nblocks_i in the .tns file)
    g = P.add_argument_group("partitions per dim")
    g.add_argument("--min-dim-npartitions", type=int, default=1,
                   help="Minimum number of partitions per dim (== nblocks_i)")
    g.add_argument("--max-dim-npartitions", type=int, default=5,
                   help="Maximum number of partitions per dim")

    # Partition sizes
    g = P.add_argument_group("partition sizes")
    g.add_argument("--min-partition-size", type=int, default=1)
    g.add_argument("--max-partition-size", type=int, default=64)
    g.add_argument("--partition-size", type=int, default=0,
                   help="Pin all partition sizes to this value (overrides min/max)")
    g.add_argument("--distribution",
                   choices=[d.value for d in Distribution],
                   default=Distribution.RANDOM.value)
    g.add_argument("--uniform-partition-sizes", action="store_true",
                   help="Use the same partition size for all partitions of a dim")
    g.add_argument("--partition-size-variance", type=float, default=0.0,
                   help="Expand min/max partition-size range by ±F around midpoint")
    g.add_argument("--free-a-partition-size", type=int, default=0,
                   help="Pin free_A partition sizes")
    g.add_argument("--free-b-partition-size", type=int, default=0,
                   help="Pin free_B partition sizes")
    g.add_argument("--contracted-partition-size", type=int, default=0,
                   help="Pin contracted partition sizes")

    # Sparsity / layout
    g = P.add_argument_group("sparsity / layout")
    g.add_argument("--density", type=float, default=0.3)
    g.add_argument("--min-nnz", type=int, default=1)
    g.add_argument("--no-shuffle", action="store_true",
                   help="Disable label shuffling (count-driven mode)")

    # Reproducibility
    g = P.add_argument_group("rng")
    g.add_argument("--seed", type=int, default=42)

    # Output
    g = P.add_argument_group("output")
    g.add_argument("--out", type=Path, default=Path("."),
                   help="Output root directory")
    g.add_argument("--name", type=str, default=None,
                   help="Base bundle name (default: 'synthetic_<seed>')")
    g.add_argument("--group", type=str, default="synthetic")
    g.add_argument("--id", dest="problem_id", type=str, default=None,
                   help="4-digit problem id (default: derived from seed)")
    g.add_argument("--start-id", type=int, default=None,
                   help="Numeric start id when --count > 1")
    g.add_argument("--count", type=int, default=1,
                   help="Emit N pairs (seed+i, id+i each)")
    g.add_argument("--source-note", type=str, default="",
                   help="Extra free-form text appended to the metadata 'source' field")

    return P


def _params_from_args(a: argparse.Namespace) -> Params:
    p = Params(
        num_free_A=a.num_free_a,
        num_free_B=a.num_free_b,
        num_contracted=a.num_contracted,
        num_batch=a.num_batch,
        min_dim_npartitions=a.min_dim_npartitions,
        max_dim_npartitions=a.max_dim_npartitions,
        min_partition_size=a.min_partition_size,
        max_partition_size=a.max_partition_size,
        uniform_partition_sizes=a.uniform_partition_sizes,
        distribution=Distribution(a.distribution),
        density=a.density,
        min_nnz=a.min_nnz,
        shuffle_labels=not a.no_shuffle,
        partition_size_variance=a.partition_size_variance,
        small_free_A_lo=a.free_a_partition_size, small_free_A_hi=a.free_a_partition_size,
        small_free_B_lo=a.free_b_partition_size, small_free_B_hi=a.free_b_partition_size,
        contracted_lo=a.contracted_partition_size, contracted_hi=a.contracted_partition_size,
        seed=a.seed,
        einsum=a.einsum,
        label_partition_size=_parse_label_map(a.label_partition_size),
        label_npartitions=_parse_label_map(a.label_npartitions),
    )
    if a.partition_size > 0:
        p.min_partition_size = a.partition_size
        p.max_partition_size = a.partition_size
        p.uniform_partition_sizes = True
        # --partition-size is a hard pin; --partition-size-variance would unpin it.
        if a.partition_size_variance > 0:
            print("warning: --partition-size pins all partitions; "
                  "ignoring --partition-size-variance", file=sys.stderr)
            p.partition_size_variance = 0.0
    return p


# Flags that only make sense in count-driven mode.
_COUNT_MODE_FLAGS = (
    "--num-free-a", "--num-free-b", "--num-contracted", "--num-batch",
    "--no-shuffle",
)


def _explicitly_passed(argv: Sequence[str], flag: str) -> bool:
    """Did the user pass `flag` (with or without `=value`) on the CLI?"""
    return any(a == flag or a.startswith(flag + "=") for a in argv)


def _die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _validate_args(args: argparse.Namespace, raw_argv: Sequence[str]) -> None:
    # Mode mutex: --einsum vs count-only flags
    if args.einsum is not None:
        conflicts = [f for f in _COUNT_MODE_FLAGS if _explicitly_passed(raw_argv, f)]
        if conflicts:
            _die(f"--einsum is incompatible with {', '.join(conflicts)} "
                 "(label roles and order come from the einsum string)")
        # Pre-validate the einsum so we never reach generate() with a bad string.
        try:
            A_labels, B_labels, out_labels = _parse_einsum(args.einsum)
            _label_roles(A_labels, B_labels, out_labels)
        except ValueError as e:
            _die(f"--einsum: {e}")
    else:
        if args.label_partition_size or args.label_npartitions:
            _die("--label-partition-size / --label-npartitions require --einsum")

    # Numeric ranges
    if args.min_dim_npartitions < 1 or args.max_dim_npartitions < 1:
        _die("--min-dim-npartitions and --max-dim-npartitions must be >= 1")
    if args.min_dim_npartitions > args.max_dim_npartitions:
        _die(f"--min-dim-npartitions ({args.min_dim_npartitions}) > "
             f"--max-dim-npartitions ({args.max_dim_npartitions})")
    if args.min_partition_size < 1 or args.max_partition_size < 1:
        _die("--min-partition-size and --max-partition-size must be >= 1")
    if args.min_partition_size > args.max_partition_size:
        _die(f"--min-partition-size ({args.min_partition_size}) > "
             f"--max-partition-size ({args.max_partition_size})")
    if args.partition_size < 0:
        _die("--partition-size must be >= 0 (0 = use min/max range)")
    if not (0.0 < args.density <= 1.0):
        _die(f"--density must be in (0, 1] (got {args.density})")
    if args.min_nnz < 1:
        _die(f"--min-nnz must be >= 1 (got {args.min_nnz})")
    if args.partition_size_variance < 0:
        _die("--partition-size-variance must be >= 0")
    for name, val in (
        ("--num-free-a", args.num_free_a),
        ("--num-free-b", args.num_free_b),
        ("--num-contracted", args.num_contracted),
    ):
        if val < 1:
            _die(f"{name} must be >= 1 (got {val})")
    if args.num_batch < 0:
        _die(f"--num-batch must be >= 0 (got {args.num_batch})")
    if args.count < 1:
        _die(f"--count must be >= 1 (got {args.count})")
    if args.problem_id is not None:
        if not (args.problem_id.isdigit() and len(args.problem_id) <= 4):
            _die(f"--id must be 1-4 digits (got '{args.problem_id}')")

    parsed_size = _parse_label_map_safe(args.label_partition_size, "--label-partition-size")
    parsed_nsec = _parse_label_map_safe(args.label_npartitions, "--label-npartitions")
    for label, v in parsed_size.items():
        if v < 1:
            _die(f"--label-partition-size: '{label}={v}' must be >= 1")
    for label, v in parsed_nsec.items():
        if v < 1:
            _die(f"--label-npartitions: '{label}={v}' must be >= 1")


def _parse_label_map_safe(spec: Optional[str], flag: str) -> Dict[str, int]:
    try:
        return _parse_label_map(spec)
    except ValueError as e:
        _die(f"{flag}: {e}")
        return {}  # unreachable


def main(argv: Optional[List[str]] = None) -> int:
    raw_argv = argv if argv is not None else sys.argv[1:]
    P = build_argparser()
    args = P.parse_args(raw_argv)
    _validate_args(args, raw_argv)
    cli_argv = ["scripts/generate_blocksparse.py"] + list(raw_argv)

    base_name = args.name or f"synthetic_{args.seed}"
    if args.start_id is not None:
        start_id = args.start_id
    elif args.problem_id is not None:
        start_id = int(args.problem_id)
    else:
        start_id = args.seed % 10000

    date = _today()
    source = "Generated by scripts/generate_blocksparse.py"
    if args.source_note:
        source += f" — {args.source_note}"

    for i in range(args.count):
        params = _params_from_args(args)
        params.seed = args.seed + i
        problem = generate(params)

        suffix = "" if args.count == 1 else f"_{i:03d}"
        name = f"{base_name}{suffix}"
        pid = f"{(start_id + i) % 10000:04d}"

        bn_A = f"{name}_A"
        bn_B = f"{name}_B"
        emit_bundle(args.out, bn_A, problem.A, bn_B, problem, pid,
                    args.group, source, "", date, cli_argv)
        emit_bundle(args.out, bn_B, problem.B, bn_A, problem, pid,
                    args.group, source, "", date, cli_argv)

        dimsA, nbA, nzA = _tensor_stats(problem.A)
        dimsB, nbB, nzB = _tensor_stats(problem.B)
        print(f"[{i + 1}/{args.count}] {name}  einsum={problem.einsum}  "
              f"A={dimsA} nblk={nbA} nnz={nzA}  "
              f"B={dimsB} nblk={nbB} nnz={nzB}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
