# generate_blocksparse.py

Generates synthetic block-sparse tensor **pairs** in the
[TensorSuite](https://github.com/tensorworld/TensorSuite) bundle format. Port
of `include/general_sparse/random_contraction.hpp` (the same logic
`fuzz_pipeline` uses), with an added einsum-driven mode and per-label
overrides.

CLI vocabulary mirrors the `.tns` block file layout:

| TNS file line                       | meaning                       | CLI knob                                                                                       |
|-------------------------------------|-------------------------------|------------------------------------------------------------------------------------------------|
| `order dim1…dimN nnz`               | total size per dim (Σ parts)  | derived — equals `dim_npartitions × partition_size`                                            |
| `nblocks1…nblocksN nnz_block`       | partitions per dim            | `--min-dim-npartitions` / `--max-dim-npartitions`, `--label-npartitions`                       |
| per-axis partition-size array       | size of each partition        | `--min-partition-size` / `--max-partition-size`, `--partition-size`, `--label-partition-size`  |

Each invocation produces two sibling bundles, `<name>_A/` and `<name>_B/`, both
operands of one contraction. Block pattern only (`values_provided: false`);
fp64 / uint64 / 0-based indexing; no `.tnsb`.

## Modes

### Einsum-driven — `--einsum`

Layout, label set, and label order come from the string.

```bash
python3 scripts/generate_blocksparse.py \
  --einsum "ade,afg->defg" \
  --label-partition-size "a=4,d=12,e=3,f=12,g=3" \
  --label-npartitions    "a=19,d=110,e=20,f=110,g=20" \
  --density 0.02 --seed 7 \
  --name demo --out out/
```

Label roles are derived by intersection:

| role        | rule                          |
|-------------|-------------------------------|
| batch       | in A ∩ B ∩ O                  |
| contracted  | in A ∩ B, not in O            |
| free_A      | in A and O, not in B          |
| free_B      | in B and O, not in A          |

Operand-only labels missing from the output are rejected — this generator does
not support "sum out an A-only or B-only label". Per-label overrides
(`--label-partition-size`, `--label-npartitions`) pin individual labels; any
label left unspecified falls back to the global min/max ranges and
`--distribution`.

### Count-driven

Label counts pick random labels from `a-z`; A/B index order is shuffled by
default. Same code path as `fuzz_pipeline --scale`.

```bash
python3 scripts/generate_blocksparse.py \
  --num-free-a 2 --num-free-b 2 --num-contracted 2 \
  --min-partition-size 8 --max-partition-size 16 \
  --max-dim-npartitions 3 --density 0.35 \
  --seed 42 --name fig1_L --out out/
```

## Flag groups

**Mode (mutually exclusive)** — `--einsum`, `--label-partition-size`,
`--label-npartitions` (label maps require `--einsum`; count-only flags
`--num-*` and `--no-shuffle` are rejected with `--einsum`).

**Count-driven rank** — `--num-free-a`, `--num-free-b`, `--num-contracted`,
`--num-batch`.

**Partitions per dim** — `--min-dim-npartitions`, `--max-dim-npartitions`
(overridable per-label via `--label-npartitions` in einsum mode).

**Partition sizes** — `--min-partition-size`, `--max-partition-size`,
`--partition-size` (hard pin; supersedes `--partition-size-variance`),
`--distribution` (`random` | `uniform` | `mixed` | `power_law`),
`--uniform-partition-sizes`, `--partition-size-variance F` (expand min/max by
±F around midpoint), `--free-a-partition-size`, `--free-b-partition-size`,
`--contracted-partition-size` (per-role bulk pins, count mode).

**Sparsity** — `--density` (fraction of the Cartesian-product block grid that
is nonzero; `nnz = max(min_nnz, ceil(density · total))`), `--min-nnz`,
`--no-shuffle`.

**Output** — `--out DIR`, `--name BASE`, `--group NAME`, `--id NNNN`,
`--start-id N`, `--count N` (emit N pairs with `seed+i`, `id+i`, suffix
`_000`…), `--source-note STR`, `--seed N`.

Run `python3 scripts/generate_blocksparse.py --help` for the full list with
defaults.

## Bundle layout

```
<name>_A/
  <name>_A_metadata.json     # TensorSuite schema, sparsity_type=block
  <name>_A.tns               # text-only block file
  README.md                  # einsum + companion + full param dump
<name>_B/
  <name>_B_metadata.json
  <name>_B.tns
  README.md
```

The `.tns` file follows the TensorSuite block-wise layout: magic line, version
and name comments, size line (`order dim1…dimN nnz`), partition-count line
(`nblocks1…nblocksN nnz_block`), one partition-size array per mode (in storage
label order), then 0-based block coordinates sorted lexicographically.

## Validation

The CLI fails on:

- mode conflicts (`--einsum` with count-only flags; label maps without
  `--einsum`),
- bad numeric ranges (`min > max` on dim-npartitions or partition-size;
  non-positive density; negative counts; non-digit `--id`),
- invalid einsums (missing `->`, repeated labels in an operand, output labels
  not present in any input, free labels that aren't in the output),
- conflicting size pinning — `--partition-size` + `--partition-size-variance`
  warns and keeps the pin.

## Reproducibility

The README of each bundle contains the einsum string, role, companion, tensor
stats, and the full parameter dict in JSON. The exact `python3
scripts/generate_blocksparse.py …` command line is also recorded. Re-running
with the same `--seed` is deterministic.
