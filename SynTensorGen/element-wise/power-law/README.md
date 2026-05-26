# KronWeave

[English](README.md) | [中文](README.zh-CN.md)

KronWeave is a CP-seeded Kronecker generator for power-law sparse tensors. It generates TensorSuite-compatible sparse tensor bundles in text (`.tns`) and binary (`.tnsb`) formats, with metadata, config snapshots, and per-bundle README files.

## Features

- CPD-seeded Kronecker generation for power-law sparse tensors.
- Baseline Zipf coordinate generation.
- TensorSuite `.tns` and `.tnsb` output writers.
- YAML configuration with validation.
- Python API and command-line interface.
- Duplicate-coordinate postprocessing with `keep`, `sum`, and `allow` policies.
- Analysis helpers for mode-wise degree plots and empirical power-law fitting.
- Tiny example configs and seeds for quick smoke tests.

## Project Layout

```text
Code/
  src/kronweave/       Python package
  cpp/                 OpenMP fastSKG backend source
  configs/             Small runnable YAML examples
  docs/                Usage notes
  seeds/               Tiny seed tensors only
  tests/               Lightweight regression tests
  example.py           Minimal Python API example
```

## Requirements

- Python 3.10+
- `numpy`
- `PyYAML`
- A C++17 compiler with OpenMP support, for example `g++`
- `make`
- Optional analysis extras: `powerlaw` and `matplotlib`

The C++ backend is built automatically the first time a fastSKG generator is used. You can also build it manually:

```bash
cd cpp
make
```

## Installation

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Install optional analysis tools with:

```bash
pip install -e ".[analysis]"
```

## Quick Start

Run the bundled Python example:

```bash
python example.py
```

This writes a small generated bundle under `generated/cp_fastskg_tiny/`.

Use the CLI:

```bash
kronweave validate-config --config configs/cp_fastskg_tiny.yaml
kronweave generate --config configs/cp_fastskg_tiny.yaml
kronweave validate-bundle --bundle generated/cp_fastskg_tiny
```

Run the Zipf baseline:

```bash
kronweave generate --config configs/zipf_tiny.yaml
```

Run fastSKG from a tiny input seed:

```bash
kronweave generate --config configs/fastskg_input_seed_tiny.yaml
```

## TensorSuite Bundle

KronWeave writes TensorSuite-style tensor bundles. See `../TensorSuite/format.html` for the TensorSuite format page. In this release, KronWeave writes element-wise coordinate sparse tensors, not block-sparse tensors.

A generated bundle is one folder per tensor:

```text
<tensor-name>/
  <tensor-name>_metadata.json
  <tensor-name>.tns
  <tensor-name>.tnsb
  README.md
  config.yaml
  optional cp_seed/
```

- `<tensor-name>_metadata.json`: canonical metadata for identity, dimensions, storage attributes, value/index types, sorting, duplicate policy, and file mappings.
- `<tensor-name>.tns`: text coordinate data. The file starts with `%%TensorSuite-TNS`, `% version`, `% name`, then a size line `<order> <dim0> ... <dimD-1> <nnz>`, followed by coordinate/value records.
- `<tensor-name>.tnsb`: binary coordinate data. It uses the same minimal cross-check header as `.tns`, followed by binary coordinate/value records using the configured `index_type`, `value_type`, and `endianness`.
- `README.md`: auxiliary generation notes that should not be placed in canonical metadata, such as duplicate statistics or generation provenance.
- `config.yaml`: the copied YAML configuration used to reproduce the bundle.
- `cp_seed/`: optional CP seed bundle written when `generator.type=cp_fastskg`.

The `.tns` and `.tnsb` headers are cross-checked against metadata after generation. Because stochastic generation samples with replacement, the final `nnz` in the metadata is the number of entries after duplicate handling and zero filtering; it can be smaller than the requested raw sample count.

## Analysis and Postprocessing

KronWeave postprocesses raw sampled coordinates before writing a final bundle:

- `duplicates.policy=keep`: keep one entry for each repeated coordinate.
- `duplicates.policy=sum`: sum values for repeated coordinates.
- `duplicates.policy=allow`: preserve raw repeated samples.
- `value`: assign constant, uniform, or clipped normal values after duplicate handling.

The analysis helpers are intentionally separate from canonical metadata:

```bash
python -m kronweave.analysis.visualizer --tns generated/cp_fastskg_tiny/cp_fastskg_tiny.tns --out-dir plots/cp_fastskg_tiny
python -m kronweave.analysis.power_law_analysis --tns generated/cp_fastskg_tiny/cp_fastskg_tiny.tns --out reports/cp_fastskg_tiny_powerlaw.json --verbose
```

`power_law_analysis.py` requires the optional `powerlaw` package. KronWeave cites the upstream implementation at [powerlaw-devs/powerlaw](https://github.com/powerlaw-devs/powerlaw), which implements methods associated with Clauset, Shalizi, and Newman, "Power-Law Distributions in Empirical Data", SIAM Review, 2009.

## Python API

```python
from kronweave import generate_from_config
from kronweave.api import validate_bundle

result = generate_from_config("configs/cp_fastskg_tiny.yaml")
validate_bundle(result["bundle_dir"])

print(result["tns"])
print(result["tnsb"])
print(result["metadata"])
```

`generate_from_config` returns paths to the generated bundle directory, `.tns`, `.tnsb`, metadata, bundle README, and copied config snapshot.

## Configuration

KronWeave is configured with YAML. A complete CP-fastSKG example is:

```yaml
name: cp_fastskg_tiny
output_dir: generated

generator:
  type: cp_fastskg
  random_seed: 42

tensor:
  sizes: [16, 16, 16]
  index_base: 0
  nnz: 200

format:
  storage: coordinate
  index_type: uint64
  endianness: little
  sorted: lexicographic
  sorted_order: [0, 1, 2]
  explicit_zeros: disallowed
  pattern_symmetry: no
  numerical_symmetry: no
  sparsity_type: element
  dense_modes: []

seed:
  source: cpd
  cpd:
    sizes: [4, 4, 4]
    rank: 2
    alpha: [1.4, 1.4, 1.4]
    noise: 0.0
    lambda_controller: softmax

fastskg:
  iter: 2

value:
  type: float64
  mode: constant
  constant: 1.0

duplicates:
  policy: keep

output:
  write_tns: true
  write_tnsb: true
  write_metadata: true
  write_readme: true
  copy_config: true
```

Top-level fields:

- `name`: tensor and bundle name. Output files use this prefix.
- `output_dir`: parent directory for generated bundles. Relative paths are resolved from the current working directory.
- `experiment.name`: backward-compatible alternative to `name` for older configs.

`generator`:

- `type`: one of `cp_fastskg`, `fastskg`, or `zipf`.
- `random_seed`: seed used by Python-side CP seed, Zipf coordinate, and value generation. The C++ fastSKG sampler may still use backend randomness.

`tensor`:

- `sizes`: final tensor dimensions. For fastSKG generators, this must equal seed dimensions raised to `fastskg.iter`.
- `index_base`: coordinate base written to TensorSuite files, either `0` or `1`.
- `nnz`: requested raw sample count. After duplicate handling, the final metadata `nnz` may be smaller.

`format`:

- `storage`: currently `coordinate`.
- `index_type`: `uint32` or `uint64` for `.tnsb` coordinate storage.
- `endianness`: currently `little` for generated TensorSuite bundles.
- `sorted`: `lexicographic` or `none`.
- `sorted_order`: 0-based mode permutation used when `sorted=lexicographic`, for example `[0, 1, 2]`. `sort_order` is accepted as an alias and canonical metadata uses `sort_order`.
- `explicit_zeros`: `allowed` or `disallowed`. When disallowed, zero-valued entries are removed before writing.
- `pattern_symmetry`: currently `no`.
- `numerical_symmetry`: currently `no`.
- `sparsity_type`: currently `element`.
- `dense_modes`: currently `[]`.
- `value_domain`: optional. If provided, it must match the domain inferred from `value`.

`seed` for Kronecker generators:

- `source`: `cpd`, `input_tns`, or `kp_from_meta`.
- `source=cpd`: requires `seed.cpd` and is used with `generator.type=cp_fastskg`.
- `source=input_tns`: uses an existing TensorSuite `.tns` seed and is used with `generator.type=fastskg`.
- `source=kp_from_meta`: expands a small TensorSuite meta-seed into a Kronecker-product seed before fastSKG.
- `input_tns`: path to an input seed for `input_tns` or `kp_from_meta`.
- `sizes`: optional expected dimensions for `input_tns`; used as a validation check.
- `expand_iter`: required for `kp_from_meta`.

`seed.cpd`:

- `sizes`: CP initiator dimensions.
- `rank`: CP rank.
- `alpha`: one exponent shared by all modes, or one exponent per mode.
- `noise`: multiplicative perturbation level for sensitivity experiments.
- `lambda_controller`: `softmax` for exponentially decayed component weights, or `none` for equal unnormalized weights.

`fastskg`:

- `iter`: Kronecker expansion depth. Final dimensions are `seed_size ** iter`.
- `nnz`: optional raw sample-count override. If omitted, `tensor.nnz` is used when present; otherwise fastSKG uses `round(W ** iter)`, where `W` is the seed weight sum.

`zipf`:

- `alpha`: one Zipf exponent shared by all modes, or one exponent per mode. Values must be greater than `1.0`.

`value`:

- `type`: `float32`, `float64`, `int32`, or `int64`.
- `mode`: `constant`, `uniform`, or `normal`.
- `constant`: value used when `mode=constant`.
- `low`, `high`: range used when `mode=uniform`.
- `mean`, `std`, `clip`: parameters for clipped normal values when `mode=normal`.
- `values_provided` or `provide_value`: top-level boolean aliases. When `false`, KronWeave writes a structural pattern-only text `.tns` without explicit values and rejects `.tnsb` output for that config.

`duplicates`:

- `policy=keep`: keep one entry for each repeated coordinate.
- `policy=sum`: sum values for repeated coordinates.
- `policy=allow`: preserve repeated raw samples.

`output`:

- `write_tns`: write the TensorSuite text tensor file.
- `write_tnsb`: write the TensorSuite binary tensor file.
- `write_metadata`: write canonical JSON metadata.
- `write_readme`: write per-bundle README.
- `copy_config`: copy the YAML config into the generated bundle.

See `configs/` for runnable tiny examples.

## Tests

```bash
pytest
```

The tests generate only small temporary tensors.

## License

MIT License. See `LICENSE`.
