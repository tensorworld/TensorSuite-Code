# Random Structural Generator

This directory contains a small Python TensorSuite-style sparse tensor generator.
It targets a uniform random mode-wise row-degree model for `Nd~d`: for each
mode, row degrees are sampled from a uniform integer range and then rebalanced
so the degree sum is exactly `tensor.nnz`.

TensorSuite file writing, metadata validation, bundle README generation, and
header cross-checks are provided by the shared repository-level
`tensorsuiteIO/` helpers.

## Usage

```bash
python random_generator.py validate-config --config configs/random_tiny.yaml
python random_generator.py generate --config configs/random_tiny.yaml
python random_generator.py validate-bundle --bundle generated/random_tiny
python example.py
```

The generated bundle contains:

- `random_tiny.tns`
- `random_tiny.tnsb`
- `random_tiny_metadata.json`
- `README.md`
- `config.yaml`

## YAML Notes

- `provide_value` and `values_provided` are boolean aliases.
- Set `provide_value: false` for structural-only output. In that case set
  `output.write_tnsb: false`, because the binary writer stores values.
- `duplicates.policy: allow` preserves the sampled mode degree sequences most
  directly. `keep` and `sum` can change the final degree distribution because
  they remove duplicate coordinates.
- `random.low` and `random.high` can be scalars or per-mode lists. Defaults are
  `0` and `2 * tensor.nnz / tensor.sizes[i]`, so the uniform degree range is
  centered on the required mean degree.
