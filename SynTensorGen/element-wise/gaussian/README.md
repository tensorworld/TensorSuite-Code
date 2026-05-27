# Random Structural Generator

This directory contains a small Python TensorSuite-style sparse tensor generator.
It targets a random row-degree pattern for each mode-i unfolding by drawing
random per-row weights for every mode and converting them to integer row
degrees.

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
