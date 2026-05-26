# seed

Seed and baseline coordinate generation helpers.

- `__init__.py`: exposes seed-generation helpers as a subpackage.
- `cpd.py`: builds dense CP-structured Kronecker initiators from sizes, rank, alpha, noise, and component-weight settings.
- `kp_seed.py`: expands a small TensorSuite meta-seed by repeated Kronecker products.
- `meta_seed.py`: writes CP seed bundles in TensorSuite formats for fastSKG input.
- `zipf.py`: samples independent truncated Zipf coordinates for the baseline generator.
