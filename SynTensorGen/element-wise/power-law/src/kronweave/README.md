# kronweave

Top-level Python package.

- `__init__.py`: exposes `generate_from_config` and the package version.
- `api.py`: orchestrates config loading, seed generation, fastSKG execution, postprocessing, output writing, and bundle validation.
- `cli.py`: defines the `kronweave` command-line interface.
- `config.py`: loads and validates YAML configuration files.
- `analysis/`: degree plotting and empirical power-law analysis helpers.
- `io/`: compatibility wrappers for the shared repository-level `tensorsuiteIO/` helpers.
- `postprocess/`: duplicate-coordinate and value-generation helpers.
- `runners/`: wrappers around external generation backends.
- `seed/`: CPD, Kronecker-product, and Zipf seed/baseline generation.
- `utils/`: reserved for shared utilities.
