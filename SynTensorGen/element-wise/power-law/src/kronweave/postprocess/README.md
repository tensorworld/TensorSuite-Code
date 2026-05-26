# postprocess

Helpers applied after raw coordinate sampling.

- `__init__.py`: marks this directory as the postprocessing subpackage.
- `duplicates.py`: validates coordinates, handles duplicate policies (`keep`, `sum`, `allow`), and computes duplicate/collision statistics.
- `values.py`: normalizes value configuration, infers the value domain, and generates constant, uniform, or clipped normal values.
