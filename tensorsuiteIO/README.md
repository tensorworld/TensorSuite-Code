# tensorsuiteIO

Shared TensorSuite bundle input and output helpers for synthetic tensor
generators.

- `metadata.py`: builds, validates, reads, and writes metadata JSON. The
  default `tensorsuite` validation profile follows the website canonical
  schema for real/synthetic and element/block tensors. The
  `synthetic_element_compact` profile is available only for older compact
  synthetic element-wise metadata.
- `readme_writer.py`: writes the per-bundle human-readable README.
- `tensorsuite.py`: parses and writes TensorSuite-TNS text files, converts to TensorSuite-TNSB binary files, reads binary samples, parses block headers, and cross-checks headers.

`read_tns(path, metadata=None)` is the compatibility loader. For element-wise
tensors it returns `TensorData` with coordinate/value arrays. For block-sparse
tensors it returns `BlockTensorData` with nonzero block coordinates and
partition-size arrays. Pass metadata explicitly, or place
`<tensor-name>_metadata.json` next to `<tensor-name>.tns` so the loader can
dispatch automatically from `sparsity_type`.

`SynTensorGen/element-wise/random` and `SynTensorGen/element-wise/power-law`
both use this shared implementation. Power-law keeps `kronweave.io.*`
compatibility wrappers, but those wrappers re-export the modules from this
directory.
