# random_tiny

Synthetic TensorSuite-style sparse tensor bundle.

- Structural distribution: `random`
- Order: `3`
- Dimensions: `[64, 64, 64]`
- Nonzeros: `4096`
- Values provided: `true`
- Text file: `random_tiny.tns`
- Binary file: `random_tiny.tnsb`
- Duplicate policy: `allowed`

For each mode-i unfolding, the row degree samples are generated from the `random` structural model before duplicate postprocessing.

Duplicate stats:

```json
{
  "raw_samples": 4096,
  "unique_nnz": null,
  "duplicate_count": null
}
```
