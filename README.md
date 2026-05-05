# TensorSuite

TensorSuite generates synthetic sparse tensors for COO and HiCOO benchmarking.
It supports configurable per-mode density, single or double precision, and a
memory threshold check before generating large tensors.

## Quick start

Generate one tensor:

```bash
python3 SynTensorGen/random_tensor_generator_limited.py output.tns D 2.0 50%1024 2%4096
```

Generate the preset tensor suite:

```bash
bash SynTensorGen/generate_all_tensors_limited.sh
```

The generated `.tns` file starts with the tensor order, then mode dimensions,
then one-based coordinates followed by a Gaussian random value.

## GitHub Pages

The TensorSuite homepage is hosted separately at:

```text
https://tensorworld.github.io/TensorSuite/
```

Website source is kept in:

```text
https://github.com/tensorworld/TensorSuite
```
