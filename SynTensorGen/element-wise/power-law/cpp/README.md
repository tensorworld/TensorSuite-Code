# cpp

OpenMP backend for stochastic Kronecker sampling.

- `fastSKG_omp.cpp`: reads a TensorSuite-TNS initiator, samples Kronecker coordinates, and writes a raw TensorSuite-TNS tensor.
- `Makefile`: builds the `run_fastSKG_omp` executable used by `kronweave.runners.fastskg`.
