# seed

Seed 和 baseline 坐标生成工具。

- `__init__.py`：将该目录标记为 seed-generation 子包。
- `cpd.py`：根据 sizes、rank、alpha、noise 和 component-weight 设置生成 dense CP-structured Kronecker initiator。
- `kp_seed.py`：对一个小的 TensorSuite meta-seed 做 repeated Kronecker product 扩展。
- `meta_seed.py`：把 CP seed 写成 TensorSuite 格式的 seed bundle，供 fastSKG 使用。
- `zipf.py`：为 baseline generator 采样 independent truncated Zipf coordinates。
