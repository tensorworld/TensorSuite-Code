# KronWeave

[English](README.md) | [中文](README.zh-CN.md)

KronWeave 是一个基于 CP seed 和 Kronecker expansion 的 power-law sparse tensor generator。它可以生成兼容 TensorSuite 的稀疏张量 bundle，包括文本格式 `.tns`、二进制格式 `.tnsb`、metadata、配置快照和 bundle README。

## 功能

- 基于 CPD seed 的 Kronecker 稀疏张量生成。
- Zipf baseline 坐标生成。
- TensorSuite `.tns` 和 `.tnsb` 输出。
- YAML 配置校验。
- Python API 和命令行接口。
- 重复坐标后处理，支持 `keep`、`sum`、`allow`。
- 分析工具，支持 mode-wise degree 图和 empirical power-law fitting。
- 只包含 tiny 级别示例配置和 seed，方便快速测试。

## 项目结构

```text
Code/
  src/kronweave/       Python 包源码
  cpp/                 OpenMP fastSKG 后端源码
  configs/             小规模可运行 YAML 示例
  docs/                使用说明
  seeds/               少量 tiny seed tensor
  tests/               轻量测试
  example.py           最小 Python API 示例
```


## 环境要求

- Python 3.10+
- `numpy`
- `PyYAML`
- 支持 OpenMP 的 C++17 编译器，例如 `g++`
- `make`
- 可选分析依赖：`powerlaw` 和 `matplotlib`

第一次运行 fastSKG generator 时会自动编译 C++ 后端。也可以手动编译：

```bash
cd cpp
make
```

## 安装

在 `Code/` 目录下执行：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

如果需要运行分析工具：

```bash
pip install -e ".[analysis]"
```

## 快速使用

运行 Python 示例：

```bash
python example.py
```

输出会写到 `generated/cp_fastskg_tiny/`。

使用命令行：

```bash
kronweave validate-config --config configs/cp_fastskg_tiny.yaml
kronweave generate --config configs/cp_fastskg_tiny.yaml
kronweave validate-bundle --bundle generated/cp_fastskg_tiny
```

运行 Zipf baseline：

```bash
kronweave generate --config configs/zipf_tiny.yaml
```

使用 tiny input seed 运行 fastSKG：

```bash
kronweave generate --config configs/fastskg_input_seed_tiny.yaml
```

## TensorSuite Bundle

KronWeave 会写出 TensorSuite 风格的 tensor bundle。TensorSuite 格式说明见 `../TensorSuite/format.html`。当前版本的 KronWeave 写出的是 element-wise coordinate sparse tensor，不是 block-sparse tensor。

一个生成后的 bundle 是一个 tensor 对应一个文件夹：

```text
<tensor-name>/
  <tensor-name>_metadata.json
  <tensor-name>.tns
  <tensor-name>.tnsb
  README.md
  config.yaml
  optional cp_seed/
```

- `<tensor-name>_metadata.json`：canonical metadata，记录 identity、dimensions、value/index types、sorting、duplicate policy 和 file mappings。
- `<tensor-name>.tns`：文本坐标数据。文件以 `%%TensorSuite-TNS`、`% version`、`% name` 开头，然后是 size line：`<order> <dim0> ... <dimD-1> <nnz>`，后面每行是 coordinate/value record。
- `<tensor-name>.tnsb`：二进制坐标数据。它有和 `.tns` 对齐的 minimal cross-check header，后面是 binary coordinate/value records，使用配置中的 `index_type`、`value_type` 和 `endianness`。
- `README.md`：记录不适合放入 canonical metadata 的辅助信息，例如 duplicate stats 和 generation provenance。
- `config.yaml`：复制保存的 YAML 配置，用于复现。
- `cp_seed/`：当 `generator.type=cp_fastskg` 时写出的可选 CP seed bundle。

生成后会对 `.tns`、`.tnsb` header 和 metadata 做 cross-check。由于 stochastic generation 是 with replacement sampling，最终 metadata 里的 `nnz` 是 duplicate handling 和 zero filtering 后的条目数，可能小于配置中的 raw sample count。

## 分析和后处理

KronWeave 会在写出 bundle 前对 raw sampled coordinates 做后处理：

- `duplicates.policy=keep`：重复坐标只保留一条。
- `duplicates.policy=sum`：重复坐标的 value 求和。
- `duplicates.policy=allow`：保留 raw repeated samples。
- `value`：在重复处理后生成 constant、uniform 或 clipped normal values。

分析工具和 canonical metadata 是分开的：

```bash
python -m kronweave.analysis.visualizer --tns generated/cp_fastskg_tiny/cp_fastskg_tiny.tns --out-dir plots/cp_fastskg_tiny
python -m kronweave.analysis.power_law_analysis --tns generated/cp_fastskg_tiny/cp_fastskg_tiny.tns --out reports/cp_fastskg_tiny_powerlaw.json --verbose
```

`power_law_analysis.py` 需要可选依赖 `powerlaw`。KronWeave 引用其上游实现 [powerlaw-devs/powerlaw](https://github.com/powerlaw-devs/powerlaw)，该工具对应 Clauset、Shalizi 和 Newman 的论文 "Power-Law Distributions in Empirical Data", SIAM Review, 2009。

## Python API

```python
from kronweave import generate_from_config
from kronweave.api import validate_bundle

result = generate_from_config("configs/cp_fastskg_tiny.yaml")
validate_bundle(result["bundle_dir"])

print(result["tns"])
print(result["tnsb"])
print(result["metadata"])
```

`generate_from_config` 会返回 bundle 目录、`.tns`、`.tnsb`、metadata、bundle README 和配置快照的路径。

## 配置说明

KronWeave 使用 YAML 配置。一个完整的 CP-fastSKG 示例是：

```yaml
name: cp_fastskg_tiny
output_dir: generated

generator:
  type: cp_fastskg
  random_seed: 42

tensor:
  sizes: [16, 16, 16]
  index_base: 0
  nnz: 200

format:
  storage: coordinate
  index_type: uint64
  endianness: little
  sorted: lexicographic
  sorted_order: [0, 1, 2]
  explicit_zeros: disallowed
  pattern_symmetry: no
  numerical_symmetry: no
  sparsity_type: element
  dense_modes: []

seed:
  source: cpd
  cpd:
    sizes: [4, 4, 4]
    rank: 2
    alpha: [1.4, 1.4, 1.4]
    noise: 0.0
    lambda_controller: softmax

fastskg:
  iter: 2

value:
  type: float64
  mode: constant
  constant: 1.0

duplicates:
  policy: keep

output:
  write_tns: true
  write_tnsb: true
  write_metadata: true
  write_readme: true
  copy_config: true
```

顶层字段：

- `name`：tensor 和 bundle 名称，输出文件会使用这个前缀。
- `output_dir`：生成 bundle 的父目录。相对路径会按当前工作目录解析。
- `experiment.name`：为了兼容旧配置保留的 `name` 替代写法。

`generator`：

- `type`：可选 `cp_fastskg`、`fastskg` 或 `zipf`。
- `random_seed`：Python 侧 CP seed、Zipf 坐标和 value generation 使用的随机种子。C++ fastSKG sampler 仍可能使用后端随机性。

`tensor`：

- `sizes`：最终 tensor 维度。对于 fastSKG generator，它必须等于 seed dimensions 的 `fastskg.iter` 次方。
- `index_base`：写入 TensorSuite 文件的坐标基准，只能是 `0` 或 `1`。
- `nnz`：请求的 raw sample count。重复坐标处理后，最终 metadata 里的 `nnz` 可能更小。

`format`：

- `storage`：当前为 `coordinate`。
- `index_type`：`.tnsb` 中坐标的存储类型，`uint32` 或 `uint64`。
- `endianness`：`.tnsb` binary records 的字节序，`little` 或 `big`。
- `sorted`：`lexicographic` 或 `none`。
- `sorted_order`：当 `sorted=lexicographic` 时使用的 0-based mode permutation，例如 `[0, 1, 2]`。
- `explicit_zeros`：`allowed` 或 `disallowed`。如果是 `disallowed`，写出前会删除零值条目。
- `pattern_symmetry`：当前为 `no`。
- `numerical_symmetry`：当前为 `no`。
- `sparsity_type`：当前为 `element`。
- `dense_modes`：当前为 `[]`。
- `value_domain`：可选；如果写了，必须和 `value` 推断出的 value domain 一致。

Kronecker generator 的 `seed`：

- `source`：可选 `cpd`、`input_tns` 或 `kp_from_meta`。
- `source=cpd`：需要 `seed.cpd`，用于 `generator.type=cp_fastskg`。
- `source=input_tns`：直接使用已有 TensorSuite `.tns` seed，用于 `generator.type=fastskg`。
- `source=kp_from_meta`：先把一个小 TensorSuite meta-seed 做 Kronecker-product expansion，再交给 fastSKG。
- `input_tns`：`input_tns` 或 `kp_from_meta` 使用的 seed 路径。
- `sizes`：可选的 input seed 维度检查。
- `expand_iter`：`kp_from_meta` 必填。

`seed.cpd`：

- `sizes`：CP initiator 维度。
- `rank`：CP rank。
- `alpha`：一个所有 mode 共享的 exponent，或每个 mode 一个 exponent。
- `noise`：用于 sensitivity experiments 的 multiplicative perturbation level。
- `lambda_controller`：`softmax` 表示 exponential decayed component weights；`none` 表示 equal unnormalized weights。

`fastskg`：

- `iter`：Kronecker expansion depth。最终维度是 `seed_size ** iter`。
- `nnz`：可选 raw sample-count override。若省略，优先使用 `tensor.nnz`；如果也没有，则 fastSKG 使用 `round(W ** iter)`，其中 `W` 是 seed weight sum。

`zipf`：

- `alpha`：一个所有 mode 共享的 Zipf exponent，或每个 mode 一个 exponent。值必须大于 `1.0`。

`value`：

- `type`：`float32`、`float64`、`int32` 或 `int64`。
- `mode`：`constant`、`uniform` 或 `normal`。
- `constant`：`mode=constant` 时使用的值。
- `low`、`high`：`mode=uniform` 时使用的范围。
- `mean`、`std`、`clip`：`mode=normal` 时使用的 clipped normal 参数。

`duplicates`：

- `policy=keep`：重复坐标只保留一条。
- `policy=sum`：重复坐标的 value 求和。
- `policy=allow`：保留 repeated raw samples。

`output`：

- `write_tns`：写出 TensorSuite text tensor file。
- `write_tnsb`：写出 TensorSuite binary tensor file。
- `write_metadata`：写出 canonical JSON metadata。
- `write_readme`：写出 per-bundle README。
- `copy_config`：把 YAML config 复制进生成的 bundle。

可运行的 tiny 示例见 `configs/`。

## 测试

```bash
pytest
```

测试只会生成临时的小规模 tensor。

## License

MIT License. See `LICENSE`.
