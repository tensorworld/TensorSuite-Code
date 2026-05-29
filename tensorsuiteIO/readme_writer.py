from __future__ import annotations

from pathlib import Path
from typing import Any


def _yaml_summary(config: dict[str, Any]) -> str:
    lines: list[str] = []
    for section in ["generator", "tensor", "seed", "fastskg", "zipf", "format", "value", "duplicates", "output"]:
        if section in config:
            lines.append(f"- `{section}`: `{config[section]}`")
    return "\n".join(lines) if lines else "- No config summary available."


def write_bundle_readme(
    readme_path: str | Path,
    tensor_name: str,
    metadata: dict,
    config: dict,
    config_path: str | None = None,
    config_snapshot_path: str | None = None,
    duplicate_stats: dict | None = None,
    generation_info: dict | None = None,
) -> None:
    readme_path = Path(readme_path)
    generator = config.get("generator", {})
    value = config.get("value", {})
    duplicates = config.get("duplicates", {})
    role = (generation_info or {}).get("role", "generated_tensor")
    duplicate_stats_text = duplicate_stats or {}
    meta_seed_text = (generation_info or {}).get("meta_seed", "Not applicable.")
    body = f"""# {tensor_name}

## Overview
- Tensor name: `{tensor_name}`
- Tensor role: `{role}`
- Generator type: `{generator.get("type", "unknown")}`
- Source: `synthetic`
- Dimensions: `{metadata.get("dimensions")}`
- NNZ: `{metadata.get("nnz")}`
- Index base: `{metadata.get("index_base")}`
- Storage format: TensorSuite-TNS / TensorSuite-TNSB
- Value mode: `{value.get("mode", "constant")}`
- Value domain: `{metadata.get("value_domain")}`
- Duplicate policy: `{duplicates.get("policy", "keep")}`

## Configuration Source
- Config file path: `{config_path or "unknown"}`
- Config snapshot: `{config_snapshot_path or "config.yaml"}`
- Random seed: `{generator.get("random_seed", "not set")}`
- Generation time: `{metadata.get("time")}`
- KronWeave version: `0.1.0`

## YAML Configuration Summary
{_yaml_summary(config)}

## Bundle Layout
- `{tensor_name}_metadata.json`
- `{tensor_name}.tns`
- `{tensor_name}.tnsb`
- `README.md`
- `config.yaml`
- optional `cp_seed/`
- optional `reports/`
- optional `plots/`

## Metadata JSON
The metadata JSON stores only basic TensorSuite metadata. Collision rates, power-law fitting, low-rank stats, mode correlation, plots, and other evaluation results are not written into metadata JSON.

## TensorSuite-TNS Format
```text
%%TensorSuite-TNS
% version: 0.1
% name: <name>
<num_modes> <dim0> ... <dimD-1> <nnz>
<coord0> ... <coordD-1> <value>
```

## TensorSuite-TNSB Format
```text
%%TensorSuite-TNSB
% version: 0.1
% name: <name>
<num_modes> <dim0> ... <dimD-1> <nnz>
[binary payload]
```

Each binary record stores `D` `{metadata.get("index_type")}` indices followed by one `{metadata.get("value_type")}` value using `{metadata.get("endianness")}` endianness.

## Header / Metadata Cross-check
Headers and metadata can be checked with:
- `version`
- `name`
- `order` / `num_modes`
- `dimensions`
- `nnz`

## Meta Seed
{meta_seed_text}

## Duplicate Handling
- `keep`: repeated coordinates keep one entry.
- `sum`: repeated coordinates have values summed.
- `allow`: repeated entries are retained.

Duplicate stats, when computed, are documentation/report data only and are not canonical metadata:

```text
{duplicate_stats_text}
```

## Value Generation
Value configuration comes from YAML:

```text
{value}
```

The metadata `value_domain` is inferred from this YAML value configuration.

## Analysis
Power-law analysis, visualization, collision analysis, and low-rank checks can be run separately. Analysis results must not be written into metadata JSON.

## Reproducibility
Reproduce this bundle from the saved `config.yaml` snapshot and the KronWeave code version used at generation time.

## Notes
- Current implementation is CPU-only.
- CUDA is not included.
- The generator controls sparse structure; complex value modeling is out of scope for v0.1.
"""
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(body, encoding="utf-8")
