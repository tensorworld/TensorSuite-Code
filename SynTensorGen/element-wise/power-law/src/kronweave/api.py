from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
from typing import Any

import yaml

from kronweave.config import load_and_validate_config
from kronweave.io.metadata import build_metadata, read_metadata, write_metadata
from kronweave.io.readme_writer import write_bundle_readme
from kronweave.io.tensorsuite import (
    cross_check_header_metadata,
    cross_check_tns_tnsb,
    parse_tns_header,
    parse_tnsb_header,
    write_tns,
    write_tnsb_from_tns,
)
from kronweave.postprocess.duplicates import handle_duplicates
from kronweave.postprocess.values import generate_values, infer_value_domain
from kronweave.runners.fastskg import run_fastskg
from kronweave.seed.kp_seed import generate_kp_seed_from_meta
from kronweave.seed.meta_seed import write_cp_seed_bundle
from kronweave.seed.zipf import generate_zipf_coords


def _resolve_output_dir(config_path: Path, output_dir: str) -> Path:
    out = Path(output_dir)
    if not out.is_absolute():
        out = Path.cwd() / out
    return out


def _value_seed(config: dict[str, Any]) -> int | None:
    seed = config.get("generator", {}).get("random_seed")
    return None if seed is None else int(seed) + 1000


def _write_config_snapshot(config: dict, bundle_dir: Path) -> Path:
    snapshot = bundle_dir / "config.yaml"
    snapshot.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return snapshot


def _resolve_input_path(config_path: Path, path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    candidate = config_path.parent / path
    if candidate.exists():
        return candidate
    return Path.cwd() / path


def _value_dtype(value_type: str):
    import numpy as np

    mapping = {
        "float32": np.float32,
        "float64": np.float64,
        "int32": np.int32,
        "int64": np.int64,
    }
    return mapping[value_type]


def _format_cfg(config: dict) -> dict:
    return config.get("format", {})


def _prepare_entries(config: dict, coords, values):
    import numpy as np

    coord_arr = np.asarray(coords, dtype=np.uint64)
    value_type = config.get("value", {}).get("type", "float64")
    value_arr = np.asarray(values).astype(_value_dtype(value_type), copy=False)
    if _format_cfg(config).get("explicit_zeros", "disallowed") == "disallowed":
        keep = value_arr != 0
        coord_arr = coord_arr[keep]
        value_arr = value_arr[keep]
    return coord_arr, value_arr


def _sort_entries(config: dict, coords, values):
    import numpy as np

    fmt = _format_cfg(config)
    if fmt.get("sorted") != "lexicographic" or len(coords) == 0:
        return coords, values
    sorted_order = [int(x) for x in fmt.get("sorted_order", [])]
    keys = tuple(coords[:, d] for d in reversed(sorted_order))
    order = np.lexsort(keys)
    return coords[order], values[order]


def _finalize_bundle(
    *,
    bundle_dir: Path,
    name: str,
    dims: list[int],
    coords,
    values,
    config: dict,
    config_path: Path,
    duplicate_stats: dict,
    generation_info: dict,
) -> dict:
    tensor_cfg = config["tensor"]
    fmt = _format_cfg(config)
    output_cfg = config.get("output", {})
    index_base = int(tensor_cfg.get("index_base", 0))
    values_provided = bool(config.get("values_provided", True))
    bundle_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = _write_config_snapshot(config, bundle_dir) if output_cfg.get("copy_config", True) else None
    tns_path = bundle_dir / f"{name}.tns"
    tnsb_path = bundle_dir / f"{name}.tnsb"
    metadata_path = bundle_dir / f"{name}_metadata.json"
    readme_path = bundle_dir / "README.md"
    if output_cfg.get("write_tns", True):
        write_tns(tns_path, name, dims, coords, values, index_base=index_base, values_provided=values_provided)
    if output_cfg.get("write_tnsb", True):
        write_tnsb_from_tns(
            tns_path,
            tnsb_path,
            index_type=fmt.get("index_type", "uint64"),
            value_type=config.get("value", {}).get("type", "float64"),
            endianness=fmt.get("endianness", "little"),
        )
    metadata = build_metadata(
        name=name,
        dims=dims,
        nnz=len(values),
        index_base=index_base,
        value_domain=infer_value_domain(config.get("value")),
        duplicate_policy=config.get("duplicates", {}).get("policy", "keep"),
        value_type=config.get("value", {}).get("type", "float64"),
        index_type=fmt.get("index_type", "uint64"),
        endianness=fmt.get("endianness", "little"),
        sorted_state=fmt.get("sorted", "lexicographic"),
        sorted_order=fmt.get("sorted_order", list(range(len(dims)))),
        explicit_zeros=fmt.get("explicit_zeros", "disallowed"),
        pattern_symmetry=fmt.get("pattern_symmetry", "no"),
        numerical_symmetry=fmt.get("numerical_symmetry", "no"),
        sparsity_type=fmt.get("sparsity_type", "element"),
        values_provided=values_provided,
        files={
            "text": f"{name}.tns" if output_cfg.get("write_tns", True) else None,
            "binary": f"{name}.tnsb" if output_cfg.get("write_tnsb", True) else None,
            "readme": "README.md" if output_cfg.get("write_readme", True) else None,
            "config": "config.yaml" if output_cfg.get("copy_config", True) else None,
        },
    )
    if output_cfg.get("write_metadata", True):
        write_metadata(metadata_path, metadata)
    if output_cfg.get("write_readme", True):
        write_bundle_readme(
            readme_path,
            name,
            metadata,
            config,
            config_path=str(config_path),
            config_snapshot_path=str(config_snapshot.name if config_snapshot else ""),
            duplicate_stats=duplicate_stats,
            generation_info=generation_info,
        )
    if tns_path.exists() and metadata_path.exists():
        cross_check_header_metadata(parse_tns_header(tns_path), read_metadata(metadata_path))
    if tnsb_path.exists() and metadata_path.exists():
        cross_check_header_metadata(parse_tnsb_header(tnsb_path), read_metadata(metadata_path))
        cross_check_tns_tnsb(tns_path, tnsb_path, metadata=read_metadata(metadata_path))
    return {
        "bundle_dir": bundle_dir,
        "tns": tns_path,
        "tnsb": tnsb_path,
        "metadata": metadata_path,
        "readme": readme_path,
        "config": config_snapshot,
    }


def _postprocess(config: dict, raw_coords, raw_values):
    policy = config.get("duplicates", {}).get("policy", "keep")
    preserve_order = _format_cfg(config).get("sorted") == "none"
    if policy == "sum":
        generated_raw_values = generate_values(len(raw_coords), config.get("value"), seed=_value_seed(config))
        result = handle_duplicates(raw_coords, generated_raw_values, policy="sum", preserve_order=preserve_order)
        return result.coords, result.values, result.stats
    result = handle_duplicates(raw_coords, raw_values, policy=policy, preserve_order=preserve_order)
    value_cfg = config.get("value") or {}
    if policy == "allow" and value_cfg.get("mode", "constant") == "constant" and float(value_cfg.get("constant", 1.0)) == 1.0:
        return result.coords, result.values, result.stats
    values = generate_values(len(result.coords), config.get("value"), seed=_value_seed(config))
    return result.coords, values, result.stats


def generate_from_config(config_path: str | Path) -> dict:
    config_path = Path(config_path).resolve()
    config = load_and_validate_config(config_path)
    name = config.get("name") or config.get("experiment", {}).get("name")
    output_dir = _resolve_output_dir(config_path, config.get("output_dir", "outputs"))
    bundle_dir = output_dir / name
    generator_type = config["generator"]["type"]
    dims = [int(x) for x in config["tensor"]["sizes"]]
    nnz = config.get("fastskg", {}).get("nnz", config["tensor"].get("nnz"))
    nnz = None if nnz is None else int(nnz)
    index_base = int(config["tensor"].get("index_base", 0))

    if generator_type in {"cp_fastskg", "fastskg"}:
        cp_seed_name = f"{name}_cp_seed"
        seed_cfg = config["seed"]
        for generated_seed_dir in [bundle_dir / "cp_seed", bundle_dir / "meta_seed"]:
            if generated_seed_dir.exists():
                shutil.rmtree(generated_seed_dir)
        seed_source = seed_cfg["source"]
        if seed_source == "cpd":
            meta = write_cp_seed_bundle(bundle_dir / "cp_seed", name=cp_seed_name, config=config, config_path=str(config_path))
            generation_info = {
                "role": "generated_tensor",
                "meta_seed": f"CPD parameters generated CP seed `{cp_seed_name}` directly; no meta seed file was read.",
            }
        elif seed_source == "input_tns":
            seed_path = _resolve_input_path(config_path, seed_cfg["input_tns"])
            seed_header = parse_tns_header(seed_path)
            meta = {
                "tns": seed_path,
                "output_dimensions": [int(s) ** int(config["fastskg"]["iter"]) for s in seed_header.dimensions],
            }
            generation_info = {
                "role": "generated_tensor",
                "meta_seed": f"Direct TensorSuite-TNS seed `{seed_path}` was used as the fastSKG input.",
            }
        else:
            seed_path = _resolve_input_path(config_path, seed_cfg["input_tns"])
            kp_name = f"{name}_kp_seed"
            kp_coords, kp_values, kp_sizes = generate_kp_seed_from_meta(seed_path, int(seed_cfg["expand_iter"]))
            kp_dir = bundle_dir / "cp_seed"
            kp_dir.mkdir(parents=True, exist_ok=True)
            kp_tns = kp_dir / f"{kp_name}.tns"
            kp_tnsb = kp_dir / f"{kp_name}.tnsb"
            write_tns(kp_tns, kp_name, kp_sizes, kp_coords, kp_values, index_base=index_base)
            if config.get("output", {}).get("write_tnsb", True):
                write_tnsb_from_tns(
                    kp_tns,
                    kp_tnsb,
                    index_type=_format_cfg(config).get("index_type", "uint64"),
                    value_type=config.get("value", {}).get("type", "float64"),
                    endianness=_format_cfg(config).get("endianness", "little"),
                )
            meta = {
                "tns": kp_tns,
                "tnsb": kp_tnsb,
                "output_dimensions": [int(s) ** int(config["fastskg"]["iter"]) for s in kp_sizes],
            }
            generation_info = {
                "role": "generated_tensor",
                "meta_seed": f"Meta seed `{seed_path}` expanded by Kronecker product into `cp_seed/{kp_name}.tns`.",
            }
        with tempfile.TemporaryDirectory(prefix="kronweave_fastskg_") as tmp:
            raw_tns = Path(tmp) / f"{name}_raw.tns"
            raw_coords, raw_values = run_fastskg(
                seed_tns=meta["tns"],
                raw_output_tns=raw_tns,
                iterations=int(config["fastskg"]["iter"]),
                nnz=nnz,
                index_base=index_base,
                name=f"{name}_raw",
            )
        coords, values, duplicate_stats = _postprocess(config, raw_coords, raw_values)
        dims = [int(x) for x in meta.get("output_dimensions", dims)] if "output_dimensions" in meta else dims
    elif generator_type == "zipf":
        raw_coords = generate_zipf_coords(
            sizes=dims,
            nnz=nnz,
            alpha=config["zipf"]["alpha"],
            random_seed=config["generator"].get("random_seed"),
        )
        raw_values = generate_values(len(raw_coords), {"mode": "constant", "constant": 1.0}, seed=None)
        coords, values, duplicate_stats = _postprocess(config, raw_coords, raw_values)
        generation_info = {"role": "baseline", "meta_seed": "Not applicable for Zipf baseline."}
    else:  # pragma: no cover - config validation prevents this
        raise ValueError(f"Unsupported generator type: {generator_type}")

    coords, values = _prepare_entries(config, coords, values)
    coords, values = _sort_entries(config, coords, values)
    return _finalize_bundle(
        bundle_dir=bundle_dir,
        name=name,
        dims=dims,
        coords=coords,
        values=values,
        config=config,
        config_path=config_path,
        duplicate_stats=duplicate_stats,
        generation_info=generation_info,
    )


def validate_bundle(bundle_dir: str | Path) -> dict:
    bundle = Path(bundle_dir)
    name = bundle.name
    metadata = read_metadata(bundle / f"{name}_metadata.json")
    files = metadata.get("files", {})
    tns = bundle / files["text"] if files.get("text") else None
    tnsb = bundle / files["binary"] if files.get("binary") else None
    if tns is not None:
        cross_check_header_metadata(parse_tns_header(tns), metadata)
    if tnsb is not None:
        cross_check_header_metadata(parse_tnsb_header(tnsb), metadata)
    if tns is not None and tnsb is not None:
        cross_check_tns_tnsb(tns, tnsb, metadata=metadata)
    if tns is None and tnsb is None:
        raise ValueError("Bundle metadata does not reference a tensor data file")
    for key in ["readme", "config"]:
        rel = files.get(key)
        if rel and not (bundle / rel).exists():
            required = bundle / rel
            raise ValueError(f"Missing required bundle file: {required}")
    return {"bundle_dir": bundle, "metadata": metadata}
