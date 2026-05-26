from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import os
import struct
from typing import Iterable, Iterator, Sequence

import numpy as np


TNS_MAGIC = "%%TensorSuite-TNS"
TNSB_MAGIC = "%%TensorSuite-TNSB"
DEFAULT_VERSION = "1.0"
INDEX_FORMATS = {
    "uint32": ("I", 4, 2**32 - 1),
    "uint64": ("Q", 8, 2**64 - 1),
}
VALUE_FORMATS = {
    "float32": ("f", 4),
    "float64": ("d", 8),
    "int32": ("i", 4),
    "int64": ("q", 8),
}
ENDIAN_FORMATS = {"little": "<", "big": ">"}


@dataclass(frozen=True)
class TensorSuiteHeader:
    magic: str
    version: str
    name: str
    num_modes: int
    dimensions: tuple[int, ...]
    nnz: int
    payload_offset: int | None = None


@dataclass
class TensorData:
    header: TensorSuiteHeader
    coords: np.ndarray
    values: np.ndarray


def _as_path(path: str | os.PathLike[str]) -> Path:
    return Path(path)


def _parse_meta_line(line: str, key: str) -> str:
    prefix = f"% {key}:"
    if not line.startswith(prefix):
        raise ValueError(f"Expected metadata line starting with {prefix!r}, got {line!r}")
    return line[len(prefix):].strip()


def _parse_size_line(line: str) -> tuple[int, tuple[int, ...], int]:
    parts = line.strip().split()
    if len(parts) < 3:
        raise ValueError("TensorSuite size line must contain num_modes, dimensions, and nnz")
    try:
        nums = [int(x) for x in parts]
    except ValueError as exc:
        raise ValueError(f"Invalid integer in size line: {line.strip()}") from exc
    num_modes = nums[0]
    if num_modes <= 0:
        raise ValueError("num_modes must be positive")
    if len(nums) != num_modes + 2:
        raise ValueError(
            f"Size line has {len(nums)} fields but expected num_modes + dimensions + nnz = {num_modes + 2}"
        )
    dims = tuple(nums[1:-1])
    nnz = nums[-1]
    if any(d <= 0 for d in dims):
        raise ValueError(f"All dimensions must be positive, got {dims}")
    if nnz < 0:
        raise ValueError("nnz must be nonnegative")
    return num_modes, dims, nnz


def parse_tns_header(path: str | os.PathLike[str]) -> TensorSuiteHeader:
    path = _as_path(path)
    with path.open("r", encoding="utf-8") as f:
        magic = f.readline().strip()
        if magic != TNS_MAGIC:
            raise ValueError(f"Invalid TNS magic line: {magic!r}")
        version = _parse_meta_line(f.readline().strip(), "version")
        name = _parse_meta_line(f.readline().strip(), "name")
        num_modes, dims, nnz = _parse_size_line(f.readline())
    return TensorSuiteHeader(magic, version, name, num_modes, dims, nnz)


def iter_tns_entries(path: str | os.PathLike[str]) -> Iterator[tuple[tuple[int, ...], float]]:
    header = parse_tns_header(path)
    with _as_path(path).open("r", encoding="utf-8") as f:
        for _ in range(4):
            next(f)
        count = 0
        for lineno, line in enumerate(f, start=5):
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) not in {header.num_modes, header.num_modes + 1}:
                raise ValueError(
                    f"Invalid entry line {lineno}: expected {header.num_modes} coordinates with optional value"
                )
            try:
                coords = tuple(int(x) for x in parts[: header.num_modes])
                value = float(parts[-1]) if len(parts) == header.num_modes + 1 else 1.0
            except ValueError as exc:
                raise ValueError(f"Invalid entry line {lineno}: {s}") from exc
            count += 1
            yield coords, value
        if count != header.nnz:
            raise ValueError(f"TNS nnz mismatch: header={header.nnz}, actual={count}")


def read_tns(path: str | os.PathLike[str]) -> TensorData:
    header = parse_tns_header(path)
    coords: list[tuple[int, ...]] = []
    values: list[float] = []
    for coord, value in iter_tns_entries(path):
        coords.append(coord)
        values.append(value)
    coord_arr = np.asarray(coords, dtype=np.uint64).reshape((len(coords), header.num_modes))
    value_arr = np.asarray(values, dtype=np.float64)
    return TensorData(header=header, coords=coord_arr, values=value_arr)


def _format_value(value: float) -> str:
    if not math.isfinite(float(value)):
        raise ValueError(f"TensorSuite values must be finite, got {value!r}")
    return f"{float(value):.16e}"


def write_tns(
    path: str | os.PathLike[str],
    name: str,
    dims: Sequence[int],
    coords: Sequence[Sequence[int]] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    version: str = DEFAULT_VERSION,
    index_base: int = 0,
    values_provided: bool = True,
) -> TensorSuiteHeader:
    if index_base not in (0, 1):
        raise ValueError("index_base must be 0 or 1")
    dims_tuple = tuple(int(d) for d in dims)
    if not dims_tuple or any(d <= 0 for d in dims_tuple):
        raise ValueError(f"Invalid dimensions: {dims}")
    coord_arr = np.asarray(coords, dtype=np.int64)
    if coord_arr.size == 0:
        coord_arr = coord_arr.reshape((0, len(dims_tuple)))
    if coord_arr.ndim != 2 or coord_arr.shape[1] != len(dims_tuple):
        raise ValueError(f"coords must have shape (nnz, {len(dims_tuple)}), got {coord_arr.shape}")
    value_arr = np.asarray(values, dtype=np.float64)
    if values_provided and (value_arr.ndim != 1 or value_arr.shape[0] != coord_arr.shape[0]):
        raise ValueError("values must be one-dimensional and match coords length")
    if np.any(coord_arr < 0):
        raise ValueError("coords must be zero-based internal nonnegative indices")
    for d, dim in enumerate(dims_tuple):
        if coord_arr.shape[0] and np.any(coord_arr[:, d] >= dim):
            raise ValueError(f"coordinate out of range for mode {d}")
    path = _as_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{TNS_MAGIC}\n")
        f.write(f"% version: {version}\n")
        f.write(f"% name: {name}\n")
        f.write(" ".join([str(len(dims_tuple)), *map(str, dims_tuple), str(coord_arr.shape[0])]) + "\n")
        for i, coord in enumerate(coord_arr):
            stored = [str(int(x) + index_base) for x in coord]
            if values_provided:
                value = value_arr[i]
                f.write(" ".join(stored) + " " + _format_value(float(value)) + "\n")
            else:
                f.write(" ".join(stored) + "\n")
    return TensorSuiteHeader(TNS_MAGIC, version, name, len(dims_tuple), dims_tuple, int(coord_arr.shape[0]))


def parse_tnsb_header(path: str | os.PathLike[str]) -> TensorSuiteHeader:
    path = _as_path(path)
    with path.open("rb") as f:
        lines: list[bytes] = []
        for _ in range(4):
            line = f.readline()
            if not line:
                raise ValueError("Unexpected EOF while reading TNSB header")
            lines.append(line)
        payload_offset = f.tell()
    magic = lines[0].decode("utf-8").strip()
    if magic != TNSB_MAGIC:
        raise ValueError(f"Invalid TNSB magic line: {magic!r}")
    version = _parse_meta_line(lines[1].decode("utf-8").strip(), "version")
    name = _parse_meta_line(lines[2].decode("utf-8").strip(), "name")
    num_modes, dims, nnz = _parse_size_line(lines[3].decode("utf-8"))
    return TensorSuiteHeader(magic, version, name, num_modes, dims, nnz, payload_offset)


def _binary_format(num_modes: int, index_type: str, value_type: str, endianness: str) -> str:
    if index_type not in INDEX_FORMATS:
        raise ValueError("index_type must be uint32 or uint64")
    if value_type not in VALUE_FORMATS:
        raise ValueError("value_type must be float32, float64, int32, or int64")
    if endianness not in ENDIAN_FORMATS:
        raise ValueError("endianness must be little or big")
    return ENDIAN_FORMATS[endianness] + (INDEX_FORMATS[index_type][0] * num_modes) + VALUE_FORMATS[value_type][0]


def _coerce_binary_value(value: float, value_type: str):
    if value_type.startswith("int"):
        return int(value)
    return float(value)


def write_tnsb_from_tns(
    tns_path: str | os.PathLike[str],
    tnsb_path: str | os.PathLike[str],
    *,
    index_type: str = "uint64",
    value_type: str = "float64",
    endianness: str = "little",
) -> TensorSuiteHeader:
    tns_header = parse_tns_header(tns_path)
    tnsb_path = _as_path(tnsb_path)
    tnsb_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = _binary_format(tns_header.num_modes, index_type, value_type, endianness)
    max_index = INDEX_FORMATS[index_type][2]
    if any(int(d) > max_index for d in tns_header.dimensions):
        raise ValueError(f"index_type={index_type} cannot store one or more dimensions")
    count = 0
    with tnsb_path.open("wb") as out:
        out.write(f"{TNSB_MAGIC}\n".encode("utf-8"))
        out.write(f"% version: {tns_header.version}\n".encode("utf-8"))
        out.write(f"% name: {tns_header.name}\n".encode("utf-8"))
        out.write(
            (" ".join([str(tns_header.num_modes), *map(str, tns_header.dimensions), str(tns_header.nnz)]) + "\n").encode(
                "utf-8"
            )
        )
        for coords, value in iter_tns_entries(tns_path):
            if any(c < 0 or c > max_index for c in coords):
                raise ValueError(f"TNSB index_type={index_type} cannot store one or more coordinates")
            out.write(struct.pack(fmt, *coords, _coerce_binary_value(value, value_type)))
            count += 1
    if count != tns_header.nnz:
        raise ValueError(f"TNSB conversion nnz mismatch: header={tns_header.nnz}, actual={count}")
    return parse_tnsb_header(tnsb_path)


def count_tnsb_records(
    path: str | os.PathLike[str],
    *,
    index_type: str = "uint64",
    value_type: str = "float64",
) -> int:
    header = parse_tnsb_header(path)
    if index_type not in INDEX_FORMATS:
        raise ValueError("index_type must be uint32 or uint64")
    if value_type not in VALUE_FORMATS:
        raise ValueError("value_type must be float32, float64, int32, or int64")
    record_size = INDEX_FORMATS[index_type][1] * header.num_modes + VALUE_FORMATS[value_type][1]
    payload_size = _as_path(path).stat().st_size - int(header.payload_offset or 0)
    if payload_size % record_size != 0:
        raise ValueError("TNSB payload size is not a multiple of record size")
    return payload_size // record_size


def read_tnsb_records(
    path: str | os.PathLike[str],
    limit: int | None = None,
    *,
    index_type: str = "uint64",
    value_type: str = "float64",
    endianness: str = "little",
) -> list[tuple[tuple[int, ...], float]]:
    header = parse_tnsb_header(path)
    fmt = _binary_format(header.num_modes, index_type, value_type, endianness)
    record_size = struct.calcsize(fmt)
    total = count_tnsb_records(path, index_type=index_type, value_type=value_type)
    if total != header.nnz:
        raise ValueError(f"TNSB nnz mismatch: header={header.nnz}, actual={total}")
    want = total if limit is None else min(limit, total)
    records: list[tuple[tuple[int, ...], float]] = []
    with _as_path(path).open("rb") as f:
        f.seek(int(header.payload_offset or 0))
        for _ in range(want):
            chunk = f.read(record_size)
            if len(chunk) != record_size:
                raise ValueError("Unexpected EOF in TNSB payload")
            unpacked = struct.unpack(fmt, chunk)
            records.append((tuple(int(x) for x in unpacked[:-1]), float(unpacked[-1])))
    return records


def cross_check_tns_tnsb(
    tns_path: str | os.PathLike[str],
    tnsb_path: str | os.PathLike[str],
    sample: int = 8,
    *,
    metadata: dict | None = None,
) -> None:
    tns_header = parse_tns_header(tns_path)
    tnsb_header = parse_tnsb_header(tnsb_path)
    if tns_header.version != tnsb_header.version:
        raise ValueError("TNS/TNSB version mismatch")
    if tns_header.name != tnsb_header.name:
        raise ValueError("TNS/TNSB name mismatch")
    if tns_header.num_modes != tnsb_header.num_modes:
        raise ValueError("TNS/TNSB num_modes mismatch")
    if tns_header.dimensions != tnsb_header.dimensions:
        raise ValueError("TNS/TNSB dimensions mismatch")
    if tns_header.nnz != tnsb_header.nnz:
        raise ValueError("TNS/TNSB nnz mismatch")
    index_type = (metadata or {}).get("index_type", "uint64")
    value_type = (metadata or {}).get("value_type", "float64")
    endianness = (metadata or {}).get("endianness", "little")
    if count_tnsb_records(tnsb_path, index_type=index_type, value_type=value_type) != tns_header.nnz:
        raise ValueError("TNSB binary record count mismatch")
    text_records = []
    for idx, record in enumerate(iter_tns_entries(tns_path)):
        if idx >= sample:
            break
        text_records.append(record)
    binary_records = read_tnsb_records(
        tnsb_path,
        limit=sample,
        index_type=index_type,
        value_type=value_type,
        endianness=endianness,
    )
    if len(text_records) != len(binary_records):
        raise ValueError("TNS/TNSB sample length mismatch")
    for a, b in zip(text_records, binary_records):
        rel_tol = 1e-6 if value_type == "float32" else 1e-12
        abs_tol = 1e-6 if value_type == "float32" else 1e-12
        if a[0] != b[0] or not math.isclose(a[1], b[1], rel_tol=rel_tol, abs_tol=abs_tol):
            raise ValueError(f"TNS/TNSB sample record mismatch: {a} != {b}")


def cross_check_header_metadata(header: TensorSuiteHeader, metadata: dict) -> None:
    if header.version != metadata.get("version"):
        raise ValueError("header.version != metadata.version")
    if header.name != metadata.get("name"):
        raise ValueError("header.name != metadata.name")
    if header.num_modes != metadata.get("order"):
        raise ValueError("header.num_modes != metadata.order")
    if list(header.dimensions) != list(metadata.get("dimensions", [])):
        raise ValueError("header.dimensions != metadata.dimensions")
    if header.nnz != metadata.get("nnz"):
        raise ValueError("header.nnz != metadata.nnz")
