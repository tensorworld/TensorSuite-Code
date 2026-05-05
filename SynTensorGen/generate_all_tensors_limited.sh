#!/bin/bash
# ==========================================
# Tensor generation suite for COO↔HiCOO testing
# Author: Zizhong Wang
# ==========================================
# Generate synthetic tensors under memory constraint
#   - Covers mixed dense/sparse modes
#   - Adapts density per dimension size and NDIMS
#   - Automatically skips too-large configurations
# ==========================================

set -euo pipefail

# -----------------------------
# Path setup
# -----------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN_SCRIPT="$SCRIPT_DIR/random_tensor_generator_limited.py"
OUT_DIR="$SCRIPT_DIR/generated_tensors"
mkdir -p "$OUT_DIR"

# -----------------------------
# Global settings
# -----------------------------
PRECISION="D"              # 'D' = double, 'S' = single
MEM_THRESHOLD_GB=200       # memory safety limit (GiB)
SIZE_PER_DIM=1000          # base dimension size

# NDIMS and recommended density combos
declare -A DENSE_DENSITY_MAP=(
  [3]=25   # 3D: dense 25%, sparse 5%
  [4]=10   # 4D: dense 10%, sparse 3%
  [5]=5   # 5D: dense 5%, sparse 1%
)
declare -A SPARSE_DENSITY_MAP=(
  [3]=5    
  [4]=3
  [5]=1
)

# NDIMS list
# NDIMS_LIST=(3 4 5)
NDIMS_LIST=(4 5)
# -----------------------------
# Function: generate tensors for one NDIMS
# -----------------------------
generate_all_for_ndim() {
  local ndim=$1
  local dense=${DENSE_DENSITY_MAP[$ndim]}
  local sparse=${SPARSE_DENSITY_MAP[$ndim]}

  echo "=== Generating ${ndim}D tensors (dense=${dense}%, sparse=${sparse}%) ==="

  local max_mask=$(( (1 << ndim) - 1 ))
  for ((mask=1; mask<=max_mask; mask++)); do
    local args=()
    local name="tensor_${ndim}D"

    for ((i=1; i<=ndim; i++)); do
      if (( (mask >> (i-1)) & 1 )); then
        density=$sparse
      else
        density=$dense
      fi
      args+=( "${density}%${SIZE_PER_DIM}" )
      name="${name}_m${i}-${density}"
    done

    local outfile="${name}.tns"
    echo "Generating ${outfile} ..."
    python3 "$GEN_SCRIPT" "$OUT_DIR/$outfile" "$PRECISION" "$MEM_THRESHOLD_GB" "${args[@]}"
    echo "✅ Done -> $OUT_DIR/$outfile"
  done

  echo "=== ${ndim}D generation done ==="
  echo
}

# -----------------------------
# Main loop
# -----------------------------
for nd in "${NDIMS_LIST[@]}"; do
  generate_all_for_ndim "$nd"
done

echo "🎉 All tensors generated successfully under $OUT_DIR/"
