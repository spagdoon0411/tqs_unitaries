#!/usr/bin/env bash
# Regenerates all TQS magnetization readings from a particular checkpoint, over both the full [-4, 4] sweep and the [-1, 0] zoom sweep.
#
# Run from the repo root. E.g.,
#
#     bash scripts/generate_devout_sun_10_readings.sh

set -euo pipefail

CKPT_DIR="checkpoints/20260703-224128_devout-sun-10"
ITERATION=135000
N_H=101
N_READINGS=20
OUT_DIR="scripts/data/devout-sun-10-135000"

mkdir -p "$OUT_DIR"

run_sweep() {
    local system_size_idx="$1"
    local L="$2"
    local h_min="$3"
    local h_max="$4"
    local tag="$5"

    python scripts/magnetization_readings.py "$CKPT_DIR" \
        --iteration "$ITERATION" \
        --system-size-idx "$system_size_idx" \
        --n-h "$N_H" \
        --n-readings "$N_READINGS" \
        --h-min "$h_min" \
        --h-max "$h_max" \
        --out "$OUT_DIR/devout-sun-10_iter${ITERATION}_L${L}_${tag}_readings.pt"
}

run_sweep 0 10 -4 4 full
run_sweep 0 10 -1 0 zoom
run_sweep 10 30 -4 4 full
run_sweep 10 30 -1 0 zoom

echo "Done. Readings written under $OUT_DIR/"
