#!/usr/bin/env bash
# Pilot screening: baseline vs expert-input fix on the CARGO train/val split
# (20 epochs each, run one after the other). Never reads the CARGO test set.
# Usage (from the ViSA folder, in Git Bash):  bash tools/run_cargo_pilot.sh [baseline] [expert_input_fix]
set -u
cd "$(dirname "$0")/.."
PY=../.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
runs=("$@")
[ ${#runs[@]} -eq 0 ] && runs=(baseline expert_input_fix)
for name in "${runs[@]}"; do
  cfg="configs/CARGO/pilot_val_${name}.yml"
  out="logs/CARGO_pilot/${name}"
  if [ -e "$out/model_final.pth" ]; then echo "skip $name: $out/model_final.pth exists"; continue; fi
  mkdir -p "$out"
  resume=()
  tag=start
  if [ -e "$out/model_latest.pth" ]; then
    # Interrupted earlier: continue from the latest periodic checkpoint.
    resume=(--resume); tag=resume
    n=1; while [ -e "$out/attempt$n" ]; do n=$((n+1)); done
    mkdir -p "$out/attempt$n"
    for f in train.stdout.log train.stderr.log source_changes.patch source_status.txt source_commit.txt; do
      [ -e "$out/$f" ] && mv "$out/$f" "$out/attempt$n/$f"
    done
  fi
  git rev-parse HEAD > "$out/source_commit.txt"
  git diff HEAD > "$out/source_changes.patch"
  git status --short > "$out/source_status.txt"
  echo "[$(date '+%F %T')] $tag $name" | tee -a logs/CARGO_pilot/runs.log
  "$PY" tools/train_net.py --config-file "$cfg" "${resume[@]}" > "$out/train.stdout.log" 2> "$out/train.stderr.log"
  echo "[$(date '+%F %T')] end $name exit=$?" | tee -a logs/CARGO_pilot/runs.log
done
