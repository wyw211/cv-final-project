#!/usr/bin/env bash
# Full 120-epoch training of the final method. Resumes from model_latest.pth if interrupted.
cd "$(dirname "$0")/.."
PY=../.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
cfg=configs/CARGO/visa_personvit_experts4.yml
out=logs/CARGO/ViSA_personvit_experts4
mkdir -p "$out"
resume=(); tag=start
if [ -e "$out/model_latest.pth" ]; then
  resume=(--resume); tag=resume
  n=1; while [ -e "$out/attempt$n" ]; do n=$((n+1)); done; mkdir -p "$out/attempt$n"
  for f in train.stdout.log train.stderr.log; do [ -e "$out/$f" ] && mv "$out/$f" "$out/attempt$n/$f"; done
fi
git diff HEAD > "$out/source_changes.patch" 2>/dev/null; git status --short > "$out/source_status.txt" 2>/dev/null
sha256sum weights/personvit_vitb_lup_teacher_e240_s12.pth > "$out/pretrain_sha256.txt"
echo "[$(date '+%F %T')] $tag full personvit_experts4" | tee -a logs/CARGO/runs.log
"$PY" -u tools/train_net.py --config-file "$cfg" "${resume[@]}" > "$out/train.stdout.log" 2> "$out/train.stderr.log"
echo "[$(date '+%F %T')] end full personvit_experts4 exit=$?" | tee -a logs/CARGO/runs.log
