#!/usr/bin/env bash
# Second pilot queue (2026-11-25): P2+P1, P2+P3, and the jx + 0.2 LR control.
cd "$(dirname "$0")/.."
nohup bash tools/run_cargo_pilot.sh personvit_experts4 personvit_pcl jx_lr02 \
  > logs/CARGO_pilot/launcher_queue2.log 2>&1 &
echo "queue started (pid $!); progress: logs/CARGO_pilot/runs.log"
