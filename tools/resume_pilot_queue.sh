#!/usr/bin/env bash
# Restart the P1 -> P3 -> P2 pilot queue after a reboot. Finished runs (model_final.pth)
# are skipped and interrupted ones resume from model_latest.pth (run_cargo_pilot.sh).
cd "$(dirname "$0")/.."
nohup bash tools/run_cargo_pilot.sh pcl_p1 experts4_p3 personvit_v1 personvit_baseline \
  > logs/CARGO_pilot/launcher_queue.log 2>&1 &
echo "queue started (pid $!); progress: logs/CARGO_pilot/runs.log"
