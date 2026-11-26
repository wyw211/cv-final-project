#!/usr/bin/env bash
# Third pilot queue (2026-11-26): night/low-light fixes for A<->A on top of the final method.
cd "$(dirname "$0")/.."
nohup bash tools/run_cargo_pilot.sh light_l1 light_l2 light_l3 \
  > logs/CARGO_pilot/launcher_queue3.log 2>&1 &
echo "queue started (pid $!); progress: logs/CARGO_pilot/runs.log"
