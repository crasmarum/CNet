#!/bin/bash
# Launch the multi-GPU (data-parallel) tiny-shakespeare example: one process per
# GPU, each pinned to a distinct device, all sharing one NCCL communicator.
#
# Usage: ./run_dp_shakespeare.sh <num_gpus> [extra dp_shakespeare flags...]
#   ./run_dp_shakespeare.sh 1
#   ./run_dp_shakespeare.sh 4 -batch 64 -steps 5000
set -eu
W="${1:-1}"; shift || true
BIN="${BIN:-./dp_shakespeare}"
rm -f /tmp/cnet_shk_id.bin /tmp/cnet_shk_id.ready
for r in $(seq 0 $((W - 1))); do
  CUDA_VISIBLE_DEVICES="$r" "$BIN" -world "$W" -rank "$r" "$@" &
done
wait
