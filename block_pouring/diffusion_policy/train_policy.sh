#!/bin/bash
# Behavior-cloning pretraining of the DPPO image diffusion policy (base policy for the demo).
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DPPO=${DPPO:-$ROOT/third_party/dppo}
export DPPO_DATA_DIR=$ROOT/data/block_pouring/diffusion_policy
export DPPO_LOG_DIR=$ROOT/logs/block_pouring/diffusion_policy
export PYTHONPATH=$DPPO:$PYTHONPATH

if [ ! -f "$DPPO_DATA_DIR/train.npz" ]; then
	python "$ROOT/block_pouring/diffusion_policy/process_dataset.py" --src "$ROOT/data/block_pouring/rollouts_0804" --out "$DPPO_DATA_DIR/train.npz"
fi
cd "$DPPO"
python script/run.py --config-name=pre_diffusion_mlp_img --config-dir="$ROOT/block_pouring/diffusion_policy" "$@"
