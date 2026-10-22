#!/usr/bin/env bash
# Usage on the GPU server: GPU=2 ./run.sh smoke.py
# The shared deezer account's LD_LIBRARY_PATH points to another user's venv (wrong cuDNN), so we override it.
set -euo pipefail
cd "$(dirname "$0")"
NV=.venv/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$NV/cudnn/lib:$NV/cu13/lib
export HF_HOME=/data/deezer/htandeaudemarsac/hf
export CUDA_VISIBLE_DEVICES=${GPU:-2}
exec .venv/bin/python "$@"
