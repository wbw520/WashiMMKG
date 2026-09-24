#!/usr/bin/env bash
# Serve a local model as an OpenAI-compatible endpoint with vLLM, for the parts of the
# pipeline that call a backbone (image alignment, question verbalisation, evaluation).
#
# Usage: tools/serve_qwen.sh [gpus] [port] [model] [gpu-memory-utilization]
#   27B on two GPUs (tensor parallel; the 52 GB checkpoint does not fit one 48 GB card):
#       tools/serve_qwen.sh 0,1 8000 Qwen/Qwen3.5-27B
#   9B on a single GPU:
#       tools/serve_qwen.sh 2   8002 Qwen/Qwen3.5-9B
#
# Optional environment:
#   VLLM_ENV      conda environment that has vllm installed (activated if set)
#   HF_HOME       Hugging Face cache holding the weights (HF_HUB_OFFLINE=1 is set, so the
#                 weights must already be there -- a hub hiccup then cannot stall startup)
#   CUDA_COMPAT   directory with a CUDA forward-compatibility libcuda, prepended only when
#                 the kernel driver is older than 570 (data-centre GPUs on an old driver)
#   IMG_LIMIT     per-prompt image cap (default 4). The server enforces it, and a prompt
#                 over the cap comes back as a 400 that the pipeline records as a failed
#                 answer rather than an error -- the run completes and is simply wrong.
#   DTYPE         e.g. float16 on pre-Ampere GPUs: the checkpoints are bf16 and vLLM
#                 refuses bf16 below compute capability 8.0 instead of falling back.
#   PARALLEL      "pipeline" splits the model by layer instead of by tensor, so workers
#                 never run an all-reduce; use it where tensor parallelism hangs.
set -eo pipefail   # no -u: conda activate trips on unbound vars

if [ -n "${VLLM_ENV:-}" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$VLLM_ENV"
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi
export HF_HUB_OFFLINE=1

DRV_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
if [ -n "${CUDA_COMPAT:-}" ] && [ -d "$CUDA_COMPAT" ] && [ -n "$DRV_MAJOR" ] \
   && [ "$DRV_MAJOR" -lt 570 ] 2>/dev/null; then
  export LD_LIBRARY_PATH="$CUDA_COMPAT:${LD_LIBRARY_PATH:-}"
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
GPUS="${1:-0,1}"
PORT="${2:-8000}"
MODEL="${3:-Qwen/Qwen3.5-27B}"
# leave headroom on a card that also holds a CLIP index or a second server
UTIL="${4:-0.90}"
export CUDA_VISIBLE_DEVICES="$GPUS"

NGPU=$(awk -F',' '{print NF}' <<< "$GPUS")
if [ "${PARALLEL:-tensor}" = pipeline ]; then
  TP=1; PP="$NGPU"
else
  TP="$NGPU"; PP=1
fi
SERVED_NAME="$(basename "$MODEL" | tr 'A-Z' 'a-z')"

exec vllm serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --pipeline-parallel-size "$PP" \
  --gpu-memory-utilization "$UTIL" \
  --dtype "${DTYPE:-auto}" \
  --max-model-len 32768 \
  --max-num-seqs 64 \
  --limit-mm-per-prompt "{\"image\":${IMG_LIMIT:-4}}" \
  --trust-remote-code
