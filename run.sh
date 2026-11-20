#!/usr/bin/env bash
# Запуск в venv проекта. ctranslate2 (faster-whisper) берёт cuBLAS/cuDNN из колёс torch.
#   ./run.sh --input DIR --clean DIR --work DIR [...]   — конвейер (run.py)
#   ./run.sh scripts/pilot_stats.py ...                — любой скрипт проекта
#   VL_VENV=.venv-cpu ./run.sh ... --device cpu        — без видеокарты
DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$DIR/${VL_VENV:-.venv}"
SP="$VENV/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="$SP/cublas/lib:$SP/cudnn/lib:$SP/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_DISABLE_TELEMETRY=1 TORCH_HOME="$DIR/models/torch" PYTHONUNBUFFERED=1
cd "$DIR"
if [[ "${1:-}" == *.py ]]; then exec "$VENV/bin/python" "$@"; fi
exec "$VENV/bin/python" "$DIR/run.py" "$@"
