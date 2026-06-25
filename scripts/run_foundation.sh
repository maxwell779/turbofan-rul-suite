#!/usr/bin/env bash
cd /g/turbofan-rul-suite
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_PROGRESS_BARS=1 CUDA_VISIBLE_DEVICES=""
echo "foundation(CPU) start $(date)"
/g/anaconda3/python.exe -m src.foundation --fd all --max_train 3000
echo "foundation done $(date)"
