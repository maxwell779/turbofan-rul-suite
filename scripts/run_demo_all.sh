#!/usr/bin/env bash
cd /g/turbofan-rul-suite
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_PROGRESS_BARS=1 CUDA_VISIBLE_DEVICES=""
PY=/g/anaconda3/python.exe
for fd in FD002 FD003 FD004; do
  echo "demo $fd"; $PY -m src.mlops.demo_model 2>/dev/null; done >/dev/null 2>&1 || true
for fd in FD002 FD003 FD004; do
  $PY -c "from src.mlops.demo_model import train; train('$fd','tcn')"
  $PY -m src.analysis.error_analysis --dir experiments/demo/${fd}_tcn
  $PY -m src.mlops.conformal --dir experiments/demo/${fd}_tcn
  $PY -m src.mlops.xai --fd $fd --model tcn
  $PY -m src.mlops.export_onnx --fd $fd --model tcn
done
echo "DEMO_ALL done"
