#!/usr/bin/env bash
# 밤샘 마스터: 축소 그리드서치(윈도 30,50) → merge → 파운데이션 LoRA(4 subset)
cd /g/turbofan-rul-suite
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_PROGRESS_BARS=1
PY=/g/anaconda3/python.exe
echo "OVERNIGHT start $(date)"
rm -f experiments/sweep/lb_shard*.csv
for k in 0 1 2 3 4 5; do
  $PY -m src.sweep --shard $k --of 6 --windows 30,50 > experiments/sweep_shard$k.log 2>&1 &
done
wait
echo "SWEEP done $(date) — merging"
$PY -m src.sweep --merge > experiments/sweep_merge.log 2>&1
echo "LoRA start $(date)"
CUDA_VISIBLE_DEVICES=0 $PY -m src.foundation_lora --fd all --epochs 15 --max_train 4000 > experiments/lora.log 2>&1
echo "OVERNIGHT done $(date)"
