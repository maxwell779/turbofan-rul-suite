#!/usr/bin/env bash
cd /g/turbofan-rul-suite
export PYTHONIOENCODING=utf-8
PY=/g/anaconda3/python.exe
echo "RESUME start $(date)"
for k in 0 1 2 3 4 5; do cp experiments/sweep/lb_shard$k.csv experiments/sweep/lb_shard${k}_prev.csv 2>/dev/null; done
for k in 0 1 2 3 4 5; do
  $PY -m src.sweep --shard $k --of 6 --windows 30,50 --resume > experiments/sweep_resume_shard$k.log 2>&1 &
done
wait
echo "RESUME sweep done $(date) — merging"
$PY -m src.sweep --merge > experiments/sweep_merge2.log 2>&1
echo "RESUME all done $(date)"
