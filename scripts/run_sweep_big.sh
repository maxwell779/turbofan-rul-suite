#!/usr/bin/env bash
cd /g/turbofan-rul-suite
export PYTHONIOENCODING=utf-8
echo "BIG SWEEP start $(date)"
for k in 0 1 2 3 4 5; do
  /g/anaconda3/python.exe -m src.sweep --shard $k --of 6 > experiments/sweep_shard$k.log 2>&1 &
done
wait
/g/anaconda3/python.exe -m src.sweep --merge > experiments/sweep_merge.log 2>&1
echo "BIG SWEEP done $(date)"
