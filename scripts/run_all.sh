#!/usr/bin/env bash
# 밤샘 학습: FD001~004 × {cnn, lstm} 풀학습 후 테스트 평가 → 리더보드
cd /g/turbofan-rul-suite
export PYTHONIOENCODING=utf-8
PY=/g/anaconda3/python.exe
echo "=== RUN_ALL start $(date) ==="
for fd in FD001 FD002 FD003 FD004; do
  for m in cnn lstm; do
    echo "----- train $fd / $m -----"
    $PY -m src.train --fd $fd --model $m --epochs 60 --batch 512
  done
done
echo "=== EVAL ALL ==="
$PY -m src.eval
echo "=== RUN_ALL done $(date) ==="
