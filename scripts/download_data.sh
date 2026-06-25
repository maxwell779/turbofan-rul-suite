#!/usr/bin/env bash
# C-MAPSS (NASA Turbofan) 데이터 받기 — data/ 에 12개 txt
cd "$(dirname "$0")/.."
mkdir -p data
BASE="https://raw.githubusercontent.com/LahiruJayasinghe/RUL-Net/master/CMAPSSData"
for f in train_FD001 test_FD001 RUL_FD001 train_FD002 test_FD002 RUL_FD002 \
         train_FD003 test_FD003 RUL_FD003 train_FD004 test_FD004 RUL_FD004; do
  curl -s --max-time 120 -o "data/${f}.txt" "$BASE/${f}.txt" && echo "ok ${f}.txt"
done
echo "done -> data/"
