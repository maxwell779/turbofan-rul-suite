#!/usr/bin/env bash
cd /g/turbofan-rul-suite
export PYTHONIOENCODING=utf-8
/g/anaconda3/python.exe -m src.sweep > experiments/sweep.log 2>&1
echo "SWEEP rc=$?" >> experiments/sweep.log
