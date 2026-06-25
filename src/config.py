"""C-MAPSS RUL 설정."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EXP = ROOT / "experiments"

COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
RUL_CAP = 125          # piecewise-linear 열화: 초기 평탄 구간은 125로 클립 (표준)
WINDOW = 30            # 슬라이딩 윈도 길이
VAL_FRAC = 0.2         # 엔진 단위 hold-out 비율 (무누수)
SEED = 42
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
