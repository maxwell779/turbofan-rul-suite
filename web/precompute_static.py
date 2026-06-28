"""정적 배포(GitHub Pages)용 API 스냅샷 생성기.

server.py 의 라우트 함수를 직접 호출해 결과를 web/static/api/*.json 으로 덤프한다.
콘솔 UI(client.js)는 VITE_STATIC=1 빌드 시 /api/* 대신 이 파일들을 읽는다.

- 파일읽기/경량 엔드포인트(runs·sweep·eda·foundation·demo_models·drift·xai·latency·
  conformal·error·engines): 전량 스냅샷.
- engine(추론): demo 모델별로 대표 유닛 N개만 큐레이션해 추론·저장(파일 수·용량 관리).

사용: turbofan-rul-suite 루트에서
  python web/precompute_static.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web"))
sys.path.insert(0, str(ROOT))
import server as S  # noqa: E402

OUT = ROOT / "web" / "static" / "api"
OUT.mkdir(parents=True, exist_ok=True)
N_UNITS = 12  # demo 모델당 큐레이션 유닛 수


def unwrap(x):
    """plain dict 또는 JSONResponse 모두 dict 로."""
    body = getattr(x, "body", None)
    if body is not None:
        return json.loads(body)
    return x


def dump(name, obj):
    p = OUT / f"{name}.json"
    json.dump(obj, open(p, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"  wrote {p.relative_to(ROOT)} ({p.stat().st_size//1024} KB)")


def main():
    fds = list(S.SUBSETS)
    # 1) 파라미터 없는 엔드포인트
    dump("runs", unwrap(S.api_runs()))
    dump("sweep", unwrap(S.api_sweep()))
    dump("eda", unwrap(S.api_eda()))
    dump("foundation", unwrap(S.api_foundation()))
    dm = unwrap(S.api_demo_models())
    dump("demo_models", dm)

    # 2) fd별: drift
    for fd in fds:
        dump(f"drift_{fd}", unwrap(S.api_drift(fd)))

    # 3) demo (fd,model) 쌍별: xai·latency·conformal·error
    pairs = []
    for name in dm.get("models", []):
        fd, model = name.split("_", 1)
        pairs.append((fd, model))
    for fd, model in pairs:
        dump(f"xai_{fd}_{model}", unwrap(S.api_xai(fd, model)))
        dump(f"latency_{fd}_{model}", unwrap(S.api_latency(fd, model)))
        dump(f"conformal_{fd}_{model}", unwrap(S.api_conformal(fd, model)))
        dump(f"error_{fd}_{model}", unwrap(S.api_error(fd, model)))

    # 4) engines(fd): 큐레이션 유닛 목록(전체에서 균등 N개) — 정적 모드 UI는 이 유닛만 제공
    fd_units = {}
    for fd, model in pairs:
        full = unwrap(S.engines(fd)).get("engines", [])
        if not full:
            fd_units[fd] = []
            dump(f"engines_{fd}", {"engines": []}); continue
        step = max(1, len(full) // N_UNITS)
        picked = full[::step][:N_UNITS]
        fd_units[fd] = [e["unit"] for e in picked]
        dump(f"engines_{fd}", {"engines": picked})

    # 5) engine(추론): demo 모델 × 큐레이션 유닛
    for fd, model in pairs:
        ok = 0
        for unit in fd_units.get(fd, []):
            res = unwrap(S.engine(fd, model, unit))
            if isinstance(res, dict) and "error" not in res:
                dump(f"engine_{fd}_{model}_{unit}", res); ok += 1
        print(f"  engine {fd}_{model}: {ok} units")

    print("DONE")


if __name__ == "__main__":
    main()
