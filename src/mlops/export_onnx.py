"""ONNX export + 지연 벤치(p50/p95/p99) + int8 동적양자화 비교.
demo 모델 사용. 출처: pytorch onnx docs, onnxruntime quantization/perf.

사용: python -m src.mlops.export_onnx [--fd FD001 --model tcn]
"""
from __future__ import annotations
import argparse, json, pickle, time
import numpy as np
import torch
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from ..model import build
from ..config import EXP

OUT = EXP / "mlops"; OUT.mkdir(parents=True, exist_ok=True)


def _bench(fn, x, warmup=20, iters=300):
    for _ in range(warmup): fn(x)
    ts = []
    for _ in range(iters):
        t = time.perf_counter(); fn(x); ts.append((time.perf_counter() - t) * 1000)
    a = np.array(ts)
    return {"p50": round(float(np.percentile(a, 50)), 4), "p95": round(float(np.percentile(a, 95)), 4),
            "p99": round(float(np.percentile(a, 99)), 4), "mean": round(float(a.mean()), 4)}


def run(fd="FD001", model="tcn"):
    d = EXP / "demo" / f"{fd}_{model}"
    b = pickle.load(open(d / "bundle.pkl", "rb")); W, F = b["window"], b["n_feat"]
    net = build(b["model"], F, W, **b["hp"]); net.load_state_dict(torch.load(d / "best.pt", map_location="cpu")); net.eval()
    dummy = torch.randn(1, W, F)
    onnx_path = d / "model.onnx"; int8_path = d / "model_int8.onnx"
    torch.onnx.export(net, dummy, str(onnx_path), input_names=["x"], output_names=["rul"],
                      dynamic_axes={"x": {0: "batch"}}, opset_version=17, dynamo=False)
    quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QInt8)

    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    sess8 = ort.InferenceSession(str(int8_path), so, providers=["CPUExecutionProvider"])
    xnp = dummy.numpy()

    # 정합성 (PyTorch vs ONNX)
    with torch.no_grad():
        y_pt = float(net(dummy).item())
    y_onnx = float(sess.run(None, {"x": xnp})[0].reshape(-1)[0])
    y_int8 = float(sess8.run(None, {"x": xnp})[0].reshape(-1)[0])

    res = {
        "fd": fd, "model": model, "window": W, "n_feat": F,
        "parity": {"pytorch": round(y_pt, 3), "onnx": round(y_onnx, 3), "int8": round(y_int8, 3),
                   "onnx_abs_err": round(abs(y_pt - y_onnx), 5), "int8_abs_err": round(abs(y_pt - y_int8), 4)},
        "size_kb": {"onnx": round((onnx_path.stat().st_size) / 1024, 1),
                    "int8": round((int8_path.stat().st_size) / 1024, 1)},
        "latency_ms_cpu_bs1": {
            "pytorch": _bench(lambda x: net(torch.tensor(x)).detach(), xnp),
            "onnx": _bench(lambda x: sess.run(None, {"x": x}), xnp),
            "onnx_int8": _bench(lambda x: sess8.run(None, {"x": x}), xnp)},
    }
    json.dump(res, open(OUT / f"latency_{fd}_{model}.json", "w"), indent=2)
    L = res["latency_ms_cpu_bs1"]
    print(f"[onnx {fd}/{model}] parity err onnx={res['parity']['onnx_abs_err']} int8={res['parity']['int8_abs_err']}", flush=True)
    print(f"  CPU bs1 p50(ms): pytorch={L['pytorch']['p50']} onnx={L['onnx']['p50']} int8={L['onnx_int8']['p50']} | int8 size {res['size_kb']['int8']}KB(onnx {res['size_kb']['onnx']}KB)", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fd", default="FD001"); ap.add_argument("--model", default="tcn")
    a = ap.parse_args(); run(a.fd, a.model)
