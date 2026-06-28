import React, { useEffect, useState } from 'react';
import { Bars, LineChart, Donut, Spark } from './charts.jsx';
import { api } from './api/client.js';

const FDS = ['FD001', 'FD002', 'FD003', 'FD004'];
const TONE = { FD001: '#3b82f6', FD002: '#f59e0b', FD003: '#22c55e', FD004: '#ef4444' };
const Card = ({ title, sub, children, span }) => (
  <div className="card" style={span ? { gridColumn: `span ${span}` } : undefined}>
    <div className="card-h sm"><div><div className="card-title">{title}</div>{sub && <div className="card-sub">{sub}</div>}</div></div>
    {children}
  </div>
);

// conformal 신뢰구간 엔진 뷰어 (SVG)
function BandChart({ cyc, pred, tru, q, alert = 30, h = 300 }) {
  if (!cyc || !cyc.length) return null;
  const w = 760, pl = 46, pr = 16, pt = 14, pb = 30;
  const xs = cyc, ally = [...pred, ...tru, ...(q ? pred.map(p => p + q) : [])];
  const y1 = Math.max(...ally, alert) * 1.05, y0 = 0;
  const X = x => pl + (x - xs[0]) / ((xs[xs.length - 1] - xs[0]) || 1) * (w - pl - pr);
  const Y = y => pt + (1 - (y - y0) / ((y1 - y0) || 1)) * (h - pt - pb);
  const path = (arr) => arr.map((y, i) => `${i ? 'L' : 'M'}${X(xs[i]).toFixed(1)} ${Y(y).toFixed(1)}`).join(' ');
  const band = q ? (pred.map((p, i) => `${i ? 'L' : 'M'}${X(xs[i]).toFixed(1)} ${Y(p + q).toFixed(1)}`).join(' ')
    + ' ' + pred.map((p, i) => `L${X(xs[pred.length - 1 - i]).toFixed(1)} ${Y(Math.max(0, pred[pred.length - 1 - i] - q)).toFixed(1)}`).join(' ') + ' Z') : null;
  return (
    <svg viewBox={`0 0 ${w} ${h}`} width="100%" height={h}>
      {[0, .25, .5, .75, 1].map((f, i) => { const yy = y0 + (y1 - y0) * f; return (
        <g key={i}><line x1={pl} y1={Y(yy)} x2={w - pr} y2={Y(yy)} stroke="#eef2f7" />
          <text x={pl - 6} y={Y(yy) + 4} textAnchor="end" fontSize="10" fill="#94a3b8">{Math.round(yy)}</text></g>); })}
      <line x1={pl} y1={Y(alert)} x2={w - pr} y2={Y(alert)} stroke="#ef4444" strokeDasharray="5 4" opacity=".6" />
      <text x={w - pr} y={Y(alert) - 4} textAnchor="end" fontSize="10" fill="#ef4444">경보 임계 {alert}</text>
      {band && <path d={band} fill="#3b82f6" opacity="0.13" />}
      <path d={path(tru)} fill="none" stroke="#22c55e" strokeWidth="2.4" />
      <path d={path(pred)} fill="none" stroke="#3b82f6" strokeWidth="2.4" strokeDasharray="5 4" />
      <text x={w / 2} y={h - 6} textAnchor="middle" fontSize="10" fill="#94a3b8">사이클 →</text>
    </svg>
  );
}

export default function App() {
  const [mode, setMode] = useState('ops');
  const [sweep, setSweep] = useState(null), [runs, setRuns] = useState([]), [found, setFound] = useState(null);
  useEffect(() => { api.sweep().then(setSweep); api.runs().then(d => setRuns(d?.runs || [])); api.foundation().then(setFound); }, []);
  const bestByFd = {};
  (sweep?.rows || []).forEach(r => { if (!bestByFd[r.fd] || r.val_rmse < bestByFd[r.fd].val_rmse) bestByFd[r.fd] = r; });  // 무누수 val 기준 best
  return (
    <div className="app">
      <div className="topbar">
        <div className="brand"><div className="logo">RUL</div><div>
          <div className="t1">예지보전 RUL Console <span style={{ color: '#3b82f6', fontSize: 11, letterSpacing: 2 }}>PdM</span></div>
          <div className="t2">NASA C-MAPSS 터보팬 · 잔여수명 예측 · 무누수·conformal·드리프트·XAI</div></div></div>
        <div style={{ display: 'flex', gap: 8 }}>
          {['ops', 'analysis'].map(m => <button key={m} onClick={() => setMode(m)}
            className="navbtn" style={{ fontWeight: 700, background: mode === m ? '#0f172a' : '#f1f5f9', color: mode === m ? '#fff' : '#334155', border: 'none', borderRadius: 8, padding: '8px 16px' }}>
            {m === 'ops' ? '운영' : '분석'}</button>)}
        </div>
      </div>
      <main className="main">
        {mode === 'ops' ? <Ops bestByFd={bestByFd} runs={runs} /> : <Analysis sweep={sweep} found={found} />}
      </main>
      <footer className="footer dim sm">PdM-RUL · C-MAPSS · 모델 8종 무누수 그리드서치 · PatchTST/조건정규화 · ONNX·conformal·XAI</footer>
    </div>
  );
}

function Ops({ bestByFd, runs }) {
  return (<>
    <section className="kpis">
      {FDS.map(fd => { const b = bestByFd[fd];
        return <div className="kpi" key={fd}><div className="bar" style={{ position: 'absolute', top: 0, left: 0, right: 0, height: 4, background: TONE[fd] }} />
          <div className="k">{fd} best RMSE</div><div className="v">{b ? b.test_rmse.toFixed(2) : '…'}</div>
          <div className="s">{b ? `${b.model} · NASA ${Math.round(b.nasa)}` : '집계중'}</div></div>; })}
    </section>
    <EngineViewer />
  </>);
}

function EngineViewer() {
  const [demo, setDemo] = useState([]);
  const [fd, setFd] = useState('FD001');
  const [model, setModel] = useState('tcn');
  const [engines, setEngines] = useState([]); const [unit, setUnit] = useState(null);
  const [data, setData] = useState(null); const [q, setQ] = useState(null);
  useEffect(() => { api.demoModels().then(d => setDemo((d?.models || []).map(s => ({ fd: s.split('_')[0], model: s.split('_').slice(1).join('_') })))); }, []);
  const models = demo.filter(d => d.fd === fd).map(d => d.model);
  useEffect(() => { if (models.length && !models.includes(model)) setModel(models[0]); }, [fd, demo]);  // fd별 가용 best 모델로
  useEffect(() => { api.engines(fd).then(d => { const es = (d?.engines || []).sort((a, b) => a.true_rul - b.true_rul); setEngines(es); if (es[0]) setUnit(es[0].unit); }); }, [fd]);
  useEffect(() => { if (unit != null) api.engine(fd, model, unit).then(setData); api.conformal(fd, model).then(c => setQ(c?.interval_halfwidth_q || null)); }, [fd, model, unit]);
  return (
    <Card title="엔진 RUL 뷰어 — conformal 신뢰구간" sub="예측(점선)·실제(초록)·90% 구간(음영)·정비 경보 임계">
      <div className="seg-nav" style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
        <select value={fd} onChange={e => setFd(e.target.value)}>{FDS.map(f => <option key={f}>{f}</option>)}</select>
        <select value={model} onChange={e => setModel(e.target.value)}>{(models.length ? models : ['lstm']).map(m => <option key={m}>{m}</option>)}</select>
        <select value={unit || ''} onChange={e => setUnit(+e.target.value)}>{engines.map(e => <option key={e.unit} value={e.unit}>엔진 {e.unit} (실제RUL {e.true_rul})</option>)}</select>
      </div>
      {data && <BandChart cyc={data.cycle} pred={data.pred_rul} tru={data.true_rul} q={q} />}
      <div className="legend"><span><i style={{ background: '#22c55e' }} />실제 RUL</span><span><i style={{ background: '#3b82f6' }} />예측 RUL</span>{q && <span><i style={{ background: '#3b82f6', opacity: .3 }} />90% conformal 구간(±{q.toFixed(0)})</span>}</div>
      {data && <div className="sensors" style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 10, marginTop: 10 }}>
        {Object.entries(data.sensors || {}).map(([k, v]) => <div key={k} className="sc" style={{ background: '#f8fafc', border: '1px solid rgba(15,23,42,.06)', borderRadius: 10, padding: 8 }}>
          <div className="lab" style={{ fontSize: 11, color: '#64748b' }}>센서 {k}</div><Spark data={v} color="#f59e0b" h={56} /></div>)}
      </div>}
    </Card>
  );
}

function Analysis({ sweep, found }) {
  const [fd, setFd] = useState('FD001');
  const [drift, setDrift] = useState(null), [xai, setXai] = useState(null), [lat, setLat] = useState(null), [err, setErr] = useState(null), [conf, setConf] = useState(null);
  const [demo, setDemo] = useState([]);
  useEffect(() => { api.demoModels().then(d => setDemo((d?.models || []).map(s => ({ fd: s.split('_')[0], model: s.split('_').slice(1).join('_') })))); }, []);
  const model = (demo.find(d => d.fd === fd)?.model) || 'tcn';  // fd별 실제 보유 데모 모델로 MLOps 표시
  useEffect(() => { api.drift(fd).then(setDrift); api.xai(fd, model).then(setXai); api.latency(fd, model).then(setLat); api.error(fd, model).then(setErr); api.conformal(fd, model).then(setConf); }, [fd, model]);
  const lb = {}; (sweep?.rows || []).forEach(r => { const k = `${r.fd}|${r.model}`; if (!lb[k] || r.test_rmse < lb[k]) lb[k] = r.test_rmse; });
  const models = ['cnn', 'lstm', 'gru', 'bilstm', 'cnnlstm', 'tcn', 'dlinear', 'patchtst'];
  const fdBest = (f, m) => { let v = null; (sweep?.rows || []).forEach(r => { if (r.fd === f && r.model === m && (v == null || r.test_rmse < v)) v = r.test_rmse; }); return v; };
  return (
    <div className="chart-grid">
      <Card title="모델 리더보드 (test RMSE, 모델별 best)" sub="무누수 val 선택 · FD×모델" span={2}>
        <Bars labels={models} series={FDS.map(f => ({ name: f, color: TONE[f], data: models.map(m => fdBest(f, m)) }))} yMin={10} yMax={45} fmt={v => v ? v.toFixed(1) : ''} height={240} />
      </Card>
      {found && <Card title="파운데이션: frozen vs LoRA" sub="MOMENT 임베딩 + head (낮을수록 좋음)">
        <Bars labels={FDS} series={[
          { name: 'frozen+Ridge', color: '#94a3b8', data: FDS.map(f => found.frozen?.[f]?.full_ridge?.rmse) },
          { name: 'LoRA', color: '#a855f7', data: FDS.map(f => found.lora?.[f]?.full?.rmse) }]} yMin={15} yMax={40} fmt={v => v ? v.toFixed(0) : ''} height={220} />
      </Card>}
      <div className="card"><div className="card-h sm"><div><div className="card-title">MLOps 패널</div><div className="card-sub">{`${fd} · best 모델 ${model}`}</div></div>
        <select value={fd} onChange={e => setFd(e.target.value)} style={{ marginLeft: 'auto' }}>{FDS.map(f => <option key={f}>{f}</option>)}</select></div>
        {conf?.empirical_coverage != null && <div className="muted" style={{ fontSize: 13, marginBottom: 8 }}>📐 Conformal: 목표 {(conf.target_coverage * 100)}% → <b>실측 {(conf.empirical_coverage * 100).toFixed(1)}%</b> (±{conf.interval_halfwidth_q} 사이클)</div>}
        {err?.by_rul_bucket && <div style={{ fontSize: 13, marginBottom: 8 }}>🎯 RUL구간 RMSE: {err.by_rul_bucket.map(b => `${b.range} ${b.rmse}`).join(' · ')} | 늦은예측 {Math.round(err.safety.late_ratio * 100)}%</div>}
        {lat?.latency_ms_cpu_bs1 && <div style={{ fontSize: 13 }}>⚡ 지연 p50(ms): PyTorch {lat.latency_ms_cpu_bs1.pytorch.p50} → ONNX <b>{lat.latency_ms_cpu_bs1.onnx.p50}</b> · int8 {lat.latency_ms_cpu_bs1.onnx_int8.p50}</div>}
      </div>
      {xai?.sensor_importance && <Card title="XAI — 센서 기여도 (IG)" sub={`${fd} · RUL 예측 주도 센서`}>
        <Bars labels={xai.sensor_importance.slice(0, 8).map(s => s.sensor)} series={[{ name: '기여', color: '#a855f7', data: xai.sensor_importance.slice(0, 8).map(s => s.importance) }]} fmt={v => v.toFixed(3)} height={200} />
      </Card>}
      {drift?.sensors && <Card title="드리프트 모니터 (PSI)" sub={`${fd} · train↔test · dataset_drift=${drift.dataset_drift}`}>
        <Bars labels={drift.sensors.slice(0, 8).map(s => s.sensor)} series={[{ name: 'PSI', color: '#ef4444', data: drift.sensors.slice(0, 8).map(s => s.psi) }]} fmt={v => v.toFixed(2)} height={200} />
      </Card>}
    </div>
  );
}
