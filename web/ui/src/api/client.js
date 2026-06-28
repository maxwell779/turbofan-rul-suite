// 콘솔 API 클라이언트.
// 기본(개발/서버 모드): FastAPI 백엔드(/api/*) 호출.
// 정적 모드(VITE_STATIC=1, GitHub Pages 배포): web/static/api/*.json 스냅샷 읽기.
//   → precompute_static.py 가 생성한 파일을 사용하므로 백엔드 없이 동작.
const j = (u) => fetch(u).then(r => r.json()).catch(() => null)
const STATIC = import.meta.env.VITE_STATIC === '1'
const B = `${import.meta.env.BASE_URL}static/api` // 정적 모드 베이스 (base 경로 인식)

export const api = STATIC ? {
  runs: () => j(`${B}/runs.json`),
  sweep: () => j(`${B}/sweep.json`),
  eda: () => j(`${B}/eda.json`),
  engines: (fd) => j(`${B}/engines_${fd}.json`),
  engine: (fd, model, unit) => j(`${B}/engine_${fd}_${model}_${unit}.json`),
  drift: (fd) => j(`${B}/drift_${fd}.json`),
  xai: (fd, m) => j(`${B}/xai_${fd}_${m}.json`),
  latency: (fd, m) => j(`${B}/latency_${fd}_${m}.json`),
  conformal: (fd, m) => j(`${B}/conformal_${fd}_${m}.json`),
  error: (fd, m) => j(`${B}/error_${fd}_${m}.json`),
  foundation: () => j(`${B}/foundation.json`),
  demoModels: () => j(`${B}/demo_models.json`),
} : {
  runs: () => j('/api/runs'), sweep: () => j('/api/sweep'), eda: () => j('/api/eda'),
  engines: (fd) => j(`/api/engines?fd=${fd}`),
  engine: (fd, model, unit) => j(`/api/engine?fd=${fd}&model=${model}&unit=${unit}`),
  drift: (fd) => j(`/api/drift?fd=${fd}`),
  xai: (fd, m) => j(`/api/xai?fd=${fd}&model=${m}`),
  latency: (fd, m) => j(`/api/latency?fd=${fd}&model=${m}`),
  conformal: (fd, m) => j(`/api/conformal?fd=${fd}&model=${m}`),
  error: (fd, m) => j(`/api/error?fd=${fd}&model=${m}`),
  foundation: () => j('/api/foundation'),
  demoModels: () => j('/api/demo_models'),
}
