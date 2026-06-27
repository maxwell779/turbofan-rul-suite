const j = (u) => fetch(u).then(r => r.json()).catch(() => null)
export const api = {
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
