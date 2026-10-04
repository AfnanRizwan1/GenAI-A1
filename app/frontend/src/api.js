// Thin client for the FastAPI backend. In Docker nginx proxies /api; in dev Vite proxies it to :8000.
const BASE = import.meta.env.VITE_API_BASE ?? '/api'

export class ApiError extends Error {
  constructor(message, status) {
    super(message)
    this.status = status
  }
}

async function request(path, options = {}) {
  let res
  try {
    res = await fetch(BASE + path, options)
  } catch {
    throw new ApiError('Cannot reach the backend. Is it running?', 0)
  }
  if (!res.ok) {
    let message = res.statusText || `HTTP ${res.status}`
    try {
      const j = await res.json()
      const d = j.detail
      message = typeof d === 'string' ? d : Array.isArray(d) ? d.map((e) => `${(e.loc || []).slice(-1)[0]}: ${e.msg}`).join('; ') : message
    } catch { /* keep status text */ }
    throw new ApiError(message, res.status)
  }
  return res.json()
}

export const getHealth = () => request('/health')
export const getSamples = () => request('/samples')
export const sampleUrl = (id) => `${BASE}/samples/${id}`

export const DEFAULT_CUSTOM = { salt_p: 0.08, blur_kernel: 5, blur_sigma: 1.5, occ_coverage: 0.2, occ_rects: 2 }

/** Multipart body for the restoration endpoints and the sketch endpoint. */
export function buildForm({ file, sampleId, corruption, severity, seed, custom, style }) {
  const f = new FormData()
  if (file) f.append('file', file, file.name || 'image.png')
  else if (sampleId) f.append('sample_id', sampleId)
  if (style !== undefined) f.append('style', String(style))
  if (corruption !== undefined) {
    f.append('corruption', corruption)
    f.append('severity', severity)
    if (seed !== '' && seed !== undefined && seed !== null) f.append('seed', String(seed))
    if (severity === 'custom') for (const [k, v] of Object.entries({ ...DEFAULT_CUSTOM, ...custom })) f.append(k, String(v))
  }
  return f
}

export const postForm = (endpoint, form) => request(endpoint, { method: 'POST', body: form })

export function downloadDataUrl(url, filename) {
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
}
