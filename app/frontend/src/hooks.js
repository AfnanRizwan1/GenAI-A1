import { useCallback, useEffect, useState } from 'react'
import { getHealth, getSamples, postForm } from './api'

/** Runs one backend call and tracks loading / error / result. */
export function useRun(endpoint) {
  const [state, setState] = useState({ loading: false, error: null, result: null })
  const run = useCallback(async (form) => {
    setState((s) => ({ ...s, loading: true, error: null }))
    try {
      const result = await postForm(endpoint, form)
      setState({ loading: false, error: null, result })
      return result
    } catch (e) {
      setState((s) => ({ ...s, loading: false, error: e.message }))
      return null
    }
  }, [endpoint])
  return { ...state, run }
}

export function useSamples() {
  const [samples, setSamples] = useState([])
  useEffect(() => { getSamples().then(setSamples).catch(() => setSamples([])) }, [])
  return samples
}

export function useHealth(intervalMs = 15000) {
  const [health, setHealth] = useState(null)
  useEffect(() => {
    let alive = true
    const tick = () => getHealth().then((h) => alive && setHealth(h)).catch(() => alive && setHealth({ status: 'down' }))
    tick()
    const id = setInterval(tick, intervalMs)
    return () => { alive = false; clearInterval(id) }
  }, [intervalMs])
  return health
}

export function useDarkMode() {
  const [dark, setDark] = useState(() => {
    try {
      const saved = localStorage.getItem('theme')
      if (saved) return saved === 'dark'
    } catch { /* ignore */ }
    return typeof window !== 'undefined' && window.matchMedia?.('(prefers-color-scheme: dark)').matches
  })
  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
    try { localStorage.setItem('theme', dark ? 'dark' : 'light') } catch { /* ignore */ }
  }, [dark])
  return [dark, setDark]
}
