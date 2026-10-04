import { useState } from 'react'
import { HashRouter, NavLink, Navigate, Route, Routes } from 'react-router-dom'
import { useDarkMode, useHealth } from './hooks'
import { HardPage, SoftPage, UniversalPage } from './pages/RestorationPages'
import SketchPage from './pages/SketchPage'

const NAV = [
  { to: '/universal', label: 'Universal Restoration', hint: 'Task 1', icon: '✦' },
  { to: '/hard', label: 'Hard-Routed Restoration', hint: 'Task 2', icon: '⑂' },
  { to: '/soft', label: 'Soft Mixture-of-Experts', hint: 'Task 3', icon: '◍' },
  { to: '/sketch', label: 'Face-to-Sketch Generator', hint: 'Task 4', icon: '✎' },
]

function StatusChip({ health }) {
  if (!health) return <span className="text-xs text-slate-400">Checking backend…</span>
  if (health.status !== 'ok') return <span className="flex items-center gap-2 text-xs text-red-600"><span className="h-2 w-2 rounded-full bg-red-500" />Backend offline</span>
  const loaded = Object.values(health.models).filter((m) => m.loaded).length
  const total = Object.keys(health.models).length
  const ok = loaded === total
  return (
    <span className="flex items-center gap-2 text-xs text-slate-600 dark:text-slate-300" data-testid="status">
      <span className={`h-2 w-2 rounded-full ${ok ? 'bg-emerald-500' : 'bg-amber-500'}`} />
      Backend online · {loaded}/{total} models loaded
    </span>
  )
}

function Shell() {
  const [open, setOpen] = useState(false)
  const [dark, setDark] = useDarkMode()
  const health = useHealth()

  const sidebar = (
    <nav className="flex h-full flex-col gap-1 p-4" aria-label="Workspaces">
      <div className="mb-6 flex items-center gap-3 px-2">
        <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-brand-600 text-lg font-bold text-white">R</div>
        <div>
          <p className="text-sm font-bold leading-tight">Restoration Studio</p>
          <p className="text-xs text-slate-500 dark:text-slate-400">Generative AI · Assignment 1</p>
        </div>
      </div>
      {NAV.map((n) => (
        <NavLink key={n.to} to={n.to} onClick={() => setOpen(false)}
          className={({ isActive }) => `flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition ${
            isActive ? 'bg-brand-50 text-brand-700 dark:bg-brand-600/20 dark:text-brand-100' : 'text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800'}`}>
          <span className="w-5 text-center text-base" aria-hidden="true">{n.icon}</span>
          <span className="flex-1">{n.label}</span>
          <span className="text-[10px] uppercase tracking-wide text-slate-400">{n.hint}</span>
        </NavLink>
      ))}
      <div className="mt-auto space-y-3 border-t border-slate-200 pt-4 dark:border-slate-800">
        <StatusChip health={health} />
        <button className="btn-secondary w-full !py-1.5 text-xs" onClick={() => setDark(!dark)} aria-label="Toggle dark mode">{dark ? '☀ Light mode' : '☾ Dark mode'}</button>
      </div>
    </nav>
  )

  return (
    <div className="min-h-screen lg:flex">
      <aside className="sticky top-0 hidden h-screen w-72 shrink-0 border-r border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900 lg:block">{sidebar}</aside>
      <div className="flex items-center justify-between border-b border-slate-200 bg-white px-4 py-3 dark:border-slate-800 dark:bg-slate-900 lg:hidden">
        <span className="text-sm font-bold">Restoration Studio</span>
        <button className="btn-secondary !py-1.5" onClick={() => setOpen(!open)} aria-expanded={open} aria-label="Menu">☰ Menu</button>
      </div>
      {open && <div className="border-b border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900 lg:hidden">{sidebar}</div>}
      <main className="min-w-0 flex-1 px-4 py-6 sm:px-8 lg:py-10">
        <div className="mx-auto max-w-6xl">
          <Routes>
            <Route path="/" element={<Navigate to="/universal" replace />} />
            <Route path="/universal" element={<UniversalPage />} />
            <Route path="/hard" element={<HardPage />} />
            <Route path="/soft" element={<SoftPage />} />
            <Route path="/sketch" element={<SketchPage />} />
            <Route path="*" element={<Navigate to="/universal" replace />} />
          </Routes>
        </div>
      </main>
    </div>
  )
}

export default function App() {
  return (
    <HashRouter>
      <Shell />
    </HashRouter>
  )
}
