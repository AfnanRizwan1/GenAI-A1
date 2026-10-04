import { useState } from 'react'
import { HashRouter, NavLink, Navigate, Route, Routes, useLocation } from 'react-router-dom'
import { Icon } from './components/Icons'
import { useDarkMode, useHealth } from './hooks'
import { HardPage, SoftPage, UniversalPage } from './pages/RestorationPages'
import SketchPage from './pages/SketchPage'

const NAV = [
  { to: '/universal', label: 'Universal Restoration', task: 'Task 1', icon: 'sliders', chip: 'Autoencoder restoration pipeline' },
  { to: '/hard', label: 'Hard-Routed Restoration', task: 'Task 2', icon: 'branch', chip: 'Classifier-routed specialists' },
  { to: '/soft', label: 'Soft Mixture-of-Experts', task: 'Task 3', icon: 'layers', chip: 'Jointly trained gate and experts' },
  { to: '/sketch', label: 'Face-to-Sketch Generator', task: 'Task 4', icon: 'pencil', chip: 'Conditional GAN · 3 styles' },
]

function StatusChips({ health }) {
  let tone = 'bg-subdued text-ink-3'
  let dot = 'bg-ink-3'
  let text = 'Checking backend…'
  if (health?.status === 'down') { tone = 'bg-bad-tint text-bad'; dot = 'bg-bad'; text = 'Backend offline' }
  else if (health?.status === 'ok') {
    const loaded = Object.values(health.models).filter((m) => m.loaded).length
    const total = Object.keys(health.models).length
    const all = loaded === total
    tone = all ? 'bg-good-tint text-good' : 'bg-warn-tint text-warn'
    dot = all ? 'bg-good' : 'bg-warn'
    text = all ? `Backend online · ${total} models loaded` : `Backend online · ${loaded} of ${total} models loaded`
  }
  return (
    <div className="space-y-2" data-testid="status">
      <span className={`chip w-full !justify-start ${tone}`}><span className={`h-1.5 w-1.5 rounded-full ${dot} ${health?.status === 'ok' ? 'animate-pulse-dot' : ''}`} />{text}</span>
      <span className="chip w-full !justify-start bg-subdued font-mono !text-[11px] text-ink-2"><Icon name="cpu" className="h-3.5 w-3.5" />ONNX Runtime · CPU</span>
    </div>
  )
}

function Shell() {
  const [open, setOpen] = useState(false)
  const [dark, setDark] = useDarkMode()
  const health = useHealth()
  const { pathname } = useLocation()
  const current = NAV.find((n) => pathname.startsWith(n.to)) ?? NAV[0]

  const sidebar = (
    <nav className="flex h-full flex-col p-5" aria-label="Workspaces">
      <div className="mb-8 flex items-center gap-3 px-1">
        <div className="flex h-10 w-10 items-center justify-center rounded-md bg-brand text-white shadow-glow"><Icon name="image" className="h-5 w-5" /></div>
        <div className="min-w-0">
          <p className="font-display text-[15px] font-extrabold leading-tight tracking-tight">Restoration Studio</p>
          <p className="whitespace-nowrap text-[11px] text-ink-3">Generative AI · Assignment 1</p>
        </div>
      </div>
      <p className="label-caps mb-2 px-3">Workspaces</p>
      <div className="space-y-1">
        {NAV.map((n) => (
          <NavLink key={n.to} to={n.to} onClick={() => setOpen(false)}
            className={({ isActive }) => `group flex items-center gap-3 rounded-md px-3 py-2.5 text-sm font-semibold transition-colors duration-150 ${
              isActive ? 'bg-brand text-white shadow-glow' : 'text-ink-2 hover:bg-subdued hover:text-ink'}`}>
            {({ isActive }) => (
              <>
                <Icon name={n.icon} className="h-5 w-5 shrink-0" />
                <span className="min-w-0 flex-1 leading-snug">{n.label}</span>
                <span className={`text-[10px] font-medium uppercase tracking-wider ${isActive ? 'text-white/70' : 'text-ink-3'}`}>{n.task}</span>
              </>
            )}
          </NavLink>
        ))}
      </div>
      <div className="mt-auto space-y-3 border-t border-line pt-5">
        <StatusChips health={health} />
        <button className="btn-secondary w-full !h-9 text-xs" onClick={() => setDark(!dark)} aria-label="Toggle dark mode">
          <Icon name={dark ? 'sun' : 'moon'} className="h-4 w-4" />{dark ? 'Light mode' : 'Dark mode'}
        </button>
      </div>
    </nav>
  )

  return (
    <div className="min-h-screen lg:flex">
      <aside className="sticky top-0 hidden h-screen w-[280px] shrink-0 border-r border-line bg-surface lg:block">{sidebar}</aside>
      <div className="flex items-center justify-between border-b border-line bg-surface px-4 py-3 lg:hidden">
        <span className="font-display text-sm font-extrabold tracking-tight">Restoration Studio</span>
        <button className="btn-secondary !h-9" onClick={() => setOpen(!open)} aria-expanded={open} aria-label="Menu"><Icon name="menu" className="h-4 w-4" />Menu</button>
      </div>
      {open && <div className="border-b border-line bg-surface lg:hidden">{sidebar}</div>}
      <main className="min-w-0 flex-1">
        <div className="hidden items-center border-b border-line bg-surface/70 px-10 py-4 backdrop-blur-md lg:flex">
          <span className="chip bg-subdued font-mono !text-[11px] font-medium text-ink-2"><Icon name="sparkles" className="h-3.5 w-3.5 text-brand" />{current.chip}</span>
        </div>
        <div className="mx-auto max-w-[1280px] px-4 py-8 sm:px-8 lg:px-10 lg:py-10">
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
