import { Icon } from './Icons'

export function Spinner({ className = 'h-4 w-4' }) {
  return (
    <svg className={`animate-spin ${className}`} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" className="opacity-25" />
      <path d="M4 12a8 8 0 018-8" stroke="currentColor" strokeWidth="4" strokeLinecap="round" className="opacity-90" />
    </svg>
  )
}

export function ErrorBanner({ message, onClose }) {
  if (!message) return null
  return (
    <div role="alert" className="flex animate-fade-up items-start gap-3 rounded-xl border border-bad/20 bg-bad-tint px-4 py-3 text-sm text-bad">
      <Icon name="alert" className="mt-0.5 h-5 w-5 shrink-0" />
      <span className="flex-1 text-ink"><strong className="font-semibold text-bad">Something went wrong: </strong>{message}</span>
      {onClose && (
        <button onClick={onClose} aria-label="Dismiss error" className="shrink-0 rounded p-0.5 text-bad hover:bg-bad/10">
          <Icon name="x" className="h-4 w-4" />
        </button>
      )}
    </div>
  )
}

const TONES = {
  brand: 'bg-brand-tint text-brand-on-tint',
  good: 'bg-good-tint text-good',
  info: 'bg-info-tint text-info',
  bad: 'bg-bad-tint text-bad',
  gray: 'bg-subdued text-ink-2',
}

export function Badge({ children, tone = 'brand', icon }) {
  return (
    <span className={`chip ${TONES[tone]}`}>
      {icon && <Icon name={icon} className="h-3.5 w-3.5" />}
      {children}
    </span>
  )
}

/** Eyebrow, title, one-sentence description and the gradient accent line. */
export function PageHeader({ eyebrow, title, subtitle }) {
  return (
    <header className="mb-8">
      <p className="label-caps !text-brand">{eyebrow}</p>
      <h1 className="mt-1 text-[2rem] font-extrabold leading-tight tracking-tight sm:text-4xl">{title}</h1>
      <p className="mt-2 max-w-3xl text-[15px] leading-relaxed text-ink-2">{subtitle}</p>
      <div className="accent-line mt-5" />
    </header>
  )
}

export function CardTitle({ icon, title, right }) {
  return (
    <div className="mb-4 flex flex-wrap items-center justify-between gap-3 border-b border-line pb-3">
      <h3 className="flex items-center gap-2 text-base font-bold tracking-tight">
        {icon && <Icon name={icon} className="h-5 w-5 text-brand" />}
        {title}
      </h3>
      {right}
    </div>
  )
}
