/** Horizontal bars for probabilities / routing weights. items: [{label, value (0..1), highlight, hint}] */
export default function Bars({ items, sort = false, ariaLabel }) {
  const rows = sort ? [...items].sort((a, b) => b.value - a.value) : items
  const top = Math.max(...rows.map((r) => r.value))
  return (
    <ul className="space-y-3.5" aria-label={ariaLabel}>
      {rows.map((r) => {
        const strongest = r.highlight ?? r.value === top
        return (
          <li key={r.label}>
            <div className="mb-1.5 flex items-baseline justify-between gap-3 text-sm">
              <span className={strongest ? 'font-bold' : 'font-medium text-ink-2'}>{r.label}</span>
              <span className={`num ${strongest ? 'font-semibold text-brand' : 'text-ink-2'}`}>{(r.value * 100).toFixed(1)}%</span>
            </div>
            <div className="h-2.5 overflow-hidden rounded-full bg-subdued">
              <div role="progressbar" aria-label={r.label} aria-valuenow={Math.round(r.value * 100)}
                   className={`h-full rounded-full transition-all duration-500 ease-out ${strongest ? 'bg-brand' : 'bg-ink-3/70'}`}
                   style={{ width: `${Math.max(r.value * 100, 0.5)}%` }} />
            </div>
            {r.hint && <p className="mt-1 text-xs text-ink-3">{r.hint}</p>}
          </li>
        )
      })}
    </ul>
  )
}
