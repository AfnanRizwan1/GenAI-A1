/** Horizontal bars for probabilities / routing weights. items: [{label, value (0..1), highlight}] */
export default function Bars({ items, sort = false, ariaLabel }) {
  const rows = sort ? [...items].sort((a, b) => b.value - a.value) : items
  const top = Math.max(...rows.map((r) => r.value))
  return (
    <ul className="space-y-3" aria-label={ariaLabel}>
      {rows.map((r) => {
        const strongest = r.highlight ?? r.value === top
        return (
          <li key={r.label}>
            <div className="mb-1 flex justify-between text-sm">
              <span className={strongest ? 'font-semibold' : ''}>{r.label}</span>
              <span className="tabular-nums text-slate-600 dark:text-slate-300">{(r.value * 100).toFixed(1)}%</span>
            </div>
            <div className="h-2.5 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800">
              <div
                role="progressbar"
                aria-label={r.label}
                aria-valuenow={Math.round(r.value * 100)}
                className={`h-full rounded-full transition-all duration-500 ${strongest ? 'bg-brand-600' : 'bg-slate-400 dark:bg-slate-500'}`}
                style={{ width: `${Math.max(r.value * 100, 0.5)}%` }}
              />
            </div>
          </li>
        )
      })}
    </ul>
  )
}
