/**
 * Segmented control (radio semantics). options: strings or {value,label}.
 * Single-row controls have a pill that glides to the active option; with `cols` the options wrap into a grid.
 */
export default function Segmented({ label, options, value, onChange, name, cols }) {
  const opts = options.map((o) => (typeof o === 'string' ? { value: o, label: o } : o))
  const idx = Math.max(0, opts.findIndex((o) => o.value === value))
  const glide = !cols
  const n = cols || opts.length
  return (
    <div>
      {label && <span className="label-caps mb-2 block" id={`seg-${name}`}>{label}</span>}
      <div role="radiogroup" aria-labelledby={label ? `seg-${name}` : undefined}
           className="relative grid gap-1 rounded-md bg-subdued p-1" style={{ gridTemplateColumns: `repeat(${n}, minmax(0, 1fr))` }}>
        {glide && (
          <span aria-hidden="true" className="pointer-events-none absolute inset-y-1 left-1 rounded-[0.5rem] bg-surface shadow-sm transition-transform duration-200 ease-[cubic-bezier(0.2,0,0,1)]"
                style={{ width: `calc((100% - 0.5rem) / ${n})`, transform: `translateX(${idx * 100}%)` }} />
        )}
        {opts.map((o) => {
          const active = value === o.value
          return (
            <button key={o.value} type="button" role="radio" aria-checked={active} onClick={() => onChange(o.value)}
              className={`relative z-10 whitespace-nowrap rounded-[0.5rem] px-2 py-2 text-[13px] font-semibold transition-colors duration-150 ${
                active ? `text-brand ${glide ? '' : 'bg-surface shadow-sm'}` : 'text-ink-2 hover:text-ink'}`}>
              {o.label}
            </button>
          )
        })}
      </div>
    </div>
  )
}
