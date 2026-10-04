/** Segmented button group; options are strings or {value,label}. */
export default function Segmented({ label, options, value, onChange, name }) {
  const opts = options.map((o) => (typeof o === 'string' ? { value: o, label: o } : o))
  return (
    <div>
      {label && <span className="label" id={`seg-${name}`}>{label}</span>}
      <div role="radiogroup" aria-labelledby={label ? `seg-${name}` : undefined} className="inline-flex flex-wrap gap-1 rounded-xl bg-slate-100 p-1 dark:bg-slate-800">
        {opts.map((o) => (
          <button
            key={o.value}
            type="button"
            role="radio"
            aria-checked={value === o.value}
            onClick={() => onChange(o.value)}
            className={`rounded-lg px-3 py-1.5 text-sm font-medium transition ${
              value === o.value
                ? 'bg-white text-brand-700 shadow-sm dark:bg-slate-950 dark:text-brand-100'
                : 'text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-white'
            }`}
          >
            {o.label}
          </button>
        ))}
      </div>
    </div>
  )
}
