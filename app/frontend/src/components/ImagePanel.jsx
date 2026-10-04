const PILL = {
  neutral: 'bg-ink/70 text-white',
  bad: 'bg-bad/85 text-white',
  good: 'bg-good/90 text-white',
  brand: 'bg-brand/90 text-white',
}

/**
 * Square image stage with a floating frosted label, optional caption and a skeleton while loading.
 * Layout classes are passed through `className` so the panel can be staggered by the caller.
 */
export default function ImagePanel({ title, src, label, tone = 'neutral', meta, caption, loading, placeholder = 'Nothing yet', className = '', style }) {
  return (
    <figure className={`card-tight animate-fade-up ${className}`} style={style}>
      <figcaption className="mb-2.5 px-1">
        <span className="block text-sm font-bold tracking-tight">{title}</span>
        <span className="num block h-4 text-[11px] text-ink-3">{meta ?? ''}</span>
      </figcaption>
      <div className="relative aspect-square w-full overflow-hidden rounded-md bg-stage">
        {loading ? (
          <div className="skeleton h-full w-full" role="status" aria-label={`Loading ${title}`} />
        ) : src ? (
          <img src={src} alt={title} className="h-full w-full object-contain" />
        ) : (
          <div className="flex h-full flex-col items-center justify-center gap-2 border-2 border-dashed border-line px-4 text-center text-sm text-ink-3">
            {placeholder}
          </div>
        )}
        {label && src && !loading && (
          <span className={`pill-label absolute left-2.5 top-2.5 ${PILL[tone]}`}>{label}</span>
        )}
      </div>
      {caption && <p className="mt-2.5 px-1 text-xs text-ink-2">{caption}</p>}
    </figure>
  )
}
