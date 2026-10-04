export default function ImagePanel({ title, src, caption, placeholder = 'Nothing yet' }) {
  return (
    <figure className="card !p-3">
      <figcaption className="mb-2 flex items-baseline justify-between px-1">
        <span className="text-sm font-semibold">{title}</span>
        {caption && <span className="text-xs text-slate-500 dark:text-slate-400">{caption}</span>}
      </figcaption>
      <div className="aspect-square w-full overflow-hidden rounded-xl bg-slate-100 dark:bg-slate-800">
        {src ? (
          <img src={src} alt={title} className="h-full w-full object-contain" />
        ) : (
          <div className="flex h-full items-center justify-center px-4 text-center text-sm text-slate-400">{placeholder}</div>
        )}
      </div>
    </figure>
  )
}
