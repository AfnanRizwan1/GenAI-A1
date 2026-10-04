import { useCallback, useRef, useState } from 'react'
import { Icon } from './Icons'

/** Before / after slider: `before` is shown left of the divider, `after` right of it. Drag or use the arrow keys. */
export default function CompareSlider({ before, after, beforeLabel = 'Input', afterLabel = 'Restored' }) {
  const ref = useRef(null)
  const [pos, setPos] = useState(50)

  const move = useCallback((clientX) => {
    const r = ref.current.getBoundingClientRect()
    setPos(Math.min(100, Math.max(0, ((clientX - r.left) / r.width) * 100)))
  }, [])

  return (
    <div className="card-tight animate-fade-up">
      <div ref={ref} className="relative mx-auto aspect-square w-full max-w-xl touch-none select-none overflow-hidden rounded-md bg-stage"
           onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); move(e.clientX) }}
           onPointerMove={(e) => e.buttons && move(e.clientX)}>
        <img src={after} alt={afterLabel} className="absolute inset-0 h-full w-full object-contain" draggable={false} />
        <img src={before} alt={beforeLabel} className="absolute inset-0 h-full w-full object-contain" draggable={false}
             style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }} />
        <span className="pill-label absolute left-2.5 top-2.5">{beforeLabel}</span>
        <span className="pill-label absolute right-2.5 top-2.5">{afterLabel}</span>
        <div className="absolute inset-y-0 w-0.5 bg-white shadow-[0_0_0_1px_rgba(15,23,42,0.35)]" style={{ left: `${pos}%` }}>
          <button type="button" role="slider" aria-label="Compare input and restored image" aria-valuemin={0} aria-valuemax={100}
                  aria-valuenow={Math.round(pos)}
                  onKeyDown={(e) => {
                    if (e.key === 'ArrowLeft') setPos((p) => Math.max(0, p - 5))
                    if (e.key === 'ArrowRight') setPos((p) => Math.min(100, p + 5))
                  }}
                  className="absolute left-1/2 top-1/2 flex h-9 w-9 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full bg-white text-brand shadow-float">
            <Icon name="compare" className="h-4 w-4" />
          </button>
        </div>
      </div>
      <p className="mt-2.5 text-center text-xs text-ink-2">Drag the handle to compare the corrupted input with the restored output.</p>
    </div>
  )
}
