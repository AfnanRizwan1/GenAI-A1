import { useEffect, useRef, useState } from 'react'
import { sampleUrl } from '../api'
import { Icon } from './Icons'

/** Upload (click or drag & drop) or pick a bundled sample. Reports {file, sampleId} through onChange. */
export default function ImageInput({ value, onChange, samples = [], sampleHint = 'Cats and dogs' }) {
  const inputRef = useRef(null)
  const [drag, setDrag] = useState(false)
  const [preview, setPreview] = useState(null)

  useEffect(() => {
    if (!value.file) {
      setPreview(null)
      return undefined
    }
    const url = URL.createObjectURL(value.file)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [value.file])

  const take = (file) => file && onChange({ file, sampleId: null })
  const shown = preview || (value.sampleId ? sampleUrl(value.sampleId) : null)

  return (
    <div className="space-y-4">
      <div>
        <span className="label-caps mb-2 block">Input image</span>
        <div
          role="button" tabIndex={0} aria-label="Upload an image"
          onClick={() => inputRef.current?.click()}
          onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && inputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => { e.preventDefault(); setDrag(false); take(e.dataTransfer.files?.[0]) }}
          className={`flex cursor-pointer items-center gap-4 rounded-md border-2 border-dashed p-4 transition-colors duration-150 ${
            drag ? 'border-brand bg-brand-tint' : 'border-line bg-subdued/50 hover:border-brand'}`}
        >
          {shown ? (
            <img src={shown} alt="Selected" className="h-[72px] w-[72px] shrink-0 rounded-md object-cover" />
          ) : (
            <span className="flex h-[72px] w-[72px] shrink-0 items-center justify-center rounded-full bg-brand-tint text-brand"><Icon name="upload" className="h-6 w-6" /></span>
          )}
          <div className="min-w-0 text-sm">
            <p className="truncate font-semibold">{value.file ? value.file.name : value.sampleId ? `Sample: ${value.sampleId}` : 'Drop an image here or browse'}</p>
            <p className="mt-0.5 text-xs leading-relaxed text-ink-3">JPEG, PNG, WebP or BMP, up to 10 MB. Resized to 128×128.</p>
          </div>
          <input ref={inputRef} type="file" accept="image/jpeg,image/png,image/webp,image/bmp" className="sr-only" data-testid="file-input"
                 onChange={(e) => { take(e.target.files?.[0]); e.target.value = '' }} />
        </div>
      </div>
      {samples.length > 0 && (
        <div>
          <div className="mb-2 flex items-baseline justify-between">
            <span className="label-caps">Or pick a sample</span>
            <span className="text-[11px] text-ink-3">{sampleHint}</span>
          </div>
          <div className="grid grid-cols-4 gap-2">
            {samples.slice(0, 8).map((s) => (
              <button key={s.id} type="button" title={s.name} aria-label={`Sample ${s.name}`} aria-pressed={value.sampleId === s.id}
                      onClick={() => onChange({ file: null, sampleId: s.id })}
                      className={`aspect-square overflow-hidden rounded-md ring-2 ring-offset-2 ring-offset-surface transition duration-150 ${
                        value.sampleId === s.id ? 'ring-brand' : 'ring-transparent hover:ring-line'}`}>
                <img src={sampleUrl(s.id)} alt={s.name} className="h-full w-full object-cover" loading="lazy" />
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
