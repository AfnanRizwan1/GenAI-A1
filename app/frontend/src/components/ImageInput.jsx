import { useEffect, useRef, useState } from 'react'
import { sampleUrl } from '../api'

/** Upload (click or drag & drop) or pick a bundled sample. Reports {file, sampleId} through onChange. */
export default function ImageInput({ value, onChange, samples = [] }) {
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
    <div className="space-y-3">
      <span className="label">Image</span>
      <div
        role="button"
        tabIndex={0}
        aria-label="Upload an image"
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); take(e.dataTransfer.files?.[0]) }}
        className={`flex cursor-pointer items-center gap-4 rounded-xl border-2 border-dashed p-4 transition ${
          drag ? 'border-brand-500 bg-brand-50 dark:bg-brand-600/10' : 'border-slate-300 hover:border-brand-500 dark:border-slate-700'
        }`}
      >
        {shown ? (
          <img src={shown} alt="Selected" className="h-20 w-20 rounded-lg object-cover" />
        ) : (
          <div className="flex h-20 w-20 items-center justify-center rounded-lg bg-slate-100 text-2xl text-slate-400 dark:bg-slate-800">⇪</div>
        )}
        <div className="text-sm">
          <p className="font-medium">{value.file ? value.file.name : value.sampleId ? `Sample: ${value.sampleId}` : 'Drop an image here or click to upload'}</p>
          <p className="text-xs text-slate-500 dark:text-slate-400">JPEG, PNG, WebP or BMP, up to 10 MB. Resized to 128×128 for the models.</p>
        </div>
        <input ref={inputRef} type="file" accept="image/jpeg,image/png,image/webp,image/bmp" className="sr-only" data-testid="file-input"
               onChange={(e) => { take(e.target.files?.[0]); e.target.value = '' }} />
      </div>
      {samples.length > 0 && (
        <div>
          <span className="label">…or pick a clean sample</span>
          <div className="flex flex-wrap gap-2">
            {samples.map((s) => (
              <button key={s.id} type="button" title={s.name} aria-label={`Sample ${s.name}`} aria-pressed={value.sampleId === s.id}
                      onClick={() => onChange({ file: null, sampleId: s.id })}
                      className={`overflow-hidden rounded-lg ring-2 transition ${value.sampleId === s.id ? 'ring-brand-600' : 'ring-transparent hover:ring-slate-300'}`}>
                <img src={sampleUrl(s.id)} alt={s.name} className="h-14 w-14 object-cover" loading="lazy" />
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
