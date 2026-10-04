import { useState } from 'react'
import { buildForm, downloadDataUrl } from '../api'
import { useRun, useSamples } from '../hooks'
import { CardTitle, ErrorBanner, PageHeader, Spinner } from '../components/Common'
import { Icon } from '../components/Icons'
import ImageInput from '../components/ImageInput'
import ImagePanel from '../components/ImagePanel'
import WebcamCapture from '../components/WebcamCapture'

// The three FS2K sketch styles: simple lines, long strokes, repeated wispy details.
const STYLES = [
  { value: 1, label: 'Style 1', hint: 'Simple lines' },
  { value: 2, label: 'Style 2', hint: 'Long strokes' },
  { value: 3, label: 'Style 3', hint: 'Wispy details' },
]

function StylePicker({ value, onChange }) {
  return (
    <div>
      <span className="label-caps mb-2 block" id="seg-style">Sketch style</span>
      <div role="radiogroup" aria-labelledby="seg-style" className="space-y-2">
        {STYLES.map((s) => {
          const on = value === s.value
          return (
            <button key={s.value} type="button" role="radio" aria-checked={on} onClick={() => onChange(s.value)}
                    className={`flex w-full items-center gap-3 rounded-md border px-4 py-3 text-left transition-colors duration-150 ${
                      on ? 'border-brand bg-brand-tint' : 'border-line bg-surface hover:border-brand/50'}`}>
              <span className={`flex h-9 w-9 items-center justify-center rounded-full ${on ? 'bg-brand text-white' : 'bg-subdued text-ink-2'}`}><Icon name="pencil" className="h-4 w-4" /></span>
              <span className="flex-1">
                <span className="block text-sm font-bold">{s.label}</span>
                <span className="block text-xs text-ink-2">{s.hint}</span>
              </span>
              <span className={`flex h-5 w-5 items-center justify-center rounded-full border-2 ${on ? 'border-brand' : 'border-line'}`}>{on && <span className="h-2.5 w-2.5 rounded-full bg-brand" />}</span>
            </button>
          )
        })}
      </div>
    </div>
  )
}

export default function SketchPage() {
  const samples = useSamples()
  const [source, setSource] = useState({ file: null, sampleId: null })
  const [style, setStyle] = useState(1)
  const { loading, error, result, run } = useRun('/sketch')
  const [hidden, setHidden] = useState(null)
  const ready = Boolean(source.file || source.sampleId)

  return (
    <div>
      <PageHeader eyebrow="Workspace 04 / Task 4" title="Face-to-Sketch Generator"
                  subtitle="A conditional GAN turns a face photograph into a pencil sketch in one of three learned styles." />
      <div className="grid gap-6 xl:grid-cols-[25rem_minmax(0,1fr)]">
        <section className="card h-fit space-y-6" aria-label="Controls">
          <CardTitle icon="sliders" title="Sketch controls" right={<span className="chip bg-subdued font-mono !text-[10px] text-ink-2">3 styles</span>} />
          <ImageInput value={source} onChange={setSource} samples={samples} sampleHint="Pets for testing" />
          <WebcamCapture onCapture={(file) => setSource({ file, sampleId: null })} />
          <StylePicker value={style} onChange={setStyle} />
          <div className="space-y-2">
            <button className="btn-primary w-full" disabled={!ready || loading} onClick={() => { setHidden(null); run(buildForm({ ...source, style })) }}>
              {loading ? <Spinner /> : <Icon name="pencil" className="h-4 w-4" />}
              {loading ? 'Generating…' : 'Generate sketch'}
            </button>
            {!ready && <p className="text-center text-xs text-ink-3">Upload a face photo or capture one with the webcam.</p>}
          </div>
        </section>

        <section className="min-w-0 space-y-6" aria-label="Results">
          <ErrorBanner message={error !== hidden ? error : null} onClose={() => setHidden(error)} />
          <div className="card !p-5">
            <CardTitle icon="image" title="Result" />
            <div className="grid gap-4 sm:grid-cols-2">
              <ImagePanel title="Original photo" src={result?.photo} label="Original" meta="128 × 128" loading={loading && !result} placeholder="Your photograph" />
              <ImagePanel title="Generated sketch" src={result?.sketch} label={result?.style ?? 'Sketch'} tone="brand" meta={result ? `${result.inference_ms.toFixed(1)} ms` : undefined}
                          loading={loading} placeholder="The sketch appears here" />
            </div>
            {result && (
              <div className="mt-5 flex flex-wrap items-center gap-3 animate-fade-up">
                <button className="btn-primary !shadow-none" onClick={() => downloadDataUrl(result.sketch, `sketch-${result.style.replace(' ', '').toLowerCase()}.png`)}>
                  <Icon name="download" className="h-4 w-4" />Download sketch
                </button>
                <span className="chip bg-subdued text-ink-2"><Icon name="pencil" className="h-3.5 w-3.5" />{result.style}</span>
                <span className="chip bg-subdued text-ink-2 num"><Icon name="timer" className="h-3.5 w-3.5" />{result.inference_ms.toFixed(1)} ms inference · {result.total_ms.toFixed(1)} ms request</span>
              </div>
            )}
          </div>
        </section>
      </div>
    </div>
  )
}
