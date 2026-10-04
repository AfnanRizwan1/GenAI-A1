import { useState } from 'react'
import { buildForm, downloadDataUrl } from '../api'
import { useRun, useSamples } from '../hooks'
import { ErrorBanner, PageHeader, Spinner } from '../components/Common'
import ImageInput from '../components/ImageInput'
import ImagePanel from '../components/ImagePanel'
import Segmented from '../components/Segmented'
import WebcamCapture from '../components/WebcamCapture'

const STYLES = [{ value: 1, label: 'Style 1' }, { value: 2, label: 'Style 2' }, { value: 3, label: 'Style 3' }]

export default function SketchPage() {
  const samples = useSamples()
  const [source, setSource] = useState({ file: null, sampleId: null })
  const [style, setStyle] = useState(1)
  const { loading, error, result, run } = useRun('/sketch')
  const [hidden, setHidden] = useState(null)
  const ready = Boolean(source.file || source.sampleId)

  return (
    <div>
      <PageHeader
        title="Face-to-Sketch Generator"
        subtitle="A conditional GAN (U-Net generator, PatchGAN discriminator) turns a face photograph into a pencil sketch in one of three learned styles."
      />
      <div className="grid gap-6 xl:grid-cols-[24rem_minmax(0,1fr)]">
        <section className="card h-fit space-y-6" aria-label="Controls">
          <ImageInput value={source} onChange={setSource} samples={samples} />
          <WebcamCapture onCapture={(file) => setSource({ file, sampleId: null })} />
          <Segmented name="style" label="Sketch style" options={STYLES} value={style} onChange={setStyle} />
          <button className="btn-primary w-full" disabled={!ready || loading} onClick={() => { setHidden(null); run(buildForm({ ...source, style })) }}>
            {loading && <Spinner />}
            {loading ? 'Generating…' : 'Generate sketch'}
          </button>
          {!ready && <p className="-mt-3 text-xs text-slate-500 dark:text-slate-400">Upload a face photo or capture one with the webcam.</p>}
        </section>

        <section className="space-y-5" aria-label="Results">
          <ErrorBanner message={error !== hidden ? error : null} onClose={() => setHidden(error)} />
          <div className="grid gap-4 sm:grid-cols-2">
            <ImagePanel title="Original photo" src={result?.photo} caption="128×128" placeholder="Your photograph" />
            <ImagePanel title="Generated sketch" src={result?.sketch} caption={result ? `${result.style} · ${result.inference_ms.toFixed(1)} ms` : ''} placeholder="The sketch appears here" />
          </div>
          {result && (
            <div className="flex flex-wrap items-center gap-3">
              <button className="btn-secondary" onClick={() => downloadDataUrl(result.sketch, `sketch-${result.style.replace(' ', '').toLowerCase()}.png`)}>⬇ Download sketch</button>
              <span className="text-sm text-slate-500 dark:text-slate-400">Inference {result.inference_ms.toFixed(1)} ms · request {result.total_ms.toFixed(1)} ms</span>
            </div>
          )}
        </section>
      </div>
    </div>
  )
}
