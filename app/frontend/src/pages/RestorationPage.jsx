import { useState } from 'react'
import { buildForm, DEFAULT_CUSTOM, downloadDataUrl } from '../api'
import { useRun, useSamples } from '../hooks'
import { ErrorBanner, PageHeader, Spinner } from '../components/Common'
import CorruptionControls from '../components/CorruptionControls'
import ImageInput from '../components/ImageInput'
import ImagePanel from '../components/ImagePanel'
import ResultDetails from '../components/ResultDetails'

/** Shared layout of the three restoration workspaces; `extra(result)` renders the task-specific card. */
export default function RestorationPage({ title, subtitle, endpoint, extra, filePrefix }) {
  const samples = useSamples()
  const [source, setSource] = useState({ file: null, sampleId: null })
  const [ctl, setCtl] = useState({ corruption: 'salt', severity: 'medium', seed: '', custom: DEFAULT_CUSTOM })
  const { loading, error, result, run } = useRun(endpoint)
  const [hidden, setHidden] = useState(null)
  const ready = Boolean(source.file || source.sampleId)

  const submit = () => {
    setHidden(null)
    run(buildForm({ ...source, ...ctl }))
  }

  return (
    <div>
      <PageHeader title={title} subtitle={subtitle} />
      <div className="grid gap-6 xl:grid-cols-[24rem_minmax(0,1fr)]">
        <section className="card h-fit space-y-6" aria-label="Controls">
          <ImageInput value={source} onChange={setSource} samples={samples} />
          <CorruptionControls value={ctl} onChange={setCtl} />
          <button className="btn-primary w-full" disabled={!ready || loading} onClick={submit}>
            {loading && <Spinner />}
            {loading ? 'Restoring…' : ctl.corruption === 'none' ? 'Restore' : 'Apply corruption and restore'}
          </button>
          {!ready && <p className="-mt-3 text-xs text-slate-500 dark:text-slate-400">Choose an image or a sample to begin.</p>}
        </section>

        <section className="space-y-5" aria-label="Results">
          <ErrorBanner message={error !== hidden ? error : null} onClose={() => setHidden(error)} />
          <div className="grid gap-4 sm:grid-cols-3">
            <ImagePanel title="Original" src={result?.original} caption="clean reference" placeholder={result ? 'No reference: the upload was used as given' : 'Your clean image'} />
            <ImagePanel title="Input to model" src={result?.input} caption="what the model sees" placeholder="The corrupted input" />
            <ImagePanel title="Restored output" src={result?.restored} caption={result ? `${result.inference_ms.toFixed(1)} ms` : ''} placeholder="The restoration" />
          </div>
          {result && (
            <>
              <div className="flex flex-wrap gap-3">
                <button className="btn-secondary" onClick={() => downloadDataUrl(result.restored, `${filePrefix}-restored.png`)}>⬇ Download restored image</button>
                <button className="btn-secondary" onClick={() => downloadDataUrl(result.input, `${filePrefix}-input.png`)}>⬇ Download input</button>
              </div>
              <div className="grid gap-4 lg:grid-cols-2">
                {extra && extra(result)}
                <ResultDetails result={result} />
              </div>
            </>
          )}
        </section>
      </div>
    </div>
  )
}
