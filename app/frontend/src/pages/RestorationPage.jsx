import { useState } from 'react'
import { buildForm, DEFAULT_CUSTOM, downloadDataUrl } from '../api'
import { useRun, useSamples } from '../hooks'
import { CardTitle, ErrorBanner, PageHeader, Spinner } from '../components/Common'
import CompareSlider from '../components/CompareSlider'
import CorruptionControls from '../components/CorruptionControls'
import { Icon } from '../components/Icons'
import ImageInput from '../components/ImageInput'
import ImagePanel from '../components/ImagePanel'
import ResultDetails, { MetricCards } from '../components/ResultDetails'
import Segmented from '../components/Segmented'

const stagger = (i) => ({ animationDelay: `${i * 60}ms` })

/** Shared layout of the three restoration workspaces; `extra(result)` renders the task-specific card. */
export default function RestorationPage({ number, title, subtitle, endpoint, extra, filePrefix, sampleHint, controlsTitle = 'Restoration controls', controlsChip }) {
  const samples = useSamples()
  const [source, setSource] = useState({ file: null, sampleId: null })
  const [ctl, setCtl] = useState({ corruption: 'salt', severity: 'medium', seed: '', custom: DEFAULT_CUSTOM })
  const [view, setView] = useState('side')
  const [copied, setCopied] = useState(false)
  const { loading, error, result, run } = useRun(endpoint)
  const [hidden, setHidden] = useState(null)
  const ready = Boolean(source.file || source.sampleId)

  const submit = () => {
    setHidden(null)
    run(buildForm({ ...source, ...ctl }))
  }

  const copyJson = async () => {
    const payload = { workspace: title, settings: result.settings, inference_ms: result.inference_ms, metrics: result.metrics,
      ...(result.probabilities && { probabilities: result.probabilities, predicted_class: result.predicted_class, selected_expert: result.selected_expert }),
      ...(result.weights && { weights: result.weights, routing_entropy: result.routing_entropy }) }
    try {
      await navigator.clipboard.writeText(JSON.stringify(payload, null, 2))
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch { /* clipboard unavailable */ }
  }

  const views = [{ value: 'side', label: 'Side by side' }, { value: 'compare', label: 'Compare' },
    ...(result?.difference ? [{ value: 'diff', label: 'Difference map' }] : [])]
  const active = views.some((v) => v.value === view) ? view : 'side'
  const applied = result?.original != null
  const caption = result && (applied ? `${result.settings.type === 'clean' ? 'No corruption' : result.settings.type}${result.settings.severity ? `, ${result.settings.severity}` : ''}` : 'Used as uploaded')

  return (
    <div>
      <PageHeader eyebrow={`Workspace 0${number} / Task ${number}`} title={title} subtitle={subtitle} />
      <div className="grid gap-6 xl:grid-cols-[25rem_minmax(0,1fr)]">
        <section className="card h-fit space-y-6" aria-label="Controls">
          <CardTitle icon="sliders" title={controlsTitle} right={controlsChip && <span className="chip bg-subdued font-mono !text-[10px] text-ink-2">{controlsChip}</span>} />
          <ImageInput value={source} onChange={setSource} samples={samples} sampleHint={sampleHint} />
          <CorruptionControls value={ctl} onChange={setCtl} />
          <div className="space-y-2">
            <button className="btn-primary w-full" disabled={!ready || loading} onClick={submit}>
              {loading ? <Spinner /> : <Icon name="sparkles" className="h-4 w-4" />}
              {loading ? 'Restoring…' : ctl.corruption === 'none' ? 'Restore' : 'Apply corruption and restore'}
            </button>
            {!ready && <p className="text-center text-xs text-ink-3">Choose an image or a sample to begin.</p>}
          </div>
        </section>

        <section className="min-w-0 space-y-6" aria-label="Results">
          <ErrorBanner message={error !== hidden ? error : null} onClose={() => setHidden(error)} />
          <div className="card !p-5">
            <CardTitle icon="image" title="Visual inspection"
                       right={result && <div className="w-full sm:w-[22rem]"><Segmented name="view" options={views} value={active} onChange={setView} /></div>} />
            {active === 'compare' && result ? (
              <CompareSlider before={result.input} after={result.restored} />
            ) : (
              <div className="grid gap-4 sm:grid-cols-3">
                {active === 'diff' ? (
                  <>
                    <ImagePanel style={stagger(0)} title="Input to model" src={result?.input} label="Corrupted input" tone="bad" meta="128 × 128" />
                    <ImagePanel style={stagger(1)} title="Restored output" src={result?.restored} label="Restored" tone="good" meta={`${result?.inference_ms.toFixed(1)} ms`} />
                    <ImagePanel style={stagger(2)} title="Difference map" src={result?.difference} label="|restored − original|" tone="brand"
                                caption={`Brighter means a larger error (full scale ${result?.difference_full_scale}).`} />
                  </>
                ) : (
                  <>
                    <ImagePanel index={1} style={stagger(0)} title="Original" src={result?.original} label="Original" meta="128 × 128 px · 3 channels" loading={loading && !result}
                                caption={result ? (applied ? 'Clean ground truth' : 'No clean reference for this upload') : undefined}
                                placeholder={result ? 'No reference: the upload was used as given' : 'Your clean image'} />
                    <ImagePanel index={2} style={stagger(1)} title="Input to model" src={result?.input} label={applied ? 'Corrupted input' : 'Input'} meta="what the model sees" tone={applied ? 'bad' : 'neutral'}
                                loading={loading && !result} caption={caption} placeholder="The corrupted input" />
                    <ImagePanel index={3} style={stagger(2)} title="Restored output" src={result?.restored} label="Restored" tone="good" meta={result ? `${result.inference_ms.toFixed(1)} ms inference` : undefined}
                                loading={loading} caption={result ? (applied ? 'Restored from the corrupted input' : 'Restored from the uploaded image') : undefined} placeholder="The restoration" />
                  </>
                )}
              </div>
            )}
            {result && (
              <div className="mt-5 flex flex-wrap gap-3">
                <button className="btn-primary !shadow-none" onClick={() => downloadDataUrl(result.restored, `${filePrefix}-restored.png`)}><Icon name="download" className="h-4 w-4" />Download restored image</button>
                <button className="btn-secondary" onClick={() => downloadDataUrl(result.input, `${filePrefix}-input.png`)}><Icon name="download" className="h-4 w-4" />Download input</button>
                <button className="btn-secondary" onClick={copyJson}><Icon name={copied ? 'check' : 'braces'} className="h-4 w-4" />{copied ? 'Copied' : 'Copy metrics as JSON'}</button>
              </div>
            )}
          </div>

          {result && (
            <>
              {extra && extra(result)}
              <MetricCards result={result} />
              <ResultDetails result={result} />
            </>
          )}
        </section>
      </div>
    </div>
  )
}
