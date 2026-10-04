import { CardTitle } from './Common'
import { Icon } from './Icons'

const TYPE_NAMES = { clean: 'None', salt: 'Salt-and-pepper', blur: 'Gaussian blur', occlusion: 'Occlusion' }
const LABELS = { probability: 'Corrupted pixels', kernel_size: 'Kernel size', sigma: 'Sigma', area_covered: 'Area covered' }
const fmt = (v) => (typeof v === 'number' && !Number.isInteger(v) ? Number(v.toFixed(3)) : v)

function Row({ k, v }) {
  return (
    <div className="flex justify-between gap-4 border-b border-line py-2 text-sm last:border-0">
      <dt className="text-ink-2">{k}</dt>
      <dd className="num text-right font-medium">{v}</dd>
    </div>
  )
}

/** Corruption settings, timing and quality metrics of the last run. */
export default function ResultDetails({ result }) {
  const s = result.settings
  return (
    <div className="card animate-fade-up" data-testid="details">
      <CardTitle icon="info" title="Run details" />
      <dl>
        <Row k="Corruption" v={TYPE_NAMES[s.type] ?? s.type} />
        {s.severity && <Row k="Severity" v={s.severity} />}
        {Object.keys(LABELS).filter((k) => s[k] !== undefined).map((k) => <Row key={k} k={LABELS[k]} v={fmt(s[k])} />)}
        {s.rectangles && <Row k="Rectangles" v={s.rectangles.length} />}
        <Row k="Seed" v={s.seed} />
        <Row k="Inference time" v={`${result.inference_ms.toFixed(1)} ms`} />
        {result.classifier_ms !== undefined && <Row k="Classifier / expert" v={`${result.classifier_ms.toFixed(1)} / ${result.expert_ms.toFixed(1)} ms`} />}
        <Row k="Input size" v={`${result.image_size} × ${result.image_size} px`} />
      </dl>
      {!result.metrics && <p className="mt-3 text-xs text-ink-3">No reference image (the upload was used as given), so no quality metrics.</p>}
    </div>
  )
}

function Gain({ value, unit = '', digits = 1 }) {
  const up = value >= 0
  return (
    <span className={`chip !px-2 !py-0.5 ${up ? 'bg-good-tint text-good' : 'bg-bad-tint text-bad'}`}>
      <Icon name="trend" className={`h-3 w-3 ${up ? '' : 'rotate-90'}`} />
      {up ? '+' : ''}{value.toFixed(digits)}{unit}
    </span>
  )
}

function Metric({ label, value, unit, foot, gain }) {
  return (
    <div className="well">
      <span className="label-caps block">{label}</span>
      <p className="mt-2 whitespace-nowrap font-mono text-[26px] font-semibold leading-none tracking-tight tabular-nums">{value}<span className="ml-1 text-sm font-medium text-ink-3">{unit}</span></p>
      <div className="mt-2 flex min-h-[1.5rem] flex-wrap items-center gap-2">
        {gain}
        {foot && <span className="num text-xs text-ink-3">{foot}</span>}
      </div>
    </div>
  )
}

/** Inference time, PSNR and SSIM (input -> restored) with improvement pills. */
export function MetricCards({ result }) {
  const m = result.metrics
  return (
    <div className="card animate-fade-up" data-testid="metrics">
      <CardTitle icon="grid" title="Performance" right={m && <span className="chip bg-good-tint text-good"><Icon name="check" className="h-3.5 w-3.5" />Measured</span>} />
      <div className="grid gap-3 sm:grid-cols-3">
        <Metric label="Inference time" value={result.inference_ms.toFixed(1)} unit="ms" foot="ONNX Runtime · CPU" />
        {m ? (
          <>
            <Metric label="PSNR" value={m.restored_psnr.toFixed(1)} unit="dB" gain={<Gain value={m.restored_psnr - m.input_psnr} unit=" dB" />} foot={`${m.input_psnr} → ${m.restored_psnr} dB`} />
            <Metric label="SSIM" value={m.restored_ssim.toFixed(3)} gain={<Gain value={m.restored_ssim - m.input_ssim} digits={3} />} foot={`${m.input_ssim} → ${m.restored_ssim}`} />
          </>
        ) : (
          <div className="well flex items-center text-xs leading-relaxed text-ink-3 sm:col-span-2">PSNR and SSIM need a clean reference. Pick a sample or upload a clean image and let the app add the corruption.</div>
        )}
      </div>
    </div>
  )
}
