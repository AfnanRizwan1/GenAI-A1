const TYPE_NAMES = { clean: 'none', salt: 'salt-and-pepper', blur: 'gaussian blur', occlusion: 'occlusion' }
const LABELS = { probability: 'Corrupted pixels', kernel_size: 'Kernel size', sigma: 'Sigma', area_covered: 'Area covered' }

function Row({ k, v }) {
  return (
    <div className="flex justify-between gap-4 border-b border-slate-100 py-1.5 text-sm last:border-0 dark:border-slate-800">
      <dt className="text-slate-500 dark:text-slate-400">{k}</dt>
      <dd className="text-right font-medium tabular-nums">{v}</dd>
    </div>
  )
}

/** Corruption settings, timing and quality metrics of the last run. */
export default function ResultDetails({ result }) {
  const s = result.settings
  const m = result.metrics
  const fmt = (v) => (typeof v === 'number' && !Number.isInteger(v) ? Number(v.toFixed(3)) : v)
  return (
    <div className="card" data-testid="details">
      <h3 className="mb-3 text-sm font-semibold">Run details</h3>
      <dl>
        <Row k="Corruption" v={TYPE_NAMES[s.type] ?? s.type} />
        {s.severity && <Row k="Severity" v={s.severity} />}
        {Object.keys(LABELS).filter((k) => s[k] !== undefined).map((k) => <Row key={k} k={LABELS[k]} v={fmt(s[k])} />)}
        {s.rectangles && <Row k="Rectangles" v={s.rectangles.length} />}
        <Row k="Seed" v={s.seed} />
        <Row k="Inference time" v={`${result.inference_ms.toFixed(1)} ms`} />
        {result.classifier_ms !== undefined && <Row k="Classifier / expert" v={`${result.classifier_ms.toFixed(1)} / ${result.expert_ms.toFixed(1)} ms`} />}
        {m && (
          <>
            <Row k="PSNR input → restored" v={`${m.input_psnr} → ${m.restored_psnr} dB`} />
            <Row k="SSIM input → restored" v={`${m.input_ssim} → ${m.restored_ssim}`} />
          </>
        )}
      </dl>
      {!m && <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">No reference image (the upload was used as given), so no quality metrics.</p>}
    </div>
  )
}
