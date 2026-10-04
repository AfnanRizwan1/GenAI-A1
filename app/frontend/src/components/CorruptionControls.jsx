import Segmented from './Segmented'

const KINDS = [
  { value: 'none', label: 'None' },
  { value: 'salt', label: 'Salt-and-pepper' },
  { value: 'blur', label: 'Gaussian blur' },
  { value: 'occlusion', label: 'Occlusion' },
]
const SEVERITIES = [
  { value: 'low', label: 'Low' },
  { value: 'medium', label: 'Medium' },
  { value: 'high', label: 'High' },
  { value: 'custom', label: 'Custom' },
]

function Slider({ id, label, min, max, step, value, onChange, format = (v) => v }) {
  return (
    <label htmlFor={id} className="block text-sm">
      <span className="mb-1 flex justify-between"><span>{label}</span><span className="tabular-nums text-slate-500">{format(value)}</span></span>
      <input id={id} type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} className="w-full accent-brand-600" />
    </label>
  )
}

/** value: {corruption, severity, seed, custom} */
export default function CorruptionControls({ value, onChange }) {
  const set = (patch) => onChange({ ...value, ...patch })
  const setCustom = (patch) => set({ custom: { ...value.custom, ...patch } })
  const c = value.custom
  return (
    <div className="space-y-4">
      <Segmented name="kind" label="Corruption" options={KINDS} value={value.corruption} onChange={(corruption) => set({ corruption })} />
      {value.corruption === 'none' ? (
        <p className="text-xs text-slate-500 dark:text-slate-400">
          No corruption is added: the image is restored as it is. Upload an already corrupted picture to test the models on unseen images.
        </p>
      ) : (
        <>
          <Segmented name="severity" label="Severity" options={SEVERITIES} value={value.severity} onChange={(severity) => set({ severity })} />
          {value.severity === 'custom' && (
            <div className="grid gap-3 rounded-xl bg-slate-50 p-4 ring-1 ring-slate-200 dark:bg-slate-950 dark:ring-slate-800 sm:grid-cols-2">
              {value.corruption === 'salt' && <Slider id="salt_p" label="Corrupted pixels" min={0.01} max={0.3} step={0.01} value={c.salt_p} onChange={(v) => setCustom({ salt_p: v })} format={(v) => `${Math.round(v * 100)}%`} />}
              {value.corruption === 'blur' && (
                <>
                  <Slider id="blur_kernel" label="Kernel size" min={3} max={11} step={2} value={c.blur_kernel} onChange={(v) => setCustom({ blur_kernel: v })} />
                  <Slider id="blur_sigma" label="Sigma" min={0.3} max={4} step={0.1} value={c.blur_sigma} onChange={(v) => setCustom({ blur_sigma: v })} format={(v) => v.toFixed(1)} />
                </>
              )}
              {value.corruption === 'occlusion' && (
                <>
                  <Slider id="occ_coverage" label="Area covered" min={0.05} max={0.5} step={0.01} value={c.occ_coverage} onChange={(v) => setCustom({ occ_coverage: v })} format={(v) => `${Math.round(v * 100)}%`} />
                  <Slider id="occ_rects" label="Rectangles" min={1} max={4} step={1} value={c.occ_rects} onChange={(v) => setCustom({ occ_rects: v })} />
                </>
              )}
            </div>
          )}
          <label className="flex items-center gap-2 text-sm">
            <span className="text-slate-600 dark:text-slate-300">Seed</span>
            <input type="number" min="0" placeholder="random" value={value.seed} onChange={(e) => set({ seed: e.target.value })}
                   className="w-28 rounded-lg border border-slate-300 bg-white px-2 py-1 text-sm dark:border-slate-700 dark:bg-slate-900" aria-label="Seed" />
            <span className="text-xs text-slate-500">same seed = same corruption</span>
          </label>
        </>
      )}
    </div>
  )
}
