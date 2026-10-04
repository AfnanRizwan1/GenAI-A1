import { Icon } from './Icons'
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
      <span className="mb-1.5 flex items-baseline justify-between"><span className="text-ink-2">{label}</span><span className="num text-xs font-semibold text-brand">{format(value)}</span></span>
      <input id={id} type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))}
             className="h-1 w-full cursor-pointer appearance-none rounded-full bg-line accent-brand" />
    </label>
  )
}

/** value: {corruption, severity, seed, custom} */
export default function CorruptionControls({ value, onChange }) {
  const set = (patch) => onChange({ ...value, ...patch })
  const setCustom = (patch) => set({ custom: { ...value.custom, ...patch } })
  const c = value.custom
  return (
    <div className="space-y-5">
      <Segmented name="kind" label="Corruption" cols={2} options={KINDS} value={value.corruption} onChange={(corruption) => set({ corruption })} />
      {value.corruption === 'none' ? (
        <p className="rounded-md bg-info-tint px-3 py-2.5 text-xs leading-relaxed text-info">
          No corruption is added: the image is restored as it is. Upload an already corrupted picture to test the models on unseen images.
        </p>
      ) : (
        <>
          <Segmented name="severity" label="Severity" options={SEVERITIES} value={value.severity} onChange={(severity) => set({ severity })} />
          {value.severity === 'custom' && (
            <div className={`well grid gap-4 ${value.corruption === 'salt' ? '' : 'sm:grid-cols-2'}`}>
              {value.corruption === 'salt' && <Slider id="salt_p" label="Corrupted pixels" min={0.01} max={0.3} step={0.01} value={c.salt_p} onChange={(v) => setCustom({ salt_p: v })} format={(v) => `${Math.round(v * 100)}%`} />}
              {value.corruption === 'blur' && (
                <>
                  <Slider id="blur_kernel" label="Kernel size" min={3} max={11} step={2} value={c.blur_kernel} onChange={(v) => setCustom({ blur_kernel: v })} format={(v) => `${v}×${v}`} />
                  <Slider id="blur_sigma" label="Sigma" min={0.3} max={4} step={0.1} value={c.blur_sigma} onChange={(v) => setCustom({ blur_sigma: v })} format={(v) => `σ = ${v.toFixed(1)}`} />
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
          <div>
            <div className="mb-2 flex items-baseline justify-between"><span className="label-caps">Seed</span><span className="text-[11px] text-ink-3">same seed = same corruption</span></div>
            <div className="flex gap-2">
              <input type="number" min="0" placeholder="random" value={value.seed} onChange={(e) => set({ seed: e.target.value })} aria-label="Seed"
                     className="num h-10 min-w-0 flex-1 rounded-md border border-line bg-subdued px-3 text-sm placeholder:text-ink-3" />
              <button type="button" className="btn-secondary !px-3" aria-label="Random seed" title="Random seed"
                      onClick={() => set({ seed: String(Math.floor(Math.random() * 1_000_000)) })}><Icon name="dice" className="h-4 w-4" /></button>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
