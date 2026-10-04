import { Badge, CardTitle } from '../components/Common'
import Bars from '../components/Bars'
import RestorationPage from './RestorationPage'

export function UniversalPage() {
  return (
    <RestorationPage
      number={1}
      title="Universal Restoration"
      subtitle="One autoencoder restores clean, salt-and-pepper, blurred and occluded images, without being told which corruption was applied."
      endpoint="/universal"
      filePrefix="universal"
    />
  )
}

const TARGETS = { clean: 'Identity bypass', 'salt-and-pepper': 'Salt-and-pepper expert', 'gaussian blur': 'Blur expert', occlusion: 'Occlusion expert' }

export function HardPage() {
  return (
    <RestorationPage
      number={2}
      title="Hard-Routed Restoration"
      subtitle="A classifier predicts the corruption and routes the image to exactly one specialist autoencoder. Clean images bypass the experts."
      endpoint="/hard"
      filePrefix="hard-routed"
      extra={(r) => (
        <div className="card animate-fade-up" data-testid="routing">
          <CardTitle icon="branch" title="Classifier probabilities" right={<span className="num text-[11px] text-ink-3">argmax routing</span>} />
          <Bars ariaLabel="Class probabilities"
                items={Object.entries(r.probabilities).map(([label, value]) => ({ label, value, highlight: label === r.predicted_class, hint: `Routing target: ${TARGETS[label]}` }))} />
          <div className="mt-5 flex flex-wrap items-center gap-2">
            <Badge icon="check">Predicted: {r.predicted_class}</Badge>
            <Badge tone={r.selected_expert.startsWith('none') ? 'gray' : 'good'}>Expert: {r.selected_expert}</Badge>
          </div>
          <p className="num mt-4 text-xs text-ink-2">
            Classifier {r.classifier_ms.toFixed(1)} ms · Expert {r.expert_ms.toFixed(1)} ms · Total {(r.classifier_ms + r.expert_ms).toFixed(1)} ms
          </p>
        </div>
      )}
    />
  )
}

function WeightStack({ ranked }) {
  return (
    <div className="flex h-3 overflow-hidden rounded-full bg-subdued" role="img" aria-label="Share of each branch in the blend">
      {ranked.map((r, i) => (
        <div key={r.branch} title={`${r.branch} ${(r.weight * 100).toFixed(1)}%`} className="h-full transition-all duration-500"
             style={{ width: `${r.weight * 100}%`, background: 'rgb(var(--primary))', opacity: 1 - i * 0.22 }} />
      ))}
    </div>
  )
}

export function SoftPage() {
  return (
    <RestorationPage
      number={3}
      title="Soft Mixture-of-Experts"
      subtitle="A gating network gives every branch a continuous weight. The output is the weighted sum of the identity branch and the three experts, trained jointly."
      endpoint="/soft"
      filePrefix="soft-moe"
      extra={(r) => {
        const top = r.contribution_ranking[0]
        const spread = top.weight < 0.6
        return (
          <div className="card animate-fade-up" data-testid="routing">
            <CardTitle icon="layers" title="Routing weights" right={<span className="chip bg-good-tint text-good num">Σ = 100.0%</span>} />
            <WeightStack ranked={r.contribution_ranking} />
            <div className="mt-5"><Bars sort ariaLabel="Routing weights" items={Object.entries(r.weights).map(([label, value]) => ({ label, value }))} /></div>
            <p className="mt-5 rounded-md bg-brand-tint px-4 py-3 text-sm leading-relaxed text-brand-on-tint">
              Strongest contribution: <strong>{top.branch}</strong> ({(top.weight * 100).toFixed(1)}%).
              {spread ? ' The weights are spread across several branches.' : ' One branch dominates this input.'}
            </p>
            <div className="well mt-4 flex items-center justify-between" data-testid="entropy">
              <div>
                <span className="label-caps">Routing entropy</span>
                <p className="num mt-1 text-xl font-semibold">{r.routing_entropy.normalized.toFixed(2)} <span className="text-xs font-medium text-ink-3">of 1 · {r.routing_entropy.nats.toFixed(3)} nat</span></p>
              </div>
              <Badge tone={spread ? 'info' : 'gray'}>{spread ? 'Blended' : 'Concentrated'}</Badge>
            </div>
          </div>
        )
      }}
    />
  )
}
