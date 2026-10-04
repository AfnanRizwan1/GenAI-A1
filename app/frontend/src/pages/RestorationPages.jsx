import Bars from '../components/Bars'
import { Badge } from '../components/Common'
import RestorationPage from './RestorationPage'

export function UniversalPage() {
  return (
    <RestorationPage
      title="Universal Restoration"
      subtitle="One autoencoder with a compressed bottleneck restores clean, salt-and-pepper, blurred and occluded images without being told which corruption was applied."
      endpoint="/universal"
      filePrefix="universal"
    />
  )
}

export function HardPage() {
  return (
    <RestorationPage
      title="Hard-Routed Restoration"
      subtitle="A classifier first predicts the corruption; exactly one specialist autoencoder then restores the image. Clean inputs bypass the experts."
      endpoint="/hard"
      filePrefix="hard-routed"
      extra={(r) => (
        <div className="card" data-testid="routing">
          <h3 className="mb-3 text-sm font-semibold">Classifier probabilities</h3>
          <Bars ariaLabel="Class probabilities" items={Object.entries(r.probabilities).map(([label, value]) => ({ label, value, highlight: label === r.predicted_class }))} />
          <div className="mt-4 flex flex-wrap gap-2">
            <Badge>Predicted: {r.predicted_class}</Badge>
            <Badge tone={r.selected_expert.startsWith('none') ? 'gray' : 'green'}>Expert: {r.selected_expert}</Badge>
          </div>
        </div>
      )}
    />
  )
}

export function SoftPage() {
  return (
    <RestorationPage
      title="Soft Mixture-of-Experts Restoration"
      subtitle="A gating network gives every branch a continuous weight; the output is the weighted sum of the identity branch and the three experts, trained jointly end to end."
      endpoint="/soft"
      filePrefix="soft-moe"
      extra={(r) => {
        const top = r.contribution_ranking[0]
        return (
          <div className="card" data-testid="routing">
            <h3 className="mb-3 text-sm font-semibold">Routing weights</h3>
            <Bars sort ariaLabel="Routing weights" items={Object.entries(r.weights).map(([label, value]) => ({ label, value }))} />
            <p className="mt-4 text-sm text-slate-600 dark:text-slate-300">
              Strongest contribution: <strong>{top.branch}</strong> ({(top.weight * 100).toFixed(1)}%).
              {top.weight < 0.6 ? ' The weights are spread across several branches.' : ' One branch dominates this input.'}
            </p>
          </div>
        )
      }}
    />
  )
}
