import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import Bars from '../components/Bars'
import { HardPage, SoftPage, UniversalPage } from '../pages/RestorationPages'
import SketchPage from '../pages/SketchPage'

const PNG = 'data:image/png;base64,iVBORw0KGgo='
const SAMPLES = [{ id: 'cat_01', name: 'cat 01', url: '/api/samples/cat_01' }]

const restoration = (extra = {}) => ({
  original: PNG, input: PNG, restored: PNG, difference: PNG, difference_full_scale: 0.25, inference_ms: 12.34, total_ms: 20, image_size: 128,
  settings: { type: 'blur', severity: 'high', seed: 4, kernel_size: 7, sigma: 2.5 },
  metrics: { input_psnr: 20.1, restored_psnr: 25.2, input_ssim: 0.71, restored_ssim: 0.83 }, ...extra,
})

/** Mock backend: GET /samples, and a queue of responses for POSTs. */
function mockBackend(posts) {
  const calls = []
  const queue = [...posts]
  const fetchMock = vi.fn(async (url, opts = {}) => {
    if ((opts.method ?? 'GET') === 'GET') {
      return { ok: true, status: 200, json: async () => (String(url).endsWith('/samples') ? SAMPLES : {}) }
    }
    calls.push({ url: String(url), form: Object.fromEntries(opts.body.entries()) })
    const next = queue.shift()
    return next.error
      ? { ok: false, status: next.status, statusText: 'err', json: async () => ({ detail: next.error }) }
      : { ok: true, status: 200, json: async () => next }
  })
  vi.stubGlobal('fetch', fetchMock)
  return calls
}

const pickSample = async (user) => user.click(await screen.findByRole('button', { name: 'Sample cat 01' }))
const runButton = () => screen.getByRole('button', { name: /apply corruption and restore/i })

describe('Bars', () => {
  it('sorts descending and highlights the strongest', () => {
    render(<Bars sort items={[{ label: 'a', value: 0.1 }, { label: 'b', value: 0.7 }, { label: 'c', value: 0.2 }]} />)
    const labels = screen.getAllByRole('progressbar').map((p) => p.getAttribute('aria-label'))
    expect(labels).toEqual(['b', 'c', 'a'])
    expect(screen.getByText('70.0%')).toBeInTheDocument()
  })
})

describe('Universal Restoration workspace', () => {
  it('runs a restoration with the chosen corruption and shows images, settings, timing and metrics', async () => {
    const calls = mockBackend([restoration()])
    const user = userEvent.setup()
    render(<UniversalPage />)
    const run = runButton()
    expect(run).toBeDisabled() // nothing selected yet
    await pickSample(user)
    await user.click(screen.getByRole('radio', { name: 'Gaussian blur' }))
    await user.click(screen.getByRole('radio', { name: 'High' }))
    await user.type(screen.getByLabelText('Seed'), '4')
    await user.click(run)

    await screen.findByRole('img', { name: 'Restored output' })
    expect(calls).toHaveLength(1)
    expect(calls[0].url).toBe('/api/universal')
    expect(calls[0].form).toMatchObject({ sample_id: 'cat_01', corruption: 'blur', severity: 'high', seed: '4' })
    const d = within(screen.getByTestId('details'))
    expect(d.getByText('Gaussian blur')).toBeInTheDocument()
    expect(d.getByText('12.3 ms')).toBeInTheDocument()
    const m = within(screen.getByTestId('metrics'))
    expect(m.getByText('20.1 → 25.2 dB')).toBeInTheDocument()
    expect(m.getByText('+5.1 dB')).toBeInTheDocument() // PSNR gain pill
    expect(m.getByText('+0.120')).toBeInTheDocument() // SSIM gain pill
    expect(screen.getByRole('button', { name: /download restored image/i })).toBeInTheDocument()
  })

  it('offers side-by-side, compare slider and difference map views', async () => {
    mockBackend([restoration()])
    const user = userEvent.setup()
    render(<UniversalPage />)
    await pickSample(user)
    await user.click(runButton())
    await screen.findByRole('img', { name: 'Restored output' })

    await user.click(screen.getByRole('radio', { name: 'Difference map' }))
    expect(screen.getByRole('img', { name: 'Difference map' })).toBeInTheDocument()
    expect(screen.getByText(/full scale 0.25/)).toBeInTheDocument()

    await user.click(screen.getByRole('radio', { name: 'Compare' }))
    const slider = screen.getByRole('slider', { name: /compare input and restored/i })
    expect(slider).toHaveAttribute('aria-valuenow', '50')
    slider.focus()
    await user.keyboard('{ArrowRight}')
    expect(slider).toHaveAttribute('aria-valuenow', '55')
    await user.keyboard('{ArrowLeft}{ArrowLeft}')
    expect(slider).toHaveAttribute('aria-valuenow', '45')

    await user.click(screen.getByRole('radio', { name: 'Side by side' }))
    expect(screen.getByRole('img', { name: 'Original' })).toBeInTheDocument()
  })

  it('hides the difference map view when there is no clean reference', async () => {
    mockBackend([restoration({ original: null, difference: null, metrics: null, settings: { type: 'clean', severity: null, seed: 1 } })])
    const user = userEvent.setup()
    render(<UniversalPage />)
    await user.upload(screen.getByTestId('file-input'), new File(['x'], 'damaged.png', { type: 'image/png' }))
    await user.click(screen.getByRole('radio', { name: 'None' }))
    await user.click(screen.getByRole('button', { name: 'Restore' }))
    await screen.findByText(/No reference image/i)
    expect(screen.queryByRole('radio', { name: 'Difference map' })).toBeNull()
    expect(screen.getByText(/PSNR and SSIM need a clean reference/i)).toBeInTheDocument()
  })

  it('shows custom sliders only for Custom severity and sends their values', async () => {
    const calls = mockBackend([restoration()])
    const user = userEvent.setup()
    render(<UniversalPage />)
    await pickSample(user)
    expect(screen.queryByLabelText(/corrupted pixels/i)).toBeNull()
    await user.click(screen.getByRole('radio', { name: 'Salt-and-pepper' }))
    await user.click(screen.getByRole('radio', { name: 'Custom' }))
    expect(screen.getByLabelText(/corrupted pixels/i)).toBeInTheDocument()
    await user.click(runButton())
    await screen.findByRole('img', { name: 'Restored output' })
    expect(calls[0].form).toMatchObject({ corruption: 'salt', severity: 'custom', salt_p: '0.08' })
  })

  it('the dice button fills in a random seed', async () => {
    mockBackend([])
    const user = userEvent.setup()
    render(<UniversalPage />)
    expect(screen.getByLabelText('Seed')).toHaveValue(null)
    await user.click(screen.getByRole('button', { name: 'Random seed' }))
    expect(Number(screen.getByLabelText('Seed').value)).toBeGreaterThanOrEqual(0)
    expect(screen.getByLabelText('Seed').value).not.toBe('')
  })

  it('restores an already corrupted upload as given (corruption: none) and notes there is no reference', async () => {
    const calls = mockBackend([restoration({ original: null, difference: null, metrics: null, settings: { type: 'clean', severity: null, seed: 1 } })])
    const user = userEvent.setup()
    render(<UniversalPage />)
    await user.upload(screen.getByTestId('file-input'), new File(['x'], 'damaged.png', { type: 'image/png' }))
    await user.click(screen.getByRole('radio', { name: 'None' }))
    await user.click(screen.getByRole('button', { name: 'Restore' }))
    await screen.findByText(/No reference image/i)
    expect(calls[0].form.file.name).toBe('damaged.png')
    expect(calls[0].form.corruption).toBe('none')
  })

  it('copies the metrics as JSON', async () => {
    mockBackend([restoration()])
    const user = userEvent.setup()   // user-event installs its own clipboard stub, so mock ours afterwards
    const writeText = vi.fn().mockResolvedValue()
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true })
    render(<UniversalPage />)
    await pickSample(user)
    await user.click(runButton())
    await user.click(await screen.findByRole('button', { name: /copy metrics as json/i }))
    await waitFor(() => expect(writeText).toHaveBeenCalledTimes(1))
    const payload = JSON.parse(writeText.mock.calls[0][0])
    expect(payload).toMatchObject({ workspace: 'Universal Restoration', inference_ms: 12.34, metrics: { restored_psnr: 25.2 }, settings: { type: 'blur', seed: 4 } })
    expect(await screen.findByRole('button', { name: /copied/i })).toBeInTheDocument()
  })

  it('shows the backend error message in a dismissible banner', async () => {
    mockBackend([{ error: "unsupported file type 'text/plain'", status: 415 }])
    const user = userEvent.setup()
    render(<UniversalPage />)
    await pickSample(user)
    await user.click(runButton())
    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent("unsupported file type 'text/plain'")
    await user.click(screen.getByRole('button', { name: /dismiss error/i }))
    await waitFor(() => expect(screen.queryByRole('alert')).toBeNull())
  })

  it('shows a loading state and disables the button while a request is running', async () => {
    let release
    vi.stubGlobal('fetch', vi.fn((url, opts = {}) => (opts.method === 'POST'
      ? new Promise((res) => { release = () => res({ ok: true, status: 200, json: async () => restoration() }) })
      : Promise.resolve({ ok: true, status: 200, json: async () => SAMPLES }))))
    const user = userEvent.setup()
    render(<UniversalPage />)
    await pickSample(user)
    await user.click(runButton())
    expect(await screen.findByRole('button', { name: /restoring/i })).toBeDisabled()
    expect(screen.getByRole('status', { name: /loading restored output/i })).toBeInTheDocument()
    release()
    await screen.findByRole('img', { name: 'Restored output' })
    expect(screen.queryByRole('status', { name: /loading/i })).toBeNull()
  })
})

describe('Hard-Routed Restoration workspace', () => {
  const hard = (cls, expert) => restoration({
    probabilities: { clean: 0.02, 'salt-and-pepper': 0.03, 'gaussian blur': 0.9, occlusion: 0.05 },
    predicted_class: cls, selected_expert: expert, classifier_ms: 3.1, expert_ms: 9.2,
  })

  it('displays the four probabilities, the routing targets, the predicted class, the selected expert and the timing split', async () => {
    mockBackend([hard('gaussian blur', 'blur specialist')])
    const user = userEvent.setup()
    render(<HardPage />)
    await pickSample(user)
    await user.click(runButton())
    const card = within(await screen.findByTestId('routing'))
    expect(card.getAllByRole('progressbar')).toHaveLength(4)
    expect(card.getByText('90.0%')).toBeInTheDocument()
    expect(card.getByText('Routing target: Identity bypass')).toBeInTheDocument()
    expect(card.getByText('Routing target: Blur expert')).toBeInTheDocument()
    expect(card.getByText('Predicted: gaussian blur')).toBeInTheDocument()
    expect(card.getByText('Expert: blur specialist')).toBeInTheDocument()
    expect(card.getByText(/Total 12\.3 ms/)).toBeInTheDocument()
    expect(within(screen.getByTestId('details')).getByText('3.1 / 9.2 ms')).toBeInTheDocument()
  })

  it('shows the identity bypass for clean predictions', async () => {
    mockBackend([hard('clean', 'none (identity bypass)')])
    const user = userEvent.setup()
    render(<HardPage />)
    await pickSample(user)
    await user.click(runButton())
    expect(await screen.findByText('Expert: none (identity bypass)')).toBeInTheDocument()
  })
})

describe('Soft Mixture-of-Experts workspace', () => {
  const soft = (w, entropy = { nats: 0.5, normalized: 0.36 }) => {
    const entries = ['identity (clean)', 'salt-and-pepper expert', 'blur expert', 'occlusion expert'].map((b, i) => [b, w[i]])
    const ranking = [...entries].sort((a, b) => b[1] - a[1]).map(([branch, weight]) => ({ branch, weight }))
    return restoration({ weights: Object.fromEntries(entries), contribution_ranking: ranking, dominant_branch: ranking[0].branch, routing_entropy: entropy })
  }
  const run = async (w, entropy) => {
    mockBackend([soft(w, entropy)])
    const user = userEvent.setup()
    render(<SoftPage />)
    await pickSample(user)
    await user.click(runButton())
    return within(await screen.findByTestId('routing'))
  }

  it('lists four sorted weights, a stacked bar, and names the dominant expert', async () => {
    const card = await run([0.05, 0.05, 0.85, 0.05])
    expect(card.getAllByRole('progressbar').map((p) => p.getAttribute('aria-label'))[0]).toBe('blur expert')
    expect(card.getByRole('img', { name: /share of each branch/i })).toBeInTheDocument()
    expect(card.getByText(/One branch dominates/)).toBeInTheDocument()
    expect(card.getByText('blur expert', { selector: 'strong' })).toBeInTheDocument()
    expect(card.getByText('Concentrated')).toBeInTheDocument()
  })

  it('says when the weights are spread across several experts and shows the routing entropy', async () => {
    const card = await run([0.1, 0.4, 0.4, 0.1], { nats: 1.19, normalized: 0.86 })
    expect(card.getByText(/spread across several branches/)).toBeInTheDocument()
    expect(card.getByText('Blended')).toBeInTheDocument()
    expect(within(card.getByTestId('entropy')).getByText(/0\.86/)).toBeInTheDocument()
    expect(within(card.getByTestId('entropy')).getByText(/1\.190 nat/)).toBeInTheDocument()
  })
})

describe('Face-to-Sketch workspace', () => {
  const sketch = (style) => ({ photo: PNG, sketch: PNG, style: `Style ${style}`, inference_ms: 33.3, total_ms: 40, image_size: 128 })

  it('offers exactly three styles with the FS2K descriptions', () => {
    mockBackend([])
    render(<SketchPage />)
    const radios = screen.getAllByRole('radio')
    expect(radios).toHaveLength(3)
    expect(radios[0]).toHaveTextContent('Simple lines')
    expect(radios[1]).toHaveTextContent('Long strokes')
    expect(radios[2]).toHaveTextContent('Wispy details')
    expect(radios[0]).toHaveAttribute('aria-checked', 'true')
  })

  it('sends the chosen style and shows photo and sketch side by side with a download button', async () => {
    const calls = mockBackend([sketch(3)])
    const user = userEvent.setup()
    render(<SketchPage />)
    await pickSample(user)
    await user.click(screen.getByRole('radio', { name: /Style 3/ }))
    expect(screen.getByRole('radio', { name: /Style 3/ })).toHaveAttribute('aria-checked', 'true')
    await user.click(screen.getByRole('button', { name: 'Generate sketch' }))
    await screen.findByRole('img', { name: 'Generated sketch' })
    expect(calls[0].url).toBe('/api/sketch')
    expect(calls[0].form).toMatchObject({ sample_id: 'cat_01', style: '3' })
    expect(screen.getByRole('img', { name: 'Original photo' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /download sketch/i })).toBeInTheDocument()
    expect(screen.getByText(/33\.3 ms inference/)).toBeInTheDocument()
  })

  it('explains when the browser has no camera access', async () => {
    mockBackend([])
    Object.defineProperty(navigator, 'mediaDevices', { value: undefined, configurable: true })
    const user = userEvent.setup()
    render(<SketchPage />)
    await user.click(screen.getByRole('button', { name: /use webcam/i }))
    expect(await screen.findByText(/cannot access a camera/i)).toBeInTheDocument()
  })

  it('reports a denied camera permission', async () => {
    mockBackend([])
    const denied = Object.assign(new Error('no'), { name: 'NotAllowedError' })
    Object.defineProperty(navigator, 'mediaDevices', { value: { getUserMedia: vi.fn().mockRejectedValue(denied) }, configurable: true })
    const user = userEvent.setup()
    render(<SketchPage />)
    await user.click(screen.getByRole('button', { name: /use webcam/i }))
    expect(await screen.findByText(/permission was denied/i)).toBeInTheDocument()
  })
})
