import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import Bars from '../components/Bars'
import { HardPage, SoftPage, UniversalPage } from '../pages/RestorationPages'
import SketchPage from '../pages/SketchPage'

const PNG = 'data:image/png;base64,iVBORw0KGgo='
const SAMPLES = [{ id: 'cat_01', name: 'cat 01', url: '/api/samples/cat_01' }]

const restoration = (extra = {}) => ({
  original: PNG, input: PNG, restored: PNG, inference_ms: 12.34, total_ms: 20, image_size: 128,
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
    const run = screen.getByRole('button', { name: /apply corruption and restore/i })
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
    expect(d.getByText('gaussian blur')).toBeInTheDocument()
    expect(d.getByText('12.3 ms')).toBeInTheDocument()
    expect(d.getByText('20.1 → 25.2 dB')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /download restored image/i })).toBeInTheDocument()
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
    await user.click(screen.getByRole('button', { name: /apply corruption and restore/i }))
    await screen.findByRole('img', { name: 'Restored output' })
    expect(calls[0].form).toMatchObject({ corruption: 'salt', severity: 'custom', salt_p: '0.08' })
  })

  it('restores an already corrupted upload as given (corruption: none) and notes there is no reference', async () => {
    const calls = mockBackend([restoration({ original: null, metrics: null, settings: { type: 'clean', severity: null, seed: 1 } })])
    const user = userEvent.setup()
    render(<UniversalPage />)
    await user.upload(screen.getByTestId('file-input'), new File(['x'], 'damaged.png', { type: 'image/png' }))
    await user.click(screen.getByRole('radio', { name: 'None' }))
    await user.click(screen.getByRole('button', { name: 'Restore' }))
    await screen.findByText(/No reference image/i)
    expect(calls[0].form.file.name).toBe('damaged.png')
    expect(calls[0].form.corruption).toBe('none')
  })

  it('shows the backend error message in a dismissible banner', async () => {
    mockBackend([{ error: "unsupported file type 'text/plain'", status: 415 }])
    const user = userEvent.setup()
    render(<UniversalPage />)
    await pickSample(user)
    await user.click(screen.getByRole('button', { name: /apply corruption and restore/i }))
    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent("unsupported file type 'text/plain'")
    await user.click(screen.getByRole('button', { name: /dismiss error/i }))
    await waitFor(() => expect(screen.queryByRole('alert')).toBeNull())
  })

  it('disables the button while a request is running', async () => {
    let release
    vi.stubGlobal('fetch', vi.fn((url, opts = {}) => (opts.method === 'POST'
      ? new Promise((res) => { release = () => res({ ok: true, status: 200, json: async () => restoration() }) })
      : Promise.resolve({ ok: true, status: 200, json: async () => SAMPLES }))))
    const user = userEvent.setup()
    render(<UniversalPage />)
    await pickSample(user)
    await user.click(screen.getByRole('button', { name: /apply corruption and restore/i }))
    expect(await screen.findByRole('button', { name: /restoring/i })).toBeDisabled()
    release()
    await screen.findByRole('img', { name: 'Restored output' })
  })
})

describe('Hard-Routed Restoration workspace', () => {
  const hard = (cls, expert) => restoration({
    probabilities: { clean: 0.02, 'salt-and-pepper': 0.03, 'gaussian blur': 0.9, occlusion: 0.05 },
    predicted_class: cls, selected_expert: expert, classifier_ms: 3.1, expert_ms: 9.2,
  })

  it('displays the four probabilities, the predicted class, the selected expert and the timing split', async () => {
    mockBackend([hard('gaussian blur', 'blur specialist')])
    const user = userEvent.setup()
    render(<HardPage />)
    await pickSample(user)
    await user.click(screen.getByRole('button', { name: /apply corruption and restore/i }))
    const card = within(await screen.findByTestId('routing'))
    expect(card.getAllByRole('progressbar')).toHaveLength(4)
    expect(card.getByText('90.0%')).toBeInTheDocument()
    expect(card.getByText('Predicted: gaussian blur')).toBeInTheDocument()
    expect(card.getByText('Expert: blur specialist')).toBeInTheDocument()
    expect(within(screen.getByTestId('details')).getByText('3.1 / 9.2 ms')).toBeInTheDocument()
  })

  it('shows the identity bypass for clean predictions', async () => {
    mockBackend([hard('clean', 'none (identity bypass)')])
    const user = userEvent.setup()
    render(<HardPage />)
    await pickSample(user)
    await user.click(screen.getByRole('button', { name: /apply corruption and restore/i }))
    expect(await screen.findByText('Expert: none (identity bypass)')).toBeInTheDocument()
  })
})

describe('Soft Mixture-of-Experts workspace', () => {
  const soft = (w) => {
    const entries = ['identity (clean)', 'salt-and-pepper expert', 'blur expert', 'occlusion expert'].map((b, i) => [b, w[i]])
    const ranking = [...entries].sort((a, b) => b[1] - a[1]).map(([branch, weight]) => ({ branch, weight }))
    return restoration({ weights: Object.fromEntries(entries), contribution_ranking: ranking, dominant_branch: ranking[0].branch })
  }
  const run = async (w) => {
    mockBackend([soft(w)])
    const user = userEvent.setup()
    render(<SoftPage />)
    await pickSample(user)
    await user.click(screen.getByRole('button', { name: /apply corruption and restore/i }))
    return within(await screen.findByTestId('routing'))
  }

  it('lists four sorted weights and names the dominant expert', async () => {
    const card = await run([0.05, 0.05, 0.85, 0.05])
    expect(card.getAllByRole('progressbar').map((p) => p.getAttribute('aria-label'))[0]).toBe('blur expert')
    expect(card.getByText(/One branch dominates/)).toBeInTheDocument()
    expect(card.getByText('blur expert', { selector: 'strong' })).toBeInTheDocument()
  })

  it('says when the weights are spread across several experts', async () => {
    const card = await run([0.1, 0.4, 0.4, 0.1])
    expect(card.getByText(/spread across several branches/)).toBeInTheDocument()
  })
})

describe('Face-to-Sketch workspace', () => {
  const sketch = (style) => ({ photo: PNG, sketch: PNG, style: `Style ${style}`, inference_ms: 33.3, total_ms: 40, image_size: 128 })

  it('sends the chosen style and shows photo and sketch side by side with a download button', async () => {
    const calls = mockBackend([sketch(3)])
    const user = userEvent.setup()
    render(<SketchPage />)
    await pickSample(user)
    expect(screen.getByRole('radio', { name: 'Style 1' })).toHaveAttribute('aria-checked', 'true')
    await user.click(screen.getByRole('radio', { name: 'Style 3' }))
    await user.click(screen.getByRole('button', { name: 'Generate sketch' }))
    await screen.findByRole('img', { name: 'Generated sketch' })
    expect(calls[0].url).toBe('/api/sketch')
    expect(calls[0].form).toMatchObject({ sample_id: 'cat_01', style: '3' })
    expect(screen.getByRole('img', { name: 'Original photo' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /download sketch/i })).toBeInTheDocument()
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
