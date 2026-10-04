import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import App from '../App'

const health = (loaded) => ({
  status: 'ok',
  models: Object.fromEntries(['universal', 'classifier', 'specialist_salt', 'specialist_blur', 'specialist_occlusion', 'soft_moe', 'sketch']
    .map((n, i) => [n, { file: `${n}.onnx`, loaded: i < loaded }])),
})

function mockGet(h) {
  vi.stubGlobal('fetch', vi.fn(async (url) => {
    if (h === 'down') throw new TypeError('Failed to fetch')
    return { ok: true, status: 200, json: async () => (String(url).endsWith('/health') ? h : []) }
  }))
}

describe('App shell', () => {
  it('links the four workspaces and navigates between them', async () => {
    mockGet(health(7))
    window.location.hash = '#/'
    const user = userEvent.setup()
    render(<App />)
    const nav = screen.getByRole('navigation', { name: 'Workspaces' })
    expect(await screen.findByRole('heading', { name: 'Universal Restoration' })).toBeInTheDocument() // default route
    for (const name of ['Universal Restoration', 'Hard-Routed Restoration', 'Soft Mixture-of-Experts', 'Face-to-Sketch Generator']) {
      expect(nav).toHaveTextContent(name)
    }
    await user.click(screen.getAllByRole('link', { name: /Hard-Routed Restoration/ })[0])
    expect(await screen.findByRole('heading', { name: 'Hard-Routed Restoration' })).toBeInTheDocument()
    await user.click(screen.getAllByRole('link', { name: /Soft Mixture-of-Experts/ })[0])
    expect(await screen.findByRole('heading', { name: 'Soft Mixture-of-Experts Restoration' })).toBeInTheDocument()
    await user.click(screen.getAllByRole('link', { name: /Face-to-Sketch Generator/ })[0])
    expect(await screen.findByRole('heading', { name: 'Face-to-Sketch Generator' })).toBeInTheDocument()
  })

  it('shows the backend status with the number of loaded models', async () => {
    mockGet(health(7))
    render(<App />)
    expect(await screen.findByText('Backend online · 7 models loaded')).toBeInTheDocument()
  })

  it('warns when only some models are loaded', async () => {
    mockGet(health(5))
    render(<App />)
    expect(await screen.findByText('Backend online · 5 of 7 models loaded')).toBeInTheDocument()
  })

  it('always shows the runtime chip', async () => {
    mockGet(health(7))
    render(<App />)
    expect(await screen.findByText('ONNX Runtime · CPU')).toBeInTheDocument()
  })

  it('shows offline when the backend cannot be reached', async () => {
    mockGet('down')
    render(<App />)
    expect(await screen.findByText('Backend offline')).toBeInTheDocument()
  })

  it('toggles dark mode on the document', async () => {
    mockGet(health(7))
    const user = userEvent.setup()
    render(<App />)
    await user.click(screen.getByRole('button', { name: /toggle dark mode/i }))
    const dark = document.documentElement.classList.contains('dark')
    await user.click(screen.getByRole('button', { name: /toggle dark mode/i }))
    expect(document.documentElement.classList.contains('dark')).toBe(!dark)
  })
})
