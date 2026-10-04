import { describe, expect, it, vi } from 'vitest'
import { ApiError, buildForm, DEFAULT_CUSTOM, downloadDataUrl, postForm } from '../api'

const entries = (form) => Object.fromEntries(form.entries())

describe('buildForm', () => {
  it('sends an uploaded file, not a sample id, and the corruption settings', () => {
    const file = new File(['x'], 'cat.png', { type: 'image/png' })
    const f = buildForm({ file, sampleId: 'cat_01', corruption: 'salt', severity: 'high', seed: '', custom: DEFAULT_CUSTOM })
    expect(f.get('file').name).toBe('cat.png')
    expect(f.has('sample_id')).toBe(false)
    expect(entries(f)).toMatchObject({ corruption: 'salt', severity: 'high' })
    expect(f.has('seed')).toBe(false) // blank seed -> backend picks a random one
    expect(f.has('salt_p')).toBe(false) // custom parameters only for severity=custom
  })

  it('sends the sample id when there is no file, and the seed when given', () => {
    const f = buildForm({ sampleId: 'cat_01', corruption: 'blur', severity: 'low', seed: 7, custom: DEFAULT_CUSTOM })
    expect(entries(f)).toMatchObject({ sample_id: 'cat_01', corruption: 'blur', severity: 'low', seed: '7' })
    expect(f.has('file')).toBe(false)
  })

  it('sends all custom parameters for severity=custom', () => {
    const f = buildForm({ sampleId: 'a', corruption: 'occlusion', severity: 'custom', seed: '', custom: { ...DEFAULT_CUSTOM, occ_rects: 3, occ_coverage: 0.3 } })
    expect(entries(f)).toMatchObject({ occ_rects: '3', occ_coverage: '0.3', salt_p: '0.08', blur_kernel: '5' })
  })

  it('sketch form carries the style and no corruption fields', () => {
    const f = buildForm({ sampleId: 'face', style: 3 })
    expect(entries(f)).toEqual({ sample_id: 'face', style: '3' })
  })
})

describe('postForm error handling', () => {
  const respond = (status, body) => vi.fn().mockResolvedValue({ ok: status < 400, status, statusText: 'Err', json: async () => body })

  it('returns the JSON body on success', async () => {
    vi.stubGlobal('fetch', respond(200, { ok: true }))
    await expect(postForm('/x', new FormData())).resolves.toEqual({ ok: true })
  })

  it('surfaces the backend detail message', async () => {
    vi.stubGlobal('fetch', respond(415, { detail: "unsupported file type 'text/plain'" }))
    await expect(postForm('/x', new FormData())).rejects.toMatchObject({ message: "unsupported file type 'text/plain'", status: 415 })
  })

  it('formats validation errors (422 list)', async () => {
    vi.stubGlobal('fetch', respond(422, { detail: [{ loc: ['body', 'salt_p'], msg: 'Input should be <= 0.5' }] }))
    await expect(postForm('/x', new FormData())).rejects.toThrow('salt_p: Input should be <= 0.5')
  })

  it('reports an unreachable backend', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')))
    const err = await postForm('/x', new FormData()).catch((e) => e)
    expect(err).toBeInstanceOf(ApiError)
    expect(err.message).toMatch(/Cannot reach the backend/)
  })
})

describe('downloadDataUrl', () => {
  it('clicks a temporary link with the file name', () => {
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {})
    downloadDataUrl('data:image/png;base64,AAAA', 'out.png')
    expect(click).toHaveBeenCalledTimes(1)
    expect(document.querySelector('a[download]')).toBeNull() // cleaned up
  })
})
