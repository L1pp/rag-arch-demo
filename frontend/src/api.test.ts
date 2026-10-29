import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  apiResourceUrl, apiUrl, cancelTask, eventsUrl, fetchTaskDownload, getAuthSession, getProtectedBlob, getStandardsPages, login,
  loginRedirectUrl, logout, resetAssistant, standardsPagesUrl, submitTask, taskDownloadUrl,
} from './api'
import type { TaskSubmission } from './types'
import { subscribeAuthInvalidation } from './authLifecycle'

afterEach(() => { vi.unstubAllGlobals(); vi.unstubAllEnvs() })

describe('frontend resource URLs', () => {
  it('keeps standards pages on the standards route and documents on the generic route', () => {
    expect(standardsPagesUrl({ std: 'GB 55023-2022', evidence: 'ev-1' })).toBe('/api/standards/pages?std=GB+55023-2022&evidence=ev-1')
    const documentUrl = standardsPagesUrl({ asset: 'asset-1', workspace_id: 'workspace-1', page: 2 })
    expect(documentUrl).toMatch(/^\/api\/documents\/pages\?/)
    expect(documentUrl).toContain('asset=asset-1')
    expect(documentUrl).toContain('workspace_id=workspace-1')
    expect(documentUrl).toContain('page=2')
    expect(standardsPagesUrl({ asset: 'asset-1', region: '0.1,0.2,0.6,0.45' })).toContain('region=0.1%2C0.2%2C0.6%2C0.45')
  })

  it('exposes the optional debug task download format', () => {
    expect(taskDownloadUrl('task/1', 'workspace/1', 'debug')).toContain('format=debug')
  })

  it('builds a same-origin login redirect without accepting an external target', () => {
    expect(loginRedirectUrl('/', '', '')).toBe('/login?next=%2F')
    expect(loginRedirectUrl('/viewer/pages', '?asset=one', '#page')).toBe(
      '/login?next=%2Fviewer%2Fpages%3Fasset%3Done%23page',
    )
  })

  it('resolves a configured API base for fetch, downloads, SSE, and returned page images', async () => {
    vi.stubEnv('VITE_API_BASE_URL', 'https://api.example.test/service/')
    const fetchMock = vi.fn().mockResolvedValue(new Response('{"pages":[]}', { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await getStandardsPages({ std: 'GB 55023-2022' })
    expect(fetchMock).toHaveBeenCalledWith(
      'https://api.example.test/service/api/standards/pages?std=GB+55023-2022',
      expect.objectContaining({ credentials: 'include' }),
    )
    expect(standardsPagesUrl({ std: 'GB 55023-2022' })).toBe('https://api.example.test/service/api/standards/pages?std=GB+55023-2022')
    expect(taskDownloadUrl('task-a', 'workspace-a', 'md')).toContain('https://api.example.test/service/api/tasks/task-a/download?')
    expect(eventsUrl('workspace-a', 'server-a', 4)).toContain('https://api.example.test/service/api/events?')
    expect(apiResourceUrl('/api/documents/page-image?asset=one')).toBe('https://api.example.test/service/api/documents/page-image?asset=one')
    expect(apiResourceUrl('/viewer/pages/history?asset=one')).toBe('/viewer/pages/history?asset=one')
    expect(apiUrl('/api/workspace')).toBe('https://api.example.test/service/api/workspace')
  })

  it('fetches private downloads and page images with credentials against the API base', async () => {
    vi.stubEnv('VITE_API_BASE_URL', 'https://api.example.test/prefix')
    const fetchMock = vi.fn().mockImplementation(() => Promise.resolve(new Response('private resource', { status: 200 })))
    vi.stubGlobal('fetch', fetchMock)

    const download = await fetchTaskDownload('task/1', 'workspace-a', 'md')
    const image = await getProtectedBlob('/api/documents/page-image?asset=asset-a&page=2')
    expect(await download.text()).toBe('private resource')
    expect(await image.text()).toBe('private resource')
    expect(fetchMock).toHaveBeenNthCalledWith(1,
      'https://api.example.test/prefix/api/tasks/task%2F1/download?workspace_id=workspace-a&format=md',
      expect.objectContaining({ credentials: 'include' }),
    )
    expect(fetchMock).toHaveBeenNthCalledWith(2,
      'https://api.example.test/prefix/api/documents/page-image?asset=asset-a&page=2',
      expect.objectContaining({ credentials: 'include' }),
    )
  })

  it('rejects unsafe API base URLs and paths', () => {
    vi.stubEnv('VITE_API_BASE_URL', 'javascript:alert(1)')
    expect(() => apiUrl('/api/workspace')).toThrow(/VITE_API_BASE_URL/)
    vi.stubEnv('VITE_API_BASE_URL', '')
    expect(() => apiUrl('//evil.example/api')).toThrow(/API 路径/)
  })

  it('uses credentialed JSON auth endpoints and treats an unauthenticated session as empty', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response('{"authenticated":true,"owner_user_id":5,"username":"alice"}', { status: 200 }))
      .mockResolvedValueOnce(new Response('{"authenticated":true,"owner_user_id":5,"username":"alice"}', { status: 200 }))
      .mockResolvedValueOnce(new Response('{"authenticated":false}', { status: 200 }))
      .mockResolvedValueOnce(new Response('{"authenticated":false}', { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await login({ username: 'alice', password: 'secret' })
    await getAuthSession()
    await logout()
    expect(await getAuthSession()).toBeNull()

    expect(fetchMock).toHaveBeenNthCalledWith(1, '/api/auth/login', expect.objectContaining({
      method: 'POST', credentials: 'include', body: JSON.stringify({ username: 'alice', password: 'secret' }),
    }))
    expect(fetchMock).toHaveBeenNthCalledWith(2, '/api/auth/session', expect.objectContaining({ credentials: 'include' }))
    expect(fetchMock).toHaveBeenNthCalledWith(3, '/api/auth/logout', expect.objectContaining({ method: 'POST', credentials: 'include' }))
  })

  it('treats a 401 session probe as unauthenticated without starting a redirect cycle', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response('{"detail":"not authenticated"}', { status: 401 }))
    vi.stubGlobal('fetch', fetchMock)
    const events: string[] = []
    const unsubscribe = subscribeAuthInvalidation(event => events.push(event.source))
    expect(await getAuthSession()).toBeNull()
    expect(events).toEqual([])
    unsubscribe()
  })

  it('passes AbortSignal through mutating task endpoints', async () => {
    const fetchMock = vi.fn().mockImplementation(() => Promise.resolve(new Response('{}', { status: 200 })))
    vi.stubGlobal('fetch', fetchMock)
    const signal = new AbortController().signal
    const submission = {
      workspace_id: 'workspace-a', assistant_id: 'chat', request_id: 'request-a', prompt: 'hello', attachment_ids: [],
    } as TaskSubmission

    await submitTask(submission, signal)
    await cancelTask('task-a', 'workspace-a', signal)
    await resetAssistant('chat', 'workspace-a', signal)

    expect(fetchMock).toHaveBeenCalledTimes(3)
    for (const [, options] of fetchMock.mock.calls) expect(options?.signal).toBe(signal)
  })
})
