import { describe, expect, it } from 'vitest'
import { loginRedirectUrl, standardsPagesUrl, taskDownloadUrl } from './api'

describe('frontend resource URLs', () => {
  it('keeps standards pages on the standards route and documents on the generic route', () => {
    expect(standardsPagesUrl({ std: 'GB 12345-2020', evidence: 'ev-1' })).toBe('/api/standards/pages?std=GB+12345-2020&evidence=ev-1')
    const documentUrl = standardsPagesUrl({ asset: 'asset-1', workspace_id: 'workspace-1', page: 2 })
    expect(documentUrl).toMatch(/^\/api\/documents\/pages\?/)
    expect(documentUrl).toContain('asset=asset-1')
    expect(documentUrl).toContain('workspace_id=workspace-1')
    expect(documentUrl).toContain('page=2')
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
})
