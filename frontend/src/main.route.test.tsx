import { describe, expect, it } from 'vitest'
import { isPageViewerPath } from './route'

describe('frontend page routing', () => {
  it('only selects the independent viewer path', () => {
    expect(isPageViewerPath('/viewer/pages')).toBe(true)
    expect(isPageViewerPath('/viewer/pages/')).toBe(true)
    expect(isPageViewerPath('/')).toBe(false)
    expect(isPageViewerPath('/viewer/history/')).toBe(false)
  })
})
