import { describe, expect, it } from 'vitest'
import { isLoginPath, isPageViewerPath, safeNextPath } from './route'

describe('frontend page routing', () => {
  it('only selects the independent viewer path', () => {
    expect(isPageViewerPath('/viewer/pages')).toBe(true)
    expect(isPageViewerPath('/viewer/pages/')).toBe(true)
    expect(isPageViewerPath('/')).toBe(false)
    expect(isPageViewerPath('/viewer/history/')).toBe(false)
  })

  it('recognizes the login route and allows only safe same-origin relative next paths', () => {
    expect(isLoginPath('/login')).toBe(true)
    expect(isLoginPath('/login/')).toBe(true)
    expect(isLoginPath('/login/other')).toBe(false)
    expect(safeNextPath('/viewer/pages?asset=one#page', 'https://frontend.example')).toBe('/viewer/pages?asset=one#page')
    for (const unsafe of ['https://evil.example', '//evil.example/path', '/\\evil.example', 'javascript:alert(1)', 'data:text/html,x']) {
      expect(safeNextPath(unsafe, 'https://frontend.example')).toBe('/')
    }
    expect(safeNextPath('/login?next=/bad', 'https://frontend.example')).toBe('/')
  })
})
