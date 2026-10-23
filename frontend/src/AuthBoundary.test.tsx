import { act, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import AuthBoundary from './AuthBoundary'
import { AUTH_INVALIDATION_CHANNEL, publishAuthInvalidation } from './authLifecycle'

const apiMock = vi.hoisted(() => ({ getAuthSession: vi.fn(), redirectToLogin: vi.fn() }))
vi.mock('./api', () => apiMock)

class FakeBroadcastChannel {
  static readonly instances = new Set<FakeBroadcastChannel>()
  private listeners = new Set<(event: MessageEvent) => void>()
  constructor(readonly name: string) { FakeBroadcastChannel.instances.add(this) }
  addEventListener(type: string, listener: EventListenerOrEventListenerObject) {
    if (type === 'message' && typeof listener === 'function') this.listeners.add(listener as (event: MessageEvent) => void)
  }
  removeEventListener(type: string, listener: EventListenerOrEventListenerObject) {
    if (type === 'message' && typeof listener === 'function') this.listeners.delete(listener as (event: MessageEvent) => void)
  }
  postMessage(data: unknown) {
    for (const peer of FakeBroadcastChannel.instances) {
      if (peer === this || peer.name !== this.name) continue
      for (const listener of peer.listeners) listener(new MessageEvent('message', { data }))
    }
  }
  close() { FakeBroadcastChannel.instances.delete(this); this.listeners.clear() }
}

const session = (owner_user_id: number, username: string) => ({ authenticated: true as const, owner_user_id, username })

describe('authenticated route boundary', () => {
  beforeEach(() => {
    FakeBroadcastChannel.instances.clear()
    vi.stubGlobal('BroadcastChannel', FakeBroadcastChannel)
    apiMock.getAuthSession.mockReset().mockResolvedValue(session(101, 'alice'))
    apiMock.redirectToLogin.mockReset()
    localStorage.clear()
  })
  afterEach(() => vi.unstubAllGlobals())

  it('clears both mounted provider tabs before revalidating a content-free remote nonce', async () => {
    localStorage.setItem('rag-arch-demo-ui-v2:101:workspace-a', JSON.stringify({ prompt: 'A 的私有草稿' }))
    let checked = 0
    apiMock.getAuthSession.mockImplementation(async () => {
      checked += 1
      if (checked > 2) {
        expect(screen.queryByTestId('private-tab-a')).not.toBeInTheDocument()
        expect(screen.queryByTestId('private-tab-b')).not.toBeInTheDocument()
        return session(202, 'bob')
      }
      return session(101, 'alice')
    })
    render(<>
      <AuthBoundary><div data-testid="private-tab-a">A 的私有任务和附件</div></AuthBoundary>
      <AuthBoundary><div data-testid="private-tab-b">B 的私有任务和附件</div></AuthBoundary>
    </>)
    expect(await screen.findByTestId('private-tab-a')).toBeInTheDocument()
    expect(await screen.findByTestId('private-tab-b')).toBeInTheDocument()

    const externalTab = new FakeBroadcastChannel(AUTH_INVALIDATION_CHANNEL)
    act(() => externalTab.postMessage('a'.repeat(36)))
    await waitFor(() => expect(screen.getByTestId('private-tab-a')).toBeInTheDocument())
    expect(screen.getByTestId('private-tab-b')).toBeInTheDocument()
    expect(localStorage.getItem('rag-arch-demo-ui-v2:101:workspace-a')).toBeNull()
    expect(apiMock.getAuthSession).toHaveBeenCalledTimes(4)
    expect(apiMock.redirectToLogin).not.toHaveBeenCalled()
    externalTab.close()
  })

  it('clears the active owner cache and redirects if the revalidated session is gone', async () => {
    localStorage.setItem('rag-arch-demo-ui-v2:101:workspace-a', JSON.stringify({ prompt: '私有草稿' }))
    apiMock.getAuthSession.mockResolvedValueOnce(session(101, 'alice')).mockResolvedValueOnce(null)
    render(<AuthBoundary><div data-testid="private-content">用户私有内容</div></AuthBoundary>)
    expect(await screen.findByTestId('private-content')).toBeInTheDocument()

    const externalTab = new FakeBroadcastChannel(AUTH_INVALIDATION_CHANNEL)
    act(() => externalTab.postMessage('b'.repeat(36)))
    await waitFor(() => expect(apiMock.redirectToLogin).toHaveBeenCalledTimes(1))
    expect(screen.queryByTestId('private-content')).not.toBeInTheDocument()
    expect(localStorage.getItem('rag-arch-demo-ui-v2:101:workspace-a')).toBeNull()
    externalTab.close()
  })

  it('blocks local invalidation immediately without treating it as a new identity check', async () => {
    localStorage.setItem('rag-arch-demo-ui-v2:101:workspace-a', JSON.stringify({ prompt: '用户草稿' }))
    render(<AuthBoundary><div data-testid="private-content">私有内容</div></AuthBoundary>)
    expect(await screen.findByTestId('private-content')).toBeInTheDocument()
    act(() => publishAuthInvalidation())
    await waitFor(() => expect(screen.queryByTestId('private-content')).not.toBeInTheDocument())
    expect(apiMock.getAuthSession).toHaveBeenCalledTimes(1)
    expect(localStorage.getItem('rag-arch-demo-ui-v2:101:workspace-a')).toBeNull()
  })

  it('clears the old owner cache when BFCache restore reveals a different server identity', async () => {
    localStorage.setItem('rag-arch-demo-ui-v2:101:workspace-a', JSON.stringify({ prompt: '旧账号草稿' }))
    apiMock.getAuthSession.mockResolvedValueOnce(session(101, 'alice')).mockResolvedValueOnce(session(202, 'bob'))
    render(<AuthBoundary><div data-testid="private-content">工作区</div></AuthBoundary>)
    expect(await screen.findByTestId('private-content')).toBeInTheDocument()

    const pageshow = Object.assign(new Event('pageshow'), { persisted: true })
    act(() => window.dispatchEvent(pageshow))
    await waitFor(() => expect(screen.getByTestId('private-content')).toBeInTheDocument())
    expect(apiMock.getAuthSession).toHaveBeenCalledTimes(2)
    expect(localStorage.getItem('rag-arch-demo-ui-v2:101:workspace-a')).toBeNull()
  })
})
