import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  AUTH_INVALIDATION_CHANNEL, AUTH_INVALIDATION_STORAGE_KEY, publishAuthInvalidation,
  subscribeAuthInvalidation,
} from './authLifecycle'

class FakeBroadcastChannel {
  static readonly instances = new Set<FakeBroadcastChannel>()
  readonly sent: unknown[] = []
  private listeners = new Set<(event: MessageEvent) => void>()
  constructor(readonly name: string) { FakeBroadcastChannel.instances.add(this) }
  addEventListener(type: string, listener: EventListenerOrEventListenerObject) {
    if (type === 'message' && typeof listener === 'function') this.listeners.add(listener as (event: MessageEvent) => void)
  }
  removeEventListener(type: string, listener: EventListenerOrEventListenerObject) {
    if (type === 'message' && typeof listener === 'function') this.listeners.delete(listener as (event: MessageEvent) => void)
  }
  postMessage(data: unknown) {
    this.sent.push(data)
    for (const peer of FakeBroadcastChannel.instances) {
      if (peer === this || peer.name !== this.name) continue
      for (const listener of peer.listeners) listener(new MessageEvent('message', { data }))
    }
  }
  close() { FakeBroadcastChannel.instances.delete(this); this.listeners.clear() }
}

afterEach(() => {
  FakeBroadcastChannel.instances.clear()
  vi.unstubAllGlobals()
})

describe('cross-tab authentication invalidation', () => {
  it('broadcasts only a random nonce and sends local invalidation separately', () => {
    vi.stubGlobal('BroadcastChannel', FakeBroadcastChannel)
    const received: string[] = []
    const unsubscribe = subscribeAuthInvalidation(event => received.push(event.source))
    const externalTab = new FakeBroadcastChannel(AUTH_INVALIDATION_CHANNEL)
    let broadcastPayload: unknown
    externalTab.addEventListener('message', event => { broadcastPayload = (event as MessageEvent).data })

    publishAuthInvalidation()
    expect(received).toEqual(['local'])
    expect(broadcastPayload).toEqual(expect.any(String))
    expect((broadcastPayload as string).length).toBeGreaterThanOrEqual(24)
    expect(JSON.stringify(broadcastPayload)).not.toMatch(/username|token|password|draft/i)

    externalTab.postMessage('f'.repeat(36))
    expect(received).toEqual(['local', 'remote'])
    window.dispatchEvent(new StorageEvent('storage', {
      key: AUTH_INVALIDATION_STORAGE_KEY, newValue: 'f'.repeat(36),
    }))
    expect(received).toEqual(['local', 'remote'])

    unsubscribe()
    externalTab.close()
  })

  it('uses the storage event fallback when BroadcastChannel is unavailable', () => {
    vi.stubGlobal('BroadcastChannel', undefined)
    const received: string[] = []
    const unsubscribe = subscribeAuthInvalidation(event => received.push(event.source))
    window.dispatchEvent(new StorageEvent('storage', {
      key: AUTH_INVALIDATION_STORAGE_KEY, newValue: 'e'.repeat(36),
    }))
    expect(received).toEqual(['remote'])
    unsubscribe()
  })
})
