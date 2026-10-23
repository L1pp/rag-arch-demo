export const AUTH_INVALIDATION_CHANNEL = 'rag-arch-auth-invalidation-v1'
export const AUTH_INVALIDATION_STORAGE_KEY = 'rag-arch-auth-invalidation-v1'

export interface AuthInvalidationEvent {
  source: 'local' | 'remote'
}

type Listener = (event: AuthInvalidationEvent) => void

const listeners = new Set<Listener>()
const seenNonces = new Set<string>()
let channel: BroadcastChannel | null = null
let listeningForStorage = false

function emit(source: AuthInvalidationEvent['source']) {
  for (const listener of [...listeners]) listener({ source })
}

function randomNonce() {
  try { return crypto.randomUUID() } catch {
    const bytes = new Uint8Array(24)
    crypto.getRandomValues(bytes)
    return Array.from(bytes, value => value.toString(16).padStart(2, '0')).join('')
  }
}

function firstSeen(nonce: string) {
  if (seenNonces.has(nonce)) return false
  seenNonces.add(nonce)
  if (seenNonces.size > 128) {
    const oldest = seenNonces.values().next().value
    if (oldest) seenNonces.delete(oldest)
  }
  return true
}

function receiveBroadcast(event: MessageEvent<unknown>) {
  // The cross-tab payload is only a random invalidation nonce. It contains no
  // identity, token, reason, draft, or other session content.
  if (typeof event.data === 'string' && event.data.length >= 24 && event.data.length <= 64 && firstSeen(event.data)) emit('remote')
}

function receiveStorage(event: StorageEvent) {
  if (event.key === AUTH_INVALIDATION_STORAGE_KEY && event.newValue
    && event.newValue.length >= 24 && event.newValue.length <= 64 && firstSeen(event.newValue)) emit('remote')
}

function openTransport() {
  if (channel || listeningForStorage || typeof window === 'undefined') return
  // Listen on storage even when BroadcastChannel works. A peer can fall back
  // temporarily, and the bounded nonce cache suppresses duplicate delivery.
  window.addEventListener('storage', receiveStorage)
  listeningForStorage = true
  if (typeof BroadcastChannel !== 'undefined') {
    try {
      channel = new BroadcastChannel(AUTH_INVALIDATION_CHANNEL)
      channel.addEventListener('message', receiveBroadcast)
      return
    } catch {
      channel = null
    }
  }
}

function closeTransportIfUnused() {
  if (listeners.size || typeof window === 'undefined') return
  if (channel) {
    channel.removeEventListener('message', receiveBroadcast)
    channel.close()
    channel = null
  }
  if (listeningForStorage) {
    window.removeEventListener('storage', receiveStorage)
    listeningForStorage = false
  }
}

export function subscribeAuthInvalidation(listener: Listener) {
  listeners.add(listener)
  openTransport()
  return () => {
    listeners.delete(listener)
    closeTransportIfUnused()
  }
}

/** Clear this tab and send a content-free invalidation to its sibling tabs. */
export function publishAuthInvalidation() {
  emit('local')
  const nonce = randomNonce()
  firstSeen(nonce)
  if (channel) {
    try { channel.postMessage(nonce) } catch { /* Storage remains a second transport. */ }
  }
  if (typeof window !== 'undefined') {
    try {
      localStorage.setItem(AUTH_INVALIDATION_STORAGE_KEY, nonce)
      localStorage.removeItem(AUTH_INVALIDATION_STORAGE_KEY)
    } catch { /* Browser storage may be disabled; the local event still applies. */ }
  }
}
