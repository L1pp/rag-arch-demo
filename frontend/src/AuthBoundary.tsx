import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import { flushSync } from 'react-dom'
import { getAuthSession, redirectToLogin } from './api'
import { subscribeAuthInvalidation } from './authLifecycle'
import { clearTenantUiForOwner } from './store'

type GateState = 'checking' | 'ready' | 'blocked' | 'error'

export default function AuthBoundary({ children }: { children: ReactNode }) {
  const [gate, setGate] = useState<GateState>('checking')
  const [error, setError] = useState('')
  const ownerUserId = useRef<number | null>(null)
  const generation = useRef(0)
  const controller = useRef<AbortController | null>(null)
  const mounted = useRef(true)

  const revalidate = useCallback(async () => {
    const currentGeneration = ++generation.current
    controller.current?.abort()
    const nextController = new AbortController()
    controller.current = nextController
    setError('')
    setGate('checking')
    try {
      const session = await getAuthSession(nextController.signal)
      if (!mounted.current || currentGeneration !== generation.current || nextController.signal.aborted) return
      if (!session) {
        clearTenantUiForOwner(ownerUserId.current)
        ownerUserId.current = null
        setGate('blocked')
        redirectToLogin()
        return
      }
      if (ownerUserId.current !== null && ownerUserId.current !== session.owner_user_id) {
        clearTenantUiForOwner(ownerUserId.current)
      }
      ownerUserId.current = session.owner_user_id
      setGate('ready')
    } catch (reason) {
      if (!mounted.current || currentGeneration !== generation.current || nextController.signal.aborted) return
      setError(reason instanceof Error ? reason.message : '暂时无法确认登录状态。')
      setGate('error')
    }
  }, [])

  const block = useCallback((recheck: boolean, clearPrivateCache = true) => {
    // Any identity boundary makes this tab's old private draft cache ineligible
    // for reuse. Drafts remain tenant-partitioned during normal sessions.
    if (clearPrivateCache) {
      clearTenantUiForOwner(ownerUserId.current)
      ownerUserId.current = null
    }
    ++generation.current
    controller.current?.abort()
    controller.current = null
    flushSync(() => setGate(recheck ? 'checking' : 'blocked'))
    if (recheck) queueMicrotask(() => void revalidate())
  }, [revalidate])

  useEffect(() => {
    mounted.current = true
    const unsubscribe = subscribeAuthInvalidation(event => {
      // A nonce only invalidates this view. Identity is always obtained from
      // the authenticated session endpoint after the screen has been removed.
      block(event.source === 'remote')
    })
    void revalidate()
    const onPageShow = (event: PageTransitionEvent) => {
      if (event.persisted) block(true, false)
    }
    window.addEventListener('pageshow', onPageShow)
    return () => {
      mounted.current = false
      ++generation.current
      controller.current?.abort()
      unsubscribe()
      window.removeEventListener('pageshow', onPageShow)
    }
  }, [block, revalidate])

  if (gate === 'ready') return <>{children}</>
  if (gate === 'error') return <main className="auth-gate" role="alert">
    <p>无法确认当前登录状态：{error}</p>
    <button type="button" onClick={() => void revalidate()}>重试</button>
  </main>
  return <main className="auth-gate" role="status">
    {gate === 'checking' ? '正在确认登录状态…' : '登录已失效，正在前往登录页…'}
  </main>
}
