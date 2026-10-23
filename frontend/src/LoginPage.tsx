import { useCallback, useEffect, useRef, useState, type FormEvent } from 'react'
import { getAuthSession, login } from './api'
import { publishAuthInvalidation, subscribeAuthInvalidation } from './authLifecycle'
import { safeNextPath } from './route'

interface LoginPageProps {
  navigate?: (path: string) => void
}

function currentNextPath() {
  return safeNextPath(new URLSearchParams(window.location.search).get('next'))
}

function browserNavigate(path: string) {
  window.location.replace(path)
}

export default function LoginPage({ navigate = browserNavigate }: LoginPageProps) {
  const nextPath = currentNextPath()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [checking, setChecking] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const requestId = useRef(0)
  const sessionController = useRef<AbortController | null>(null)
  const loginController = useRef<AbortController | null>(null)
  const mounted = useRef(true)

  const checkSession = useCallback(async () => {
    const generation = ++requestId.current
    sessionController.current?.abort()
    const controller = new AbortController()
    sessionController.current = controller
    setChecking(true)
    try {
      const session = await getAuthSession(controller.signal)
      if (!mounted.current || generation !== requestId.current || controller.signal.aborted) return
      if (session) navigate(nextPath)
      else setError(null)
    } catch (reason) {
      if (!mounted.current || generation !== requestId.current || controller.signal.aborted) return
      setError(reason instanceof Error ? reason.message : '暂时无法确认登录状态，请重试。')
    } finally {
      if (mounted.current && generation === requestId.current && !controller.signal.aborted) setChecking(false)
    }
  }, [navigate, nextPath])

  useEffect(() => {
    mounted.current = true
    const unsubscribe = subscribeAuthInvalidation(event => {
      if (event.source === 'remote') void checkSession()
    })
    void checkSession()
    return () => {
      mounted.current = false
      ++requestId.current
      sessionController.current?.abort()
      loginController.current?.abort()
      unsubscribe()
    }
  }, [checkSession])

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const cleanUsername = username.trim()
    if (!cleanUsername || !password || submitting || checking) return
    loginController.current?.abort()
    const controller = new AbortController()
    loginController.current = controller
    setSubmitting(true)
    setError(null)
    try {
      await login({ username: cleanUsername, password }, controller.signal)
      if (!mounted.current || controller.signal.aborted) return
      publishAuthInvalidation()
      navigate(nextPath)
    } catch (reason) {
      if (!controller.signal.aborted && mounted.current) {
        setError(reason instanceof Error ? reason.message : '登录未成功，请检查账号和密码。')
      }
    } finally {
      if (mounted.current && !controller.signal.aborted) setSubmitting(false)
    }
  }

  return <main className="login-page">
    <section className="login-card" aria-labelledby="login-title">
      <div className="login-brand"><span className="brand-symbol">建</span><div><strong>建筑资料助手</strong><span>安全工作区</span></div></div>
      <div className="login-copy">
        <div className="eyebrow">账号登录</div>
        <h1 id="login-title">登录你的工作区</h1>
        <p>登录后继续使用聊天、规范问答和方案核查。</p>
      </div>
      <form className="login-form" onSubmit={submit}>
        <label htmlFor="login-username">用户名</label>
        <input id="login-username" name="username" autoComplete="username" autoFocus
          value={username} onChange={event => setUsername(event.target.value)} disabled={submitting}
          required maxLength={128} />
        <label htmlFor="login-password">密码</label>
        <input id="login-password" name="password" type="password" autoComplete="current-password"
          value={password} onChange={event => setPassword(event.target.value)} disabled={submitting}
          required />
        {error && <div className="login-error" role="alert">{error}</div>}
        <button type="submit" disabled={checking || submitting || !username.trim() || !password}>
          {checking ? '正在确认登录状态…' : submitting ? '正在登录…' : '登录'}
        </button>
      </form>
      <p className="login-footnote">请使用分配给你的账号。工作区数据按账号隔离。</p>
    </section>
  </main>
}
