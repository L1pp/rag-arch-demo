import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import LoginPage from './LoginPage'

const mocks = vi.hoisted(() => ({
  getAuthSession: vi.fn(),
  login: vi.fn(),
  publishAuthInvalidation: vi.fn(),
  listener: null as null | ((event: { source: 'local' | 'remote' }) => void),
}))

vi.mock('./api', () => ({ getAuthSession: mocks.getAuthSession, login: mocks.login }))
vi.mock('./authLifecycle', () => ({
  publishAuthInvalidation: mocks.publishAuthInvalidation,
  subscribeAuthInvalidation: (listener: typeof mocks.listener) => {
    mocks.listener = listener
    return () => { mocks.listener = null }
  },
}))

describe('React login page', () => {
  beforeEach(() => {
    window.history.replaceState(null, '', '/login?next=%2Fviewer%2Fpages%3Fasset%3Dplan-a%23page')
    mocks.getAuthSession.mockReset().mockResolvedValue(null)
    mocks.login.mockReset().mockResolvedValue({ authenticated: true, owner_user_id: 12, username: 'alice' })
    mocks.publishAuthInvalidation.mockReset()
    mocks.listener = null
  })

  it('posts credentials as JSON and returns to a safe same-origin next path after login', async () => {
    const navigate = vi.fn()
    render(<LoginPage navigate={navigate} />)
    const username = await screen.findByLabelText('用户名')
    fireEvent.change(username, { target: { value: 'alice' } })
    fireEvent.change(screen.getByLabelText('密码'), { target: { value: 'secret' } })
    fireEvent.click(screen.getByRole('button', { name: '登录' }))

    await waitFor(() => expect(mocks.login).toHaveBeenCalledWith({ username: 'alice', password: 'secret' }, expect.any(AbortSignal)))
    expect(mocks.publishAuthInvalidation).toHaveBeenCalledTimes(1)
    expect(navigate).toHaveBeenCalledWith('/viewer/pages?asset=plan-a#page')
  })

  it('rechecks the server session after a remote invalidation without trusting its payload', async () => {
    mocks.getAuthSession.mockResolvedValueOnce(null).mockResolvedValueOnce({
      authenticated: true, owner_user_id: 21, username: 'bob',
    })
    const navigate = vi.fn()
    render(<LoginPage navigate={navigate} />)
    await screen.findByLabelText('用户名')
    mocks.listener?.({ source: 'remote' })
    await waitFor(() => expect(navigate).toHaveBeenCalledWith('/viewer/pages?asset=plan-a#page'))
    expect(mocks.getAuthSession).toHaveBeenCalledTimes(2)
  })

  it('falls back to the workbench for an external next URL', async () => {
    window.history.replaceState(null, '', '/login?next=https%3A%2F%2Fevil.example')
    mocks.getAuthSession.mockResolvedValue({ authenticated: true, owner_user_id: 30, username: 'alice' })
    const navigate = vi.fn()
    render(<LoginPage navigate={navigate} />)
    await waitFor(() => expect(navigate).toHaveBeenCalledWith('/'))
  })
})
