import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { WorkspaceProvider, useWorkspace } from './WorkspaceContext'
import { STORAGE_KEY, tenantStorageKey } from './store'
import { assistantIds, type TaskSnapshot, type WorkspaceSnapshot } from './types'

const apiMock = vi.hoisted(() => ({
  getWorkspace: vi.fn(),
  getUploads: vi.fn(),
  getPlans: vi.fn(),
  submitTask: vi.fn(),
  cancelTask: vi.fn(),
  resetAssistant: vi.fn(),
  uploadAsset: vi.fn(),
  redirectToLogin: vi.fn(),
  eventsUrl: vi.fn((workspaceId: string) => `/api/events?workspace_id=${workspaceId}`),
  ApiError: class extends Error { status = 0 },
}))

vi.mock('./api', () => apiMock)

class FakeEventSource {
  static readonly CLOSED = 2
  readyState = 1
  onopen: ((event: Event) => void) | null = null
  onerror: ((event: Event) => void) | null = null
  close = vi.fn(() => { this.readyState = FakeEventSource.CLOSED })
  listeners = new Map<string, (event: Event) => void>()
  constructor(readonly url: string, readonly options?: EventSourceInit) {}
  addEventListener(type: string, listener: EventListenerOrEventListenerObject) {
    if (typeof listener === 'function') this.listeners.set(type, listener as (event: Event) => void)
  }
}

function workspace(owner: number, workspaceId: string): WorkspaceSnapshot {
  return {
    owner_user_id: owner,
    workspace_id: workspaceId,
    server_instance_id: 'server-shared',
    cursor: 0,
    mode: 'infrastructure_demo',
    assistants: assistantIds.map(assistant_id => ({
      assistant_id,
      title: assistant_id,
      tasks: [],
      active_task_id: null,
    })),
  }
}

function task(overrides: Partial<TaskSnapshot> = {}): TaskSnapshot {
  return {
    task_id: 'task-user-a', assistant_id: 'chat', request_id: 'request-user-a', prompt: '用户 A 的任务',
    status: 'completed', stage: 'completed', detail: '', created_at: '2026-10-19T00:00:00Z',
    started_at: '2026-10-19T00:00:00Z', finished_at: '2026-10-19T00:00:01Z', elapsed_seconds: 1,
    duration_seconds: null, updated_seq: 1, result: null, error: null, simulation: false,
    ...overrides,
  }
}

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(done => { resolve = done })
  return { promise, resolve }
}

function Harness() {
  const { state, dispatch, refresh, send } = useWorkspace()
  return <>
    <span data-testid="owner">{state.ownerUserId ?? 'none'}</span>
    <span data-testid="workspace">{state.workspaceId ?? 'none'}</span>
    <span data-testid="prompt">{state.assistants.chat.draft.prompt}</span>
    <span data-testid="attachments">{state.assistants.chat.draft.attachment_ids.join(',')}</span>
    <span data-testid="uploads">{state.uploads.map(upload => upload.asset_id).join(',')}</span>
    <span data-testid="tasks">{state.assistants.chat.tasks.map(item => item.task_id).join(',')}</span>
    <button onClick={() => dispatch({ type: 'draft', id: 'chat', patch: {
      prompt: '用户 A 的本地草稿', attachment_ids: ['asset-a'],
    } })}>编辑草稿</button>
    <button onClick={() => void send('chat')}>发送</button>
    <button onClick={() => void refresh()}>刷新工作区</button>
  </>
}

describe('tenant workspace bootstrap', () => {
  afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals() })

  beforeEach(() => {
    localStorage.clear()
    apiMock.getWorkspace.mockReset()
    apiMock.getUploads.mockReset()
    apiMock.getPlans.mockReset()
    apiMock.submitTask.mockReset()
    apiMock.cancelTask.mockReset()
    apiMock.resetAssistant.mockReset()
    apiMock.uploadAsset.mockReset()
    apiMock.redirectToLogin.mockReset()
    vi.stubGlobal('EventSource', FakeEventSource)
  })

  it('clears a revoked session and ignores late SSE and HTTP without resubmitting', async () => {
    apiMock.getWorkspace.mockResolvedValue(workspace(101, 'workspace-a'))
    apiMock.getUploads.mockResolvedValue({ uploads: [{ asset_id: 'asset-a', filename: 'A.txt', size_bytes: 10 }] })
    apiMock.getPlans.mockResolvedValue({ plans: [], defaults: { max_steps: 50, max_tool_calls: 150 } })
    const pending = deferred<TaskSnapshot>()
    apiMock.submitTask.mockReturnValue(pending.promise)
    const sources: FakeEventSource[] = []
    vi.stubGlobal('EventSource', class extends FakeEventSource {
      constructor(url: string, options?: EventSourceInit) { super(url, options); sources.push(this) }
    })
    render(<WorkspaceProvider><Harness/></WorkspaceProvider>)
    await waitFor(() => expect(sources.length).toBe(1))
    expect(sources[0].options).toEqual({ withCredentials: true })
    fireEvent.click(screen.getByRole('button', { name: '编辑草稿' }))
    fireEvent.click(screen.getByRole('button', { name: '发送' }))
    await waitFor(() => expect(apiMock.submitTask).toHaveBeenCalledTimes(1))
    const signal = apiMock.submitTask.mock.calls[0][1] as AbortSignal
    const source = sources[0]
    act(() => source.listeners.get('auth_required')!(new Event('auth_required')))
    await waitFor(() => expect(screen.getByTestId('owner')).toHaveTextContent('none'))
    expect(source.close).toHaveBeenCalled()
    expect(signal.aborted).toBe(true)
    expect(screen.getByTestId('prompt')).toBeEmptyDOMElement()
    expect(screen.getByTestId('uploads')).toBeEmptyDOMElement()
    expect(apiMock.redirectToLogin).toHaveBeenCalledTimes(1)
    await act(async () => {
      source.listeners.get('task.updated')!(new MessageEvent('task.updated', {
        data: JSON.stringify(task()), lastEventId: 'server-shared:1',
      }))
      pending.resolve(task())
      await pending.promise
    })
    expect(screen.getByTestId('tasks')).toBeEmptyDOMElement()
    expect(screen.getByTestId('owner')).toHaveTextContent('none')
    expect(apiMock.submitTask).toHaveBeenCalledTimes(1)
  })

  it('backs off terminal SSE failures before refreshing instead of looping on quota rejection', async () => {
    apiMock.getWorkspace.mockResolvedValue(workspace(101, 'workspace-a'))
    apiMock.getUploads.mockResolvedValue({ uploads: [] })
    apiMock.getPlans.mockResolvedValue({ plans: [], defaults: { max_steps: 50, max_tool_calls: 150 } })
    const sources: FakeEventSource[] = []
    vi.stubGlobal('EventSource', class extends FakeEventSource {
      constructor(url: string) { super(url); sources.push(this) }
    })
    render(<WorkspaceProvider><Harness/></WorkspaceProvider>)
    await waitFor(() => expect(sources.length).toBe(1))
    vi.useFakeTimers()
    sources[0].readyState = FakeEventSource.CLOSED
    act(() => sources[0].onerror!(new Event('error')))
    expect(apiMock.getWorkspace).toHaveBeenCalledTimes(1)
    await act(async () => { await vi.advanceTimersByTimeAsync(999) })
    expect(apiMock.getWorkspace).toHaveBeenCalledTimes(1)
    await act(async () => { await vi.advanceTimersByTimeAsync(1) })
    expect(apiMock.getWorkspace.mock.calls.length).toBeGreaterThan(1)
    expect(apiMock.submitTask).not.toHaveBeenCalled()
  })

  it('does not submit a replacement while the current task is recovering', async () => {
    const snapshot = workspace(101, 'workspace-a')
    const recovering = task({ status: 'recovering', stage: 'recovering', finished_at: null })
    snapshot.assistants[0].tasks = [recovering]
    snapshot.assistants[0].active_task_id = recovering.task_id
    apiMock.getWorkspace.mockResolvedValue(snapshot)
    apiMock.getUploads.mockResolvedValue({ uploads: [] })
    apiMock.getPlans.mockResolvedValue({ plans: [], defaults: { max_steps: 50, max_tool_calls: 150 } })
    render(<WorkspaceProvider><Harness/></WorkspaceProvider>)
    await waitFor(() => expect(screen.getByTestId('tasks')).toHaveTextContent(recovering.task_id))
    fireEvent.click(screen.getByRole('button', { name: '编辑草稿' }))
    await waitFor(() => expect(screen.getByTestId('prompt')).toHaveTextContent('用户 A 的本地草稿'))
    fireEvent.click(screen.getByRole('button', { name: '发送' }))
    await act(async () => { await Promise.resolve() })
    expect(apiMock.submitTask).not.toHaveBeenCalled()
  })

  it('aborts the old tenant request, closes its SSE, and ignores a late response after switching users', async () => {
    const userA = workspace(101, 'workspace-a')
    const userB = workspace(202, 'workspace-b')
    const lateSubmission = deferred<TaskSnapshot>()
    apiMock.getWorkspace.mockResolvedValueOnce(userA).mockResolvedValueOnce(userB)
    apiMock.getUploads.mockResolvedValueOnce({ uploads: [{ asset_id: 'asset-a', filename: 'A.txt', size_bytes: 10 }] })
      .mockResolvedValueOnce({ uploads: [{ asset_id: 'asset-b', filename: 'B.txt', size_bytes: 20 }] })
    apiMock.getPlans.mockResolvedValue({ plans: [], defaults: { max_steps: 50, max_tool_calls: 150 } })
    apiMock.submitTask.mockReturnValue(lateSubmission.promise)
    localStorage.setItem(tenantStorageKey(202, 'workspace-b'), JSON.stringify({
      owner_user_id: 202,
      workspace_id: 'workspace-b',
      selected_assistant: 'plan',
      drafts: { chat: { prompt: '用户 B 的草稿', attachment_ids: ['asset-b'] } },
    }))

    const sources: FakeEventSource[] = []
    const NativeEventSource = FakeEventSource
    vi.stubGlobal('EventSource', class extends NativeEventSource {
      constructor(url: string, options?: EventSourceInit) { super(url, options); sources.push(this) }
    })
    render(<WorkspaceProvider><Harness/></WorkspaceProvider>)
    await waitFor(() => expect(screen.getByTestId('owner')).toHaveTextContent('101'))
    await waitFor(() => expect(screen.getByTestId('uploads')).toHaveTextContent('asset-a'))
    fireEvent.click(screen.getByRole('button', { name: '编辑草稿' }))
    await waitFor(() => expect(JSON.parse(localStorage.getItem(tenantStorageKey(101, 'workspace-a')) || '{}'))
      .toMatchObject({ owner_user_id: 101, workspace_id: 'workspace-a' }))
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: '发送' }))
    await waitFor(() => expect(apiMock.submitTask).toHaveBeenCalledTimes(1))
    const oldSignal = apiMock.submitTask.mock.calls[0][1] as AbortSignal
    expect(oldSignal.aborted).toBe(false)

    fireEvent.click(screen.getByRole('button', { name: '刷新工作区' }))
    await waitFor(() => expect(screen.getByTestId('owner')).toHaveTextContent('202'))
    await waitFor(() => expect(screen.getByTestId('attachments')).toHaveTextContent('asset-b'))
    await waitFor(() => expect(screen.getByTestId('uploads')).toHaveTextContent('asset-b'))
    expect(oldSignal.aborted).toBe(true)
    expect(sources.length).toBeGreaterThanOrEqual(2)
    expect(sources[0].close).toHaveBeenCalled()
    expect(screen.getByTestId('tasks')).toBeEmptyDOMElement()
    expect(screen.getByTestId('prompt')).toHaveTextContent('用户 B 的草稿')

    await act(async () => {
      lateSubmission.resolve(task())
      await lateSubmission.promise
    })
    await waitFor(() => expect(screen.getByTestId('owner')).toHaveTextContent('202'))
    expect(screen.getByTestId('tasks')).toBeEmptyDOMElement()
    expect(screen.getByTestId('prompt')).toHaveTextContent('用户 B 的草稿')
  })
})
