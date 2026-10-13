import { createContext, useCallback, useContext, useEffect, useReducer, useRef, type ReactNode } from 'react'
import * as api from './api'
import { createRerunSubmission, createSubmission, currentTask, loadInitialState, persistableUi, reducer, STORAGE_KEY, type Action, type SubmissionOverrides, type WorkspaceState } from './store'
import { isActive, type AssistantId, type TaskSnapshot } from './types'

interface WorkspaceContextValue {
  state: WorkspaceState
  dispatch: React.Dispatch<Action>
  refresh: () => Promise<void>
  send: (id: AssistantId, retry?: boolean, overrides?: SubmissionOverrides) => Promise<void>
  rerun: (id: AssistantId, task: TaskSnapshot) => Promise<void>
  uploadFiles: (id: AssistantId, files: File[]) => Promise<void>
  cancel: (id: AssistantId, taskId: string) => Promise<void>
  reset: (id: AssistantId) => Promise<void>
}
const Context = createContext<WorkspaceContextValue | null>(null)
const messageOf = (error: unknown) => error instanceof Error ? error.message : '请求未完成，请重试。'

export function WorkspaceProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, undefined, loadInitialState)
  const stateRef = useRef(state)
  stateRef.current = state
  const sourceRef = useRef<EventSource | null>(null)
  const refreshId = useRef(0)
  const controllerRef = useRef<AbortController | null>(null)
  const mounted = useRef(true)

  const refresh = useCallback(async () => {
    const generation = ++refreshId.current
    sourceRef.current?.close()
    sourceRef.current = null
    controllerRef.current?.abort()
    const controller = new AbortController()
    controllerRef.current = controller
    dispatch({ type: 'loading' })
    dispatch({ type: 'resourcesLoading', value: true })
    try {
      const snapshot = await api.getWorkspace(controller.signal)
      if (mounted.current && generation === refreshId.current) {
        dispatch({ type: 'snapshot', snapshot, now: Date.now() })
        try {
          const [uploads, plans] = await Promise.all([
            api.getUploads(snapshot.workspace_id, controller.signal),
            api.getPlans(controller.signal),
          ])
          if (mounted.current && generation === refreshId.current) {
            dispatch({ type: 'resources', uploads: uploads.uploads, plans: plans.plans, defaults: plans.defaults })
          }
        } catch (resourceError) {
          if (!controller.signal.aborted && mounted.current && generation === refreshId.current) {
            dispatch({ type: 'resourceError', message: messageOf(resourceError) })
          }
        }
      }
    } catch (error) {
      if (!controller.signal.aborted && mounted.current && generation === refreshId.current) {
        dispatch({ type: 'bootstrapError', message: messageOf(error) })
        dispatch({ type: 'resourceError', message: messageOf(error) })
      }
    }
  }, [])

  useEffect(() => {
    mounted.current = true
    void refresh()
    return () => {
      mounted.current = false
      ++refreshId.current
      controllerRef.current?.abort()
      sourceRef.current?.close()
    }
  }, [refresh])

  useEffect(() => {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(persistableUi(state))) }
    catch { /* Storage may be unavailable; in-memory operation still works. */ }
  }, [state.workspaceId, state.instanceId, state.selected, state.assistants])

  useEffect(() => {
    if (!state.loaded || state.loading || !state.workspaceId || !state.instanceId) return
    const instanceId = state.instanceId
    // Cursor changes and assistant navigation deliberately do not recreate SSE.
    const source = new EventSource(api.eventsUrl(state.workspaceId, instanceId, state.cursor))
    sourceRef.current = source
    let disposed = false
    let checkingServer = false
    const current = () => !disposed && sourceRef.current === source && mounted.current
    source.onopen = () => { if (current()) dispatch({ type: 'connection', value: 'live' }) }
    source.onerror = () => {
      if (!current()) return
      dispatch({ type: 'connection', value: 'reconnecting' })
      // Native retry preserves Last-Event-ID. A restarted backend, however,
      // rejects the old workspace; check its identity without resubmitting work.
      if (checkingServer) return
      checkingServer = true
      void api.getWorkspace().then(snapshot => {
        if (current() && (snapshot.server_instance_id !== instanceId
            || snapshot.workspace_id !== state.workspaceId || source.readyState === EventSource.CLOSED)) {
          source.close()
          void refresh()
        }
      }).catch(() => {
        if (current() && source.readyState === EventSource.CLOSED) {
          dispatch({ type: 'bootstrapError', message: '连接已断开，请重新连接服务。' })
        }
      }).finally(() => { checkingServer = false })
    }
    source.addEventListener('task.updated', (event) => {
      if (!current()) return
      const message = event as MessageEvent<string>
      if (message.lastEventId && !message.lastEventId.startsWith(instanceId + ':')) {
        source.close()
        void refresh()
        return
      }
      try {
        dispatch({ type: 'task', task: JSON.parse(message.data) as TaskSnapshot,
          instanceId, cursor: message.lastEventId, now: Date.now() })
      } catch {
        source.close()
        void refresh()
      }
    })
    source.addEventListener('assistant.reset', (event) => {
      if (!current()) return
      const message = event as MessageEvent<string>
      try {
        const payload = JSON.parse(message.data) as { assistant_id: AssistantId }
        dispatch({ type: 'assistantReset', id: payload.assistant_id, instanceId, cursor: message.lastEventId })
      } catch { void refresh() }
    })
    source.addEventListener('snapshot_required', () => {
      if (!current()) return
      source.close()
      void refresh()
    })
    return () => {
      disposed = true
      source.close()
      if (sourceRef.current === source) sourceRef.current = null
    }
    // The snapshot epoch owns the connection; task updates and tab switches do not.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.loaded, state.loading, state.workspaceId, state.instanceId, state.connectionEpoch, refresh])

  const send = useCallback(async (id: AssistantId, retry = false, overrides: SubmissionOverrides = {}) => {
    const snapshot = stateRef.current
    const assistant = snapshot.assistants[id]
    if (!snapshot.workspaceId || !snapshot.instanceId || assistant.submitting || assistant.resetting || isActive(currentTask(assistant)) || snapshot.uploadingAssistant === id) return
    const draft = assistant.draft
    if (!retry && !(overrides.prompt ?? draft.prompt).trim()) return
    const payload = createSubmission(assistant, snapshot.workspaceId, crypto.randomUUID(), retry, overrides)
    dispatch({ type: 'submitStart', payload })
    try {
      const task = await api.submitTask(payload)
      if (!mounted.current) return
      dispatch({ type: 'task', task, instanceId: snapshot.instanceId, now: Date.now() })
      dispatch({ type: 'inspect', id, taskId: task.task_id })
    } catch (error) {
      if (!mounted.current) return
      dispatch({ type: 'submitFailed', id, requestId: payload.request_id,
        message: messageOf(error), instanceId: snapshot.instanceId })
      if (error instanceof api.ApiError && [404, 409].includes(error.status)) await refresh()
    }
  }, [refresh])

  const rerun = useCallback(async (id: AssistantId, task: TaskSnapshot) => {
    if (!['failed', 'cancelled'].includes(task.status)) return
    const snapshot = stateRef.current
    const assistant = snapshot.assistants[id]
    if (task.assistant_id !== id || !snapshot.workspaceId || !snapshot.instanceId
      || assistant.submitting || assistant.resetting || isActive(currentTask(assistant))
      || snapshot.uploadingAssistant === id || snapshot.resourcesLoading) return
    const payload = createRerunSubmission(task, snapshot.workspaceId, crypto.randomUUID())
    if (!payload.prompt) return
    dispatch({ type: 'submitStart', payload })
    try {
      const nextTask = await api.submitTask(payload)
      if (!mounted.current) return
      dispatch({ type: 'task', task: nextTask, instanceId: snapshot.instanceId, now: Date.now() })
      dispatch({ type: 'inspect', id, taskId: nextTask.task_id })
    } catch (error) {
      if (!mounted.current) return
      dispatch({ type: 'submitFailed', id, requestId: payload.request_id,
        message: messageOf(error), instanceId: snapshot.instanceId })
      if (error instanceof api.ApiError && [404, 409].includes(error.status)) await refresh()
    }
  }, [refresh])

  useEffect(() => {
    const revalidateRestoredPage = (event: PageTransitionEvent) => {
      if (event.persisted) void refresh()
    }
    window.addEventListener('pageshow', revalidateRestoredPage)
    return () => window.removeEventListener('pageshow', revalidateRestoredPage)
  }, [refresh])

  const uploadFiles = useCallback(async (id: AssistantId, files: File[]) => {
    const snapshot = stateRef.current
    if (!snapshot.workspaceId || !files.length || snapshot.uploadingAssistant !== null) return
    const assistant = snapshot.assistants[id]
    const attachmentIds = [...assistant.draft.attachment_ids]
    dispatch({ type: 'uploading', id, value: true })
    try {
      for (const file of files) {
        const upload = await api.uploadAsset(snapshot.workspaceId, file)
        if (!mounted.current) return
        dispatch({ type: 'uploadAdded', upload })
        if (!attachmentIds.includes(upload.asset_id)) attachmentIds.push(upload.asset_id)
        dispatch({ type: 'draft', id, patch: { attachment_ids: [...attachmentIds] } })
      }
    } finally {
      if (mounted.current) dispatch({ type: 'uploading', id, value: false })
    }
  }, [])

  const cancel = useCallback(async (id: AssistantId, taskId: string) => {
    const snapshot = stateRef.current
    if (!snapshot.workspaceId || !snapshot.instanceId || snapshot.assistants[id].cancelling) return
    dispatch({ type: 'operation', id, operation: 'cancelling', value: true })
    try {
      const task = await api.cancelTask(taskId, snapshot.workspaceId)
      if (mounted.current) dispatch({ type: 'task', task, instanceId: snapshot.instanceId, now: Date.now() })
    } catch (error) {
      if (mounted.current) dispatch({ type: 'operation', id, operation: 'cancelling', value: false,
        error: messageOf(error), instanceId: snapshot.instanceId })
      if (error instanceof api.ApiError && [404, 409].includes(error.status)) await refresh()
    }
  }, [refresh])

  const reset = useCallback(async (id: AssistantId) => {
    const snapshot = stateRef.current
    const assistant = snapshot.assistants[id]
    if (!snapshot.workspaceId || !snapshot.instanceId || assistant.submitting || assistant.resetting || isActive(currentTask(assistant))) return
    dispatch({ type: 'operation', id, operation: 'resetting', value: true })
    try {
      const workspace = await api.resetAssistant(id, snapshot.workspaceId)
      if (mounted.current && snapshot.instanceId === stateRef.current.instanceId) {
        dispatch({ type: 'snapshot', snapshot: workspace, now: Date.now() })
      }
    } catch (error) {
      if (mounted.current) dispatch({ type: 'operation', id, operation: 'resetting', value: false,
        error: messageOf(error), instanceId: snapshot.instanceId })
      if (error instanceof api.ApiError && [404, 409].includes(error.status)) await refresh()
    }
  }, [refresh])

  return <Context.Provider value={{ state, dispatch, refresh, send, rerun, uploadFiles, cancel, reset }}>{children}</Context.Provider>
}

export function useWorkspace() {
  const context = useContext(Context)
  if (!context) throw new Error('WorkspaceProvider is required')
  return context
}
