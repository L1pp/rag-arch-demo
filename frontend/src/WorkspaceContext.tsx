import { createContext, useCallback, useContext, useEffect, useReducer, useRef, type ReactNode } from 'react'
import * as api from './api'
import { publishAuthInvalidation } from './authLifecycle'
import { createRerunSubmission, createSubmission, currentTask, loadInitialState, loadTenantUi, persistableUi, reducer, tenantStorageKey, type Action, type SubmissionOverrides, type WorkspaceState } from './store'
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
const isAbortError = (error: unknown) => !!error && typeof error === 'object' && 'name' in error
  && (error as { name?: unknown }).name === 'AbortError'

interface TenantToken {
  ownerUserId: number | null
  workspaceId: string | null
  epoch: number
}

function sameTenant(left: TenantToken, right: TenantToken) {
  return left.ownerUserId === right.ownerUserId && left.workspaceId === right.workspaceId
}

export function WorkspaceProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, undefined, loadInitialState)
  const stateRef = useRef(state)
  stateRef.current = state
  const sourceRef = useRef<EventSource | null>(null)
  const reconnectDelayRef = useRef(1000)
  const refreshId = useRef(0)
  const refreshControllerRef = useRef<AbortController | null>(null)
  const requestControllersRef = useRef(new Set<AbortController>())
  const tenantRef = useRef<TenantToken>({ ownerUserId: state.ownerUserId, workspaceId: state.workspaceId, epoch: 0 })
  const mounted = useRef(true)

  const registerController = () => {
    const controller = new AbortController()
    requestControllersRef.current.add(controller)
    return controller
  }
  const releaseController = (controller: AbortController) => {
    requestControllersRef.current.delete(controller)
    if (refreshControllerRef.current === controller) refreshControllerRef.current = null
  }
  const requestIsCurrent = (token: TenantToken) => {
    const currentTenant = tenantRef.current
    return mounted.current && token.epoch === currentTenant.epoch && sameTenant(token, currentTenant)
  }
  const stateMatchesTenant = (currentState: WorkspaceState) => sameTenant(tenantRef.current, {
    ownerUserId: currentState.ownerUserId,
    workspaceId: currentState.workspaceId,
    epoch: tenantRef.current.epoch,
  })

  const refresh = useCallback(async () => {
    const generation = ++refreshId.current
    sourceRef.current?.close()
    sourceRef.current = null
    refreshControllerRef.current?.abort()
    const controller = registerController()
    refreshControllerRef.current = controller
    dispatch({ type: 'loading' })
    dispatch({ type: 'resourcesLoading', value: true })
    try {
      const snapshot = await api.getWorkspace(controller.signal)
      if (mounted.current && generation === refreshId.current && !controller.signal.aborted) {
        const incomingTenant: TenantToken = {
          ownerUserId: snapshot.owner_user_id ?? null,
          workspaceId: snapshot.workspace_id,
          epoch: tenantRef.current.epoch,
        }
        const tenantChanged = !sameTenant(tenantRef.current, incomingTenant)
        if (tenantChanged) {
          tenantRef.current = { ...incomingTenant, epoch: tenantRef.current.epoch + 1 }
          for (const pending of requestControllersRef.current) {
            if (pending !== controller) pending.abort()
          }
        }
        const token = tenantRef.current
        dispatch({ type: 'snapshot', snapshot, now: Date.now(),
          savedUi: tenantChanged ? loadTenantUi(token.ownerUserId, token.workspaceId!) : undefined })
        try {
          const [uploads, plans] = await Promise.all([
            api.getUploads(snapshot.workspace_id, controller.signal),
            api.getPlans(controller.signal),
          ])
          if (requestIsCurrent(token) && generation === refreshId.current && !controller.signal.aborted) {
            dispatch({ type: 'resources', uploads: uploads.uploads, plans: plans.plans, defaults: plans.defaults })
          }
        } catch (resourceError) {
          if (!controller.signal.aborted && requestIsCurrent(token) && generation === refreshId.current) {
            dispatch({ type: 'resourceError', message: messageOf(resourceError) })
          }
        }
      }
    } catch (error) {
      if (!controller.signal.aborted && mounted.current && generation === refreshId.current) {
        dispatch({ type: 'bootstrapError', message: messageOf(error) })
        dispatch({ type: 'resourceError', message: messageOf(error) })
      }
    } finally {
      releaseController(controller)
    }
  }, [])

  useEffect(() => {
    mounted.current = true
    void refresh()
    return () => {
      mounted.current = false
      ++refreshId.current
      for (const controller of requestControllersRef.current) controller.abort()
      requestControllersRef.current.clear()
      refreshControllerRef.current = null
      sourceRef.current?.close()
    }
  }, [refresh])

  useEffect(() => {
    const tenant = tenantRef.current
    if (!state.loaded || state.ownerUserId === null || !state.workspaceId
      || !sameTenant(tenant, { ownerUserId: state.ownerUserId, workspaceId: state.workspaceId, epoch: tenant.epoch })) return
    try { localStorage.setItem(tenantStorageKey(state.ownerUserId, state.workspaceId), JSON.stringify(persistableUi(state))) }
    catch { /* Storage may be unavailable; in-memory operation still works. */ }
  }, [state.loaded, state.ownerUserId, state.workspaceId, state.instanceId, state.selected, state.assistants])

  useEffect(() => {
    if (!state.loaded || state.loading || !state.workspaceId || !state.instanceId) return
    const token = tenantRef.current
    if (!sameTenant(token, { ownerUserId: state.ownerUserId, workspaceId: state.workspaceId, epoch: token.epoch })) return
    const instanceId = state.instanceId
    // Cursor changes and assistant navigation deliberately do not recreate SSE.
    const source = new EventSource(api.eventsUrl(state.workspaceId, instanceId, state.cursor), { withCredentials: true })
    sourceRef.current = source
    let disposed = false
    let checkingServer = false
    let checkController: AbortController | null = null
    let retryTimer: ReturnType<typeof setTimeout> | null = null
    const current = () => !disposed && sourceRef.current === source && requestIsCurrent(token)
    source.onopen = () => {
      if (current()) {
        reconnectDelayRef.current = 1000
        dispatch({ type: 'connection', value: 'live' })
      }
    }
    source.onerror = () => {
      if (!current()) return
      dispatch({ type: 'connection', value: 'reconnecting' })
      // Native retry preserves Last-Event-ID. A restarted backend, however,
      // rejects the old workspace; check its identity without resubmitting work.
      if (checkingServer) return
      checkingServer = true
      const checkServer = () => {
      retryTimer = null
      if (!current()) { checkingServer = false; return }
      checkController = registerController()
      void api.getWorkspace(checkController.signal).then(snapshot => {
        if (current() && (snapshot.server_instance_id !== instanceId
            || snapshot.workspace_id !== state.workspaceId || (snapshot.owner_user_id ?? null) !== state.ownerUserId
            || source.readyState === EventSource.CLOSED)) {
          source.close()
          void refresh()
        }
      }).catch(() => {
        if (current() && source.readyState === EventSource.CLOSED) {
          dispatch({ type: 'bootstrapError', message: '连接已断开，请重新连接服务。' })
        }
      }).finally(() => {
        if (checkController) releaseController(checkController)
        checkController = null
        checkingServer = false
      })
      }
      // A terminal HTTP rejection (including connection quotas) must not
      // create a snapshot/reconnect request loop. Native retries retain ID.
      if (source.readyState === EventSource.CLOSED) {
        retryTimer = setTimeout(checkServer, reconnectDelayRef.current)
        reconnectDelayRef.current = Math.min(reconnectDelayRef.current * 2, 30000)
      } else checkServer()
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
    source.addEventListener('auth_required', () => {
      if (!current()) return
      // Revoke this view before navigating. Late HTTP/SSE callbacks must not
      // restore data belonging to the expired session while navigation runs.
      disposed = true
      source.close()
      sourceRef.current = null
      ++refreshId.current
      tenantRef.current = { ownerUserId: null, workspaceId: null, epoch: tenantRef.current.epoch + 1 }
      for (const controller of requestControllersRef.current) controller.abort()
      dispatch({ type: 'identityExpired' })
      publishAuthInvalidation()
      api.redirectToLogin()
    })
    return () => {
      disposed = true
      source.close()
      if (retryTimer !== null) clearTimeout(retryTimer)
      checkController?.abort()
      if (sourceRef.current === source) sourceRef.current = null
    }
    // The snapshot epoch owns the connection; task updates and tab switches do not.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.loaded, state.loading, state.ownerUserId, state.workspaceId, state.instanceId, state.connectionEpoch, refresh])

  const send = useCallback(async (id: AssistantId, retry = false, overrides: SubmissionOverrides = {}) => {
    const snapshot = stateRef.current
    const token = tenantRef.current
    const assistant = snapshot.assistants[id]
    if (!stateMatchesTenant(snapshot) || !requestIsCurrent(token) || !snapshot.workspaceId || !snapshot.instanceId
      || assistant.submitting || assistant.resetting || isActive(currentTask(assistant)) || snapshot.uploadingAssistant === id) return
    const draft = assistant.draft
    if (!retry && !(overrides.prompt ?? draft.prompt).trim()) return
    const payload = createSubmission(assistant, snapshot.workspaceId, crypto.randomUUID(), retry, overrides)
    const controller = registerController()
    dispatch({ type: 'submitStart', payload })
    try {
      const task = await api.submitTask(payload, controller.signal)
      if (!requestIsCurrent(token) || controller.signal.aborted) return
      dispatch({ type: 'task', task, instanceId: snapshot.instanceId, now: Date.now() })
      dispatch({ type: 'inspect', id, taskId: task.task_id })
    } catch (error) {
      if (!requestIsCurrent(token) || controller.signal.aborted || isAbortError(error)) return
      dispatch({ type: 'submitFailed', id, requestId: payload.request_id,
        message: messageOf(error), instanceId: snapshot.instanceId })
      if (error instanceof api.ApiError && [404, 409].includes(error.status)) await refresh()
    } finally {
      releaseController(controller)
    }
  }, [refresh])

  const rerun = useCallback(async (id: AssistantId, task: TaskSnapshot) => {
    if (!['failed', 'cancelled'].includes(task.status)) return
    const snapshot = stateRef.current
    const token = tenantRef.current
    const assistant = snapshot.assistants[id]
    if (!stateMatchesTenant(snapshot) || !requestIsCurrent(token) || task.assistant_id !== id || !snapshot.workspaceId || !snapshot.instanceId
      || assistant.submitting || assistant.resetting || isActive(currentTask(assistant))
      || snapshot.uploadingAssistant === id || snapshot.resourcesLoading) return
    const payload = createRerunSubmission(task, snapshot.workspaceId, crypto.randomUUID())
    if (!payload.prompt) return
    const controller = registerController()
    dispatch({ type: 'submitStart', payload })
    try {
      const nextTask = await api.submitTask(payload, controller.signal)
      if (!requestIsCurrent(token) || controller.signal.aborted) return
      dispatch({ type: 'task', task: nextTask, instanceId: snapshot.instanceId, now: Date.now() })
      dispatch({ type: 'inspect', id, taskId: nextTask.task_id })
    } catch (error) {
      if (!requestIsCurrent(token) || controller.signal.aborted || isAbortError(error)) return
      dispatch({ type: 'submitFailed', id, requestId: payload.request_id,
        message: messageOf(error), instanceId: snapshot.instanceId })
      if (error instanceof api.ApiError && [404, 409].includes(error.status)) await refresh()
    } finally {
      releaseController(controller)
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
    const token = tenantRef.current
    if (!stateMatchesTenant(snapshot) || !requestIsCurrent(token) || !snapshot.workspaceId || !files.length || snapshot.uploadingAssistant !== null) return
    const assistant = snapshot.assistants[id]
    const attachmentIds = [...assistant.draft.attachment_ids]
    const controller = registerController()
    dispatch({ type: 'uploading', id, value: true })
    try {
      for (const file of files) {
        const upload = await api.uploadAsset(snapshot.workspaceId, file, controller.signal)
        if (!requestIsCurrent(token) || controller.signal.aborted) return
        dispatch({ type: 'uploadAdded', upload })
        if (!attachmentIds.includes(upload.asset_id)) attachmentIds.push(upload.asset_id)
        dispatch({ type: 'draft', id, patch: { attachment_ids: [...attachmentIds] } })
      }
    } catch (error) {
      if (!requestIsCurrent(token) || controller.signal.aborted || isAbortError(error)) return
      throw error
    } finally {
      if (requestIsCurrent(token)) dispatch({ type: 'uploading', id, value: false })
      releaseController(controller)
    }
  }, [])

  const cancel = useCallback(async (id: AssistantId, taskId: string) => {
    const snapshot = stateRef.current
    const token = tenantRef.current
    if (!stateMatchesTenant(snapshot) || !requestIsCurrent(token) || !snapshot.workspaceId || !snapshot.instanceId || snapshot.assistants[id].cancelling) return
    const controller = registerController()
    dispatch({ type: 'operation', id, operation: 'cancelling', value: true })
    try {
      const task = await api.cancelTask(taskId, snapshot.workspaceId, controller.signal)
      if (requestIsCurrent(token) && !controller.signal.aborted) {
        dispatch({ type: 'task', task, instanceId: snapshot.instanceId, now: Date.now() })
      }
    } catch (error) {
      if (requestIsCurrent(token) && !controller.signal.aborted && !isAbortError(error)) dispatch({ type: 'operation', id, operation: 'cancelling', value: false,
        error: messageOf(error), instanceId: snapshot.instanceId })
      if (requestIsCurrent(token) && error instanceof api.ApiError && [404, 409].includes(error.status)) await refresh()
    } finally {
      releaseController(controller)
    }
  }, [refresh])

  const reset = useCallback(async (id: AssistantId) => {
    const snapshot = stateRef.current
    const token = tenantRef.current
    const assistant = snapshot.assistants[id]
    if (!stateMatchesTenant(snapshot) || !requestIsCurrent(token) || !snapshot.workspaceId || !snapshot.instanceId
      || assistant.submitting || assistant.resetting || isActive(currentTask(assistant))) return
    const controller = registerController()
    dispatch({ type: 'operation', id, operation: 'resetting', value: true })
    try {
      const workspace = await api.resetAssistant(id, snapshot.workspaceId, controller.signal)
      if (!requestIsCurrent(token) || controller.signal.aborted) return
      if (!sameTenant(token, {
        ownerUserId: workspace.owner_user_id ?? null,
        workspaceId: workspace.workspace_id,
        epoch: token.epoch,
      })) {
        void refresh()
        return
      }
      if (snapshot.instanceId === stateRef.current.instanceId) {
        dispatch({ type: 'snapshot', snapshot: workspace, now: Date.now() })
      }
    } catch (error) {
      if (requestIsCurrent(token) && !controller.signal.aborted && !isAbortError(error)) dispatch({ type: 'operation', id, operation: 'resetting', value: false,
        error: messageOf(error), instanceId: snapshot.instanceId })
      if (requestIsCurrent(token) && error instanceof api.ApiError && [404, 409].includes(error.status)) await refresh()
    } finally {
      releaseController(controller)
    }
  }, [refresh])

  return <Context.Provider value={{ state, dispatch, refresh, send, rerun, uploadFiles, cancel, reset }}>{children}</Context.Provider>
}

export function useWorkspace() {
  const context = useContext(Context)
  if (!context) throw new Error('WorkspaceProvider is required')
  return context
}
