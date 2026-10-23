import {
  assistantIds,
  isActive,
  type AssistantId,
  type AssistantSnapshot,
  type PlanAction,
  type PlanDefaults,
  type RegisteredPlan,
  type TaskSnapshot,
  type TaskSubmission,
  type UploadAsset,
  type WorkspaceSnapshot,
} from './types'

/** Legacy shared key: retained only as a name for migration tests; it is never restored or written. */
export const STORAGE_KEY = 'rag-arch-demo-ui-v1'
const TENANT_STORAGE_PREFIX = 'rag-arch-demo-ui-v2'
export const MAX_PROMPT_CHARS = 20000
export const DEFAULT_PLAN_DEFAULTS: PlanDefaults = { max_steps: 50, max_tool_calls: 150 }
export const titles: Record<AssistantId, string> = {
  chat: '聊天助手', standards: '规范库问答', plan: '方案辅助核查',
}
export interface Draft {
  prompt: string
  duration_seconds: number
  fail_at_end: boolean
  attachment_ids: string[]
  registered_plan_id: string | null
  plan_action: PlanAction
  use_embeddings: boolean
  max_steps: number
  max_tool_calls: number
}
export interface AssistantState extends AssistantSnapshot {
  draft: Draft
  pending: TaskSubmission | null
  submitting: boolean
  resetting: boolean
  cancelling: boolean
  error: string | null
  inspectedTaskId: string | null
  resetSeq: number
}
export interface WorkspaceState {
  ownerUserId: number | null
  workspaceId: string | null
  instanceId: string | null
  mode: WorkspaceSnapshot["mode"]
  modelName: string | null
  cursor: number
  selected: AssistantId
  assistants: Record<AssistantId, AssistantState>
  receivedAt: Record<string, number>
  loaded: boolean
  loading: boolean
  connection: 'connecting' | 'live' | 'reconnecting' | 'offline'
  bootstrapError: string | null
  notice: string | null
  connectionEpoch: number
  uploads: UploadAsset[]
  plans: RegisteredPlan[]
  planDefaults: PlanDefaults
  resourcesLoading: boolean
  resourcesError: string | null
  uploadingAssistant: AssistantId | null
}
interface SavedUi {
  owner_user_id?: number
  workspace_id?: string
  server_instance_id?: string
  selected_assistant?: AssistantId
  drafts?: Partial<Record<AssistantId, SavedDraft>>
}
interface SavedDraft {
  prompt?: string
  attachment_ids?: string[]
  registered_plan_id?: string | null
  plan_action?: PlanAction
  use_embeddings?: boolean
  max_steps?: number
  max_tool_calls?: number
}
export interface SubmissionOverrides {
  prompt?: string
  attachment_ids?: string[]
  registered_plan_id?: string | null
  plan_action?: PlanAction
  use_embeddings?: boolean
  max_steps?: number
  max_tool_calls?: number
}
export function tenantStorageKey(ownerUserId: number, workspaceId: string): string {
  return `${TENANT_STORAGE_PREFIX}:${encodeURIComponent(String(ownerUserId))}:${encodeURIComponent(workspaceId)}`
}

export function clearTenantUi(ownerUserId: number | null | undefined, workspaceId: string | null | undefined) {
  if (typeof ownerUserId !== 'number' || !Number.isInteger(ownerUserId) || ownerUserId < 1 || !workspaceId) return
  try { localStorage.removeItem(tenantStorageKey(ownerUserId, workspaceId)) }
  catch { /* Storage may be unavailable; the active view is still cleared. */ }
}

export function clearTenantUiForOwner(ownerUserId: number | null | undefined) {
  if (typeof ownerUserId !== 'number' || !Number.isInteger(ownerUserId) || ownerUserId < 1) return
  const prefix = `${TENANT_STORAGE_PREFIX}:${encodeURIComponent(String(ownerUserId))}:`
  try {
    for (let index = localStorage.length - 1; index >= 0; index -= 1) {
      const key = localStorage.key(index)
      if (key?.startsWith(prefix)) localStorage.removeItem(key)
    }
  } catch { /* Storage may be unavailable; the active view is still cleared. */ }
}

/** Read a tenant's UI cache only after the server has confirmed both identity keys. */
export function loadTenantUi(ownerUserId: number | null | undefined, workspaceId: string): SavedUi | undefined {
  if (typeof ownerUserId !== 'number' || !Number.isInteger(ownerUserId) || ownerUserId < 1) return undefined
  try {
    const raw = localStorage.getItem(tenantStorageKey(ownerUserId, workspaceId))
    if (!raw) return undefined
    const saved = JSON.parse(raw) as SavedUi
    if (saved.owner_user_id !== ownerUserId || saved.workspace_id !== workspaceId) return undefined
    return saved
  } catch {
    return undefined
  }
}
export const cursorSequence = (cursor: number | string): number => {
  const value = typeof cursor === 'number' ? cursor : Number(cursor.split(':').at(-1))
  return Number.isFinite(value) ? value : 0
}
function emptyAssistant(id: AssistantId, draft?: Partial<Draft>): AssistantState {
  const maxSteps = typeof draft?.max_steps === 'number' ? Math.min(80, Math.max(3, Math.floor(draft.max_steps))) : DEFAULT_PLAN_DEFAULTS.max_steps
  const maxToolCalls = typeof draft?.max_tool_calls === 'number' ? Math.min(200, Math.max(1, Math.floor(draft.max_tool_calls))) : DEFAULT_PLAN_DEFAULTS.max_tool_calls
  return {
    assistant_id: id, title: titles[id], active_task_id: null, tasks: [], execution_mode: "simulation",
    draft: {
      prompt: typeof draft?.prompt === 'string' ? draft.prompt.slice(0, MAX_PROMPT_CHARS) : '',
      duration_seconds: typeof draft?.duration_seconds === 'number'
        ? Math.min(120, Math.max(1, draft.duration_seconds)) : 5,
      fail_at_end: draft?.fail_at_end === true,
      attachment_ids: Array.isArray(draft?.attachment_ids) ? draft.attachment_ids.filter((value): value is string => typeof value === 'string') : [],
      registered_plan_id: typeof draft?.registered_plan_id === 'string' ? draft.registered_plan_id : null,
      plan_action: draft?.plan_action === 'review' ? 'review' : 'question',
      use_embeddings: draft?.use_embeddings === true,
      max_steps: maxSteps,
      max_tool_calls: maxToolCalls,
    },
    pending: null, submitting: false, resetting: false, cancelling: false,
    error: null, inspectedTaskId: null, resetSeq: 0,
  }
}
export function initialState(saved: SavedUi = {}): WorkspaceState {
  return {
    ownerUserId: saved.owner_user_id ?? null,
    workspaceId: saved.workspace_id ?? null, instanceId: saved.server_instance_id ?? null,
    selected: assistantIds.includes(saved.selected_assistant!) ? saved.selected_assistant! : 'chat',
    // Keep text drafts, but never restore a previous test's slow/failing scenario.
    assistants: Object.fromEntries(assistantIds.map(id => {
      const savedDraft = saved.drafts?.[id]
      return [id, emptyAssistant(id, {
        prompt: savedDraft?.prompt,
        attachment_ids: savedDraft?.attachment_ids,
        registered_plan_id: savedDraft?.registered_plan_id,
        plan_action: savedDraft?.plan_action,
        use_embeddings: savedDraft?.use_embeddings,
        max_steps: savedDraft?.max_steps,
        max_tool_calls: savedDraft?.max_tool_calls,
      })]
    })) as Record<AssistantId, AssistantState>,
    mode: "infrastructure_demo", modelName: null,
    cursor: 0, receivedAt: {}, loaded: false, loading: true, connection: 'connecting',
    bootstrapError: null, notice: null, connectionEpoch: 0,
    uploads: [], plans: [], planDefaults: DEFAULT_PLAN_DEFAULTS,
    resourcesLoading: true, resourcesError: null, uploadingAssistant: null,
  }
}
export function loadInitialState(): WorkspaceState {
  // Authentication is resolved by the first workspace snapshot. Never restore
  // the old unscoped v1 cache before the server identifies its owner.
  return initialState()
}
export function persistableUi(state: WorkspaceState): SavedUi {
  return {
    ...(state.ownerUserId ? { owner_user_id: state.ownerUserId } : {}),
    workspace_id: state.workspaceId ?? undefined,
    server_instance_id: state.instanceId ?? undefined,
      selected_assistant: state.selected,
      drafts: Object.fromEntries(assistantIds.map(id => {
        const draft = state.assistants[id].draft
        return [id, {
          prompt: draft.prompt,
          attachment_ids: draft.attachment_ids,
          registered_plan_id: draft.registered_plan_id ?? undefined,
          plan_action: draft.plan_action,
          use_embeddings: draft.use_embeddings,
          max_steps: draft.max_steps,
          max_tool_calls: draft.max_tool_calls,
        }]
      })),
  }
}
export function currentTask(assistant: AssistantState): TaskSnapshot | undefined {
  return assistant.tasks.find(t => t.task_id === assistant.active_task_id)
    ?? assistant.tasks.findLast(isActive)
    ?? assistant.tasks.at(-1)
}
function sameArray(left: unknown, right: unknown) {
  return Array.isArray(left) && Array.isArray(right) && left.length === right.length && left.every((value, index) => value === right[index])
}

export function createSubmission(
  assistant: AssistantState,
  workspaceId: string,
  requestId: string,
  retry = false,
  overrides: SubmissionOverrides = {},
): TaskSubmission {
  const simulated = assistant.execution_mode !== 'model'
  const { draft, pending } = assistant
  const prompt = (overrides.prompt ?? draft.prompt).trim()
  const attachmentIds = [...(overrides.attachment_ids ?? draft.attachment_ids)]
  const registeredPlanId = overrides.registered_plan_id !== undefined ? overrides.registered_plan_id : draft.registered_plan_id
  const planAction = overrides.plan_action ?? draft.plan_action
  const isPlanReview = assistant.assistant_id === 'plan' && planAction === 'review'
  // Keep the legacy request field for API/history compatibility. Chat retrieval
  // is selected automatically from extracted attachment text on the server.
  const useEmbeddings = assistant.assistant_id === 'chat'
    ? false : overrides.use_embeddings ?? draft.use_embeddings
  const maxSteps = overrides.max_steps ?? draft.max_steps
  const maxToolCalls = overrides.max_tool_calls ?? draft.max_tool_calls
  const sameOptions = pending
    && pending.prompt === prompt
    && sameArray(pending.attachment_ids, attachmentIds)
    && (pending.registered_plan_id ?? null) === (registeredPlanId ?? null)
    && (pending.plan_action ?? 'question') === planAction
    && (pending.use_embeddings ?? false) === useEmbeddings
    && (!isPlanReview || (
      (pending.max_steps ?? DEFAULT_PLAN_DEFAULTS.max_steps) === maxSteps
      && (pending.max_tool_calls ?? DEFAULT_PLAN_DEFAULTS.max_tool_calls) === maxToolCalls
    ))
    && (!simulated || (pending.duration_seconds === draft.duration_seconds && pending.fail_at_end === draft.fail_at_end))
  if (pending && (retry || sameOptions)) return pending
  const common: TaskSubmission = {
    workspace_id: workspaceId, assistant_id: assistant.assistant_id,
    request_id: requestId, prompt, attachment_ids: attachmentIds,
    use_embeddings: useEmbeddings,
  }
  if (assistant.assistant_id === 'plan') {
    common.registered_plan_id = registeredPlanId ?? null
    common.plan_action = planAction
    if (isPlanReview) {
      common.max_steps = maxSteps
      common.max_tool_calls = maxToolCalls
    }
  }
  if (simulated) {
    common.duration_seconds = draft.duration_seconds
    common.fail_at_end = draft.fail_at_end
  }
  return common
}

export function createRerunSubmission(task: TaskSnapshot, workspaceId: string, requestId: string): TaskSubmission {
  const common: TaskSubmission = {
    workspace_id: workspaceId,
    assistant_id: task.assistant_id,
    request_id: requestId,
    prompt: task.prompt.trim(),
    attachment_ids: [...(task.attachment_ids ?? [])],
    use_embeddings: task.assistant_id === 'chat' ? false : task.use_embeddings ?? false,
  }
  if (task.assistant_id === 'plan') {
    common.registered_plan_id = task.registered_plan_id ?? null
    common.plan_action = task.plan_action ?? 'question'
    if (task.max_steps != null) common.max_steps = task.max_steps
    if (task.max_tool_calls != null) common.max_tool_calls = task.max_tool_calls
  }
  return common
}

function withTask(assistant: AssistantState, incoming: TaskSnapshot): AssistantState {
  if (incoming.updated_seq <= assistant.resetSeq) return assistant
  const old = assistant.tasks.find(t => t.task_id === incoming.task_id)
  if (old && old.updated_seq >= incoming.updated_seq) return assistant
  const tasks = [...assistant.tasks.filter(t => t.task_id !== incoming.task_id), incoming]
    .sort((a, b) => a.created_at.localeCompare(b.created_at) || a.updated_seq - b.updated_seq)
  const active = tasks.findLast(isActive)
  return { ...assistant, tasks, active_task_id: active?.task_id ?? null }
}
function acknowledge(assistant: AssistantState): AssistantState {
  const pending = assistant.pending
  if (!pending || !assistant.tasks.some(t => t.request_id === pending.request_id)) return assistant
  return {
    ...assistant, pending: null, submitting: false, error: null,
    draft: assistant.draft.prompt === pending.prompt ? { ...assistant.draft, prompt: '' } : assistant.draft,
  }
}
export type Action =
  | { type: 'select'; id: AssistantId }
  | { type: 'draft'; id: AssistantId; patch: Partial<Draft> }
  | { type: 'loading' }
  | { type: 'identityExpired' }
  | { type: 'snapshot'; snapshot: WorkspaceSnapshot; now: number; savedUi?: SavedUi }
  | { type: 'bootstrapError'; message: string }
  | { type: 'connection'; value: WorkspaceState['connection'] }
  | { type: 'noticeDismissed' }
  | { type: 'task'; task: TaskSnapshot; instanceId: string; cursor?: string; now: number }
  | { type: 'assistantReset'; id: AssistantId; instanceId: string; cursor: string }
  | { type: 'submitStart'; payload: TaskSubmission }
  | { type: 'submitFailed'; id: AssistantId; requestId: string; message: string; instanceId: string }
  | { type: 'operation'; id: AssistantId; operation: 'resetting' | 'cancelling'; value: boolean; error?: string; instanceId?: string }
  | { type: 'inspect'; id: AssistantId; taskId: string }
  | { type: 'resourcesLoading'; value: boolean }
  | { type: 'resources'; uploads: UploadAsset[]; plans: RegisteredPlan[]; defaults: PlanDefaults }
  | { type: 'resourceError'; message: string }
  | { type: 'uploadAdded'; upload: UploadAsset }
  | { type: 'uploading'; id: AssistantId; value: boolean }

export function reducer(state: WorkspaceState, action: Action): WorkspaceState {
  const change = (id: AssistantId, assistant: AssistantState) => ({
    ...state, assistants: { ...state.assistants, [id]: assistant },
  })
  switch (action.type) {
    case 'identityExpired': return { ...initialState(), loading: false, resourcesLoading: false,
      connection: 'offline', bootstrapError: '登录已失效，请重新登录。', connectionEpoch: state.connectionEpoch + 1 }
    case 'select': return { ...state, selected: action.id }
    case 'draft': return change(action.id, {
      ...state.assistants[action.id], draft: { ...state.assistants[action.id].draft, ...action.patch },
    })
    case 'loading': return { ...state, loading: true, bootstrapError: null, connection: 'connecting' }
    case 'bootstrapError': return { ...state, loading: false, bootstrapError: action.message, connection: 'offline' }
    case 'connection': return { ...state, connection: action.value }
    case 'noticeDismissed': return { ...state, notice: null }
    case 'snapshot': {
      const snapshot = action.snapshot
      const ownerUserId = snapshot.owner_user_id ?? null
      // Authenticated workspaces are isolated by both stable keys. Legacy
      // ownerless mode has only one user, so preserve its drafts on restarts.
      const tenantChanged = state.ownerUserId !== ownerUserId
        || (ownerUserId !== null && state.workspaceId !== snapshot.workspace_id)
      const base = tenantChanged ? initialState(action.savedUi) : state
      const sameWorkspace = base.workspaceId === snapshot.workspace_id
      const restarted = !tenantChanged && !!base.instanceId && base.instanceId !== snapshot.server_instance_id
      const same = !tenantChanged && sameWorkspace && base.instanceId === snapshot.server_instance_id
      const seq = cursorSequence(snapshot.cursor)
      const receivedAt = same ? { ...base.receivedAt } : {}
      const assistants = { ...base.assistants }
      for (const id of assistantIds) {
        const previous = base.assistants[id]
        const remote = snapshot.assistants.find(a => a.assistant_id === id)
        if (!remote) continue
        if (same && previous.resetSeq > seq) continue
        let assistant = {
          ...emptyAssistant(id, previous.draft), ...remote,
          pending: same ? previous.pending : null,
          submitting: same ? previous.submitting : false,
          inspectedTaskId: same ? previous.inspectedTaskId : null,
          resetSeq: same ? previous.resetSeq : 0,
        }
        assistant.tasks = [...remote.tasks].sort((a, b) => a.created_at.localeCompare(b.created_at))
        for (const task of assistant.tasks) receivedAt[task.task_id] = action.now
        if (same) {
          // A late GET must not undo a newer POST/SSE update.
          for (const task of previous.tasks) {
            if (task.updated_seq > seq) assistant = withTask(assistant, task)
          }
        }
        assistant.active_task_id = assistant.tasks.findLast(isActive)?.task_id ?? null
        if (!assistant.tasks.some(t => t.task_id === assistant.inspectedTaskId)) {
          assistant.inspectedTaskId = assistant.tasks.at(-1)?.task_id ?? null
        }
        assistants[id] = acknowledge(assistant)
      }
      return {
        ...base, ownerUserId, workspaceId: snapshot.workspace_id, instanceId: snapshot.server_instance_id,
        mode: snapshot.mode, modelName: snapshot.model_name ?? null,
        cursor: same ? Math.max(base.cursor, seq) : seq, assistants, receivedAt,
        loaded: true, loading: false, bootstrapError: null, connection: 'connecting',
        notice: restarted ? '后台服务已重启，旧运行状态已清空。磁盘日志不会自动恢复为本次对话。' : base.notice,
        connectionEpoch: base.connectionEpoch + 1,
      }
    }
    case 'task': {
      if (action.instanceId !== state.instanceId) return state
      const id = action.task.assistant_id
      const previous = state.assistants[id]
      const merged = withTask(previous, action.task)
      const assistant = acknowledge({
        ...merged,
        inspectedTaskId: merged.inspectedTaskId ?? action.task.task_id,
        cancelling: isActive(action.task) ? merged.cancelling : false,
      })
      const newer = !previous.tasks.some(t => t.task_id === action.task.task_id && t.updated_seq >= action.task.updated_seq)
      return {
        ...change(id, assistant),
        cursor: action.cursor ? Math.max(state.cursor, cursorSequence(action.cursor)) : state.cursor,
        receivedAt: newer ? { ...state.receivedAt, [action.task.task_id]: action.now } : state.receivedAt,
      }
    }
    case 'assistantReset': {
      if (action.instanceId !== state.instanceId) return state
      const seq = cursorSequence(action.cursor)
      const old = state.assistants[action.id]
      if (seq <= old.resetSeq) return state
      const assistant = { ...emptyAssistant(action.id, old.draft), title: old.title, execution_mode: old.execution_mode ?? "simulation", resetSeq: seq }
      return { ...change(action.id, assistant), cursor: Math.max(seq, state.cursor) }
    }
    case 'submitStart':
      return change(action.payload.assistant_id, {
        ...state.assistants[action.payload.assistant_id],
        pending: action.payload, submitting: true, error: null,
      })
    case 'submitFailed': {
      if (action.instanceId !== state.instanceId) return state
      const old = state.assistants[action.id]
      if (old.tasks.some(t => t.request_id === action.requestId) || old.pending?.request_id !== action.requestId) return state
      return change(action.id, { ...old, submitting: false, error: action.message })
    }
    case 'operation': {
      if (action.instanceId && action.instanceId !== state.instanceId) return state
      return change(action.id, {
        ...state.assistants[action.id], [action.operation]: action.value, error: action.error ?? null,
      })
    }
    case 'inspect': return change(action.id, { ...state.assistants[action.id], inspectedTaskId: action.taskId })
    case 'resourcesLoading': return { ...state, resourcesLoading: action.value, resourcesError: action.value ? null : state.resourcesError }
    case 'resources': {
      const validUploads = new Set(action.uploads.map(upload => upload.asset_id))
      const validPlans = new Set(action.plans.map(plan => plan.plan_id))
      const assistants = Object.fromEntries(assistantIds.map(id => {
        const assistant = state.assistants[id]
        const maxSteps = assistant.draft.max_steps === state.planDefaults.max_steps
          ? Math.min(80, Math.max(3, Math.floor(action.defaults.max_steps))) : assistant.draft.max_steps
        const maxToolCalls = assistant.draft.max_tool_calls === state.planDefaults.max_tool_calls
          ? Math.min(200, Math.max(1, Math.floor(action.defaults.max_tool_calls))) : assistant.draft.max_tool_calls
        return [id, {
          ...assistant,
          draft: {
            ...assistant.draft,
            attachment_ids: assistant.draft.attachment_ids.filter(assetId => validUploads.has(assetId)),
            registered_plan_id: assistant.draft.registered_plan_id && validPlans.has(assistant.draft.registered_plan_id)
              ? assistant.draft.registered_plan_id : null,
            max_steps: maxSteps,
            max_tool_calls: maxToolCalls,
          },
        }]
      })) as Record<AssistantId, AssistantState>
      return {
        ...state, assistants, uploads: action.uploads, plans: action.plans,
        planDefaults: action.defaults, resourcesLoading: false, resourcesError: null,
      }
    }
    case 'resourceError': return { ...state, resourcesLoading: false, resourcesError: action.message }
    case 'uploadAdded': {
      if (state.uploads.some(upload => upload.asset_id === action.upload.asset_id)) return state
      return { ...state, uploads: [...state.uploads, action.upload], resourcesError: null }
    }
    case 'uploading': return { ...state, uploadingAssistant: action.value ? action.id : state.uploadingAssistant === action.id ? null : state.uploadingAssistant }
  }
}
