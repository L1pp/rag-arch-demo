import type {
  PlansResponse,
  StandardsCatalog,
  StandardsCandidatesResponse,
  StandardsPagesResponse,
  TaskSubmission,
  TaskSnapshot,
  UploadAsset,
  UploadsResponse,
  WorkspaceSnapshot,
} from './types'

export class ApiError extends Error {
  status: number
  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

export function loginRedirectUrl(pathname = window.location.pathname, search = window.location.search, hash = window.location.hash) {
  const next = `${pathname || '/'}${search}${hash}`
  return `/login?${new URLSearchParams({ next })}`
}

function redirectToLogin() {
  window.location.replace(loginRedirectUrl())
}

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(url, options)
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error
    throw new ApiError('暂时无法连接服务，请稍后重试。', 0)
  }
  if (!response.ok) {
    if (response.status === 401) redirectToLogin()
    let message = `请求未成功（HTTP ${response.status}）。`
    try {
      const body = await response.json() as { detail?: unknown }
      if (typeof body.detail === 'string') message = body.detail
    } catch { /* Keep the HTTP message when the response is not JSON. */ }
    throw new ApiError(message, response.status)
  }
  return response.json() as Promise<T>
}
const json = (body: unknown): RequestInit => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
})
export const getWorkspace = (signal?: AbortSignal) =>
  request<WorkspaceSnapshot>('/api/workspace', { signal })
export const submitTask = (body: TaskSubmission) =>
  request<TaskSnapshot>('/api/tasks', json(body))
export const cancelTask = (taskId: string, workspaceId: string) =>
  request<TaskSnapshot>(`/api/tasks/${encodeURIComponent(taskId)}/cancel`, json({ workspace_id: workspaceId }))
export const resetAssistant = (assistantId: string, workspaceId: string) =>
  request<WorkspaceSnapshot>(`/api/assistants/${encodeURIComponent(assistantId)}/reset`, json({ workspace_id: workspaceId }))
export const taskDownloadUrl = (taskId: string, workspaceId: string, format: 'json' | 'md' | 'debug') =>
  `/api/tasks/${encodeURIComponent(taskId)}/download?${new URLSearchParams({ workspace_id: workspaceId, format })}`
export const eventsUrl = (workspaceId: string, instanceId: string, cursor: number | string) =>
  `/api/events?${new URLSearchParams({ workspace_id: workspaceId, server_instance_id: instanceId, after: String(cursor) })}`

export const getStandardsCatalog = (signal?: AbortSignal) =>
  request<StandardsCatalog>('/api/standards/catalog', { signal })

export const getStandardCandidates = (query: string, signal?: AbortSignal) =>
  request<StandardsCandidatesResponse>(`/api/standards/candidates?${new URLSearchParams({ query })}`, { signal })

export interface StandardsPagesQuery {
  std?: string
  evidence?: string
  source_ref?: string
  page?: number
  asset?: string
  plan_id?: string
  workspace_id?: string
}

export const standardsPagesUrl = ({ std, evidence, source_ref, page, asset, plan_id, workspace_id }: StandardsPagesQuery) => {
  const params = new URLSearchParams()
  if (std) params.set('std', std)
  if (evidence) params.set('evidence', evidence)
  if (source_ref) params.set('source_ref', source_ref)
  if (typeof page === 'number' && Number.isFinite(page)) params.set('page', String(Math.max(1, Math.floor(page))))
  if (asset) params.set('asset', asset)
  if (plan_id) params.set('plan_id', plan_id)
  if (workspace_id) params.set('workspace_id', workspace_id)
  return `${std ? '/api/standards/pages' : '/api/documents/pages'}?${params}`
}

export const getStandardsPages = (query: StandardsPagesQuery, signal?: AbortSignal) =>
  request<StandardsPagesResponse>(standardsPagesUrl(query), { signal })

export const getUploads = (workspaceId: string, signal?: AbortSignal) =>
  request<UploadsResponse>(`/api/uploads?${new URLSearchParams({ workspace_id: workspaceId })}`, { signal })

export const uploadAsset = (workspaceId: string, file: File, signal?: AbortSignal) =>
  request<UploadAsset>(`/api/uploads?${new URLSearchParams({ workspace_id: workspaceId, filename: file.name })}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/octet-stream' },
    body: file,
    signal,
  })

export const getPlans = (signal?: AbortSignal) =>
  request<PlansResponse>('/api/plans', { signal })
