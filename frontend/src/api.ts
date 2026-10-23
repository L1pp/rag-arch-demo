import type {
  AuthLogin,
  AuthLogout,
  AuthSession,
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
import { publishAuthInvalidation } from './authLifecycle'
import { safeNextPath } from './route'

export class ApiError extends Error {
  status: number
  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

export function loginRedirectUrl(pathname = window.location.pathname, search = window.location.search, hash = window.location.hash) {
  const next = safeNextPath(`${pathname || '/'}${search}${hash}`)
  return `/login?${new URLSearchParams({ next })}`
}

export function redirectToLogin() {
  window.location.replace(loginRedirectUrl())
}

/** API origin plus an optional reverse-proxy prefix; route paths append `/api/...`. */
function apiBasePrefix() {
  const configured = (import.meta.env.VITE_API_BASE_URL || '').trim()
  if (!configured) return ''
  let base: URL
  try { base = new URL(configured) } catch {
    throw new ApiError('VITE_API_BASE_URL 必须是 http(s) 地址或留空。', 0)
  }
  if (!['http:', 'https:'].includes(base.protocol) || !base.hostname || base.username || base.password
    || base.search || base.hash || configured.startsWith('//') || configured.includes('?') || configured.includes('#')) {
    throw new ApiError('VITE_API_BASE_URL 只能包含不带凭据、查询参数或片段的 http(s) 地址。', 0)
  }
  const prefix = base.pathname.replace(/\/+$/, '')
  return `${base.origin}${prefix}`
}

export function apiUrl(path: string) {
  if (!path.startsWith('/') || path.startsWith('//') || /[\\\u0000-\u001f\u007f]/.test(path)) {
    throw new ApiError('API 路径无效。', 0)
  }
  return `${apiBasePrefix()}${path}`
}

export function apiResourceUrl(value: string) {
  if (value.startsWith('/viewer/')) return value
  if (value.startsWith('/') && !value.startsWith('//')) return apiUrl(value)
  try {
    const parsed = new URL(value)
    return ['http:', 'https:'].includes(parsed.protocol) ? value : ''
  } catch {
    return ''
  }
}

type RequestPolicy = { handleUnauthorized?: boolean }

async function authenticatedFetch(url: string, options: RequestInit = {}, policy: RequestPolicy = {}) {
  let response: Response
  try {
    response = await fetch(url, { ...options, credentials: 'include' })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error
    throw new ApiError('暂时无法连接服务，请稍后重试。', 0)
  }
  if (!response.ok) {
    if (response.status === 401 && policy.handleUnauthorized !== false) {
      // Subscribers synchronously erase tenant UI, close SSE, abort requests,
      // and advance their epoch before navigation starts.
      publishAuthInvalidation()
      redirectToLogin()
    }
    let message = `请求未成功（HTTP ${response.status}）。`
    try {
      const body = await response.json() as { detail?: unknown }
      if (typeof body.detail === 'string') message = body.detail
    } catch { /* Keep the HTTP message when the response is not JSON. */ }
    throw new ApiError(message, response.status)
  }
  return response
}

async function request<T>(path: string, options: RequestInit = {}, policy: RequestPolicy = {}): Promise<T> {
  const response = await authenticatedFetch(apiUrl(path), options, policy)
  return response.json() as Promise<T>
}

export async function getProtectedBlob(resource: string, signal?: AbortSignal) {
  const url = apiResourceUrl(resource)
  if (!url) throw new ApiError('资源地址无效。', 0)
  let resourceOrigin: string
  try { resourceOrigin = new URL(url, window.location.origin).origin } catch {
    throw new ApiError('资源地址无效。', 0)
  }
  const apiOrigin = apiBasePrefix() ? new URL(apiBasePrefix()).origin : window.location.origin
  if (resourceOrigin !== apiOrigin) throw new ApiError('资源地址与当前 API 服务不匹配。', 0)
  return (await authenticatedFetch(url, { signal })).blob()
}

export async function fetchTaskDownload(taskId: string, workspaceId: string, format: 'json' | 'md' | 'debug', signal?: AbortSignal) {
  const path = `/api/tasks/${encodeURIComponent(taskId)}/download?${new URLSearchParams({ workspace_id: workspaceId, format })}`
  const response = await authenticatedFetch(apiUrl(path), { signal })
  return response.blob()
}

export async function downloadTask(taskId: string, workspaceId: string, format: 'json' | 'md' | 'debug', signal?: AbortSignal) {
  const blob = await fetchTaskDownload(taskId, workspaceId, format, signal)
  if (signal?.aborted) return
  const blobUrl = URL.createObjectURL(blob)
  const link = document.createElement('a')
  const fileName = `task-${taskId}.${format === 'md' ? 'md' : 'json'}`
  link.href = blobUrl
  link.download = fileName
  document.body.appendChild(link)
  link.click()
  link.remove()
  window.setTimeout(() => URL.revokeObjectURL(blobUrl), 0)
}
const json = (body: unknown): RequestInit => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
})

export const login = (body: AuthLogin, signal?: AbortSignal) =>
  request<AuthSession>('/api/auth/login', { ...json(body), signal }, { handleUnauthorized: false })

export async function getAuthSession(signal?: AbortSignal): Promise<AuthSession | null> {
  try {
    const session = await request<AuthSession | AuthLogout>('/api/auth/session', { signal }, { handleUnauthorized: false })
    return session.authenticated ? session as AuthSession : null
  } catch (error) {
    if (error instanceof ApiError && error.status === 401) return null
    throw error
  }
}

export const logout = (signal?: AbortSignal) =>
  request<AuthLogout>('/api/auth/logout', { ...json({}), signal })

export const getWorkspace = (signal?: AbortSignal) =>
  request<WorkspaceSnapshot>('/api/workspace', { signal })
export const submitTask = (body: TaskSubmission, signal?: AbortSignal) =>
  request<TaskSnapshot>('/api/tasks', { ...json(body), signal })
export const cancelTask = (taskId: string, workspaceId: string, signal?: AbortSignal) =>
  request<TaskSnapshot>(`/api/tasks/${encodeURIComponent(taskId)}/cancel`, { ...json({ workspace_id: workspaceId }), signal })
export const resetAssistant = (assistantId: string, workspaceId: string, signal?: AbortSignal) =>
  request<WorkspaceSnapshot>(`/api/assistants/${encodeURIComponent(assistantId)}/reset`, { ...json({ workspace_id: workspaceId }), signal })
export const taskDownloadUrl = (taskId: string, workspaceId: string, format: 'json' | 'md' | 'debug') =>
  apiUrl(`/api/tasks/${encodeURIComponent(taskId)}/download?${new URLSearchParams({ workspace_id: workspaceId, format })}`)
export const eventsUrl = (workspaceId: string, instanceId: string, cursor: number | string) =>
  apiUrl(`/api/events?${new URLSearchParams({ workspace_id: workspaceId, server_instance_id: instanceId, after: String(cursor) })}`)

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

const standardsPagesPath = ({ std, evidence, source_ref, page, asset, plan_id, workspace_id }: StandardsPagesQuery) => {
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
export const standardsPagesUrl = (query: StandardsPagesQuery) => apiUrl(standardsPagesPath(query))

export const getStandardsPages = (query: StandardsPagesQuery, signal?: AbortSignal) =>
  request<StandardsPagesResponse>(standardsPagesPath(query), { signal })

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
