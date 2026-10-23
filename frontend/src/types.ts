import type { components } from './generated/api'

export type AuthLogin = components['schemas']['AuthLogin']
export type AuthSession = components['schemas']['AuthSessionSnapshot']
export type AuthLogout = components['schemas']['AuthLogout']
export type TaskSnapshot = components['schemas']['TaskSnapshot']
export type AssistantSnapshot = components['schemas']['AssistantSnapshot']
export type WorkspaceSnapshot = components['schemas']['WorkspaceSnapshot']
export type SubmitTaskRequest = components['schemas']['SubmitTask']
export type AssistantId = TaskSnapshot['assistant_id']
export type TaskStatus = TaskSnapshot['status']
export const assistantIds: AssistantId[] = ['chat', 'standards', 'plan']
export const isActive = (task?: TaskSnapshot) =>
  !!task && ['queued', 'running', 'cancelling', 'recovering'].includes(task.status)

export interface StandardCatalogEntry {
  catalog_id: string
  std_no: string
  title: string
  library_class: string
  text_available: boolean
  original_available: boolean
  page_url: string | null
  history_url: string
}

export interface StandardCatalogMatch extends StandardCatalogEntry {
  candidate_only?: boolean
  match_method?: string
  match_notes?: string[]
}

export interface StandardsCatalog {
  entries: StandardCatalogEntry[]
  total: number
}

export interface StandardCitation {
  evidence_id: string
  std_no: string
  title: string
  clause_no?: string | null
  text: string
  page_url: string | null
  history_url: string
}

/** Extra fields returned for a real standards answer before generated types catch up. */
export type StandardTaskResult = NonNullable<TaskSnapshot['result']> & {
  citations?: StandardCitation[]
  catalog_matches?: StandardCatalogMatch[]
  answer_scope?: string
  status?: string
  tool_calls?: unknown
}

export interface StandardsCandidatesResponse {
  query: string
  matches: StandardCatalogMatch[]
  warnings: string[]
  semantic_used: boolean
}

export type PlanAction = 'review' | 'question'

export interface UploadAsset {
  asset_id: string
  filename: string
  size_bytes: number
}

export interface UploadsResponse {
  uploads: UploadAsset[]
}

export interface RegisteredPlan {
  plan_id: string
  label: string
  filename: string
  page_url?: string | null
}

export interface PlanDefaults {
  max_steps: number
  max_tool_calls: number
}

export interface PlansResponse {
  plans: RegisteredPlan[]
  defaults: PlanDefaults
}

export interface SubmissionFields {
  attachment_ids: string[]
  registered_plan_id?: string | null
  plan_action?: PlanAction
  use_embeddings?: boolean
  max_steps?: number
  max_tool_calls?: number
}

export type TaskSubmission = SubmitTaskRequest & SubmissionFields
export type TaskSnapshotSubmission = TaskSnapshot & Partial<SubmissionFields>

export interface TaskFile {
  asset_id: string
  filename: string
  size_bytes: number
  page_url?: string | null
}

export type AppTaskResult = StandardTaskResult & {
  files?: TaskFile[]
  document?: Record<string, unknown> | null
  report?: Record<string, unknown> | null
  report_markdown?: string | null
  source_coverage?: Record<string, unknown> | null
  evidence_ids?: string[]
  report_format?: string | null
  file_mode?: 'direct' | 'retrieval' | 'pages_only' | string | null
  uses_embeddings?: boolean
}

export interface StandardsPageSource {
  source_ref: string
  filename: string
  pages: number[]
}

export interface StandardsPageImage {
  page: number
  document_page?: number | string | null
  image_url: string
}

export interface StandardsPagesResponse {
  std_no: string
  title: string
  source_ref: string
  filename: string
  page_count: number
  sources: StandardsPageSource[]
  pages: StandardsPageImage[]
  warning?: string
}
