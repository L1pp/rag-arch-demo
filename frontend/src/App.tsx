import { useEffect, useLayoutEffect, useRef, useState, type FormEvent } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { ApiError, downloadTask, getStandardCandidates, getStandardsCatalog, logout as logoutApi, redirectToLogin, taskDownloadUrl } from './api'
import { publishAuthInvalidation } from './authLifecycle'
import ResourceControls, { formatBytes } from './ResourceControls'
import { clearTenantUi, currentTask, MAX_PROMPT_CHARS, type AssistantState } from './store'
import {
  assistantIds,
  isActive,
  type AssistantId,
  type AppTaskResult,
  type PageImageSource,
  type StandardCatalogEntry,
  type StandardCatalogMatch,
  type StandardCitation,
  type StandardTaskResult,
  type TaskFile,
  type TaskSnapshot,
  type TaskSnapshotSubmission,
  type TaskStatus,
} from './types'
import { useWorkspace } from './WorkspaceContext'

type IconName = AssistantId | 'send' | 'stop' | 'reset' | 'download' | 'clock' | 'check' | 'close' | 'arrow'
function Icon({ name, size = 20 }: { name: IconName; size?: number }) {
  const paths: Record<IconName, React.ReactNode> = {
    chat: <><path d="M20 11.5a7.5 7.5 0 0 1-7.5 7.5H5l-3 3V11.5A7.5 7.5 0 0 1 9.5 4h3a7.5 7.5 0 0 1 7.5 7.5Z"/><path d="M7 10h8M7 14h5"/></>,
    standards: <><path d="M5 3h11l3 3v15H5zM15 3v5h4"/><path d="M9 12h6M9 16h6"/></>,
    plan: <><path d="m12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6l8-3Z"/><path d="m8.5 12 2.5 2.5 4.5-5"/></>,
    send: <><path d="m4 12 16-8-5 16-4-6-7-2Z"/><path d="m11 14 9-10"/></>,
    stop: <rect x="6" y="6" width="12" height="12" rx="2"/>,
    reset: <><path d="M3 10a9 9 0 1 1 1 7M3 4v6h6"/></>,
    download: <><path d="M12 3v12m-4-4 4 4 4-4M4 16v5h16v-5"/></>,
    clock: <><circle cx="12" cy="12" r="9"/><path d="M12 7v6l4 2"/></>,
    check: <path d="m5 12 4 4L19 6"/>,
    close: <path d="m6 6 12 12M6 18 12-12"/>,
    arrow: <path d="M5 12h14m-5-5 5 5-5 5"/>,
  }
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>
}
const statusLabels: Record<TaskStatus, string> = {
  queued: '等待执行', running: '运行中', cancelling: '正在停止', recovering: '确认执行状态中',
  completed: '已完成', failed: '失败', cancelled: '已停止',
}
const stageLabels: Record<string, string> = {
  queued: '等待执行', starting: '正在启动', preparing: '准备演示',
  processing: '模拟处理', working: '模拟处理中', finishing: '准备结束',
  files: '处理附件', reading: '读取资料', tool: '调用工具', assembling: '整理结果',
  analyzing: '模拟分析', summarizing: '模拟汇总',
  completed: '演示完成', failed: '模拟失败', cancelled: '演示已停止', cancelling: '响应停止请求',
}
const modelStages: Record<string, string> = {
  queued: '等待处理', starting: '准备对话', preparing: '准备对话',
  processing: '正在回答', working: '正在回答', model: '模型正在回答',
  files: '处理附件', reading: '读取资料', tool: '调用工具', assembling: '整理结果',
  retrieval: '检索资料', calling_tool: '调用工具',
  calling_model: '模型正在回答', model_call: '模型正在回答',
  completed: '回答已返回', failed: '回答未完成', cancelled: '已停止', cancelling: '响应停止请求',
}
function stageLabel(task: TaskSnapshot) {
  const labels = task.simulation === false ? modelStages : stageLabels
  return labels[task.stage] || task.stage || statusLabels[task.status]
}
function Badge({ task }: { task?: TaskSnapshot }) {
  const status = task?.status
  return <span className={'status-badge ' + (status ?? 'idle')}>
    <span className="status-dot" />{status ? statusLabels[status] : '待开始'}
  </span>
}
function Avatar({ id, small = false }: { id: AssistantId; small?: boolean }) {
  return <span className={'assistant-avatar ' + (small ? 'small' : '')} data-kind={id}><Icon name={id} size={small ? 17 : 22}/></span>
}
function formatTime(value?: string | null) {
  return value ? new Date(value).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' }) : ''
}
function Elapsed({ task }: { task: TaskSnapshot }) {
  const { state } = useWorkspace()
  const [now, setNow] = useState(Date.now())
  const active = isActive(task)
  useEffect(() => {
    if (!active) return
    const timer = window.setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(timer)
  }, [active])
  const receivedAt = state.receivedAt[task.task_id] ?? now
  const seconds = Math.max(0, Math.floor(task.elapsed_seconds + (active && task.started_at ? Math.max(0, now - receivedAt) / 1000 : 0)))
  return <span className="elapsed">{String(Math.floor(seconds / 60)).padStart(2, '0')}:{String(seconds % 60).padStart(2, '0')}</span>
}
function Sidebar() {
  const { state, dispatch } = useWorkspace()
  const activeCount = assistantIds.filter(id => isActive(currentTask(state.assistants[id]))).length
  return <aside className="sidebar">
    <div className="brand"><span className="brand-symbol">建</span><div><strong>建筑资料助手</strong><span>工作台</span></div></div>
    <div className="nav-eyebrow">我的助手 <span>03</span></div>
    <nav aria-label="助手导航">
      {assistantIds.map(id => {
        const assistant = state.assistants[id]
        const task = currentTask(assistant)
        return <button key={id} className={'assistant-nav ' + (id === state.selected ? 'selected' : '')}
          aria-current={id === state.selected ? 'page' : undefined} onClick={() => dispatch({ type: 'select', id })}>
          <Avatar id={id} small />
          <span className="nav-content"><strong>{assistant.title}</strong><Badge task={task}/></span>
          {id === state.selected && <span className="nav-indicator"/>}
        </button>
      })}
    </nav>
    <div className="sidebar-note"><span className="note-icon">↗</span><strong>各自运行，随时切换</strong><p>切换助手不会中断任务。<br/>回复和进度保留在各自会话中。</p></div>
    <div className="sidebar-bottom"><div><span className={'connection-dot ' + state.connection}/><span>{state.connection === 'live' ? '实时连接正常' : state.connection === 'reconnecting' ? '事件流正在重连' : state.connection === 'connecting' ? '连接服务中' : '服务未连接'}</span></div>
      <small>{!state.loaded ? '正在连接工作区…' : activeCount ? activeCount + ' 个助手运行中' : '可开始新的对话或核查'}</small></div>
  </aside>
}
function EmptyState({ id }: { id: AssistantId }) {
  const { state, dispatch } = useWorkspace()
  if (!state.loaded) return <div className="empty-state"><p>正在连接工作区…</p></div>
  const model = state.assistants[id].execution_mode === 'model'
  const text = model ? id === 'standards'
    ? ['从一条规范问题开始', '查阅规范条文、版本信息和原页，回答附有来源。', 'GB 55023-2022 中支撑脚手架施工荷载有哪些要求？']
    : id === 'plan'
      ? ['从一份方案开始核查', '核查方案文字内容，并可在追问中按需读取原页。', '请先选择方案，再提出需要核查的问题。']
      : ['今天想聊些什么？', '可以直接聊天，也可以添加文件后提问。', '你好，请简单介绍一下你能帮我做什么。'] : {
    chat: ['从一个模拟任务开始', '先测试独立会话、任务进度和即时切换。这里暂时不会调用模型。', '帮我模拟一次基础对话。'],
    standards: ['规范问答，也有独立会话', '这里演示规范助手的任务状态，尚未连接规范库或检索功能。', '模拟一次规范问题处理。'],
    plan: ['为方案核查准备一个工作区', '这里仅模拟处理过程，不会读取施工方案或生成正式核查结论。', '模拟一次方案核查任务。'],
  }[id]
  return <div className="empty-state"><div className="empty-orbit"><Avatar id={id}/><span/><span/></div>
    <div className="eyebrow">一个会话，一份进度</div><h1>{text[0]}</h1><p>{text[1]}</p>
    <button className="example-button" onClick={() => dispatch({ type: 'draft', id, patch: { prompt: text[2] } })}>{text[2]}<Icon name="arrow" size={16}/></button>
    <div className="empty-features"><span>独立状态</span><i/><span>即时切换</span><i/><span>可停止任务</span></div>
  </div>
}

export function StandardsCatalogPanel() {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [entries, setEntries] = useState<StandardCatalogEntry[]>([])
  const [candidateMatches, setCandidateMatches] = useState<StandardCatalogMatch[]>([])
  const [candidateWarnings, setCandidateWarnings] = useState<string[]>([])
  const [candidateError, setCandidateError] = useState<string | null>(null)
  const [candidateLoading, setCandidateLoading] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [reload, setReload] = useState(0)
  const candidateController = useRef<AbortController | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    setLoading(true)
    setError(null)
    void getStandardsCatalog(controller.signal)
      .then(payload => setEntries(Array.isArray(payload.entries) ? payload.entries : []))
      .catch(reason => {
        if (controller.signal.aborted) return
        setEntries([])
        setError(reason instanceof Error ? reason.message : '规范目录暂时不可用，请稍后重试。')
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })
    return () => controller.abort()
  }, [reload])

  useEffect(() => () => candidateController.current?.abort(), [])

  const lookupCandidates = async () => {
    const value = query.trim()
    if (!value) {
      setCandidateMatches([])
      setCandidateWarnings([])
      setCandidateError(null)
      return
    }
    candidateController.current?.abort()
    const controller = new AbortController()
    candidateController.current = controller
    setCandidateLoading(true)
    setCandidateError(null)
    setCandidateWarnings([])
    try {
      const response = await getStandardCandidates(value, controller.signal)
      if (controller.signal.aborted) return
      setCandidateMatches(Array.isArray(response.matches) ? response.matches : [])
      setCandidateWarnings(Array.isArray(response.warnings) ? response.warnings : [])
    } catch (reason) {
      if (controller.signal.aborted) return
      setCandidateMatches([])
      setCandidateError(reason instanceof Error ? reason.message : '候选查询暂时不可用。')
    } finally {
      if (!controller.signal.aborted) setCandidateLoading(false)
    }
  }

  const normalized = query.trim().toLocaleLowerCase()
  const filtered = normalized
    ? entries.filter(entry => `${entry.std_no} ${entry.title}`.toLocaleLowerCase().includes(normalized))
    : entries

  return <section className="standards-catalog" aria-label="规范目录">
    <div className="standards-catalog-heading">
      <div>
        <div className="standards-catalog-title"><strong>规范目录</strong><span>{loading ? '加载中…' : error ? '加载失败' : `${filtered.length} / ${entries.length} 条`}</span></div>
        <p>按标准号或中文名查找规范，可打开原页和历史关系。</p>
      </div>
      <button type="button" className="catalog-toggle" aria-expanded={open} onClick={() => setOpen(value => !value)}>
        {open ? '收起规范目录' : '展开规范目录'}
      </button>
    </div>
    {open && <div className="standards-catalog-body">
      <div className="standards-search-row">
        <label className="standards-search">
          <span aria-hidden="true">⌕</span>
          <span className="sr-only">搜索规范号或中文名</span>
          <input aria-label="搜索规范号或中文名" value={query} onChange={event => setQuery(event.target.value)}
            onKeyDown={event => { if (event.key === 'Enter') { event.preventDefault(); void lookupCandidates() } }} placeholder="搜索标准号或中文名…" />
          {query && <button type="button" aria-label="清空规范搜索" onClick={() => setQuery('')}>×</button>}
        </label>
        <button type="button" className="candidate-search-button" disabled={candidateLoading || !query.trim()} onClick={() => void lookupCandidates()}>
          {candidateLoading ? '查找中…' : '查找候选'}
        </button>
      </div>
      {candidateError && <div className="catalog-status catalog-error" role="alert"><span>{candidateError}</span></div>}
      {!candidateError && candidateWarnings.map((warning, index) => <div className="catalog-status catalog-warning" role="status" key={index}>{warning}</div>)}
      {!candidateLoading && !candidateError && query.trim() && !candidateMatches.length && candidateWarnings.length === 0 && <div className="catalog-status">暂无候选结果。</div>}
      {!!candidateMatches.length && <section className="candidate-results" aria-label="规范候选结果">
        <div className="candidate-results-heading"><strong>候选结果</strong><span>候选不等于已取证条文</span></div>
        <ul>
          {candidateMatches.map(candidate => <li key={candidate.catalog_id || candidate.std_no}>
            <div className="candidate-item-main"><div><strong>{candidate.std_no}</strong><span className={candidate.candidate_only ? 'candidate-badge' : 'matched-badge'}>{candidate.candidate_only ? '候选' : '已登记匹配'}</span></div><p>{candidate.title}</p>
              {!!candidate.match_notes?.length && <ul className="candidate-notes">{candidate.match_notes.slice(0, 3).map((note, index) => <li key={index}>{note}</li>)}</ul>}
            </div>
            <div className="catalog-item-links">{candidate.page_url && <a href={candidate.page_url} target="_blank" rel="noopener noreferrer">查看原页↗</a>}{candidate.history_url && <a href={candidate.history_url} target="_blank" rel="noopener noreferrer">历史关系↗</a>}</div>
          </li>)}
        </ul>
      </section>}
      {loading && <div className="catalog-status" role="status">正在加载规范目录…</div>}
      {!loading && error && <div className="catalog-status catalog-error" role="alert"><span>{error}</span><button type="button" onClick={() => setReload(value => value + 1)}>重试</button></div>}
      {!loading && !error && !entries.length && <div className="catalog-status">暂无可用规范。</div>}
      {!loading && !error && entries.length > 0 && !filtered.length && <div className="catalog-status">没有匹配的规范。</div>}
      {!loading && !error && !!filtered.length && <ul className="standards-catalog-list">
        {filtered.map(entry => {
          return <li key={entry.catalog_id || entry.std_no} className="standards-catalog-item">
            <div className="catalog-item-main">
              <div className="catalog-item-heading"><strong>{entry.std_no}</strong></div>
              <p>{entry.title}</p>
              <div className="catalog-item-meta"><span>{entry.text_available ? '文本可检索' : '暂无文本'}</span><span>{entry.original_available ? '原页可查看' : '暂无原页'}</span></div>
            </div>
            <div className="catalog-item-links">
              {entry.page_url && <a href={entry.page_url} target="_blank" rel="noopener noreferrer">查看原页↗</a>}
              {entry.history_url && <a href={entry.history_url} target="_blank" rel="noopener noreferrer">历史关系↗</a>}
            </div>
          </li>
        })}
      </ul>}
    </div>}
  </section>
}

function StandardCitationCards({ task }: { task: TaskSnapshot }) {
  const [open, setOpen] = useState(false)
  if (task.assistant_id !== 'standards' && task.assistant_id !== 'plan') return null
  const result = task.result as StandardTaskResult | null | undefined
  const citations = result?.citations ?? []
  if (!citations.length) return null
  return <section className="citation-list" aria-label="规范依据"><details open={open}>
    <summary className="citation-list-heading" onClick={event => { event.preventDefault(); setOpen(value => !value) }}><strong>规范依据</strong><span>{citations.length} 条证据</span></summary>
    {open && <div className="citation-cards">
      {citations.map((citation: StandardCitation) => <article className="citation-card" key={citation.evidence_id}>
        <div className="citation-card-heading"><strong>{citation.std_no}</strong>{citation.clause_no && <span>第 {citation.clause_no} 条</span>}</div>
        <p className="citation-card-title">{citation.title}</p>
        <p className="citation-card-text">{citation.text || '未提供条文摘录。'}</p>
        <div className="citation-card-links">
          {citation.page_url && <a href={citation.page_url} target="_blank" rel="noopener noreferrer">查看证据页↗</a>}
          {citation.history_url && <a href={citation.history_url} target="_blank" rel="noopener noreferrer">历史关系↗</a>}
          <span title={citation.evidence_id}>证据 {citation.evidence_id}</span>
        </div>
      </article>)}
    </div>}
  </details></section>
}

function CatalogMatchCards({ task }: { task: TaskSnapshot }) {
  if (task.assistant_id !== 'standards' && task.assistant_id !== 'plan') return null
  const result = task.result as StandardTaskResult | null | undefined
  const matches = result?.catalog_matches ?? []
  if (!matches.length) return null
  return <section className="citation-list catalog-match-list" aria-label="相关规范候选"><details>
    <summary className="citation-list-heading"><strong>相关规范候选</strong><span>{matches.length} 条</span></summary>
    <div className="candidate-results">
      <p className="candidate-results-heading"><span>候选匹配不等于已取证条文。</span></p>
      <ul>{matches.map((match: StandardCatalogMatch) => <li key={match.catalog_id || match.std_no}>
        <div className="candidate-item-main"><div><strong>{match.std_no}</strong><span className={match.candidate_only ? 'candidate-badge' : 'matched-badge'}>{match.candidate_only ? '候选' : '已登记匹配'}</span></div><p>{match.title}</p>
          {!!match.match_notes?.length && <ul className="candidate-notes">{match.match_notes.slice(0, 3).map((note, index) => <li key={index}>{note}</li>)}</ul>}
        </div>
        <div className="catalog-item-links">{match.page_url && <a href={match.page_url} target="_blank" rel="noopener noreferrer">查看原页↗</a>}{match.history_url && <a href={match.history_url} target="_blank" rel="noopener noreferrer">历史关系↗</a>}</div>
      </li>)}</ul>
    </div>
  </details></section>
}

function isOcrWarning(value: unknown): value is string {
  return typeof value === 'string' && /ocr/i.test(value)
}

function safeOcrWarning(warning: string): string {
  // Provider diagnostics can contain credentials, temporary links, or server paths.
  // Keep those details out of both the visible notice and the expanded details.
  return /https?:\/\/|\b(?:api[\s_-]?key|access[\s_-]?key|authorization|bearer|token|secret|password|endpoint|key)\b|密钥|(?:[A-Za-z]:[\\/]|\/(?:opt|etc|home|tmp|data\/runtime)\/)/i.test(warning)
    ? 'OCR 状态暂不可显示，请查看原页。'
    : warning
}

function OcrNotice({ result }: { result: AppTaskResult | null | undefined }) {
  if (!result) return null
  const warnings = Array.isArray(result.warnings) ? result.warnings.filter(isOcrWarning).map(safeOcrWarning) : []
  const documentPages = Array.isArray(result.document?.ocr_pages) ? result.document.ocr_pages : []
  const fileGroups = (Array.isArray(result.files) ? result.files : []).flatMap((file, index) => {
    if (!file || typeof file !== 'object' || !Array.isArray(file.ocr_pages) || !file.ocr_pages.length) return []
    const filename = typeof file.filename === 'string' ? file.filename.split(/[\\/]/).at(-1)?.trim() : ''
    const label = filename && safeOcrWarning(filename) === filename ? `附件 ${index + 1}（${filename}）` : `附件 ${index + 1}`
    return [{ label, pages: file.ocr_pages }]
  })
  const groups = [{ label: '', pages: documentPages }, ...fileGroups]
  if (!warnings.length && !groups.some(group => group.pages.length)) return null

  const pageDescriptions = groups.flatMap(({ label, pages }) => pages.flatMap((item: unknown) => {
    if (!item || typeof item !== 'object') return []
    const record = item as Record<string, unknown>
    if (typeof record.page !== 'number' || !Number.isInteger(record.page) || record.page < 1) return []
    const page = record.page_numbering === 'system_rendered_page'
      ? `系统排版第 ${record.page} 页` : `第 ${record.page} 页`
    const state = record.status === 'completed' ? '已识别，待核对'
      : record.status === 'failed' ? '识别未成功'
      : record.status === 'unknown' ? '识别状态未确认'
      : record.status === 'not_attempted' ? '未执行识别'
      : '识别情况待核对'
    return [`${label ? `${label} · ` : ''}${page}：${state}`]
  }))
  const distinctWarnings = [...new Set(warnings)]
  return <aside className="catalog-status catalog-warning" role="status" aria-label="OCR 识别提示">
    <div><strong>OCR 识别提醒</strong><p>识别文字可能有错误或遗漏，关键数字、单位和表格请核对原页。</p>
      {distinctWarnings.map((warning, index) => <p key={index}>{warning}</p>)}
      {!!pageDescriptions.length && <p>{[...new Set(pageDescriptions)].join('；')}。</p>}
    </div>
  </aside>
}

function PageReviewDetails({ task }: { task: TaskSnapshot }) {
  const result = task.result as AppTaskResult | undefined
  const rawPages: unknown[] = result?.page_sources ?? []
  const pages = rawPages.filter((source): source is PageImageSource => {
    if (!source || typeof source !== 'object') return false
    const row = source as Record<string, unknown>
    return row.sent_to_model === true && typeof row.page === 'number' && Number.isInteger(row.page) && row.page > 0
  })
  if (!pages.length) return null
  return <details className="response-details"><summary>已提供模型的页图/区域 · {pages.length} 项</summary>
    <p>以下记录页图投递，不代表模型已正确识别；关键数字仍需结合原页核对。</p>
    <ul>{pages.map((source, index) => {
      const region = Array.isArray(source.region) && source.region.length === 4
      const url = typeof source.page_url === 'string' && source.page_url.startsWith('/viewer/pages?') ? source.page_url : null
      const numbering = /\.docx?$/i.test(source.filename ?? '') ? '系统排版第' : '第'
      return <li key={source.image_id || source.evidence_id || index}>
        <span>{source.filename || '来源文件'} · {numbering}{source.page}页{region ? ' · 局部区域' : ''} · {source.automatic ? '首次自动附入' : '按需读取'} · {source.cited ? '已引用' : '未引用'}</span>
        {url && <> <a href={url} target="_blank" rel="noopener noreferrer">{region ? '查看区域↗' : '查看原页↗'}</a></>}
      </li>
    })}</ul>
  </details>
}

function ResultDetails({ task }: { task: TaskSnapshot }) {
  const warnings = task.result?.warnings ?? []
  const links = (task.result?.web_sources ?? []).flatMap(source => {
    const url = typeof source.url === 'string' ? source.url : ''
    if (!/^https?:\/\//i.test(url)) return []
    return [{ url, title: typeof source.title === 'string' ? source.title : url }]
  })
  if (!warnings.length && !links.length) return null
  return <details className="response-details"><summary>来源与说明</summary>
    {warnings.map((warning, index) => <p key={index}>{isOcrWarning(warning) ? safeOcrWarning(warning) : warning}</p>)}
    {!!links.length && <ul>{links.map((link, index) => <li key={index}><a href={link.url} target="_blank" rel="noopener noreferrer">{link.title}</a></li>)}</ul>}
  </details>
}

function MarkdownBody({ text }: { text: string }) {
  return <div className="markdown-body"><ReactMarkdown remarkPlugins={[remarkGfm]} skipHtml>{text}</ReactMarkdown></div>
}

function TaskSelectionSummary({ task, state }: { task: TaskSnapshot; state: ReturnType<typeof useWorkspace>['state'] }) {
  const submission = task as TaskSnapshotSubmission
  const attachmentIds = submission.attachment_ids ?? []
  const files = attachmentIds.map(assetId => state.uploads.find(asset => asset.asset_id === assetId)?.filename || assetId)
  const plan = submission.registered_plan_id ? state.plans.find(item => item.plan_id === submission.registered_plan_id) : null
  if (!files.length && !plan && submission.plan_action !== 'review') return null
  return <div className="task-selection-summary" aria-label="本轮提交选择">
    <span>本轮选择</span>
    {files.length > 0 && <span>附件：{files.join('、')}</span>}
    {plan && <span>方案：{plan.label}</span>}
    {submission.plan_action === 'review' && <span>整份核查报告</span>}
    {submission.use_embeddings && <span>向量检索</span>}
    {task.assistant_id === 'plan' && (submission.max_steps || submission.max_tool_calls) && <span>{submission.max_steps ?? 50} steps · {submission.max_tool_calls ?? 150} calls</span>}
  </div>
}

function simpleReportText(value: unknown): string {
  if (typeof value === 'string' || typeof value === 'number') return String(value)
  if (Array.isArray(value)) return value.map(simpleReportText).filter(Boolean).join('；')
  if (value && typeof value === 'object') {
    const record = value as Record<string, unknown>
    for (const key of ['text', 'description', 'message', 'reason', 'label', 'name']) {
      if (record[key] != null) return simpleReportText(record[key])
    }
  }
  return ''
}

function FindingCard({ finding, index }: { finding: unknown; index: number }) {
  if (!finding || typeof finding !== 'object') return <li className="finding-card"><p>{simpleReportText(finding) || `发现 ${index + 1}`}</p></li>
  const record = finding as Record<string, unknown>
  const title = simpleReportText(record.title ?? record.name) || `发现 ${index + 1}`
  const verdict = simpleReportText(record.verdict ?? record.conclusion ?? record.status)
  const description = simpleReportText(record.description ?? record.summary)
  const conditions = simpleReportText(record.conditions ?? record.condition)
  const recommendation = simpleReportText(record.recommendation ?? record.action)
  const quote = simpleReportText(record.quote ?? record.excerpt ?? record.citation)
  const position = simpleReportText(record.location ?? record.page ?? record.page_no ?? record.document_page ?? record.paragraph)
  return <li className="finding-card"><strong>{title}</strong>
    {verdict && <span className="finding-verdict">{verdict}</span>}
    {description && <p>{description}</p>}
    {conditions && <p><b>条件：</b>{conditions}</p>}
    {recommendation && <p><b>建议：</b>{recommendation}</p>}
    {(quote || position) && <details className="finding-evidence"><summary>依据与位置</summary>{quote && <p>“{quote}”</p>}{position && <p>{position}</p>}</details>}
  </li>
}

function CoverageSummary({ coverage }: { coverage: unknown }) {
  if (!coverage) return null
  if (typeof coverage === 'string' || Array.isArray(coverage)) return <p>{simpleReportText(coverage)}</p>
  if (typeof coverage !== 'object') return null
  const record = coverage as Record<string, unknown>
  const proportion = simpleReportText(record.proportion ?? record.coverage_ratio ?? record.covered_ratio ?? record.covered)
  const provided = record.provided_units ?? record.providedUnits
  const total = record.total_units ?? record.totalUnits
  const partial = record.partially_provided_units ?? record.partiallyProvidedUnits
  const unread = simpleReportText(record.unread_pages ?? record.unreadPages ?? record.unread)
  const units = provided != null && total != null
    ? `完整提供 ${simpleReportText(provided)}/${simpleReportText(total)} 个可读单元`
      + (partial ? `；部分提供 ${simpleReportText(partial)} 个单元` : '')
    : ''
  const parts = [
    proportion && `覆盖比例：${proportion}`,
    units,
    unread && `未读页：${unread}`,
  ].filter(Boolean)
  return parts.length ? <p>{parts.join('；')}</p> : null
}

function planReportContent(result: AppTaskResult | null | undefined) {
  const report = result?.report && typeof result.report === 'object' ? result.report : null
  const exportedMarkdown = result?.report_markdown?.trim() ? result.report_markdown : null
  // Follow-up metadata alone is not a report; let the ordinary reply render.
  if (!report && !exportedMarkdown) return null
  const natural = report?.report_format === 'markdown' || result?.report_format === 'markdown'
  const answer = simpleReportText(report?.answer)
  const markdown = natural ? answer || result?.text || exportedMarkdown : exportedMarkdown
  const findings = Array.isArray(report?.findings) ? report.findings : report?.findings ? [report.findings] : []
  const coverage = report?.source_coverage ?? report?.coverage ?? result?.source_coverage
  const hasCoverage = coverage != null
  const limitations = Array.isArray(report?.limitations) ? report.limitations : report?.limitations ? [report.limitations] : []
  if (!markdown && !answer && !findings.length && !hasCoverage && !limitations.length) return null
  return { markdown, answer, findings, hasCoverage, coverage, limitations, natural }
}

function PlanReportDetails({ content }: { content: ReturnType<typeof planReportContent> }) {
  if (!content) return null
  const { markdown, answer, findings, hasCoverage, coverage, limitations, natural } = content
  if (markdown) return <section className="plan-report" aria-label="方案核查报告">
    <MarkdownBody text={markdown}/>
    {natural && hasCoverage && <details className="plan-report-section"><summary>文字提供范围</summary><CoverageSummary coverage={coverage}/></details>}
    {natural && limitations.length > 0 && <details className="plan-report-section"><summary>资料与运行说明</summary><ul className="report-list">{limitations.map((item, index) => <li key={index}>{simpleReportText(item)}</li>)}</ul></details>}
  </section>
  return <section className="plan-report" aria-label="方案核查报告">
    {answer && <div className="plan-report-section"><strong>回答</strong><MarkdownBody text={answer}/></div>}
    {findings.length > 0 && <details className="plan-report-section plan-report-collapsible"><summary><strong>发现（{findings.length}）</strong></summary><ul className="finding-list">{findings.map((finding, index) => <FindingCard finding={finding} index={index} key={index}/>)}</ul></details>}
    {hasCoverage && <div className="plan-report-section"><strong>覆盖</strong><CoverageSummary coverage={coverage}/></div>}
    {limitations.length > 0 && <div className="plan-report-section"><strong>局限</strong><ul className="report-list">{limitations.map((item, index) => <li key={index}>{simpleReportText(item)}</li>)}</ul></div>}
  </section>
}

function TaskFiles({ task }: { task: TaskSnapshot }) {
  const result = task.result as AppTaskResult | null | undefined
  const files = (result?.files ?? []) as TaskFile[]
  if (!files.length) return null
  return <section className="task-files" aria-label="任务文件"><strong>相关文件</strong>
    <ul>{files.map(file => <li key={file.asset_id}><span>{file.filename} · {formatBytes(file.size_bytes)}</span>
      {file.page_url && <a href={file.page_url} target="_blank" rel="noopener noreferrer">查看原件↗</a>}
    </li>)}</ul>
  </section>
}

function TaskDownloadLink({ taskId, workspaceId, format, children }: {
  taskId: string
  workspaceId: string
  format: 'json' | 'md' | 'debug'
  children: React.ReactNode
}) {
  const [error, setError] = useState<string | null>(null)
  const controller = useRef<AbortController | null>(null)
  useEffect(() => () => controller.current?.abort(), [])
  return <>
    <a href={taskDownloadUrl(taskId, workspaceId, format)} download onClick={event => {
      event.preventDefault()
      controller.current?.abort()
      controller.current = new AbortController()
      const signal = controller.current.signal
      setError(null)
      void downloadTask(taskId, workspaceId, format, signal).catch(reason => {
        if (!signal.aborted) setError(reason instanceof Error ? reason.message : '下载失败，请稍后重试。')
      })
    }}>{children}</a>
    {error && <span className="download-error" role="alert">{error}</span>}
  </>
}

function TaskReadMode({ task }: { task: TaskSnapshot }) {
  const result = task.result as AppTaskResult | null | undefined
  const usesEmbeddings = result?.uses_embeddings ?? (task as TaskSnapshotSubmission).use_embeddings ?? false
  const mode = result?.file_mode
  const label = mode === 'direct' ? '全文直读'
    : mode === 'pages_only' ? '仅原页'
    : mode === 'retrieval' ? (usesEmbeddings ? '向量混合检索' : '关键词检索')
    : usesEmbeddings ? '向量混合检索' : null
  return label ? <span className="task-read-mode">本轮：{label}</span> : null
}

type StandardQaNotice = { title: string; message: string }
function standardQaNotice(task: TaskSnapshot): StandardQaNotice | null {
  if (task.assistant_id !== 'standards' || task.simulation !== false || task.status !== 'completed') return null
  const result = task.result as (StandardTaskResult & { termination_reason?: unknown }) | null | undefined
  if (!result) return null
  const reason = typeof result.termination_reason === 'string' ? result.termination_reason.trim().toLowerCase() : ''
  const status = typeof result.status === 'string' ? result.status.trim().toLowerCase() : ''
  if (reason === 'citation_rejected' || status === 'citation_rejected') return {
    title: '引用未通过核验',
    message: '下方保留模型原文供检查；该内容未经引用核验，不能视为已验证结论。',
  }
  if (reason === 'invalid_result' || status === 'invalid_result') return {
    title: '输出格式未通过核验',
    message: '下方保留模型原文供检查；该结果未通过格式校验，不能视为已验证答案。',
  }
  if (reason === 'insufficient_evidence') return {
    title: '资料不足以支持结论',
    message: '下方内容未得到充分证据支持，不应作为已核实答案。',
  }
  if (!reason && status === 'insufficient_evidence') return {
    title: '回答尚未通过核验',
    message: '后端未提供具体核验原因；下方原文仅供检查，不能视为已验证结论。',
  }
  if (reason && reason !== 'answered') return {
    title: '回答未通过完整核验',
    message: '下方原文仅供检查，不应作为已验证结论。',
  }
  if (reason === 'answered' && status && !['answered', 'grounded'].includes(status)) return {
    title: '回答状态未通过核验',
    message: '结果状态与完成标记不一致，下方原文不应作为已验证结论。',
  }
  return null
}

function TaskMessages({ task, assistant, onInspect }: { task: TaskSnapshot; assistant: AssistantState; onInspect: () => void }) {
  const { state, dispatch, rerun } = useWorkspace()
  const active = isActive(task)
  const assistantBusy = assistant.submitting || assistant.resetting || isActive(currentTask(assistant)) || state.uploadingAssistant === assistant.assistant_id || state.resourcesLoading
  const simulated = task.simulation !== false
  const result = task.result as AppTaskResult | null | undefined
  const planReport = task.assistant_id === 'plan' ? planReportContent(result) : null
  const validationNotice = standardQaNotice(task)
  const replyText = planReport ? null : result?.text
  return <section className="turn" aria-label={'任务：' + task.prompt}>
    <div className="user-message"><span className="message-time">{formatTime(task.created_at)}</span><div>{task.prompt}</div></div>
    <div className="assistant-message"><Avatar id={task.assistant_id} small/>
      <div className="assistant-message-body">
        <div className="message-author"><strong>{assistant.title}</strong><span>{simulated ? '模拟回复' : task.result?.model || state.modelName || '模型回复'}</span></div>
        {validationNotice && <div className="inline-error" role="alert"><strong>{validationNotice.title}</strong> {validationNotice.message}</div>}
        {task.status === 'completed' ? replyText
          ? <div className="reply-text"><MarkdownBody text={replyText}/></div>
          : planReport ? null
          : <div className="reply-text">{simulated ? '演示任务已完成，服务器未返回文本。' : '服务端未返回回答文本。'}</div>
          : task.status === 'failed' ? <div className="task-outcome failure"><strong>{simulated ? '本次演示未完成' : '本次回答未完成'}</strong><p>{task.error || task.detail || (simulated ? '服务器返回模拟失败。' : '模型请求未完成，请重试。')}</p></div>
          : task.status === 'cancelled' ? <div className="task-outcome"><strong>任务已停止</strong><p>{task.detail || (simulated ? '本次模拟运行已结束，其他助手不受影响。' : '本次请求已停止，其他助手不受影响。')}</p></div>
          : <div className="running-message"><span className="activity-bars"><i/><i/><i/></span><div><strong>{stageLabel(task)}</strong><p>{task.detail || '等待服务器提供下一步进度。'}</p></div></div>}
        <OcrNotice result={result}/>
        <TaskSelectionSummary task={task} state={state}/>
        <TaskReadMode task={task}/>
        <PlanReportDetails content={planReport}/>
        <TaskFiles task={task}/>
        <PageReviewDetails task={task}/>
        <ResultDetails task={task}/>
        {task.status === 'completed' && <><StandardCitationCards task={task}/><CatalogMatchCards task={task}/></>}
        <div className="message-meta"><Badge task={task}/><span><Icon name="clock" size={13}/><Elapsed task={task}/></span>
          {(['failed', 'cancelled'] as TaskStatus[]).includes(task.status) && <button className="text-button" disabled={assistantBusy}
            onClick={() => void rerun(task.assistant_id, task)}>重新运行此任务</button>}
          <button className="text-button" onClick={() => { dispatch({ type: 'inspect', id: task.assistant_id, taskId: task.task_id }); onInspect() }}>查看详情 <span>↗</span></button>
          {!active && state.workspaceId && <TaskDownloadLink taskId={task.task_id} workspaceId={state.workspaceId} format="md">{simulated ? '下载演示记录' : '下载对话记录'}</TaskDownloadLink>}
        </div>
      </div>
    </div>
  </section>
}
function Inspector({ assistant }: { assistant: AssistantState }) {
  const { state, dispatch, cancel, rerun } = useWorkspace()
  const task = assistant.tasks.find(t => t.task_id === assistant.inspectedTaskId) ?? currentTask(assistant)
  const busy = assistant.submitting || assistant.resetting || isActive(currentTask(assistant)) || state.uploadingAssistant === assistant.assistant_id || state.resourcesLoading
  const model = assistant.execution_mode === 'model'
  return <aside className="inspector">
    <div className="panel-label">{model ? '请求详情' : '任务详情'} <span>LIVE</span></div>
    <div className="task-detail-card">
      {task ? <>
        <div className="detail-status"><Badge task={task}/><Icon name={task.status === 'completed' ? 'check' : 'clock'} size={18}/></div>
        <div className="timer-label">{task.simulation === false ? '本次请求耗时' : <>模拟设定 {task.duration_seconds ?? 5} 秒 · 已等待</>}</div><div className="large-timer"><Elapsed key={task.task_id} task={task}/><span>分 : 秒</span></div>
        <div className="stage-box"><span>当前阶段</span><strong>{stageLabel(task)}</strong><p>{task.detail || '服务器暂未提供阶段说明。'}</p></div>
        {task.error && <div className="inline-error" role="alert">{task.error}</div>}
        {isActive(task) && <button className="stop-button" disabled={assistant.cancelling || task.status === 'cancelling'}
          onClick={() => void cancel(assistant.assistant_id, task.task_id)}><Icon name="stop" size={15}/>{assistant.cancelling || task.status === 'cancelling' ? '正在停止…' : '停止任务'}</button>}
        {(['failed', 'cancelled'] as TaskStatus[]).includes(task.status) && <button className="rerun-button" disabled={busy}
          onClick={() => void rerun(assistant.assistant_id, task)}>重新运行此任务</button>}
        <div className="task-id" title={task.task_id}>任务 <code>{task.task_id.slice(0, 8)}</code><span>更新 #{task.updated_seq}</span></div>
        {!isActive(task) && state.workspaceId && <div className="downloads"><span>{task.simulation === false ? '下载对话记录' : '下载演示记录'}</span><div>
          <TaskDownloadLink taskId={task.task_id} workspaceId={state.workspaceId} format="md"><Icon name="download" size={14}/>Markdown</TaskDownloadLink>
          <TaskDownloadLink taskId={task.task_id} workspaceId={state.workspaceId} format="json">JSON</TaskDownloadLink>
          {Boolean(task.result?.audit_log) && <TaskDownloadLink taskId={task.task_id} workspaceId={state.workspaceId} format="debug">调试JSON</TaskDownloadLink>}</div></div>}
      </> : <div className="no-task"><Icon name="clock" size={28}/><strong>还没有任务</strong><p>发送后，当前阶段与实际耗时会显示在这里。</p></div>}
    </div>
    {!model && <details className="demo-settings">
      <summary>演示设置<span>下次等待 {assistant.draft.duration_seconds} 秒</span></summary>
      <label className="setting-label" htmlFor={'duration-' + assistant.assistant_id}>模拟时长 <span>1–120 秒</span></label>
      <div className="duration-field"><input id={'duration-' + assistant.assistant_id} type="number" min={1} max={120}
        value={assistant.draft.duration_seconds} disabled={busy}
        onChange={e => dispatch({ type: 'draft', id: assistant.assistant_id, patch: { duration_seconds: Math.min(120, Math.max(1, Number(e.target.value) || 1)) } })}/><span>秒</span></div>
      <label className="checkbox-label"><input type="checkbox" checked={assistant.draft.fail_at_end} disabled={busy}
        onChange={e => dispatch({ type: 'draft', id: assistant.assistant_id, patch: { fail_at_end: e.target.checked } })}/><span>结束时模拟失败<small>用于确认错误状态与提示</small></span></label>
      <p className="setting-note">仅用于模拟等待；刷新后恢复5秒正常完成，不保留失败测试开关。</p>
    </details>}
      <div className="demo-footnote"><span className="tiny-square"/><p>{model
      ? assistant.assistant_id === 'standards'
        ? '规范目录和原页可随时查看。'
        : assistant.assistant_id === 'plan'
          ? '请选择登记方案或上传单个文件。'
          : '可直接聊天，或添加附件后提问。'
      : '本助手仍是模拟任务，尚未接入资料处理。'}</p></div>
  </aside>
}
export default function App() {
  const { state, dispatch, refresh, send, reset } = useWorkspace()
  const [loggingOut, setLoggingOut] = useState(false)
  const [logoutError, setLogoutError] = useState<string | null>(null)
  const id = state.selected
  const assistant = state.assistants[id]
  const model = !state.loaded || assistant.execution_mode === 'model'
  const [inspectorOpen, setInspectorOpen] = useState(false)
  const attachmentCount = assistant.draft.attachment_ids.length
  const active = isActive(currentTask(assistant))
  const busy = assistant.submitting || assistant.resetting || active || state.uploadingAssistant === id || state.resourcesLoading
  const canSend = state.loaded && !state.loading && !busy && !!assistant.draft.prompt.trim()
  const bottom = useRef<HTMLDivElement>(null)
  const messages = useRef<HTMLDivElement>(null)
  const scrollPositions = useRef<Partial<Record<AssistantId, number>>>({})
  const lastView = useRef({ id, count: assistant.tasks.length })
  useLayoutEffect(() => {
    if (messages.current) messages.current.scrollTop = scrollPositions.current[id] ?? messages.current.scrollHeight
  }, [id])
  useEffect(() => {
    const previous = lastView.current
    lastView.current = { id, count: assistant.tasks.length }
    if (previous.id === id && previous.count < assistant.tasks.length) bottom.current?.scrollIntoView({ block: 'end' })
  }, [id, assistant.tasks.length, assistant.tasks.at(-1)?.status])
  const submit = (event: FormEvent) => {
    event.preventDefault()
    if (canSend) void send(id)
  }
  const signOut = async () => {
    if (loggingOut) return
    setLoggingOut(true)
    setLogoutError(null)
    try {
      await logoutApi()
      clearTenantUi(state.ownerUserId, state.workspaceId)
      publishAuthInvalidation()
      redirectToLogin()
    } catch (reason) {
      if (reason instanceof ApiError && reason.status === 401) return
      setLogoutError(reason instanceof Error ? reason.message : '退出登录未完成，请重试。')
      setLoggingOut(false)
    }
  }
  const headingSubtitle = !state.loaded ? '正在连接服务…' : model
    ? id === 'standards' ? '查阅规范条文与版本信息' : id === 'plan' ? '文字核查与方案追问' : attachmentCount ? `${attachmentCount} 个附件` : '直接聊天，支持可选附件'
    : '模拟任务 · 资料尚未接入'
  return <div className="workbench">
    <Sidebar/>
    <main className="workspace">
      {state.notice && <div className="notice-banner" role="status"><span>{state.notice}</span><button aria-label="关闭重启提示" onClick={() => dispatch({ type: 'noticeDismissed' })}><Icon name="close" size={16}/></button></div>}
      {state.bootstrapError && <div className="connection-error" role="alert"><span>{state.bootstrapError}</span><button onClick={() => void refresh()} disabled={state.loading}>重新连接</button></div>}
      <header className="workspace-header"><div className="heading-group"><Avatar id={id}/><div><h2>{assistant.title}</h2><p>{headingSubtitle}</p></div></div>
        <div className="header-actions"><button className="reset-button" aria-expanded={inspectorOpen} onClick={() => setInspectorOpen(value => !value)}>{inspectorOpen ? '收起详情' : '任务详情'}</button>
        <button className="reset-button" onClick={() => void reset(id)} disabled={!state.loaded || busy || !assistant.tasks.length}><Icon name="reset" size={15}/>{assistant.resetting ? '正在重置…' : '重置此助手'}</button>
        <button className="reset-button" type="button" onClick={() => void signOut()} disabled={loggingOut}>
          {loggingOut ? '正在退出…' : '退出'}
        </button></div>
      </header>
      {logoutError && <div className="submit-error" role="alert">{logoutError}</div>}
      <div className="workspace-body">
        <div className="chat-column">
          {id === 'standards' && <StandardsCatalogPanel/>}
          <div className="messages" ref={messages} onScroll={e => { scrollPositions.current[id] = e.currentTarget.scrollTop }} aria-live="polite" aria-busy={active}>
            {assistant.tasks.length ? assistant.tasks.map(task => <TaskMessages key={task.task_id} task={task} assistant={assistant} onInspect={() => setInspectorOpen(true)}/>) : <EmptyState id={id}/>}
            <div ref={bottom}/>
          </div>
          <form className="composer-wrap" onSubmit={submit}>
            {assistant.error && <div className="submit-error" role="alert"><span>{assistant.error} 草稿已保留。</span>
              {assistant.pending && !busy && <button type="button" onClick={() => void send(id, true)}>重试上次提交</button>}</div>}
            <div className={'composer ' + (active ? 'has-active-task' : '')}>
              <ResourceControls id={id} busy={busy}/>
              <label className="sr-only" htmlFor="prompt">消息</label>
              <textarea id="prompt" maxLength={MAX_PROMPT_CHARS} rows={2} value={assistant.draft.prompt}
                placeholder={active ? '本助手正在运行，可先写草稿，或切换到另一个助手…' : model ? id === 'plan' ? '输入方案核查问题…' : '发送消息…' : '输入消息，发送一个模拟任务…'}
                onChange={e => dispatch({ type: 'draft', id, patch: { prompt: e.target.value } })}
                onKeyDown={e => {
                  if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
                    e.preventDefault()
                    if (canSend) e.currentTarget.form?.requestSubmit()
                  }
                }}/>
              <div className="composer-bottom"><span className="composer-hint">{active ? <><span className="live-dot"/>任务运行中，其他助手仍可使用</> : 'Enter 发送 · Shift + Enter 换行'}</span>
                <button type="submit" className="send-button" disabled={!canSend}>{assistant.submitting ? '正在提交' : active ? (model ? '回答中' : '任务运行中') : (model ? '发送' : '发送模拟任务')}<Icon name="send" size={16}/></button></div>
            </div>
            <div className="composer-disclaimer"><span>{model
              ? id === 'standards' ? '回答附规范依据。' : id === 'plan' ? '整份核查包含文字与图表；图表尽力核对，疑点会注明。' : attachmentCount ? '附件将在发送时读取。' : '可以直接提问。'
              : '当前助手为模拟任务，不会调用模型或读取资料。'}</span><span>{assistant.draft.prompt.length}/{MAX_PROMPT_CHARS}</span></div>
          </form>
        </div>
        {inspectorOpen && <Inspector assistant={assistant}/>}
      </div>
    </main>
  </div>
}
