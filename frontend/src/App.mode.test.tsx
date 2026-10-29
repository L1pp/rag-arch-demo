import { beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import App from './App'
import { initialState, reducer, tenantStorageKey } from './store'
import { assistantIds, type TaskSnapshot, type WorkspaceSnapshot } from './types'
import { useWorkspace } from './WorkspaceContext'

let context: ReturnType<typeof useWorkspace>
vi.mock('./WorkspaceContext', () => ({ useWorkspace: () => context }))
const apiMock = vi.hoisted(() => ({
  ApiError: class extends Error { constructor(message: string, readonly status: number) { super(message) } },
  downloadTask: vi.fn(), getStandardCandidates: vi.fn(), getStandardsCatalog: vi.fn(), logout: vi.fn(), redirectToLogin: vi.fn(),
  taskDownloadUrl: vi.fn((taskId: string, workspaceId: string, format: string) => `/api/tasks/${taskId}/download?workspace_id=${workspaceId}&format=${format}`),
}))
vi.mock('./api', () => apiMock)

function workspace(): WorkspaceSnapshot {
  return {
    workspace_id: 'work', server_instance_id: 'server', cursor: 1,
    mode: 'chat_preview', model_name: 'test-chat-model',
    assistants: assistantIds.map(id => ({
      assistant_id: id, title: { chat: '聊天助手', standards: '规范库问答', plan: '方案辅助核查' }[id],
      execution_mode: 'model', active_task_id: null, tasks: [],
    })),
  }
}
beforeEach(() => {
  apiMock.getStandardCandidates.mockReset()
  apiMock.downloadTask.mockReset().mockResolvedValue(undefined)
  apiMock.getStandardsCatalog.mockReset().mockResolvedValue({ entries: [] })
  apiMock.logout.mockReset().mockResolvedValue({ authenticated: false })
  apiMock.redirectToLogin.mockReset()
  const state = reducer(initialState(), { type: 'snapshot', snapshot: workspace(), now: Date.now() })
  state.resourcesLoading = false
  context = {
    state, dispatch: vi.fn(), refresh: vi.fn(async () => {}),
    send: vi.fn(async () => {}), rerun: vi.fn(async () => {}), uploadFiles: vi.fn(async () => {}), cancel: vi.fn(async () => {}), reset: vi.fn(async () => {}),
  }
})

describe('execution mode presentation', () => {
  it('logs out through JSON and removes the active tenant draft cache', async () => {
    context.state.ownerUserId = 77
    context.state.workspaceId = 'workspace-77'
    const cacheKey = tenantStorageKey(77, 'workspace-77')
    localStorage.setItem(cacheKey, JSON.stringify({ owner_user_id: 77, workspace_id: 'workspace-77', drafts: {} }))

    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: '退出' }))

    await waitFor(() => expect(apiMock.logout).toHaveBeenCalledTimes(1))
    expect(localStorage.getItem(cacheKey)).toBeNull()
    expect(apiMock.redirectToLogin).toHaveBeenCalledTimes(1)
  })

  it('shows real chat without demo controls or a fake reply label', () => {
    const task: TaskSnapshot = {
      task_id: 'real-task', assistant_id: 'chat', request_id: 'real-request', prompt: '你好',
      status: 'completed', stage: 'completed', detail: '回答已完成', created_at: '2026-09-14T13:00:00Z',
      started_at: '2026-09-14T13:00:00Z', finished_at: '2026-09-14T13:00:01Z',
      elapsed_seconds: 1, duration_seconds: null, updated_seq: 1, simulation: false, error: null,
      result: {
        text: '真实回答 <img src="https://example.invalid/no-fetch.png">', simulation: false,
        model: 'test-chat-model', warnings: ['这是一条测试说明。'],
        web_sources: [{ title: '测试来源', url: 'https://example.test/source' }],
        native_tool_usage: {}, token_usage: {}, file_mode: 'direct', uses_embeddings: false,
      },
    }
    context.state.assistants.chat.tasks = [task]
    context.state.assistants.chat.draft.prompt = '继续聊天'
    render(<App/>)
    expect(screen.queryByText('三个助手均已接入真实模型')).not.toBeInTheDocument()
    expect(screen.getByText('直接聊天，支持可选附件')).toBeInTheDocument()
    expect(screen.queryByText('模拟回复')).not.toBeInTheDocument()
    expect(screen.queryByText(/演示设置/)).not.toBeInTheDocument()
    expect(screen.queryByText(/模拟设定/)).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '任务详情' }))
    expect(screen.getByText('回答已返回')).toBeInTheDocument()
    expect(screen.getByText('本轮：全文直读')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: '下载对话记录' })).toHaveAttribute('href', expect.stringContaining('format=md'))
    fireEvent.click(screen.getByText('来源与说明'))
    expect(screen.getByText('这是一条测试说明。')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: '测试来源' })).toHaveAttribute('href', 'https://example.test/source')
    expect(document.querySelectorAll('img')).toHaveLength(0)
    fireEvent.click(screen.getByRole('button', { name: '发送' }))
    expect(context.send).toHaveBeenCalledWith('chat')
  })

  it('shows the standards assistant as a real BM25-backed model without demo controls', () => {
    context.state.selected = 'standards'
    context.state.assistants.standards.draft.prompt = 'GB 55023-2022 中支撑脚手架施工荷载有哪些要求？'
    render(<App/>)
    expect(screen.getByText('查阅规范条文与版本信息')).toBeInTheDocument()
    expect(screen.queryByText(/演示设置/)).not.toBeInTheDocument()
    expect(screen.queryByRole('spinbutton', { hidden: true })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: '发送' })).toBeEnabled()
    expect(screen.queryByRole('button', { name: '发送模拟任务' })).not.toBeInTheDocument()
    expect(screen.getByText('回答附规范依据。')).toBeInTheDocument()
  })

  it('renders real standards citations as evidence cards with original-page links', () => {
    const task: TaskSnapshot = {
      task_id: 'standards-task', assistant_id: 'standards', request_id: 'standards-request', prompt: '防水要求',
      status: 'completed', stage: 'completed', detail: '回答已完成', created_at: '2026-09-14T13:00:00Z',
      started_at: '2026-09-14T13:00:00Z', finished_at: '2026-09-14T13:00:01Z', elapsed_seconds: 1,
      duration_seconds: null, updated_seq: 1, simulation: false,
      result: {
        text: '请按规范施工。', simulation: false, model: 'test-standards-model', warnings: [], web_sources: [],
        native_tool_usage: {}, token_usage: {}, citations: [{
          evidence_id: 'ev-1', std_no: 'GB 55008-2021', title: '工程防水通用规范', clause_no: '4.2.1',
          text: '防水工程应满足设计要求。', page_url: '/viewer/pages?std=gb-55008&evidence=ev-1', history_url: '/viewer/history/?std=gb-55008',
        }], catalog_matches: [], answer_scope: 'clause', status: 'grounded', corpus_id: 'standards', grounded: true, tool_calls: 1,
      },
    }
    context.state.selected = 'standards'
    context.state.assistants.standards.execution_mode = 'model'
    context.state.assistants.standards.tasks = [task]
    render(<App/>)
    expect(screen.getByRole('region', { name: '规范依据' })).toBeInTheDocument()
    fireEvent.click(screen.getByText('规范依据'))
    const pageLink = screen.getByRole('link', { name: '查看证据页↗' })
    expect(pageLink).toHaveAttribute('target', '_blank')
    expect(pageLink).toHaveAttribute('rel', 'noopener noreferrer')
    expect(screen.getByText('防水工程应满足设计要求。')).toBeInTheDocument()
  })

  it('renders a real plan report and files without falling back to demo copy', () => {
    const task: TaskSnapshot = {
      task_id: 'plan-task', assistant_id: 'plan', request_id: 'plan-request', prompt: '生成整份方案核查报告',
      status: 'completed', stage: 'completed', detail: '方案核查已完成', created_at: '2026-09-14T13:00:00Z',
      started_at: '2026-09-14T13:00:00Z', finished_at: '2026-09-14T13:00:01Z', elapsed_seconds: 1,
      duration_seconds: null, updated_seq: 1, simulation: false,
      result: {
        text: '方案核查摘要', simulation: false, model: 'test-plan-model', warnings: [], web_sources: [],
        native_tool_usage: {}, token_usage: {}, report_markdown: '# 方案核查摘要\n\n总体可执行\n\n## 发现\n\n发现一项待补充说明\n\n图像页未完整读取',
        report: { answer: '总体可执行', findings: ['发现一项待补充说明'], coverage: '已覆盖主要章节', limitations: '图像页未完整读取' },
        citations: [{ evidence_id: 'plan-ev-1', std_no: 'GB 55023-2022', title: '建筑施工脚手架安全技术统一标准', clause_no: '6.2.1', text: '规范引用摘要', page_url: '/viewer/pages?std=gb-55023&evidence=plan-ev-1', history_url: '/viewer/history/?std=gb-55023' }],
        files: [{ asset_id: 'report-1', filename: '方案核查报告.md', size_bytes: 1024, page_url: '/viewer/pages?asset=plan-1&workspace_id=work' }],
      },
    }
    context.state.selected = 'plan'
    context.state.assistants.plan.tasks = [task]
    context.state.assistants.plan.draft.registered_plan_id = 'P01'
    context.state.plans = [{ plan_id: 'P01', label: 'P01 主体方案', filename: 'P01.pdf' }]
    render(<App/>)
    expect(screen.getByRole('region', { name: '方案核查报告' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: '方案核查摘要' })).toBeInTheDocument()
    expect(screen.getAllByText('总体可执行')).toHaveLength(1)
    expect(screen.getAllByText('图像页未完整读取')).toHaveLength(1)
    expect(screen.getByText('发现一项待补充说明')).toBeInTheDocument()
    expect(screen.queryByText('发现（1）')).not.toBeInTheDocument()
    expect(screen.queryByRole('link', { name: '查看证据页↗' })).not.toBeInTheDocument()
    fireEvent.click(screen.getByText('规范依据'))
    expect(screen.getByRole('link', { name: '查看证据页↗' })).toHaveAttribute('target', '_blank')
    expect(screen.getByRole('link', { name: '查看原件↗' })).toHaveAttribute('target', '_blank')
    expect(screen.getByText('整份核查包含文字与图表；图表尽力核对，疑点会注明。')).toBeInTheDocument()
  })

  it('shows confirmed page-image records and region links without claiming recognition accuracy', () => {
    const task: TaskSnapshot = {
      task_id: 'plan-images', assistant_id: 'plan', request_id: 'image-request', prompt: '核查参数',
      status: 'completed', stage: 'completed', detail: '完成', created_at: '2026-10-28T13:00:00Z',
      started_at: null, finished_at: null, elapsed_seconds: 1, duration_seconds: null, updated_seq: 1,
      result: { text: '关键数字待复核', page_sources: [
        { image_id: 'whole', filename: '方案.docx', page: 7, sent_to_model: true, automatic: true, cited: false,
          page_url: '/viewer/pages?asset=private&page=7' },
        { image_id: 'region', filename: '方案.docx', page: 7, sent_to_model: true, region: [0.1, 0.2, 0.6, 0.45], cited: true,
          page_url: '/viewer/pages?asset=private&page=7&region=0.1,0.2,0.6,0.45' },
        { image_id: 'unconfirmed', filename: '未确认.pdf', page: 8, sent_to_model: false },
      ] },
    }
    context.state.selected = 'plan'
    context.state.assistants.plan.tasks = [task]
    render(<App />)
    fireEvent.click(screen.getByText('已提供模型的页图/区域 · 2 项'))
    expect(screen.getByText('以下记录页图投递，不代表模型已正确识别；关键数字仍需结合原页核对。')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: '查看区域↗' })).toHaveAttribute('href', expect.stringContaining('region='))
    expect(screen.getByText(/系统排版第7页 · 局部区域/)).toHaveTextContent('已引用')
    expect(screen.queryByText('未确认.pdf')).not.toBeInTheDocument()
  })

  it.each([
    { name: '旧结构化报告', result: { report: { answer: '旧结构回答', findings: ['旧结构发现'] } }, expected: '旧结构回答', rawVisible: false },
    { name: '旧纯文字报告', result: { report: null }, expected: '旧版纯文字回复', rawVisible: true },
    { name: '空报告和空白 Markdown', result: { report: {}, report_markdown: ' \n ' }, expected: '旧版纯文字回复', rawVisible: true },
    { name: '没有可读字段的报告', result: { report: { answer: {}, findings: [], limitations: [] } }, expected: '旧版纯文字回复', rawVisible: true },
    { name: '仅 Markdown 的可读模型原文', result: { report: null, report_markdown: '# 原文保留\n\n未符合结构要求的模型原文。' }, expected: '未符合结构要求的模型原文。', rawVisible: false },
  ])('preserves $name as a report fallback', ({ result, expected, rawVisible }) => {
    const task: TaskSnapshot = {
      task_id: 'fallback-plan', assistant_id: 'plan', request_id: 'fallback-request', prompt: '方案核查',
      status: 'completed', stage: 'completed', detail: '方案核查已完成', created_at: '2026-09-14T13:00:00Z',
      started_at: '2026-09-14T13:00:00Z', finished_at: '2026-09-14T13:00:01Z', elapsed_seconds: 1,
      duration_seconds: null, updated_seq: 1, simulation: false,
      result: { text: '旧版纯文字回复', simulation: false, model: 'test-plan-model', warnings: [],
        web_sources: [], native_tool_usage: {}, token_usage: {}, ...result },
    }
    context.state.selected = 'plan'
    context.state.assistants.plan.tasks = [task]
    render(<App/>)
    expect(screen.getByText(expected)).toBeInTheDocument()
    if (!rawVisible) expect(screen.queryByText('旧版纯文字回复')).not.toBeInTheDocument()
    expect(screen.queryByText('服务端未返回回答文本。')).not.toBeInTheDocument()
  })

  it('offers frozen failed-task rerun without using the changed draft', () => {
    const task: TaskSnapshot = {
      task_id: 'failed-task', assistant_id: 'chat', request_id: 'old-request', prompt: '原始附件问题',
      status: 'failed', stage: 'failed', detail: '后台处理失败', created_at: '2026-09-14T13:00:00Z',
      started_at: '2026-09-14T13:00:00Z', finished_at: '2026-09-14T13:00:01Z', elapsed_seconds: 1,
      duration_seconds: null, updated_seq: 1, simulation: false, attachment_ids: ['asset-old'], use_embeddings: true,
      result: null,
    }
    context.state.selected = 'chat'
    context.state.assistants.chat.tasks = [task]
    context.state.assistants.chat.draft.prompt = '用户后来改过的草稿'
    render(<App/>)
    fireEvent.click(screen.getByRole('button', { name: '任务详情' }))
    const rerunButtons = screen.getAllByRole('button', { name: '重新运行此任务' })
    expect(rerunButtons.length).toBe(2)
    fireEvent.click(rerunButtons[0])
    expect(context.rerun).toHaveBeenCalledWith('chat', task)
    expect(context.send).not.toHaveBeenCalled()
  })
})
