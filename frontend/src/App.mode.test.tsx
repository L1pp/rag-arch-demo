import { beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import App from './App'
import { initialState, reducer } from './store'
import { assistantIds, type TaskSnapshot, type WorkspaceSnapshot } from './types'
import { useWorkspace } from './WorkspaceContext'

let context: ReturnType<typeof useWorkspace>
vi.mock('./WorkspaceContext', () => ({ useWorkspace: () => context }))

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
  const state = reducer(initialState(), { type: 'snapshot', snapshot: workspace(), now: Date.now() })
  state.resourcesLoading = false
  context = {
    state, dispatch: vi.fn(), refresh: vi.fn(async () => {}),
    send: vi.fn(async () => {}), rerun: vi.fn(async () => {}), uploadFiles: vi.fn(async () => {}), cancel: vi.fn(async () => {}), reset: vi.fn(async () => {}),
  }
})

describe('execution mode presentation', () => {
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
    context.state.assistants.standards.draft.prompt = 'DEMO-STD-01 中施工荷载有哪些要求？'
    render(<App/>)
    expect(screen.getByText('查阅规范条文与版本信息')).toBeInTheDocument()
    expect(screen.queryByText(/演示设置/)).not.toBeInTheDocument()
    expect(screen.queryByRole('spinbutton', { hidden: true })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: '发送' })).toBeEnabled()
    expect(screen.queryByRole('button', { name: '发送模拟任务' })).not.toBeInTheDocument()
    expect(screen.getByText('回答附规范依据。')).toBeInTheDocument()
  })

  it.each([
    { name: '引用被拒绝', status: 'insufficient_evidence', termination_reason: 'citation_rejected', title: '引用未通过核验', disallowed: '资料不足以支持结论' },
    { name: '格式错误', status: 'insufficient_evidence', termination_reason: 'invalid_result', title: '输出格式未通过核验', disallowed: '资料不足' },
    { name: '确实缺少证据', status: 'insufficient_evidence', termination_reason: 'insufficient_evidence', title: '资料不足以支持结论', disallowed: null },
    { name: '只有状态没有原因', status: 'insufficient_evidence', termination_reason: '', title: '回答尚未通过核验', disallowed: '资料不足以支持结论' },
    { name: '正常回答', status: 'answered', termination_reason: 'answered', title: null, disallowed: null },
  ])('marks $name correctly and preserves the answer text', ({ status, termination_reason, title, disallowed }) => {
    const task: TaskSnapshot = {
      task_id: 'validation-task', assistant_id: 'standards', request_id: 'validation-request', prompt: '自编规范示例问题',
      status: 'completed', stage: 'completed', detail: '回答已完成', created_at: '2026-09-14T13:00:00Z',
      started_at: '2026-09-14T13:00:00Z', finished_at: '2026-09-14T13:00:01Z', elapsed_seconds: 1,
      duration_seconds: null, updated_seq: 1, simulation: false,
      result: {
        text: '这是保留的模型原文。', simulation: false, model: 'test-model', warnings: [], web_sources: [],
        native_tool_usage: {}, token_usage: {}, citations: [], catalog_matches: [], status, termination_reason,
      },
    }
    context.state.selected = 'standards'
    context.state.assistants.standards.execution_mode = 'model'
    context.state.assistants.standards.tasks = [task]
    render(<App />)

    expect(screen.getByText('这是保留的模型原文。')).toBeInTheDocument()
    if (title) {
      const notice = screen.getByRole('alert')
      expect(notice).toHaveTextContent(title)
      if (termination_reason === 'citation_rejected' || termination_reason === 'invalid_result') {
        expect(notice).toHaveTextContent('下方保留模型原文供检查')
      }
      if (disallowed) expect(notice).not.toHaveTextContent(disallowed)
    } else {
      expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    }
  })
  it('renders standards citations as evidence cards with original-page links', () => {
    const task: TaskSnapshot = {
      task_id: 'standards-task', assistant_id: 'standards', request_id: 'standards-request', prompt: '防水要求',
      status: 'completed', stage: 'completed', detail: '回答已完成', created_at: '2026-09-14T13:00:00Z',
      started_at: '2026-09-14T13:00:00Z', finished_at: '2026-09-14T13:00:01Z', elapsed_seconds: 1,
      duration_seconds: null, updated_seq: 1, simulation: false,
      result: {
        text: '请按规范施工。', simulation: false, model: 'test-standards-model', warnings: [], web_sources: [],
        native_tool_usage: {}, token_usage: {}, citations: [{
          evidence_id: 'ev-1', std_no: 'DEMO-STD-02', title: '自编防水示例规范', clause_no: '示例条款一',
          text: '防水工程应满足设计要求。', page_url: '/viewer/pages?std=demo-std-02&evidence=ev-1', history_url: '/viewer/history/?std=demo-std-02',
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
        citations: [{ evidence_id: 'plan-ev-1', std_no: 'DEMO-STD-01', title: '自编施工方案示例引用', clause_no: '示例条款二', text: '规范引用摘要', page_url: '/viewer/pages?std=demo-std-01&evidence=plan-ev-1', history_url: '/viewer/history/?std=demo-std-01' }],
        files: [{ asset_id: 'report-1', filename: '方案核查报告.md', size_bytes: 1024, page_url: '/viewer/pages?asset=plan-1&workspace_id=work' }],
      },
    }
    context.state.selected = 'plan'
    context.state.assistants.plan.tasks = [task]
    context.state.assistants.plan.draft.registered_plan_id = 'DEMO-PLAN-01'
    context.state.plans = [{ plan_id: 'DEMO-PLAN-01', label: 'DEMO-PLAN-01 自编方案', filename: 'DEMO-PLAN-01.pdf' }]
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
    expect(screen.getByText('整份核查覆盖文字；追问可按需读图。')).toBeInTheDocument()
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
