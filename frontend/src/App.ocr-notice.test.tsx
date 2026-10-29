import { beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, within } from '@testing-library/react'
import App from './App'
import { initialState, reducer } from './store'
import { assistantIds, type TaskSnapshot, type WorkspaceSnapshot } from './types'
import { useWorkspace } from './WorkspaceContext'

let context: ReturnType<typeof useWorkspace>
vi.mock('./WorkspaceContext', () => ({ useWorkspace: () => context }))
vi.mock('./api', () => ({
  ApiError: class extends Error { constructor(message: string, readonly status: number) { super(message) } },
  downloadTask: vi.fn(async () => undefined), getStandardCandidates: vi.fn(async () => ({ matches: [] })),
  getStandardsCatalog: vi.fn(async () => ({ entries: [] })), logout: vi.fn(async () => ({ authenticated: false })),
  redirectToLogin: vi.fn(), taskDownloadUrl: vi.fn(() => '/api/tasks/test/download'),
}))

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

function task(assistant: 'chat' | 'plan', extras: Record<string, unknown>): TaskSnapshot {
  return {
    task_id: 'ocr-task', assistant_id: assistant, request_id: 'ocr-request', prompt: '读取方案',
    status: 'completed', stage: 'completed', detail: '回答已完成', created_at: '2026-10-28T00:00:00Z',
    started_at: '2026-10-28T00:00:00Z', finished_at: '2026-10-28T00:00:01Z', elapsed_seconds: 1,
    duration_seconds: null, updated_seq: 1, simulation: false, error: null,
    result: {
      text: '模型回答原文。', simulation: false, model: 'test-model', warnings: [],
      web_sources: [], native_tool_usage: {}, token_usage: {}, ...extras,
    } as NonNullable<TaskSnapshot['result']>,
  }
}

beforeEach(() => {
  const state = reducer(initialState(), { type: 'snapshot', snapshot: workspace(), now: Date.now() })
  state.resourcesLoading = false
  context = {
    state, dispatch: vi.fn(), refresh: vi.fn(async () => {}), send: vi.fn(async () => {}),
    rerun: vi.fn(async () => {}), uploadFiles: vi.fn(async () => {}), cancel: vi.fn(async () => {}), reset: vi.fn(async () => {}),
  }
})

describe('OCR notices in task results', () => {
  it('shows an OCR warning next to the chat answer without opening details and preserves the original-page link', () => {
    context.state.assistants.chat.tasks = [task('chat', {
      warnings: ['第 2 页 OCR 已识别，文字需核对原页。', '普通附加说明。'],
      files: [{ asset_id: 'doc-1', filename: '方案.pdf', size_bytes: 100, page_url: '/viewer/pages?asset=doc-1&page=2' }],
    })]
    render(<App />)

    const notice = screen.getByRole('status', { name: 'OCR 识别提示' })
    expect(notice).toBeVisible()
    expect(within(notice).getByText('第 2 页 OCR 已识别，文字需核对原页。')).toBeVisible()
    expect(notice).toHaveTextContent('关键数字、单位和表格请核对原页')
    expect(screen.getByRole('link', { name: '查看原件↗' })).toHaveAttribute('href', '/viewer/pages?asset=doc-1&page=2')
    expect(screen.getByText('来源与说明').closest('details')).not.toHaveAttribute('open')
    expect(screen.getByText('模型回答原文。')).toBeVisible()
  })

  it('shows plan page statuses even if no warning was supplied, including system-rendered Word pages', () => {
    context.state.selected = 'plan'
    context.state.assistants.plan.tasks = [task('plan', {
      document: { ocr_pages: [
        { page: 1, status: 'completed', recognition_verified: false, page_numbering: 'system_rendered_page' },
        { page: 2, status: 'failed' }, { page: 3, status: 'unknown' }, { page: 4, status: 'not_attempted' },
      ] },
      files: [{ asset_id: 'doc-2', filename: '方案.docx', size_bytes: 100, page_url: '/viewer/pages?asset=doc-2&page=1' }],
    })]
    render(<App />)

    const notice = screen.getByRole('status', { name: 'OCR 识别提示' })
    expect(notice).toBeVisible()
    expect(notice).toHaveTextContent('系统排版第 1 页：已识别，待核对')
    expect(notice).toHaveTextContent('第 2 页：识别未成功')
    expect(notice).toHaveTextContent('第 3 页：识别状态未确认')
    expect(notice).toHaveTextContent('第 4 页：未执行识别')
    expect(screen.getByRole('link', { name: '查看原件↗' })).toHaveAttribute('href', '/viewer/pages?asset=doc-2&page=1')
  })

  it('shows per-file OCR pages without warnings while retaining document-level status', () => {
    context.state.assistants.chat.tasks = [task('chat', {
      warnings: [],
      document: { ocr_pages: [{ page: 7, status: 'unknown', recognition_verified: false }] },
      files: [
        { asset_id: 'file-1', filename: '自编示例甲.pdf', size_bytes: 100, ocr_pages: [
          { page: 1, status: 'completed', recognition_verified: false, reason: 'https://example.test/private?token=demo' },
          { page: 2, status: 'failed', recognition_verified: false, reason: '/tmp/synthetic-ocr/private.log' },
        ] },
        { asset_id: 'file-2', filename: '自编示例乙.docx', size_bytes: 200, ocr_pages: [
          { page: 1, status: 'not_attempted', page_numbering: 'system_rendered_page', recognition_verified: false },
        ] },
      ],
    })]
    render(<App />)

    const notice = screen.getByRole('status', { name: 'OCR 识别提示' })
    expect(notice).toBeVisible()
    expect(notice).toHaveTextContent('第 7 页：识别状态未确认')
    expect(notice).toHaveTextContent('附件 1（自编示例甲.pdf） · 第 1 页：已识别，待核对')
    expect(notice).toHaveTextContent('附件 1（自编示例甲.pdf） · 第 2 页：识别未成功')
    expect(notice).toHaveTextContent('附件 2（自编示例乙.docx） · 系统排版第 1 页：未执行识别')
    expect(notice).toHaveTextContent('关键数字、单位和表格请核对原页')
    expect(within(notice).queryByText(/example\.test|token=demo|synthetic-ocr/)).not.toBeInTheDocument()
  })

  it('does not show an OCR notice when attachment pages are absent or empty', () => {
    context.state.assistants.chat.tasks = [task('chat', {
      warnings: [], files: [
        { asset_id: 'file-1', filename: '自编示例甲.pdf', size_bytes: 100, ocr_pages: [] },
        { asset_id: 'file-2', filename: '自编示例乙.pdf', size_bytes: 200 },
      ],
    })]
    render(<App />)
    expect(screen.queryByRole('status', { name: 'OCR 识别提示' })).not.toBeInTheDocument()
  })

  it('keeps ordinary native-text warnings in the existing collapsed section without adding an OCR notice', () => {
    context.state.assistants.chat.tasks = [task('chat', { warnings: ['普通文字提取说明。'], document: { ocr_pages: [] } })]
    render(<App />)

    expect(screen.queryByRole('status', { name: 'OCR 识别提示' })).not.toBeInTheDocument()
    fireEvent.click(screen.getByText('来源与说明'))
    expect(screen.getByText('普通文字提取说明。')).toBeVisible()
  })

  it('does not expose OCR provider URLs or credentials in either notice', () => {
    context.state.assistants.chat.tasks = [task('chat', { warnings: [
      'OCR 失败：https://example.test/secret?token=private',
      'OCR 失败，API Key: private-key',
      'OCR 失败，文件 /tmp/synthetic-ocr/page.pdf',
    ] })]
    render(<App />)

    const notice = screen.getByRole('status', { name: 'OCR 识别提示' })
    expect(notice).toHaveTextContent('OCR 状态暂不可显示，请查看原页。')
    fireEvent.click(screen.getByText('来源与说明'))
    expect(screen.queryByText(/example\.test|token=private|private-key|\/tmp\/synthetic-ocr/)).not.toBeInTheDocument()
  })
})
