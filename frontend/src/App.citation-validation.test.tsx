import { beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
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

function task(result: Record<string, unknown>, overrides: Partial<TaskSnapshot> = {}): TaskSnapshot {
  return {
    task_id: 'qa-task', assistant_id: 'standards', request_id: 'qa-request', prompt: '查询测试条文',
    status: 'completed', stage: 'completed', detail: '回答已完成', created_at: '2026-10-23T00:00:00Z',
    started_at: '2026-10-23T00:00:00Z', finished_at: '2026-10-23T00:00:01Z', elapsed_seconds: 1,
    duration_seconds: null, updated_seq: 1, simulation: false, error: null,
    result: {
      text: '模型回答原文，供核对。', simulation: false, model: 'test-model', warnings: [],
      web_sources: [], native_tool_usage: {}, token_usage: {}, ...result,
    } as NonNullable<TaskSnapshot['result']>,
    ...overrides,
  }
}

beforeEach(() => {
  const state = reducer(initialState(), { type: 'snapshot', snapshot: workspace(), now: Date.now() })
  state.resourcesLoading = false
  state.selected = 'standards'
  context = {
    state, dispatch: vi.fn(), refresh: vi.fn(async () => {}), send: vi.fn(async () => {}),
    rerun: vi.fn(async () => {}), uploadFiles: vi.fn(async () => {}), cancel: vi.fn(async () => {}), reset: vi.fn(async () => {}),
  }
})

describe('standards answer validation notices', () => {
  it('warns when citation validation rejects a completed model answer while retaining its text', () => {
    context.state.assistants.standards.tasks = [task({
      status: 'insufficient_evidence', termination_reason: 'citation_rejected',
    })]

    render(<App />)

    expect(screen.getByRole('alert')).toHaveTextContent('引用未通过核验')
    expect(screen.getByText('模型回答原文，供核对。')).toBeInTheDocument()
    expect(screen.getByRole('alert')).toHaveTextContent('不能视为已验证结论')
  })

  it('warns on insufficient evidence even when no termination reason is supplied', () => {
    context.state.assistants.standards.tasks = [task({ status: 'insufficient_evidence' })]

    render(<App />)

    expect(screen.getByRole('alert')).toHaveTextContent('回答尚未通过核验')
  })

  it('does not warn for a grounded answer or a simulated task', () => {
    context.state.assistants.standards.tasks = [task({ status: 'grounded', termination_reason: 'answered' })]
    const { rerender } = render(<App />)
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()

    context.state.assistants.standards.tasks = [task({
      status: 'insufficient_evidence', termination_reason: 'citation_rejected',
    }, { simulation: true })]
    rerender(<App />)
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })
})
