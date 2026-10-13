import { beforeEach, expect, it, vi } from 'vitest'
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
    mode: 'chat_preview', model_name: 'natural-model',
    assistants: assistantIds.map(id => ({
      assistant_id: id,
      title: { chat: '聊天助手', standards: '规范库问答', plan: '方案辅助核查' }[id],
      execution_mode: 'model', active_task_id: null, tasks: [],
    })),
  }
}

beforeEach(() => {
  const state = reducer(initialState(), { type: 'snapshot', snapshot: workspace(), now: Date.now() })
  state.resourcesLoading = false
  state.selected = 'plan'
  context = {
    state, dispatch: vi.fn(), refresh: vi.fn(async () => {}),
    send: vi.fn(async () => {}), rerun: vi.fn(async () => {}), uploadFiles: vi.fn(async () => {}),
    cancel: vi.fn(async () => {}), reset: vi.fn(async () => {}),
  }
})

it('renders only the natural answer and keeps program appendices folded', () => {
  const task: TaskSnapshot = {
    task_id: 'natural-plan', assistant_id: 'plan', request_id: 'natural-request',
    prompt: '生成方案报告', status: 'completed', stage: 'completed', detail: '已完成',
    created_at: '2026-09-22T13:00:00Z', started_at: '2026-09-22T13:00:00Z',
    finished_at: '2026-09-22T13:00:01Z', elapsed_seconds: 1, duration_seconds: null,
    updated_seq: 1, simulation: false,
    result: {
      text: '# 自然报告\n\n正文中描述了一个需要关注的条件。',
      simulation: false, model: 'natural-model', warnings: [], web_sources: [],
      native_tool_usage: {}, token_usage: {},
      report_markdown: '# 自然报告\n\n正文中描述了一个需要关注的条件。\n\n---\n\n运行记录（程序生成）：',
      report: {
        answer: '# 自然报告\n\n正文中描述了一个需要关注的条件。',
        report_format: 'markdown',
        findings: [],
        source_coverage: { provided_units: 4, total_units: 6, partially_provided_units: 1 },
        limitations: ['运行说明附录'],
      },
      citations: [], files: [],
    },
  }
  context.state.assistants.plan.tasks = [task]
  render(<App />)

  expect(screen.getByRole('region', { name: '方案核查报告' })).toBeInTheDocument()
  expect(screen.getByRole('heading', { name: '自然报告' })).toBeInTheDocument()
  expect(screen.getByText('文字提供范围')).toBeInTheDocument()
  expect(screen.getByText('资料与运行说明')).toBeInTheDocument()
  expect(screen.queryByText('运行记录（程序生成）：')).not.toBeInTheDocument()
  const coverageDetails = screen.getByText('文字提供范围').closest('details')
  const limitationsDetails = screen.getByText('资料与运行说明').closest('details')
  expect(coverageDetails).not.toHaveAttribute('open')
  expect(limitationsDetails).not.toHaveAttribute('open')
  fireEvent.click(screen.getByText('文字提供范围'))
  expect(coverageDetails).toHaveAttribute('open')
  expect(screen.getByText(/完整提供 4\/6 个可读单元/)).toBeInTheDocument()
  fireEvent.click(screen.getByText('资料与运行说明'))
  expect(limitationsDetails).toHaveAttribute('open')
  expect(screen.getByText('运行说明附录')).toBeInTheDocument()
  expect(screen.getByText('正文中描述了一个需要关注的条件。')).toBeInTheDocument()
  expect(screen.queryByText(/发现（0）/)).not.toBeInTheDocument()
})

it('shows a follow-up text response when source coverage has no report body', () => {
  const result = {
    text: '追问直接回答正文。', simulation: false, model: 'natural-model', warnings: [],
    web_sources: [], native_tool_usage: {}, token_usage: {},
    source_coverage: { provided_units: 1, total_units: 1 },
  } as unknown as NonNullable<TaskSnapshot['result']>
  const task: TaskSnapshot = {
    task_id: 'natural-question', assistant_id: 'plan', request_id: 'natural-question-request',
    prompt: '请解释这一处', status: 'completed', stage: 'completed', detail: '已完成',
    created_at: '2026-09-22T13:00:00Z', started_at: '2026-09-22T13:00:00Z',
    finished_at: '2026-09-22T13:00:01Z', elapsed_seconds: 1, duration_seconds: null,
    updated_seq: 1, simulation: false, result,
  }
  context.state.assistants.plan.tasks = [task]
  render(<App />)

  expect(screen.getByText('追问直接回答正文。')).toBeInTheDocument()
  expect(screen.queryByRole('region', { name: '方案核查报告' })).not.toBeInTheDocument()
})
