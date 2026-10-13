import { describe, expect, it } from 'vitest'
import { createRerunSubmission, createSubmission, initialState, persistableUi, reducer } from './store'
import { assistantIds, type AssistantId, type TaskSnapshot, type WorkspaceSnapshot } from './types'

const instance = 'server-one'
const task = (overrides: Partial<TaskSnapshot> = {}): TaskSnapshot => ({
  task_id: 'task-chat', assistant_id: 'chat', request_id: 'request-chat',
  prompt: '已提交的历史任务', status: 'running', stage: 'processing',
  detail: '服务器正在模拟处理', created_at: '2026-09-14T01:00:00Z',
  started_at: '2026-09-14T01:00:00Z', finished_at: null,
  elapsed_seconds: 2, duration_seconds: 5, updated_seq: 4, result: null, error: null, simulation: true,
  ...overrides,
})
const snapshot = (
  tasks: Partial<Record<AssistantId, TaskSnapshot[]>> = {},
  serverInstance = instance,
  cursor = 4,
): WorkspaceSnapshot => ({
  workspace_id: serverInstance === instance ? 'workspace-one' : 'workspace-two',
  server_instance_id: serverInstance, cursor, mode: 'infrastructure_demo',
  assistants: assistantIds.map(id => ({
    assistant_id: id, title: { chat: '聊天助手', standards: '规范库问答', plan: '方案辅助核查' }[id],
    tasks: tasks[id] ?? [],
    active_task_id: tasks[id]?.find(t => ['queued', 'running', 'cancelling'].includes(t.status))?.task_id ?? null,
  })),
})
const load = (workspace: WorkspaceSnapshot) =>
  reducer(initialState(), { type: 'snapshot', snapshot: workspace, now: 1000 })

describe('workspace reducer', () => {
  it('ignores late HTTP task state and advances global cursor only from SSE', () => {
    let state = load(snapshot({ chat: [task()] }))
    state = reducer(state, {
      type: 'task', instanceId: instance, cursor: instance + ':8', now: 2000,
      task: task({ status: 'completed', updated_seq: 8, result: { text: '模拟完成', simulation: true } }),
    })
    state = reducer(state, {
      type: 'task', instanceId: instance, now: 3000,
      task: task({ status: 'running', updated_seq: 5 }),
    })
    expect(state.assistants.chat.tasks[0].status).toBe('completed')
    expect(state.assistants.chat.tasks[0].updated_seq).toBe(8)
    expect(state.cursor).toBe(8)
    state = reducer(state, {
      type: 'task', instanceId: instance, now: 4000,
      task: task({ task_id: 'task-standards', assistant_id: 'standards', updated_seq: 12 }),
    })
    expect(state.assistants.standards.tasks[0].updated_seq).toBe(12)
    expect(state.cursor).toBe(8) // HTTP seq12 does not imply that SSE seq9–11 were seen.
  })

  it('does not revive reset history when an older task response arrives', () => {
    let state = load(snapshot({ chat: [task({ status: 'completed', updated_seq: 8 })] }, instance, 8))
    state = reducer(state, {
      type: 'assistantReset', id: 'chat', instanceId: instance, cursor: instance + ':9',
    })
    state = reducer(state, { type: 'task', instanceId: instance, now: 2000, task: task({ updated_seq: 8 }) })
    expect(state.assistants.chat.tasks).toEqual([])
    expect(state.assistants.chat.active_task_id).toBeNull()
    expect(state.assistants.chat.resetSeq).toBe(9)
    expect(state.cursor).toBe(9)
    state = reducer(state, {
      type: 'task', instanceId: instance, cursor: instance + ':10', now: 3000,
      task: task({ task_id: 'new-chat-task', request_id: 'new-request', updated_seq: 10 }),
    })
    expect(state.assistants.chat.tasks.map(t => t.task_id)).toEqual(['new-chat-task'])
  })

  it('restores all three assistants from the server snapshot while keeping UI drafts', () => {
    const initial = initialState({
      workspace_id: 'workspace-one', server_instance_id: instance, selected_assistant: 'plan',
      drafts: {
        chat: { prompt: '聊天草稿' },
        standards: { prompt: '规范草稿' },
        plan: { prompt: '方案草稿' },
      },
    })
    let state = reducer(initial, {
      type: 'snapshot', now: 2000,
      snapshot: snapshot({
        chat: [task({ status: 'completed', updated_seq: 5 })],
        standards: [task({ task_id: 'task-standards', assistant_id: 'standards', updated_seq: 6 })],
        plan: [task({ task_id: 'task-plan', assistant_id: 'plan', status: 'failed', updated_seq: 7, error: '预设失败' })],
      }, instance, 7),
    })
    expect(assistantIds.map(id => state.assistants[id].tasks.length)).toEqual([1, 1, 1])
    expect(state.assistants.standards.active_task_id).toBe('task-standards')
    expect(state.assistants.chat.draft).toEqual({
      prompt: '聊天草稿', duration_seconds: 5, fail_at_end: false, attachment_ids: [], registered_plan_id: null,
      plan_action: 'question', use_embeddings: false, max_steps: 50, max_tool_calls: 150,
    })
    expect(state.assistants.standards.draft.fail_at_end).toBe(false)
    expect(state.selected).toBe('plan')
    const tasksBefore = assistantIds.map(id => state.assistants[id].tasks)
    state = reducer(state, { type: 'select', id: 'chat' })
    expect(assistantIds.map(id => state.assistants[id].tasks)).toEqual(tasksBefore)
    expect(state.assistants.plan.draft.prompt).toBe('方案草稿')
  })

  it('clears business history after a server restart and never persists tasks or reports', () => {
    let state = load(snapshot({
      chat: [task({ status: 'completed', result: { text: '历史模拟报告文本', simulation: true } })],
    }))
    state = reducer(state, { type: 'draft', id: 'chat', patch: { prompt: '本地草稿' } })
    const saved = persistableUi(state)
    expect(Object.keys(saved).sort()).toEqual(['drafts', 'selected_assistant', 'server_instance_id', 'workspace_id'])
    const serialized = JSON.stringify(saved)
    expect(serialized).not.toContain('已提交的历史任务')
    expect(serialized).not.toContain('历史模拟报告文本')
    expect(serialized).not.toContain('request_id')
    expect(serialized).not.toContain('tasks')
    expect(serialized).not.toContain('duration_seconds')
    expect(serialized).not.toContain('fail_at_end')
    state = reducer(state, { type: 'snapshot', snapshot: snapshot({}, 'server-two', 0), now: 5000 })
    expect(assistantIds.every(id => state.assistants[id].tasks.length === 0)).toBe(true)
    expect(state.receivedAt).toEqual({})
    expect(state.workspaceId).toBe('workspace-two')
    expect(state.notice).toContain('后台服务已重启')
    expect(state.assistants.chat.draft.prompt).toBe('本地草稿')
    state = reducer(state, { type: 'task', task: task({ updated_seq: 999 }), instanceId: instance, now: 6000 })
    expect(state.assistants.chat.tasks).toEqual([])
  })
  it('keeps real chat mode through refresh/reset and omits simulation parameters', () => {
    const workspace = snapshot({}, instance, 4)
    workspace.mode = 'chat_preview'
    workspace.model_name = 'test-chat-model'
    workspace.assistants[0].execution_mode = 'model'
    workspace.assistants[1].execution_mode = 'simulation'
    workspace.assistants[2].execution_mode = 'simulation'
    let state = load(workspace)
    state = reducer(state, { type: 'draft', id: 'chat', patch: { prompt: '真实聊天', duration_seconds: 120, fail_at_end: true } })
    const request = createSubmission(state.assistants.chat, 'workspace-one', 'new-real-request')
    expect(request).toEqual({ workspace_id: 'workspace-one', assistant_id: 'chat', request_id: 'new-real-request', prompt: '真实聊天', attachment_ids: [], use_embeddings: false })
    state = reducer(state, { type: 'submitStart', payload: request })
    state = reducer(state, { type: 'submitFailed', id: 'chat', requestId: request.request_id, message: '暂时失败', instanceId: instance })
    expect(createSubmission(state.assistants.chat, 'workspace-one', 'unused-new-id', true).request_id).toBe(request.request_id)
    state = reducer(state, { type: 'assistantReset', id: 'chat', instanceId: instance, cursor: instance + ':5' })
    expect(state.mode).toBe('chat_preview')
    expect(state.modelName).toBe('test-chat-model')
    expect(state.assistants.chat.execution_mode).toBe('model')
    const demo = createSubmission(state.assistants.standards, 'workspace-one', 'demo-request')
    expect(demo.duration_seconds).toBe(5)
    expect(demo.fail_at_end).toBe(false)
  })

  it('captures attachments and plan settings, while retrying the exact failed payload', () => {
    const workspace = snapshot({}, instance, 4)
    workspace.assistants[0].execution_mode = 'model'
    workspace.assistants[2].execution_mode = 'model'
    let state = load(workspace)
    state = reducer(state, { type: 'resources', uploads: [
      { asset_id: 'asset-1', filename: '方案.pdf', size_bytes: 1024 },
    ], plans: [{ plan_id: 'DEMO-PLAN-01', label: 'DEMO-PLAN-01 自编方案', filename: 'DEMO-PLAN-01.pdf' }], defaults: { max_steps: 50, max_tool_calls: 150 } })
    state = reducer(state, { type: 'draft', id: 'chat', patch: { prompt: '结合附件回答', attachment_ids: ['asset-1'], use_embeddings: true } })
    const chatRequest = createSubmission(state.assistants.chat, 'workspace-one', 'request-with-file')
    expect(chatRequest).toMatchObject({ attachment_ids: ['asset-1'], use_embeddings: false })

    state = reducer(state, { type: 'draft', id: 'plan', patch: {
      prompt: '方案问题', registered_plan_id: 'DEMO-PLAN-01', plan_action: 'review', max_steps: 60, max_tool_calls: 180,
    } })
    const planRequest = createSubmission(state.assistants.plan, 'workspace-one', 'request-plan')
    expect(planRequest).toMatchObject({ registered_plan_id: 'DEMO-PLAN-01', plan_action: 'review', max_steps: 60, max_tool_calls: 180 })
    state = reducer(state, { type: 'submitStart', payload: chatRequest })
    state = reducer(state, { type: 'draft', id: 'chat', patch: { attachment_ids: [] } })
    expect(createSubmission(state.assistants.chat, 'workspace-one', 'new-request', true)).toEqual(chatRequest)
  })

  it('reruns legacy chat history with server-selected attachment retrieval', () => {
    const legacy = task({ assistant_id: 'chat', status: 'failed', prompt: '旧任务',
      attachment_ids: ['asset-old'], use_embeddings: true })
    expect(createRerunSubmission(legacy, 'workspace-one', 'new-chat-rerun')).toMatchObject({
      assistant_id: 'chat', attachment_ids: ['asset-old'], use_embeddings: false,
    })
  })

  it('sends the adjustable budget for full reviews and leaves questions to server defaults', () => {
    const state = initialState()
    const assistant = state.assistants.plan
    assistant.execution_mode = 'model'
    const review = createSubmission(assistant, 'workspace-one', 'review-request', false, { plan_action: 'review' })
    expect(review).toMatchObject({ plan_action: 'review', max_steps: 50, max_tool_calls: 150 })

    assistant.draft.max_steps = 60
    assistant.draft.max_tool_calls = 180
    const question = createSubmission(assistant, 'workspace-one', 'question-request', false, {
      plan_action: 'question', max_steps: 80, max_tool_calls: 200,
    })
    expect(question.plan_action).toBe('question')
    expect(question).not.toHaveProperty('max_steps')
    expect(question).not.toHaveProperty('max_tool_calls')
  })

  it('keeps an unchanged pending question when only the report budget changes', () => {
    const assistant = initialState().assistants.plan
    assistant.execution_mode = 'model'
    assistant.draft.prompt = '请说明这一项'
    assistant.pending = createSubmission(assistant, 'workspace-one', 'pending-question')
    assistant.draft.max_steps = 60
    assistant.draft.max_tool_calls = 180

    expect(createSubmission(assistant, 'workspace-one', 'unused-new-id')).toBe(assistant.pending)
  })

  it('retries an existing pending question with its original explicit budget', () => {
    const assistant = initialState().assistants.plan
    assistant.execution_mode = 'model'
    assistant.pending = {
      ...createSubmission(assistant, 'workspace-one', 'legacy-pending-question'),
      max_steps: 50, max_tool_calls: 150,
    }
    assistant.draft.prompt = '新的草稿'
    assistant.draft.max_steps = 60
    assistant.draft.max_tool_calls = 180

    const retried = createSubmission(assistant, 'workspace-one', 'unused-new-id', true)
    expect(retried).toBe(assistant.pending)
    expect(retried).toMatchObject({ max_steps: 50, max_tool_calls: 150 })
  })

  it('preserves saved question budgets when rerunning legacy and current task snapshots', () => {
    for (const budget of [{ max_steps: 50, max_tool_calls: 150 }, { max_steps: 8, max_tool_calls: 20 }]) {
      const failed = task({ assistant_id: 'plan', status: 'failed', plan_action: 'question', ...budget })
      const payload = createRerunSubmission(failed, 'workspace-one', 'rerun-question')
      expect(payload).toMatchObject({ plan_action: 'question', ...budget })
    }
  })

  it('does not invent a report budget when the retried snapshot has no saved limits', () => {
    for (const planAction of ['question', 'review'] as const) {
      const failed = task({ assistant_id: 'plan', status: 'failed', plan_action: planAction,
        max_steps: null, max_tool_calls: null })
      const payload = createRerunSubmission(failed, 'workspace-one', 'rerun-without-budget')
      expect(payload).not.toHaveProperty('max_steps')
      expect(payload).not.toHaveProperty('max_tool_calls')
    }
  })

  it('removes invalid attachment and plan selections when refreshed resources change', () => {
    let state = load(snapshot({}, instance, 4))
    state = reducer(state, { type: 'draft', id: 'chat', patch: { attachment_ids: ['expired', 'kept'] } })
    state = reducer(state, { type: 'draft', id: 'plan', patch: { registered_plan_id: 'DEMO-PLAN-01' } })
    state = reducer(state, { type: 'resources', uploads: [{ asset_id: 'kept', filename: 'ok.txt', size_bytes: 10 }], plans: [], defaults: { max_steps: 50, max_tool_calls: 150 } })
    expect(state.assistants.chat.draft.attachment_ids).toEqual(['kept'])
    expect(state.assistants.plan.draft.registered_plan_id).toBeNull()
  })

  it('builds a fresh rerun payload from the failed task snapshot instead of the current draft', () => {
    const failed = task({
      assistant_id: 'plan', status: 'failed', prompt: '原任务问题', request_id: 'old-request',
      attachment_ids: ['asset-old'], registered_plan_id: 'DEMO-PLAN-02', plan_action: 'review',
      use_embeddings: true, max_steps: 42, max_tool_calls: 123,
    })
    const payload = createRerunSubmission(failed, 'workspace-one', 'new-rerun-request')
    expect(payload).toMatchObject({
      assistant_id: 'plan', request_id: 'new-rerun-request', prompt: '原任务问题', attachment_ids: ['asset-old'],
      registered_plan_id: 'DEMO-PLAN-02', plan_action: 'review', use_embeddings: true, max_steps: 42, max_tool_calls: 123,
    })
    expect(payload.request_id).not.toBe(failed.request_id)
  })

})
