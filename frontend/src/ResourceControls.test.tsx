import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import ResourceControls, { MAX_UPLOAD_BYTES, validateUploadBatch } from './ResourceControls'
import { initialState } from './store'
import { useWorkspace } from './WorkspaceContext'

let context: ReturnType<typeof useWorkspace>
vi.mock('./WorkspaceContext', () => ({ useWorkspace: () => context }))

beforeEach(() => {
  const state = initialState()
  state.workspaceId = 'workspace-one'
  state.loaded = true
  state.loading = false
  state.assistants.plan.execution_mode = 'model'
  state.plans = [{ plan_id: 'P01', label: 'P01 主体方案', filename: 'P01.pdf' }]
  context = {
    state,
    dispatch: vi.fn(),
    refresh: vi.fn(async () => {}),
    send: vi.fn(async () => {}),
    rerun: vi.fn(async () => {}),
    uploadFiles: vi.fn(async () => {}),
    cancel: vi.fn(async () => {}),
    reset: vi.fn(async () => {}),
  }
})

describe('resource controls', () => {
  it('accepts exact 10 MiB and rejects a single or batch over the limit', () => {
    const exact = new File([new Uint8Array(MAX_UPLOAD_BYTES)], 'exact.txt')
    const over = new File([new Uint8Array(MAX_UPLOAD_BYTES + 1)], 'over.txt')
    const six = new File([new Uint8Array(6 * 1024 * 1024)], 'six.txt')
    const fourPlus = new File([new Uint8Array(4 * 1024 * 1024 + 1)], 'four-plus.txt')
    expect(validateUploadBatch([exact])).toBeNull()
    expect(validateUploadBatch([over])).toContain('单个文件')
    expect(validateUploadBatch([six, fourPlus])).toContain('本批附件总大小')
    expect(validateUploadBatch([new File(['x'], 'script.exe')])).toContain('仅支持')
  })

  it('uploads optional chat attachments without making them required', async () => {
    render(<ResourceControls id="chat" busy={false}/>)
    expect(screen.queryByText('启用向量检索')).not.toBeInTheDocument()
    expect(screen.getByText('提取正文合计≤20万估算Token时全文阅读，超过后自动混合检索。')).toBeInTheDocument()
    const file = new File(['hello'], 'notes.txt', { type: 'text/plain' })
    fireEvent.change(screen.getByLabelText('选择附件'), { target: { files: [file] } })
    await waitFor(() => expect(context.uploadFiles).toHaveBeenCalledWith('chat', [file]))
  })

  it('selects a registered plan and submits the full review action', () => {
    context.state.assistants.plan.draft.registered_plan_id = 'P01'
    render(<ResourceControls id="plan" busy={false}/>)
    fireEvent.click(screen.getByRole('button', { name: '生成整份方案核查报告' }))
    expect(context.send).toHaveBeenCalledWith('plan', false, { prompt: '生成整份方案核查报告', plan_action: 'review' })
  })

  it('clears upload selection when choosing a registered plan', () => {
    context.state.assistants.plan.draft.attachment_ids = ['asset-1']
    render(<ResourceControls id="plan" busy={false}/>)
    fireEvent.change(screen.getByRole('combobox', { name: '已登记方案' }), { target: { value: 'P01' } })
    expect(context.dispatch).toHaveBeenCalledWith({ type: 'draft', id: 'plan', patch: { registered_plan_id: 'P01', attachment_ids: [] } })
  })
})
