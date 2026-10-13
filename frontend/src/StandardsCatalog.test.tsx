import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { StandardsCatalogPanel } from './App'
import { getStandardCandidates, getStandardsCatalog } from './api'

vi.mock('./api', async () => {
  const actual = await vi.importActual<typeof import('./api')>('./api')
  return { ...actual, getStandardsCatalog: vi.fn(), getStandardCandidates: vi.fn() }
})

const catalog = {
  entries: [
    {
      catalog_id: 'demo-std-01', std_no: 'DEMO-STD-01', title: '自编防水示例规范', library_class: '自编演示资料',
      text_available: true, original_available: true, page_url: '/original/demo-std-01', history_url: '/viewer/history/?std=demo-std-01',
    },
    {
      catalog_id: 'demo-std-02', std_no: 'DEMO-STD-02', title: '自编防火示例规范', library_class: '自编演示资料',
      text_available: true, original_available: false, page_url: null, history_url: '/viewer/history/?std=demo-std-02',
    },
  ],
  total: 2,
}

const catalogMock = vi.mocked(getStandardsCatalog)
const candidateMock = vi.mocked(getStandardCandidates)

function openCatalog() {
  const view = render(<StandardsCatalogPanel />)
  fireEvent.click(screen.getByRole('button', { name: '展开规范目录' }))
  return view
}

describe('standards catalog', () => {
  beforeEach(() => {
    catalogMock.mockResolvedValue(catalog)
    candidateMock.mockReset()
  })

  it('filters by standard number/title and keeps source links in new tabs', async () => {
    openCatalog()
    expect(await screen.findByText('DEMO-STD-01')).toBeInTheDocument()
    const original = screen.getByRole('link', { name: '查看原页↗' })
    const history = screen.getAllByRole('link', { name: '历史关系↗' })[0]
    expect(original).toHaveAttribute('target', '_blank')
    expect(original).toHaveAttribute('rel', 'noopener noreferrer')
    expect(history).toHaveAttribute('target', '_blank')
    expect(history).toHaveAttribute('rel', 'noopener noreferrer')

    fireEvent.change(screen.getByRole('textbox', { name: '搜索规范号或中文名' }), { target: { value: '防火' } })
    expect(screen.getByText('DEMO-STD-02')).toBeInTheDocument()
    expect(screen.queryByText('DEMO-STD-01')).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '收起规范目录' }))
    expect(screen.queryByRole('textbox', { name: '搜索规范号或中文名' })).not.toBeInTheDocument()
  })

  it('shows empty and request-error states without creating a task', async () => {
    catalogMock.mockResolvedValueOnce({ entries: [], total: 0 })
    const { unmount } = openCatalog()
    expect(await screen.findByText('暂无可用规范。')).toBeInTheDocument()
    unmount()

    catalogMock.mockRejectedValueOnce(new Error('目录服务不可用'))
    openCatalog()
    expect(await screen.findByText('目录服务不可用')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '重试' })).toBeInTheDocument()
    await waitFor(() => expect(catalogMock).toHaveBeenCalled())
  })

  it('only queries semantic candidates on button/Enter and displays reasons without confidence claims', async () => {
    candidateMock.mockResolvedValue({
      query: 'DEMO-STD-03 旧版', semantic_used: true, warnings: [],
      matches: [{ ...catalog.entries[0], candidate_only: true, match_method: 'identifier_alias', match_notes: ['编号年份与目录候选接近'] }],
    })
    openCatalog()
    const input = screen.getByRole('textbox', { name: '搜索规范号或中文名' })
    fireEvent.change(input, { target: { value: 'DEMO-STD-03 旧版' } })
    expect(candidateMock).not.toHaveBeenCalled()
    fireEvent.keyDown(input, { key: 'Enter' })
    await waitFor(() => expect(candidateMock).toHaveBeenCalledWith('DEMO-STD-03 旧版', expect.any(AbortSignal)))
    expect(screen.getByText('候选')).toBeInTheDocument()
    expect(screen.getByText('编号年份与目录候选接近')).toBeInTheDocument()
    expect(screen.queryByText(/置信度/)).not.toBeInTheDocument()
  })
})
