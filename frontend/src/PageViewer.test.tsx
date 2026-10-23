import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import PageViewer from './PageViewer'
import { getProtectedBlob, getStandardsPages } from './api'

vi.mock('./api', async (importOriginal) => ({
  ...await importOriginal<typeof import('./api')>(),
  getStandardsPages: vi.fn(),
  getProtectedBlob: vi.fn(),
}))

class TestURL extends URL {
  static createObjectURL = vi.fn(() => 'blob:page-viewer-test')
  static revokeObjectURL = vi.fn()
}

const pagesFor = (sourceRef = 'source-a') => ({
  std_no: 'GB 55008-2021', title: '工程防水通用规范', source_ref: sourceRef,
  filename: `${sourceRef}.pdf`, page_count: 8,
  sources: [
    { source_ref: 'source-a', filename: '正式版.pdf', pages: [2, 5, 8] },
    { source_ref: 'source-b', filename: '扫描版.pdf', pages: [1, 2] },
  ],
  pages: [
    { page: sourceRef === 'source-a' ? 2 : 1, document_page: sourceRef === 'source-a' ? 12 : 1, image_url: `/images/${sourceRef}-1.png` },
    { page: sourceRef === 'source-a' ? 5 : 2, document_page: sourceRef === 'source-a' ? 15 : 2, image_url: `/images/${sourceRef}-2.png` },
  ],
})

const pagesMock = vi.mocked(getStandardsPages)
const imageMock = vi.mocked(getProtectedBlob)

afterEach(() => { vi.unstubAllGlobals(); vi.unstubAllEnvs() })

describe('independent standards page viewer', () => {
  beforeEach(() => {
    window.history.replaceState(null, '', '/viewer/pages?std=gb-55008&evidence=evidence-1&source_ref=source-a&page=2')
    pagesMock.mockResolvedValue(pagesFor())
    imageMock.mockReset().mockResolvedValue(new Blob(['page image']))
    vi.stubGlobal('URL', TestURL)
  })

  it('renders without WorkspaceProvider, shows mapped evidence pages, and changes source/page', async () => {
    render(<PageViewer />)
    expect(await screen.findByRole('heading', { name: '工程防水通用规范' })).toBeInTheDocument()
    expect(screen.getByText((_, element) => !!element && element.classList.contains('viewer-page-summary') && element.textContent?.includes('本条涉及 3 页') === true)).toBeInTheDocument()
    expect(await screen.findAllByRole('img')).toHaveLength(2)
    expect(screen.getByRole('combobox', { name: '选择页码' })).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: '下一页' }))
    await waitFor(() => expect(pagesMock).toHaveBeenLastCalledWith(expect.objectContaining({ page: 5, evidence: 'evidence-1', source_ref: 'source-a' }), expect.any(AbortSignal)))

    fireEvent.change(screen.getByRole('combobox', { name: '来源' }), { target: { value: 'source-b' } })
    await waitFor(() => expect(pagesMock).toHaveBeenLastCalledWith(expect.objectContaining({ page: 1, evidence: 'evidence-1', source_ref: 'source-b' }), expect.any(AbortSignal)))
  })

  it('lets the backend choose the mapped page when an evidence link has no page parameter', async () => {
    window.history.replaceState(null, '', '/viewer/pages?std=gb-55008&evidence=evidence-1')
    pagesMock.mockReset()
    pagesMock.mockImplementation(async query => {
      expect(query).toMatchObject({ std: 'gb-55008', evidence: 'evidence-1' })
      expect(query).not.toHaveProperty('page')
      return pagesFor()
    })
    render(<PageViewer />)
    expect(await screen.findByAltText('工程防水通用规范 第 2 页')).toBeInTheDocument()
    expect(pagesMock).toHaveBeenCalledTimes(1)
  })

  it('reports a single image failure locally and displays metadata errors with retry', async () => {
    render(<PageViewer />)
    const image = await screen.findByAltText('工程防水通用规范 第 2 页')
    fireEvent.error(image)
    expect(await screen.findByText('当前页面暂不支持')).toBeInTheDocument()

    pagesMock.mockRejectedValueOnce(new Error('后端没有对应原页'))
    fireEvent.click(screen.getByRole('button', { name: '下一页' }))
    expect(await screen.findByText('后端没有对应原页')).toBeInTheDocument()
    expect(screen.getByRole('alert')).toHaveTextContent('当前页面暂不支持')
  })

  it('keeps the previous source selector after a source metadata error without showing stale images', async () => {
    pagesMock.mockResolvedValueOnce(pagesFor())
    render(<PageViewer />)
    expect(await screen.findByAltText('工程防水通用规范 第 2 页')).toBeInTheDocument()

    pagesMock.mockRejectedValueOnce(new Error('副本元数据不可用'))
    fireEvent.change(screen.getByRole('combobox', { name: '来源' }), { target: { value: 'source-b' } })
    expect(await screen.findByText('副本元数据不可用')).toBeInTheDocument()
    expect(screen.getByRole('combobox', { name: '来源' })).toBeEnabled()
    expect(screen.queryByAltText('工程防水通用规范 第 2 页')).not.toBeInTheDocument()

    pagesMock.mockResolvedValueOnce(pagesFor('source-a'))
    fireEvent.change(screen.getByRole('combobox', { name: '来源' }), { target: { value: 'source-a' } })
    await waitFor(() => expect(pagesMock).toHaveBeenLastCalledWith(expect.objectContaining({ source_ref: 'source-a', page: 2 }), expect.any(AbortSignal)))
  })

  it('shows an explicit empty-source state', async () => {
    pagesMock.mockResolvedValueOnce({ ...pagesFor(), pages: [] })
    render(<PageViewer />)
    expect(await screen.findByText('当前来源暂无可显示的页图。')).toBeInTheDocument()
  })

  it('passes private page image URLs to the credentialed API resource fetcher', async () => {
    vi.stubEnv('VITE_API_BASE_URL', 'https://api.example.test/prefix')
    pagesMock.mockResolvedValueOnce({
      ...pagesFor(),
      pages: [{ page: 2, document_page: 12, image_url: '/api/documents/page-image?asset=private-a&page=2' }],
    })
    render(<PageViewer />)
    const image = await screen.findByAltText('工程防水通用规范 第 2 页') as HTMLImageElement
    expect(image.src).toBe('blob:page-viewer-test')
    expect(imageMock).toHaveBeenCalledWith('/api/documents/page-image?asset=private-a&page=2', expect.any(AbortSignal))
  })
})
