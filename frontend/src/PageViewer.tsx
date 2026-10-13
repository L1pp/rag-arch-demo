import { useEffect, useState } from 'react'
import { getStandardsPages, type StandardsPagesQuery } from './api'
import type { StandardsPageImage, StandardsPageSource, StandardsPagesResponse } from './types'

interface ViewerQuery extends StandardsPagesQuery {
  page?: number
}

function hasSource(query: ViewerQuery) {
  return !!(query.std || query.asset || query.plan_id)
}

function readViewerQuery(): ViewerQuery {
  const params = new URLSearchParams(window.location.search)
  const rawPage = params.get('page')
  const parsedPage = rawPage == null ? Number.NaN : Number(rawPage)
  const query: ViewerQuery = {
    std: params.get('std')?.trim() || undefined,
    asset: params.get('asset')?.trim() || undefined,
    plan_id: params.get('plan_id')?.trim() || undefined,
    workspace_id: params.get('workspace_id')?.trim() || undefined,
    evidence: params.get('evidence')?.trim() || undefined,
    source_ref: params.get('source_ref')?.trim() || undefined,
  }
  if (Number.isFinite(parsedPage)) query.page = Math.max(1, Math.floor(parsedPage))
  return query
}

function writeViewerQuery(query: ViewerQuery) {
  const params = new URLSearchParams()
  if (query.std) params.set('std', query.std)
  if (query.asset) params.set('asset', query.asset)
  if (query.plan_id) params.set('plan_id', query.plan_id)
  if (query.workspace_id) params.set('workspace_id', query.workspace_id)
  if (query.evidence) params.set('evidence', query.evidence)
  if (query.source_ref) params.set('source_ref', query.source_ref)
  if (typeof query.page === 'number' && Number.isFinite(query.page)) params.set('page', String(query.page))
  window.history.replaceState(null, '', `${window.location.pathname}?${params}`)
}

function errorMessage(reason: unknown) {
  return reason instanceof Error ? reason.message : '原页暂时无法加载，请稍后重试。'
}

function currentPage(data: StandardsPagesResponse | null, queryPage?: number) {
  const firstPage = data?.pages[0]?.page
  return typeof firstPage === 'number' && Number.isFinite(firstPage) ? firstPage : queryPage ?? 1
}

function sourceLabel(query: ViewerQuery) {
  return query.std || query.asset || query.plan_id || ''
}

function sourceOptionsFor(data: StandardsPagesResponse): StandardsPageSource[] {
  const options = [...data.sources]
  if (data.source_ref && !options.some(source => source.source_ref === data.source_ref)) {
    options.unshift({ source_ref: data.source_ref, filename: data.filename, pages: [] })
  }
  return options
}

function PageImage({ image, title, zoom }: { image: StandardsPageImage; title: string; zoom: number | 'fit' }) {
  const [failed, setFailed] = useState(false)
  const scale = zoom === 'fit' ? '100%' : `${zoom}%`
  const documentPage = image.document_page == null ? '' : `，原文页 ${image.document_page}`
  return <figure className="viewer-page-card">
    <div className="viewer-page-label">第 {image.page} 页{documentPage}</div>
    {failed || !image.image_url
      ? <div className="viewer-image-error" role="img" aria-label={`${title} 第 ${image.page} 页暂不支持`}>当前页面暂不支持</div>
      : <div className="viewer-image-wrap"><img src={image.image_url} alt={`${title} 第 ${image.page} 页`} style={{ width: scale }} onError={() => setFailed(true)} /></div>}
  </figure>
}

export default function PageViewer() {
  const [query, setQuery] = useState<ViewerQuery>(() => readViewerQuery())
  const [data, setData] = useState<StandardsPagesResponse | null>(null)
  const [sourceOptions, setSourceOptions] = useState<StandardsPageSource[]>([])
  const [loading, setLoading] = useState(hasSource(query))
  const [error, setError] = useState(hasSource(query) ? null : '缺少 std、asset 或 plan_id 参数，无法打开规范原页。')
  const [zoom, setZoom] = useState<number | 'fit'>('fit')
  const [reload, setReload] = useState(0)

  useEffect(() => {
    if (!hasSource(query)) {
      setLoading(false)
      setError('缺少 std、asset 或 plan_id 参数，无法打开规范原页。')
      return
    }
    const controller = new AbortController()
    setLoading(true)
    setError(null)
    void getStandardsPages(query, controller.signal)
      .then(payload => {
        setData(payload)
        setSourceOptions(sourceOptionsFor(payload))
      })
      .catch(reason => {
        if (controller.signal.aborted) return
        setData(null)
        setError(errorMessage(reason))
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })
    return () => controller.abort()
  }, [query, reload])

  useEffect(() => {
    if (!data) return
    const previousTitle = document.title
    document.title = `${data.std_no || sourceLabel(query)} · 原页`
    return () => { document.title = previousTitle }
  }, [data, query.std, query.asset, query.plan_id])

  const selectedSource = query.source_ref || data?.source_ref || ''
  const selectedSourceInfo = sourceOptions.find(source => source.source_ref === selectedSource)
  const mappedPages = query.evidence && selectedSourceInfo?.pages.length ? selectedSourceInfo.pages : null
  const page = currentPage(data, query.page)
  const mappedIndex = mappedPages ? mappedPages.indexOf(page) : -1
  const pageIndex = mappedPages && mappedIndex >= 0 ? mappedIndex : Math.max(0, page - 1)
  const canGoPrevious = (mappedPages ? pageIndex > 0 : page > 1) && !loading
  const canGoNext = !!data && (mappedPages ? pageIndex < mappedPages.length - 1 : data.page_count > page) && !loading
  const previousPage = mappedPages ? mappedPages[Math.max(0, pageIndex - 1)] : Math.max(1, page - 1)
  const nextPage = mappedPages ? mappedPages[Math.min(mappedPages.length - 1, pageIndex + 1)] : page + 1
  const pageOptions = mappedPages ?? (data && data.page_count > 0 && data.page_count <= 300
    ? Array.from({ length: data.page_count }, (_, index) => index + 1)
    : [])

  const navigate = (next: Partial<ViewerQuery>) => {
    const updated = { ...query, ...next }
    setQuery(updated)
    writeViewerQuery(updated)
  }

  return <main className="viewer-page" aria-busy={loading}>
    <header className="viewer-header">
      <div className="viewer-heading">
        <a className="viewer-back" href="/">← 工作台</a>
        <div className="viewer-eyebrow">原页浏览器</div>
        <h1>{loading ? '正在加载原页…' : data?.title || '规范原页'}</h1>
        <p>{data?.std_no || sourceLabel(query) || '未指定来源'}{data?.filename ? <> · {data.filename}</> : null}</p>
      </div>
      <div className="viewer-header-meta"><span>每次最多显示两页</span>{query.evidence && <span>证据 {query.evidence}</span>}</div>
    </header>

    <section className="viewer-toolbar" aria-label="原页控制">
      <label className="viewer-source-field" htmlFor="viewer-source"><span>来源</span>
        <select id="viewer-source" value={selectedSource} disabled={loading || !sourceOptions.length} onChange={event => {
          const source = sourceOptions.find(option => option.source_ref === event.target.value)
          navigate({ source_ref: event.target.value, page: source?.pages[0] ?? 1 })
        }}>
          {!sourceOptions.length && <option value="">暂无来源</option>}
          {sourceOptions.map(source => <option key={source.source_ref} value={source.source_ref}>{source.filename || source.source_ref}{source.pages.length ? ` · ${source.pages.length} 页` : ''}</option>)}
        </select>
      </label>
      <div className="viewer-pagination" aria-label="页码翻页">
        <button type="button" onClick={() => navigate({ page: previousPage })} disabled={!canGoPrevious}>上一页</button>
        {!!pageOptions.length && <label className="viewer-page-select"><span className="sr-only">选择页码</span><select aria-label="选择页码" value={page} onChange={event => navigate({ page: Number(event.target.value) })}>{pageOptions.map(option => <option key={option} value={option}>第 {option} 页</option>)}</select></label>}
        <span>第 {page} / {data?.page_count || '—'} 页</span>
        <button type="button" onClick={() => navigate({ page: nextPage })} disabled={!canGoNext}>下一页</button>
      </div>
      <div className="viewer-zoom" aria-label="缩放">
        <button type="button" onClick={() => setZoom(value => value === 'fit' ? 90 : Math.max(50, value - 10))} disabled={zoom !== 'fit' && zoom <= 50}>−</button>
        <button type="button" className={zoom === 100 ? 'active' : ''} onClick={() => setZoom(100)} aria-pressed={zoom === 100}>100%</button>
        <button type="button" className={zoom === 'fit' ? 'active' : ''} onClick={() => setZoom('fit')} aria-pressed={zoom === 'fit'}>适宽</button>
        <button type="button" onClick={() => setZoom(value => value === 'fit' ? 110 : Math.min(180, value + 10))} disabled={zoom !== 'fit' && zoom >= 180}>＋</button>
      </div>
    </section>

    {loading && <div className="viewer-status" role="status">正在加载原页…</div>}
    {!loading && error && <div className="viewer-status viewer-error" role="alert"><div><strong>当前页面暂不支持</strong><span>{error}</span></div><button type="button" onClick={() => setReload(value => value + 1)}>重试</button></div>}
    {!loading && !error && data?.warning && <div className="viewer-status viewer-warning" role="status">{data.warning}</div>}
    {!loading && !error && data && <div className="viewer-page-summary" role="status">{query.evidence && mappedPages ? `本条涉及 ${mappedPages.length} 页` : `来源共 ${data.page_count} 页`} · 当前首屏显示 {Math.min(2, data.pages.length)} 页</div>}
    {!loading && !error && data && !data.pages.length && <div className="viewer-empty" role="status">当前来源暂无可显示的页图。</div>}
    {!loading && !error && data && !!data.pages.length && <section className="viewer-pages" aria-label="规范页图">
      {data.pages.slice(0, 2).map(image => <PageImage key={`${image.page}-${image.image_url}`} image={image} title={data.title} zoom={zoom}/>)}
    </section>}
  </main>
}
