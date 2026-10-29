import { useRef, useState, type ChangeEvent } from 'react'
import { useWorkspace } from './WorkspaceContext'
import type { AssistantId, UploadAsset } from './types'

export const MAX_UPLOAD_BYTES = 10 * 1024 * 1024
const SUPPORTED_UPLOAD_EXTENSIONS = ['.pdf', '.doc', '.docx', '.txt', '.md']

export function formatBytes(value: number) {
  if (value < 1024) return `${value} B`
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`
  return `${(value / (1024 * 1024)).toFixed(1)} MB`
}

export function validateUploadBatch(files: File[], existingBytes = 0, maxFiles = Number.POSITIVE_INFINITY) {
  if (files.length > maxFiles) return `本助手最多选择 ${maxFiles} 个文件。`
  if (files.some(file => !SUPPORTED_UPLOAD_EXTENSIONS.some(extension => file.name.toLowerCase().endsWith(extension)))) return '仅支持 PDF、DOC、DOCX、TXT、MD 文件。'
  const batchBytes = files.reduce((total, file) => total + file.size, 0)
  if (files.some(file => file.size > MAX_UPLOAD_BYTES)) return '单个文件不能超过 10 MiB。'
  if (batchBytes > MAX_UPLOAD_BYTES || existingBytes + batchBytes > MAX_UPLOAD_BYTES) return '本批附件总大小不能超过 10 MiB。'
  return null
}

function AssetList({ assets, onRemove, disabled }: { assets: UploadAsset[]; onRemove: (assetId: string) => void; disabled: boolean }) {
  if (!assets.length) return null
  return <div className="selected-assets" aria-label="已选择附件">
    {assets.map(asset => <span className="selected-asset" key={asset.asset_id}>
      <span title={asset.filename}>{asset.filename}</span><small>{formatBytes(asset.size_bytes)}</small>
      <button type="button" aria-label={`移除附件 ${asset.filename}`} disabled={disabled} onClick={() => onRemove(asset.asset_id)}>×</button>
    </span>)}
  </div>
}

function AvailableAssets({ assets, disabled, onSelect }: { assets: UploadAsset[]; disabled: boolean; onSelect: (asset: UploadAsset) => void }) {
  if (!assets.length) return null
  return <details className="available-assets">
    <summary>已上传文件<span>{assets.length} 个</span></summary>
    <div>{assets.map(asset => <button type="button" key={asset.asset_id} disabled={disabled} onClick={() => onSelect(asset)}>
      <span>{asset.filename}</span><small>{formatBytes(asset.size_bytes)}</small>
    </button>)}</div>
  </details>
}

export default function ResourceControls({ id, busy }: { id: AssistantId; busy: boolean }) {
  const { state, dispatch, send, uploadFiles } = useWorkspace()
  const assistant = state.assistants[id]
  const inputRef = useRef<HTMLInputElement>(null)
  const [error, setError] = useState<string | null>(null)
  if (id === 'standards') return null

  const selectedAssets = assistant.draft.attachment_ids
    .map(assetId => state.uploads.find(asset => asset.asset_id === assetId))
    .filter((asset): asset is UploadAsset => !!asset)
  const selectedBytes = selectedAssets.reduce((total, asset) => total + asset.size_bytes, 0)
  const availableAssets = state.uploads.filter(asset => !assistant.draft.attachment_ids.includes(asset.asset_id))
  const sourceSelected = id === 'plan' && (!!assistant.draft.registered_plan_id || assistant.draft.attachment_ids.length > 0)
  const selectedPlan = assistant.draft.registered_plan_id
    ? state.plans.find(plan => plan.plan_id === assistant.draft.registered_plan_id)
    : undefined
  const planLabel = selectedPlan?.label || assistant.draft.registered_plan_id || ''
  const uploading = state.uploadingAssistant !== null

  const handleFiles = async (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files ?? [])
    event.target.value = ''
    if (!files.length) return
    if (id === 'plan' && assistant.draft.attachment_ids.length + files.length > 1) {
      setError('方案只能选择一个文件，请先移除当前文件。')
      return
    }
    const validation = validateUploadBatch(files, selectedBytes, id === 'plan' ? 1 : Number.POSITIVE_INFINITY)
    if (validation) {
      setError(validation)
      return
    }
    setError(null)
    if (id === 'plan' && assistant.draft.registered_plan_id) {
      dispatch({ type: 'draft', id, patch: { registered_plan_id: null } })
    }
    try {
      await uploadFiles(id, files)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '上传未完成，请重试。')
    }
  }

  const removeAsset = (assetId: string) => {
    dispatch({ type: 'draft', id, patch: { attachment_ids: assistant.draft.attachment_ids.filter(value => value !== assetId) } })
  }

  const selectExisting = (asset: UploadAsset) => {
    if (id === 'plan') {
      dispatch({ type: 'draft', id, patch: { registered_plan_id: null, attachment_ids: [asset.asset_id] } })
      return
    }
    if (selectedBytes + asset.size_bytes > MAX_UPLOAD_BYTES) {
      setError('本批附件总大小不能超过 10 MiB。')
      return
    }
    dispatch({ type: 'draft', id, patch: { attachment_ids: [...assistant.draft.attachment_ids, asset.asset_id] } })
  }

  return <section className={`resource-controls resource-controls-${id}`} aria-label={id === 'plan' ? '方案来源与设置' : '聊天附件'}>
    {id === 'chat' && <>
      <div className="resource-toolbar">
        <label className="upload-button">
          <input ref={inputRef} aria-label="选择附件" type="file" multiple accept=".pdf,.doc,.docx,.txt,.md,application/pdf,application/msword,text/plain,application/vnd.openxmlformats-officedocument.wordprocessingml.document" disabled={busy || uploading} onChange={handleFiles}/>
          <span>添加附件</span>
        </label>
        {state.uploadingAssistant === id && <span className="resource-uploading">上传中…</span>}
      </div>
      <p className="resource-note">提取正文合计≤20万估算Token时全文阅读，超过后自动混合检索。</p>
      <AssetList assets={selectedAssets} onRemove={removeAsset} disabled={busy}/>
      <AvailableAssets assets={availableAssets} disabled={busy} onSelect={selectExisting}/>
      {error && <div className="resource-error" role="alert">{error}</div>}
      {state.resourcesError && <div className="resource-error" role="alert">{state.resourcesError}</div>}
    </>}
    {id === 'plan' && <>
      <details className="plan-resource-picker" open={!sourceSelected}>
      <summary>{sourceSelected ? `方案：${planLabel || selectedAssets[0]?.filename || '已选择'}` : '选择方案或上传文件'}</summary>
      <div className="plan-source-row">
        <label className="plan-source-field" htmlFor="registered-plan"><span>已登记方案</span>
          <select id="registered-plan" value={assistant.draft.registered_plan_id ?? ''} disabled={busy || state.resourcesLoading}
            onChange={event => dispatch({ type: 'draft', id, patch: { registered_plan_id: event.target.value || null, attachment_ids: [] } })}>
            <option value="">选择 P01 / P02 / R01…</option>
            {state.plans.map(plan => <option value={plan.plan_id} key={plan.plan_id}>{plan.label} · {plan.filename}</option>)}
          </select>
        </label>
        <label className="upload-button">
          <input ref={inputRef} aria-label="选择方案文件" type="file" accept=".pdf,.doc,.docx,.txt,.md,application/pdf,application/msword,text/plain,application/vnd.openxmlformats-officedocument.wordprocessingml.document" disabled={busy || uploading} onChange={handleFiles}/>
          <span>上传方案</span>
        </label>
      </div>
      {planLabel && <div className="resource-selected-plan">当前方案：{planLabel}{selectedPlan?.page_url && <> · <a href={selectedPlan.page_url} target="_blank" rel="noopener noreferrer">查看原页↗</a></>}</div>}
      <AssetList assets={selectedAssets} onRemove={removeAsset} disabled={busy}/>
      <AvailableAssets assets={availableAssets} disabled={busy} onSelect={selectExisting}/>
      </details>
      {error && <div className="resource-error" role="alert">{error}</div>}
      {state.resourcesError && <div className="resource-error" role="alert">{state.resourcesError}</div>}
      <button type="button" className="review-button" disabled={busy || !sourceSelected}
        onClick={() => void send(id, false, { prompt: '生成整份方案核查报告', plan_action: 'review' })}>生成整份方案核查报告</button>
      <details className="plan-settings">
        <summary>方案核查设置<span>{assistant.draft.max_steps} steps · {assistant.draft.max_tool_calls} calls</span></summary>
        <div className="plan-settings-fields">
          <label>最大步骤<input type="number" min={3} max={80} value={assistant.draft.max_steps} disabled={busy}
            onChange={event => dispatch({ type: 'draft', id, patch: { max_steps: Math.min(80, Math.max(3, Number(event.target.value) || 3)) } })}/></label>
          <label>最大工具调用<input type="number" min={1} max={200} value={assistant.draft.max_tool_calls} disabled={busy}
            onChange={event => dispatch({ type: 'draft', id, patch: { max_tool_calls: Math.min(200, Math.max(1, Number(event.target.value) || 1)) } })}/></label>
        </div>
        <p>仅适用于整份方案核查报告；修改只影响下一次报告生成。</p>
      </details>
    </>}
  </section>
}
