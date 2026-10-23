# 架构与代码阅读路线

[English](architecture_en.md) | [简体中文](architecture.md) | [Web 启动说明](web_setup.md)

当前公开实现将 React 前端、FastAPI API、PostgreSQL 状态库和 AI worker 分开运行。使用者在一个部署中创建多个账号，各账号数据按所有权隔离。仓库不包含内部语料、来源原件、数据库、模型密钥或生成后的索引包。

~~~~mermaid
flowchart LR
    U[浏览器 / React] --> G[HTTPS 网关或 Vite 开发服务器]
    G --> A[FastAPI API 进程]
    A --> D[(PostgreSQL 用户、会话、任务、事件、额度、审计)]
    D --> W[独立 AI worker]
    W --> M[异步模型、搜索、嵌入、重排调用]
    W --> L[受限本地处理与受监督子进程]
    U --> E[可续传任务事件流]
    A --> E
    W --> E
    C[使用者准备的语料和原件] --> W
~~~~

## 请求与任务生命周期

FastAPI 负责认证、账号内 API 请求、任务提交、上传下载边界和事件流。持久 AI 任务不会在请求进程中执行。PostgreSQL 保存任务状态和有序事件；一个或多个 worker 领取排队任务，并持久记录进度和终态。客户端可携带事件游标重新连接并续传。每账号助手任务槽和全站执行上限由共享数据库状态约束，因此增加 API 或 worker 进程不会绕过额度。

每个账号的任务、附件、历史和审计记录按账号隔离。部署运维者仍管理数据库和主机；本项目不提供托管式多租户服务或身份管理平台。

Worker 使用项目内的应用工作流与检索代码。聊天、网页搜索、嵌入和重排等网络请求采用异步编排。同步 CPU、文件和资料转换工作由有界本地执行资源或受监督子进程处理。离线 CLI 仍是独立的同步样例入口。

## 共享检索数据

检索链路组合 BM25 与稠密向量候选，通过 RRF 融合后按配置重排并展开证据。公共语料记录、BM25 统计和稠密向量支持不可变只读内存映射包。多个进程可共享操作系统文件页，避免各自构建完整的 Python 内存副本；完整语料记录与定位信息按需解码。检索行为保持可读，改变的是共享数据的加载方式。

公开代码包含 mmap 实现，但不包含语料和派生包。使用者按项目格式准备自有数据，再基于自己的语料和兼容的向量缓存构建索引包。源数据更新后，应构建匹配的语料与索引包，再重启 worker。公开命令和资料边界见[启动说明](web_setup.md)。

## 源码阅读位置

| 层次 | 主要文件 | 阅读重点 |
|---|---|---|
| React 前端 | frontend/src/ | 登录边界、工作区状态、任务请求、事件续传、结果和下载 |
| API 与认证 | rag_arch/web/app.py、auth.py、contracts.py | API-only 启动、会话、授权、请求校验、健康与就绪检查 |
| PostgreSQL 状态 | rag_arch/web/db.py、task_repository.py、persistent_tasks.py | 显式数据库迁移、按所有权查询、持久任务状态流转 |
| 事件与 worker | rag_arch/web/events.py、task_worker.py、worker_process.py | 有序事件、游标续传、租约、执行和恢复 |
| 附件与资料 | rag_arch/web/documents.py、document_routes.py、standards_resources.py | 账号附件归属与只读共享语料资源 |
| 异步 Provider 边界 | rag_arch/async_support.py、providers.py、web_search.py | 异步网络调用与有界本地工作 |
| 检索与证据 | rag_arch/retrieval.py、binary_bm25.py、binary_vectors.py、binary_corpus.py、evidence_index.py | 混合候选、mmap 包、证据查询和按需解码 |
| Agent 与报告 | rag_arch/standard_tools.py、graph_agent.py、plan_review_agent.py、plan_output.py | 证据工具链、方案引句核验、引用检查和报告输出 |
| 独立 Web 前端 | frontend/ | React 开发服务器与生产构建 |

## 资料和运行边界

可工作的部署需要 PostgreSQL、运维者提供的模型设置、项目格式的语料，以及查看来源页所需的原件。使用者可准备 data/corpus/clauses.jsonl、data/corpus/tables.jsonl，并按工作流需要添加目录、状态和来源映射。API 凭据和运行文件保存在本地。

README 中的截图于 2026-10-13 使用虚构离线结果拍摄，展示较早版本的 React 界面，不包含当前账号隔离、PostgreSQL 持久任务或 worker 生命周期。它们只作界面示意，不是部署验证证据。
