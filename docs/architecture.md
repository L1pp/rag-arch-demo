# 架构与代码阅读路线

[English](architecture_en.md) | [简体中文](architecture.md) | [Web 启动说明](web_setup.md)

本仓库默认展示 React + FastAPI 主工作台的代码结构。完整规范库、原件、运行数据库和模型密钥不在仓库内，页面源码可阅读，但主工作台的真实资料问答与原页查看需要使用者自行准备兼容资源。

~~~~mermaid
flowchart LR
    U[浏览器 / React] --> A[FastAPI 页面与认证]
    A --> B[工作区、任务、SSE、取消、下载 API]
    B --> C[TaskService / 助手执行器]
    C --> D[标准问答、资料问答、方案核查 runners]
    D --> E[BM25 + 向量检索]
    E --> F[RRF 融合 / 重排]
    F --> G[证据工具与 Agent]
    G --> H[来源与方案引句核验]
    H --> I[Markdown / JSON 报告]
    J[使用者准备的语料、元数据、原件] --> D
~~~~

| 链路 | 主要源码 | 阅读重点 |
|---|---|---|
| React 页面 | frontend/src/App.tsx、api.ts、store.ts、WorkspaceContext.tsx | 工作区状态、助手任务、事件订阅、报告显示和调用边界 |
| FastAPI 入口 | rag_arch/web/app.py、contracts.py | 页面与静态资源、登录、工作区和任务 API、SSE、取消与下载 |
| 任务与事件 | rag_arch/web/task_service.py | 单进程任务生命周期、进度事件、取消和当前运行快照 |
| 上传与资料页面 | rag_arch/web/document_routes.py、documents.py、standards_routes.py | 上传文件管理、来源页查看、资料目录和状态接口 |
| 历史记录 | rag_arch/web/history.py | 本地历史页面与记录读取接口 |
| 后端工作流 | rag_arch/web/chat_runner.py、standards_runner.py、plan_runner.py | 请求如何转入聊天、资料问答和方案核查链路 |
| 文档处理 | rag_arch/loaders.py、splitting.py、plan_documents.py | TXT/MD/PDF/DOCX 读取、行号/页码、扫描页缺口和切分 |
| 检索与重排 | rag_arch/retrieval.py、evidence_index.py、reranking.py、providers.py | 中文与数字分词、BM25、向量归一化、RRF 候选池、重排服务 |
| 证据与问答 | rag_arch/standard_tools.py、graph_agent.py、page_qa.py | 搜索和按条号读取证据、LangGraph 工具循环、最终引用约束 |
| 方案核查 | rag_arch/plan_review_agent.py、plan_review_context.py、plan_review_evidence.py | 全文优先、分批处理、方案引句定位、证据组装和覆盖 |
| 核验与输出 | rag_arch/plan_output.py、plan_review_report.py、model_input_view.py | 引用与工程判断区分、来源回查、Markdown/JSON 报告 |

FastAPI 启动时读取配置并建立任务服务，通过同一来源提供已构建的 React 静态文件和 API。前端打包与启动方式见[Web 启动说明](web_setup.md)。代码保留任务状态、SSE 进度、取消和文件下载的接口；这是本地工作台，不是托管服务或多租户产品。

运行资料问答至少需要使用者准备项目格式的 data/corpus/clauses.jsonl 与 data/corpus/tables.jsonl。目录摘要、目录记录、版本关系、状态记录及 PDF/DOCX 等原件按需要放入相应的数据目录。模型 API、嵌入服务和重排服务也由使用者配置。各项资源不随本仓库发布，路径与最小准备步骤见[Web 启动说明](web_setup.md)。

demo/ 与 examples/ 仍提供自编的离线 CLI 小样例；Streamlit 小页面现作为独立无密钥样例运行。离线替身只覆盖样例设计的主题，不能据此声称完成任意工程方案审查。现有截图记录的是首次发布的 Streamlit 页面，不是当前 React 工作台。
