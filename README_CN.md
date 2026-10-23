# RAG Arch Demo｜React + FastAPI 建筑资料助手

[English](README.md) | [简体中文](README_CN.md)

本仓库展示建筑资料助手当前的 React + FastAPI 实现。React 前端、FastAPI API、PostgreSQL 任务库和 AI worker 分开运行。代码展示账号内任务如何经过资料处理、混合检索、证据引用、方案核查、可续传进度事件和报告导出。

公开目录包含应用代码与自编虚构样例，不含完整规范库、来源原件、API 凭据、数据库或生成后的语料/向量包。使用者自行准备 PostgreSQL、模型配置、语料，以及需要查看来源页时所需的原件。Web 工作台用于展示工程实现，不是克隆即可运行的托管服务。请看[启动说明](docs/web_setup.md)。

## 当前 Web 架构

在同一部署中，每个账号的任务、附件、历史和审计记录彼此隔离。PostgreSQL 持久保存认证、任务和事件状态；独立 API 进程接收请求，AI worker 领取并执行持久任务。客户端断线后可以携带游标重新订阅任务事件。数据库约束每个账号的助手任务槽和全站执行上限。

Worker 保留检索和 Agent 链路。稠密向量、BM25 索引和语料记录可通过只读内存映射包共享，完整语料记录按需解码。模型、搜索、嵌入和重排等网络编排在 worker 中异步执行；本地计算和文件处理由受限的进程资源管理。[架构说明](docs/architecture.md)将这些链路映射到源码。

## 启动本地工作台

按[启动步骤](docs/web_setup.md)准备 Python 与前端依赖，显式执行 PostgreSQL 数据库迁移、创建账号，并分别启动 API、worker 和 React 开发服务器。真实资料流程需要使用者提供模型配置和资料。

自编离线 CLI 仍提供一个小型确定性样例：

~~~~bash
python -m demo --output output
~~~~

结果写入 output/ 下的 Markdown 和 JSON 文件。替身仅演示自编样例，不核查任意方案。

## 贡献与公开范围

我主导完成建筑资料助手的实现，并有权公开本仓库选取的代码。源码包含资料读取与来源定位、BM25 和向量检索、RRF 融合、重排、证据工具、Agent 工具调用、方案引句核验、引用校验和报告生成。课程借鉴说明与第三方组件信息保留在[来源说明](THIRD_PARTY_NOTICES.md)。

公开样例资料为自编虚构内容。完整内部语料、客户材料、运行状态、私有评测输入，以及基于这些资料生成的派生产物均未发布。公开的历史评测数字汇总在[效果说明](docs/evaluation.md)；原始输入和报告不公开。

## 较早版本的 React 界面示意

以下截图于 2026-10-13 使用虚构离线样例拍摄，展示的是较早版本的 React 界面，不包含当前账号、PostgreSQL 任务库或 worker 链路。截图不代表真实模型服务或生产部署已验证。

![较早版本工作台示意](docs/screenshots/react-01-workbench.png)
![较早版本证据页与自编离线结果](docs/screenshots/react-02-evidence.png)
![较早版本方案核查页与自编离线结果](docs/screenshots/react-03-plan-review.png)

## 仓库导航

- frontend/：React 应用。
- rag_arch/web/：FastAPI API、认证、持久化、任务、事件、上传与下载。
- rag_arch/：资料处理、检索、证据、Agent、方案核查和报告。
- scripts/：显式数据库迁移及本地语料/索引准备。
- demo/、examples/：自编离线 CLI 样例和虚构输入。
- docs/：架构、启动、效果边界、发布范围、历史界面截图和验证记录。

建议先读[架构说明](docs/architecture.md)、[启动说明](docs/web_setup.md)和[发布说明](docs/publication.md)。2026-10-12 和 2026-10-13 的验证文档仅记录较早版本的实现。本次迁移的验证记录见[这里](docs/web_validation_20261023.md)，将在最终测试和审阅证据确定后补全。

## 许可证

本项目自有代码采用 [MIT 许可](LICENSE)。第三方依赖遵循各自许可；不保证依赖及产物满足所有商业用途，使用者应自行核对相关授权。组件与来源见[第三方说明](THIRD_PARTY_NOTICES.md)。
