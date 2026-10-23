# RAG Arch Demo｜建筑资料助手

[English](README.md) | [简体中文](README_CN.md)

RAG Arch Demo 是建筑资料助手，可依据来源证据回答问题，并核查方案中的引句、生成问题清单报告。本仓库公开其中一部分工程实现，便于阅读和交流。

## 核心流程与架构

- 处理使用者提供的 PDF 和 DOCX 文件，并保留来源位置。
- 结合 BM25 与稠密检索，通过 RRF 融合候选，再进行重排。
- 基于证据和引用回答问题，核验方案引句，并生成问题清单报告。

默认 Web 工作台采用 React、FastAPI、PostgreSQL 和独立 AI worker，包含账号内任务隔离、可续传任务事件和共享只读检索包。可从[架构与代码阅读路线](docs/zh-CN/architecture.md)了解模块关系。

## 证据页面示意

![较早版 React 界面的证据结果](docs/screenshots/react-02-evidence.png)

截图于 2026-10-13 使用自编虚构资料和离线样例拍摄。画面来自较早版本，仅作界面示意，不验证当前多账号行为或在线模型服务。其他历史截图见[文档导航](docs/zh-CN/README.md)。

## Web 工作台

Web 部署需要使用者自行准备 PostgreSQL、模型凭据和语料。安装和启动步骤见[启动说明](docs/zh-CN/setup.md)。

## 离线样例

需要 Python 3.12 或更新版本。自编 CLI 使用确定性替身和虚构输入，不会核查任意方案。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m demo --output output
```

Linux/macOS 可用 `source .venv/bin/activate` 激活环境。Markdown 和 JSON 结果写入 `output/`。

## 更多文档

[文档导航](docs/zh-CN/README.md)（[English](docs/README.md)）集中列出架构、启动、评测、发布范围、截图和验证记录。贡献前请看[贡献指南](.github/CONTRIBUTING.md)。

## 许可证

项目自有代码采用 [MIT 许可](LICENSE)。第三方组件遵循各自条款，详见[第三方说明](docs/third_party_notices.md)。
