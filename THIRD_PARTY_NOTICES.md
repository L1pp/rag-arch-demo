# 实现来源与第三方依赖

核心 Python 实现来自作者本地建筑资料 RAG 项目的当前工作代码，按运行时导入依赖清单选取；没有迁移原 Git 历史。对应文件见 `docs/core_manifest.json`。演示适配层及 `examples` 中的资料、方案为本演示版新编写。

原源码保留的来源说明：

- `rag_arch/retrieval.py`：借鉴课程 FlashRAG 检索器的 record/search 分离思路；中文分词、数值词元校验、余弦检索、RRF 和候选重排属于本项目实现。
- `rag_arch/providers.py`：借鉴课程聊天助手的 create_llm 与 RAG embedding factory 组织方式；本项目增加独立配置、服务适配与运行记录。

本仓库未复制 FlashRAG 包、课程资料、教学课件或第三方数据集全文。源码注释中的课程来源不能作为对第三方代码的重新授权。第三方库遵循各自许可证：NumPy、HTTPX、python-dotenv、LangChain、LangGraph、Streamlit 等。PDF 读取及当前模块导入链使用 PyMuPDF，其许可条件应按使用方式检查；请勿将本演示的发布解释为对这些依赖的重新授权。

本项目代码采用 MIT 许可证，完整条款见 `LICENSE`。第三方依赖继续遵循各自许可证，课程来源说明与原有署名保留。
