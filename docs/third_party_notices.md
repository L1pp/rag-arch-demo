# Sources and Third-Party Dependencies / 来源与第三方依赖

The core code was selected from the author's building-document RAG project. The demo adapter and fictional samples were written for this release. File origins and later adaptations are recorded in [the source manifest](core_manifest.json).

## Source acknowledgements / 来源说明

- `rag_arch/retrieval.py`: retains the record/search design from the course FlashRAG retriever. Chinese tokenization, numeric-token checks, cosine retrieval, RRF, and reranking are project adaptations.
- `rag_arch/providers.py`: draws on the course's `create_llm` and RAG embedding factory, with separate provider configuration and project runtime handling.

核心代码选自作者的建筑资料 RAG 项目，演示适配与虚构素材为本版编写。上述检索器和客户端保留课程设计借鉴说明；未收录 FlashRAG 包、课程课件或第三方数据集全文。文件来源与后续适配见[源码清单](core_manifest.json)。

## Third-party components / 第三方组件

Original project code uses [MIT](../LICENSE). Third-party components retain their own licenses; consult the terms for the versions you use.

- **PyMuPDF**: required PDF dependency, offered under AGPLv3 or a commercial license. [Official licensing](https://pymupdf.io/licensing).
- **Other dependencies**: React, FastAPI, Uvicorn, NumPy, HTTPX, python-dotenv, LangChain, LangGraph, Pydantic, and Pillow. Python versions are pinned in [requirements/web.txt](../requirements/web.txt); frontend versions are recorded in [package.json](../frontend/package.json) and [package-lock.json](../frontend/package-lock.json). Their upstream license texts remain applicable.

自有代码采用 MIT，第三方组件遵循各自许可。PyMuPDF 为必需的 PDF 依赖，提供 AGPLv3 或商业许可；使用者可按上述官方链接及各依赖版本的许可文本核对授权。
