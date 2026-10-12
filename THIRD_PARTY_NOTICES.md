# 实现来源与第三方依赖

核心 Python 实现来自作者本地建筑资料 RAG 项目的当前工作代码，按运行时导入依赖清单选取；没有迁移原 Git 历史。对应文件见 `docs/core_manifest.json`。演示适配层及 `examples` 中的资料、方案为本演示版新编写。

原源码保留的来源说明：

- `rag_arch/retrieval.py`：借鉴课程 FlashRAG 检索器的 record/search 分离思路；中文分词、数值词元校验、余弦检索、RRF 和候选重排属于本项目实现。
- `rag_arch/providers.py`：借鉴课程聊天助手的 create_llm 与 RAG embedding factory 组织方式；本项目增加独立配置、服务适配与运行记录。

本仓库未复制 FlashRAG 包、课程资料、教学课件或第三方数据集全文。课程来源说明与原有署名保留；这些说明不构成对第三方内容的重新授权。

## 版权与许可范围

[MIT 许可证](LICENSE) 仅适用于本项目作者拥有相应权利的自有代码。第三方依赖、素材和标识的权利属于各自权利人，仍遵循各自许可。本项目不为第三方的使用方式或合规性提供保证，也不保证所有依赖、组合应用或生成产物均已获得适合商业用途的授权。使用者应根据实际使用、分发和部署方式，自行核对并履行相关许可要求；本声明不替代第三方许可，也不免除其要求。

当前 PyMuPDF 是必需依赖，官方提供 AGPLv3 与商业许可两条路径。其使用条件应查阅 [PyMuPDF 官方许可说明](https://pymupdf.io/licensing)。其他依赖应以各自发布版本的许可文本为准。

## Copyright and licensing scope

The [MIT License](LICENSE) covers only project-authored code for which the authors hold the relevant rights. Third-party dependencies, materials, and marks remain the property of their respective rights holders and subject to their own licenses. This project provides no assurance about third-party use or compliance, and does not guarantee that dependencies, combined applications, or generated outputs are cleared for commercial use. Users must verify and comply with the applicable terms for their intended use, distribution, and deployment. This notice does not replace or waive third-party license requirements.

PyMuPDF is currently a required dependency, offered under AGPLv3 or a commercial license; see its [official licensing page](https://pymupdf.io/licensing). Consult the license text of each installed version for other dependencies.
