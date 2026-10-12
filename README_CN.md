# RAG Arch Demo · 可追溯的资料问答与方案核查

[English](README.md) | [简体中文](README_CN.md)

把“查到资料”到“报告中引用了什么”的过程留在代码和输出里：读取资料与方案，执行 BM25＋向量检索、RRF 融合和重排，通过工具取得证据，校验方案引句，再输出带来源的报告。

这是建筑资料助手的独立公开演示版，面向代码阅读、面试交流和小样例运行。核心链路来自实际项目；全新目录和首次 Git 提交，不携带原历史。完整规范库、真实方案和运行日志不在本仓库。

## 先跑一次

运行条件：Python 3.12，Windows/macOS/Linux，浏览器。默认离线模式无需密钥、无需下载模型、不会请求外部模型服务。

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m demo --output output
python -m streamlit run app.py --server.address 127.0.0.1
```

命令行输出 `output/report.md`、`output/standard_answer.json` 和 `output/plan_review.json`。页面支持资料查看、资料问答、方案核查及报告下载。提交中的 [样例报告](examples/sample_report.md) 是自编输入的离线演示输出。

```bash
python -m demo --mode offline --standards examples/standards.json --plan examples/demo_plan.json --output output
python -m unittest discover -s tests -v
python scripts/verify_public_tree.py
```

离线向量、重排与聊天均为确定性替身，实际调用保留的检索器、证据工具和 Agent 图循环。它用于展示数据与调用如何流动，只覆盖样例设计的主题，不能审查任意工程方案。所有 `DEMO` 资料和数值均为自编虚构，不是工程规范。

## 在线运行

复制 `.env.example` 为 `.env`，用自己的服务地址、模型名和 API 密钥替换占位符。聊天模型须支持工具调用；向量服务须兼容原 OpenAIEmbeddings 适配器；云重排使用保留的百炼适配器。模型选择与参数上限由你使用的服务决定。

```bash
python -m demo --mode live --output output
```

在线模式需要网络与对应服务权限，可能产生费用。它使用同一自编资料，不读取原项目配置。未经实际模型调用验证的在线配置不能当作已完成的服务部署。页面的在线开关也须主动选择。

## 核心实现与贡献范围

我主导完成建筑资料助手的实现，并有权公开本版选取的代码。本仓库重点展示我负责的资料读取与来源定位、混合检索与重排、证据工具、工具循环、长方案上下文、引用核验和报告交付；检索器与客户端借鉴的课程设计在原注释和来源说明中保留。

[架构与阅读路线](docs/architecture.md) 将每一环节映射到源码。适合沿下面顺序追问：

1. `loaders.py` / `splitting.py`：文字怎么提取，位置怎么保留，扫描页缺口怎么提示。
2. `retrieval.py` / `evidence_index.py`：中文和数字怎么分词，BM25/余弦/RRF 的分数区别，候选池如何进入重排。
3. `standard_tools.py` / `graph_agent.py`：模型取得了哪些证据，引用 ID 如何限制在本轮取得范围。
4. `plan_review_agent.py` / `plan_review_evidence.py`：全文、分批和定稿证据如何连接，方案引句如何核验。
5. `plan_review_report.py` / `model_input_view.py`：如何交付来源，如何减少重复元数据而保留正文和回查能力。

`demo/` 提供样例与运行适配；`rag_arch/` 保留业务代码，文件来源见 [源码清单](docs/core_manifest.json)。保留课程借鉴注释，第三方依赖与授权边界见 [来源说明](THIRD_PARTY_NOTICES.md)。本版未增加多用户、权限系统或大型框架。

## 演示截图

截图均来自本版自编资料与离线运行，没有真实资料或客户信息。

![自编资料与演示入口](docs/screenshots/01-library.png)
![检索与证据引用](docs/screenshots/02-qa.png)
![方案核查报告](docs/screenshots/03-review.png)

## 现有效果与限制

历史 60 题检索回归中，同一 RRF 候选池加入重排后，Recall@8 为 0.8444→0.8722，完整证据为 44/60→47/60；Top-4 完整证据同时由 40/60 降至 39/60。固定 240 题 AECBench 问答中，可解码选项正确率为 146/240→184/240。

五方案整案评测采用 59 个固定开发参考项，结果是参考覆盖，不是全报告准确率；未证明 RAG 稳定提高整案技术发现。范围、原生复核对照、不利结果和不能归因的变化见 [效果说明](docs/evaluation.md)。公开版不携带原始评测输入，也不宣称小 DEMO 复现这些成绩。

引用校验能检查来源与引句，不能自动证明工程适用性、结论正确或缺陷发现完整。文字层、图表和复杂计算的缺口需要人工核对。

## 目录

```text
app.py                  Streamlit 演示页面
rag_arch/               保留的业务实现
demo/                   离线/在线运行适配
examples/               自编资料、方案与样例报告
docs/                   架构、效果、公开范围、截图
scripts/                公开文件检查
tests/                  有限离线链路验证
.env.example            仅占位符
requirements.txt        已验证的直接依赖版本
```

[首次发布验证](docs/validation.md) 记录安装、离线测试与界面检查结果。[发布范围与复查](docs/publication.md) 说明清理边界。`.env`、输出、日志、向量缓存、附件和压缩包默认忽略；将来新增素材仍需复查。

## 许可证

本项目代码采用 [MIT 许可证](LICENSE)。第三方依赖保留各自许可证，课程来源说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
