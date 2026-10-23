# React + FastAPI 本地启动说明

[English](../setup.md) | [简体中文](setup.md) | [架构说明](architecture.md)

公开工作台由 React 前端、FastAPI API、PostgreSQL 数据库和独立 AI worker 组成。它用于展示源码，不是打包好的托管服务。使用者自行准备 PostgreSQL、模型密钥、语料，以及需要打开来源页时用到的原件。

## 运行条件

- Python 3.12。
- Node.js 22.12 或更高版本；前端锁文件也支持 Node 20.19 及以上的 20.x 版本。
- API、worker、数据库迁移和账号管理进程均可连接的 PostgreSQL。
- 运维者提供的模型设置和语料文件。
- 本机浏览器。

项目脚本不会创建 PostgreSQL 服务端或导入语料。数据库凭据和模型密钥不要提交到 Git 或日志。

## 安装 Python 与前端依赖

在仓库根目录执行：

~~~~powershell
python -m venv .venv-web
.\.venv-web\Scripts\Activate.ps1
python -m pip install -r requirements/web.txt
python -m pip install --no-deps -e .
Push-Location frontend
npm ci
Pop-Location
~~~~

Linux/macOS 可使用 source .venv-web/bin/activate 激活环境。Python 包、API 和 worker 共用该环境。

## 配置 PostgreSQL 并执行迁移

在迁移、账号管理、API 和 worker 所用进程环境中配置数据库 URL 与 schema。RAG_ARCH_DATABASE_* 变量不从模型设置文件 .env 加载。

~~~~powershell
$env:RAG_ARCH_DATABASE_URL = 'postgresql://<应用用户>:<密码>@<主机>:5432/<数据库>'
$env:RAG_ARCH_DATABASE_SCHEMA = 'rag_arch'
python -m scripts.migrate_web schema
~~~~

先准备数据库和用户，再显式执行迁移。启动 API 不会自动建表，也不会迁移旧数据。启动 API 或 worker 前，应应用所有待迁移版本。不要把真实连接串写入提交文件。

本地开发实例完成迁移后，可添加账号：

~~~~powershell
python -m rag_arch.web.auth_cli add YOUR_USERNAME
~~~~

命令会要求输入两次密码。在新开的账号管理、API 或 worker 终端中，也要设置相同的数据库环境变量。

## 提供模型设置和语料

将 .env.example 复制为 .env，填写需要使用的服务地址、模型和 API 密钥。聊天模型须支持工作流所用的工具；使用向量检索或重排功能时，也要提供相应服务。各服务地址和能力取决于使用者选择；仓库不含可用凭据，也不声称线上服务已验证。

规范问答需要项目格式的 data/corpus/clauses.jsonl 与 data/corpus/tables.jsonl。按需添加目录、状态、版本及来源映射。只有需要打开来源页的流程才需要原件。方案文件可通过网页上传。请勿把私人资料或运行文件提交到 Git。

代码支持为公共语料、BM25 索引和稠密向量准备只读 mmap 包。语料较大时，可用自有数据构建并部署配套的包：

~~~~powershell
python -m scripts.build_core_corpus_mmap
python -m scripts.build_core_bm25
~~~~

向量转换命令只转换已有且兼容的向量缓存，不会创建向量或调用模型：

~~~~powershell
python -m scripts.convert_core_vectors
~~~~

API 与每个 worker 应使用同一份源语料及匹配版本的索引包。源语料更新后，需重建派生包并重启 worker 载入。完整内部数据及对应资源包不在本仓库中。

## 启动 API、worker 与 React 前端

从仓库根目录打开三个终端，并在每个终端设置数据库 URL 与 schema。本地 HTTP 开发时，在 API 终端配置页面来源与 Cookie 设置，再启动 API：

~~~~powershell
$env:RAG_ARCH_DATABASE_URL = 'postgresql://<应用用户>:<密码>@<主机>:5432/<数据库>'
$env:RAG_ARCH_DATABASE_SCHEMA = 'rag_arch'
$env:RAG_ARCH_AUTH_COOKIE_SECURE = 'false'
$env:RAG_ARCH_FRONTEND_ORIGINS = 'http://127.0.0.1:5173'
.\scripts\start_web.ps1
~~~~

在第二个终端设置相同的数据库变量并启动 worker：

~~~~powershell
.\scripts\start_worker.ps1
~~~~

在第三个终端启动前端开发服务器：

~~~~powershell
Push-Location frontend
npm run dev
~~~~

打开 http://127.0.0.1:5173 并登录。API 监听 127.0.0.1:8000，Vite 开发服务器将 API 请求转发给它。任务状态与事件保存在数据库中；API 与 worker 必须使用相同的 schema。生产环境应构建前端，并通过 HTTPS 网关统一提供前端和 API；部署前需审阅 Cookie、页面来源、数据库角色、worker 和数据目录设置。

## 离线 CLI 样例

自编 CLI 可在不启动 Web 数据库、不提供在线 API 凭据或私有语料时运行小型离线样例：

~~~~powershell
python -m demo --output output
~~~~

它会在 output/ 下写入 Markdown 和 JSON。确定性替身只覆盖自编虚构样例，不是通用方案审查服务。
