# React + FastAPI 本地启动说明

[English](../setup.md) | [简体中文](setup.md) | [架构说明](architecture.md)

公开工作台是持续开发中的实现快照，展示 React、FastAPI、PostgreSQL 和 worker 的主要工程实现，并非完整交付。使用者需自行准备 PostgreSQL、模型密钥、语料及所需原件，并完成相应的环境集成。

## 运行条件

- Python 3.12。
- Node.js 22.12 或更高版本；前端锁文件也支持 Node 20.19 及以上的 20.x 版本。
- API、worker、数据库迁移和账号管理进程均可连接的 PostgreSQL。
- 运维者提供的模型设置和语料文件。
- 本机浏览器。
- Linux 下处理旧 DOC 或预览 Word 页面需要 LibreOffice Writer：将 `soffice` 加入 `PATH`，或在进程环境中配置 `RAG_ARCH_LIBREOFFICE_EXECUTABLE`。Windows 下使用 DOCX，页面预览需要 Microsoft Word。

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

Linux 下可在 POSIX shell 中安装同一套环境与前端依赖：

~~~~sh
python -m venv .venv-web
source .venv-web/bin/activate
python -m pip install -r requirements/web.txt
python -m pip install --no-deps -e .
cd frontend
npm ci
cd ..
~~~~

Python 包、API 和 worker 共用该环境。

## 配置 PostgreSQL 并执行迁移

在迁移、账号管理、API 和 worker 所用进程环境中配置数据库 URL 与 schema。RAG_ARCH_DATABASE_* 变量不从模型设置文件 .env 加载。

~~~~powershell
$env:RAG_ARCH_DATABASE_URL = 'postgresql://<应用用户>:<密码>@<主机>:5432/<数据库>'
$env:RAG_ARCH_DATABASE_SCHEMA = 'rag_arch'
python -m scripts.migrate_web schema
~~~~

Linux 下使用 `export` 配置相同的变量，再执行迁移：

~~~~sh
export RAG_ARCH_DATABASE_URL='postgresql://<应用用户>:<密码>@<主机>:5432/<数据库>'
export RAG_ARCH_DATABASE_SCHEMA='rag_arch'
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

API 与每个 worker 应使用同一份源语料及匹配版本的索引包。打开语料 mmap 包时不会检查原始源目录是否存在或仍与包匹配，结构有效的旧包仍可能加载；运维者需管理匹配版本。源语料更新后，需重建匹配的语料、BM25 和向量包并重启 worker 载入。完整内部数据及对应资源包不在本仓库中。

## 可选的页面 OCR 与图文核查

云 OCR 默认关闭。启用时，在 worker 进程环境或本地 `.env` 中设置 `RAG_ARCH_OCR_ENABLED=true`、`RAG_OCR_BASE_URL` 和 `RAG_OCR_API_KEY`。当前适配器固定使用 `qwen-vl-ocr-2025-11-20`，只接受阿里云北京普通按量 HTTPS 服务，不使用 Token Plan。OCR 地址或密钥留空时会回退到聊天地址或密钥，它们仍须符合 OCR 服务要求。`RAG_ARCH_OCR_TIMEOUT_SECONDS` 控制请求超时，默认 120 秒，范围 1–300 秒。

只有已领取的 worker 任务可向 OCR 提交候选页。`RAG_ARCH_OCR_MAX_PAGES_PER_TASK` 从 worker 进程环境读取，默认 100，范围 1–500。识别状态始终标记为未核验（`recognition_verified=false`），页面会展示提示。启用后页图会发送到配置的服务，可能产生服务费用。

图文方案核查可在图像预算内将整页或局部区域提供给支持视觉输入的模型。结果记录图片获取、投递、返回和引用状态，但不能保证视觉识读正确。`deepseek-v4.1-flash-1017` 快照使用专门的 Responses 参数适配；使用者需自行配置兼容服务及凭据。

OCR 缓存也保留失败结果；同一资产不会在配置修改后自动重新识别。需要重新尝试时，先核对已发请求的状态、修正配置，待任务安全结束后重启相关 worker，再重新上传为新资产并手动选择它提交；这是显式的新请求，可能产生费用。不要通过自动删除未知状态缓存来重发。

## 启动 API、worker 与 React 前端

在 Windows 上从仓库根目录打开三个 PowerShell 终端，并在每个终端设置数据库 URL 与 schema。本地 HTTP 开发时，在 API 终端配置页面来源与 Cookie 设置，再启动 API：

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

Linux 下在每个终端激活 `.venv-web`，并使用 `export` 设置相同的数据库 URL 与 schema；API 终端还要按上述含义设置页面来源与 Cookie 环境变量。分别在两个终端启动 API 和 worker：

~~~~sh
python -m uvicorn rag_arch.web.app:app --host 127.0.0.1 --port 8000 --workers 1 --proxy-headers --forwarded-allow-ips 127.0.0.1 --timeout-graceful-shutdown 5
python -m rag_arch.web.worker_process run --concurrency 3
~~~~

第三个终端进入 `frontend/`，执行 `npm run dev`。打开 http://127.0.0.1:5173 并登录。API 监听 127.0.0.1:8000，Vite 开发服务器将 API 请求转发给它。任务状态与事件保存在数据库中；API 与 worker 必须使用相同的 schema。生产环境应构建前端，并通过 HTTPS 网关统一提供前端和 API；部署前需审阅 Cookie、页面来源、数据库角色、worker 和数据目录设置。

## 恢复中断任务

本机 worker 已退出时，运维者可在该 worker 所在主机配置好数据库环境，然后执行一次维护命令：

~~~~sh
python -m rag_arch.web.worker_process recover --isolate-interrupted
~~~~

`--cleanup-owned-resources` 仅允许清理部分身份已核实的子进程；LibreOffice 进程组只核验退出，不由该命令终止。无法确认退出时继续保留未知状态和任务槽。

默认冷却期为 120 秒。只有确认本机 worker 和已登记的子进程资源退出后，操作才会隔离中断任务；不会自动重试，外部工作状态仍为未知。

## 离线 CLI 样例

自编 CLI 可在不启动 Web 数据库、不提供在线 API 凭据或私有语料时运行小型离线样例：

~~~~powershell
python -m demo --output output
~~~~

它会在 output/ 下写入 Markdown 和 JSON。确定性替身只覆盖自编虚构样例，不是通用方案审查服务。
