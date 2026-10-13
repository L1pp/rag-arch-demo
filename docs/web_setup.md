# React + FastAPI 本地启动说明

[English](web_setup_en.md) | [简体中文](web_setup.md) | [架构说明](architecture.md)

本说明启动仓库中的 React/FastAPI 主工作台。公开仓库没有提供完整规范库、真实原件、API 密钥或运行数据库，所以这里给出的是自备资源后的本地启动路径，不承诺克隆后即可完成真实资料问答或方案核查。

## 运行条件

- Python 3.12。
- Node.js 22.12 或更高版本；锁文件也允许 Node 20.19 及以上的 20.x 版本。
- 本机浏览器。
- 你自行提供的模型配置和规范资料。需要向量检索或重排时，还需准备对应服务和可用的向量数据。

## 安装依赖并构建前端

在仓库根目录执行：

~~~~powershell
python -m venv .venv-web
.\.venv-web\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
Push-Location frontend
npm ci
npm run build
Pop-Location
~~~~

macOS/Linux 激活虚拟环境的命令是 source .venv-web/bin/activate。构建完成后，FastAPI 从 frontend/dist/ 提供 React 页面和静态资源。

## 配置模型服务

复制配置模板并填写自己的服务地址、模型和密钥：

~~~~powershell
Copy-Item .env.example .env
~~~~

编辑 .env，至少配置要使用的聊天服务。聊天模型需要支持工具调用。若启用向量检索和重排，再分别填写 Embedding 与 Reranker 配置。地址、模型能力和参数上限由你选择的服务决定；本仓库不附带可用密钥，也不代表线上服务已验证。

不要把真实的 .env、密钥或服务日志提交到 Git。

## 准备自己的规范资料

资料问答至少需要项目格式的两个文件：

- data/corpus/clauses.jsonl
- data/corpus/tables.jsonl

字段和加载规则可从 rag_arch/standards.py 与 rag_arch/evidence_index.py 查看。按需添加目录摘要、目录记录、版本关系和状态记录。需要从证据跳转到原件页时，还要准备相应原件和代码要求的来源映射；没有原件的条目不能打开原页。相关加载与位置处理见 rag_arch/web/standards_resources.py、rag_arch/page_sources.py 和 rag_arch/standard_status.py。

如果使用历史/准备情况页面，还需按项目格式提供 data/meta/history/active_preparation.json。输入方案可以通过网页上传。认证库、附件、历史数据、索引和缓存写入本地数据目录，由使用者自行维护；请勿提交包含私人资料的运行目录。

DOCX 正文提取不要求 Word。原页分页预览使用 Windows 上已安装的 Microsoft Word 和 `scripts/render_docx_pages.ps1`；本次未验证这一可选流程。

## 创建本地账号并启动

先为当前机器创建登录账号。命令会交互输入两次密码，不会把密码写进命令历史：

~~~~powershell
python -m rag_arch.web.auth_cli add YOUR_USERNAME
~~~~

本地使用 HTTP 时，在启动后端的同一终端显式关闭 Secure Cookie。该变量由进程环境读取，不是 .env 中的模型设置：

~~~~powershell
$env:RAG_ARCH_AUTH_COOKIE_SECURE = 'false'
python -m uvicorn rag_arch.web.app:app --host 127.0.0.1 --port 8000 --workers 1
~~~~

打开 http://127.0.0.1:8000 并登录。Windows 也可在依赖、前端构建和环境变量准备好后运行 .\start_web.ps1 -Port 8000。

认证只控制本地登录。账号共享同一工作区和运行数据，当前代码不提供账号间的数据隔离；示例只绑定环回地址供本机使用。

## 保留的样例入口

无需真实资料库即可先看自编离线 CLI 路径：

~~~~powershell
python -m demo --output output
~~~~

独立的 Streamlit 无密钥样例需额外安装 requirements-demo.txt，启动命令见[README](../README.md#separate-sample-paths)。两种样例都使用确定性替身，范围仅限自编输入。
