# 2026-10-13 React/FastAPI 源码迁移验证

[English](web_validation_20261013_en.md) | [简体中文](web_validation_20261013.md)

历史范围说明（2026-10-23 更新）：本页记录第一阶段 React/FastAPI 源码迁移。当时保留了独立 Streamlit 样例，Web 工作台也还是较早的单进程结构。Streamlit 现已退役并移除；以下测试不验证当前 PostgreSQL 多用户架构。请看[本次迁移验证记录](web_validation_20261023.md)。

当时默认入口已改为 React/FastAPI 工作台；streamlit_demo.py、离线 CLI 和自编样例作为独立入口保留。没有引入原 Git 历史或私有语料。

- 新建独立 Python 3.12.10 Web 环境安装主依赖及 editable 包；为样例测试添加可选 Streamlit 依赖，pip check 通过。
- 从公开后端契约生成前端类型。8 组、47 项前端测试通过：5 组首次通过，修正 JSX 闭合后其余3组通过；TypeScript 检查和 Vite 构建通过。
- 原18项样例回归在新环境全部通过；离线 CLI 生成 Markdown/JSON，socket 连接尝试为0。
- 3项 FastAPI 测试使用注入的内存资源通过，覆盖方案配置、工作区/任务核验状态透传、任务失败；状态透传测试同时检查 Markdown 下载保留引用拒绝警告。
- 源码与配置检查未发现可确认的密钥、私网主机或个人路径。具体历史来源目录改为通用的 data/meta/history/active_preparation.json，由使用者提供。
- 保留原有 Word 分页辅助源码，本次未运行 Word。

这些验证覆盖当时的源码构建、部分 API 契约及自编样例；不代表真实模型服务、完整规范库、浏览器登录、原件查看或生产部署已验证。该阶段关联的截图是历史 Streamlit 样例；当时没有制作 React 截图或接管浏览器。
