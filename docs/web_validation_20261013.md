# 2026-10-13 React/FastAPI 源码迁移验证

[English](web_validation_20261013_en.md) | [简体中文](web_validation_20261013.md)

默认入口已改为现有 React/FastAPI 工作台。`streamlit_demo.py`、离线 CLI 和自编样例独立保留；没有引入原 Git 历史或私有语料。

- 新建独立 Python 3.12.10 Web 环境安装主依赖及 editable 包；为样例测试添加可选 Streamlit 依赖，`pip check` 通过。
- 从公开后端契约生成前端类型。8 组、47 项前端测试通过：5 组首次通过，修正 JSX 闭合后其余3组通过；TypeScript 检查和 Vite 构建通过。
- 原18项样例回归在新环境全部通过；离线 CLI 生成 Markdown/JSON，socket 连接尝试为0。
- 3项 FastAPI 测试使用注入的内存资源通过，覆盖方案配置、工作区/任务核验状态透传、任务失败；状态透传测试同时检查 Markdown 下载保留引用拒绝警告。
- 源码与配置检查未发现可确认的密钥、私网主机或个人路径。具体历史来源目录改为通用的 `data/meta/history/active_preparation.json`，由使用者提供。
- 保留原有 Word 分页辅助源码，本次未运行 Word。

验证范围是源码构建、选定API契约及自编样例，不代表真实模型服务、完整规范库、真实浏览器登录、原页功能或生产部署已经验证。现有截图仍为历史 Streamlit 样例，本次没有接管浏览器或制作 React 截图。
