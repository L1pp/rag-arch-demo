"""Minimal Streamlit interface for the public demo."""
from __future__ import annotations

import json

import streamlit as st

from demo.runtime import (
    DEFAULT_PLAN, DEFAULT_STANDARDS, PUBLIC_MATERIAL_NOTICE,
    _read_json, build_runtime, make_plan_document, render_plan_report,
    result_to_dict, run_plan_review, run_standard_question,
)

st.set_page_config(page_title="资料检索与方案核查演示", page_icon="📚", layout="wide")
st.title("资料检索与方案核查演示")
st.caption("核心运行链路来自仓库内的 EvidenceIndex、StandardTools 和 Agent 实现。")
st.info(PUBLIC_MATERIAL_NOTICE)

mode_label = st.radio(
    "运行方式",
    ["离线脚本演示", "在线模型（需配置本仓库 .env）"],
    horizontal=True,
    help="默认离线，无网络请求。在线模式只在点击运行后读取本演示仓库自己的配置。",
)
mode = "live" if mode_label.startswith("在线") else "offline"
if mode == "live":
    st.warning("在线模式会将演示输入发送到已配置的模型、向量和重排服务。请只使用本仓库的虚构样例资料。")

material_tab, qa_tab, plan_tab = st.tabs(["资料", "问答", "方案核查"])

with material_tab:
    standards_data = _read_json(DEFAULT_STANDARDS)
    plan_data = _read_json(DEFAULT_PLAN)
    left, right = st.columns(2)
    with left:
        st.subheader(standards_data["standard"].get("title", "样例资料"))
        st.caption(standards_data.get("disclaimer", PUBLIC_MATERIAL_NOTICE))
        for clause in standards_data.get("clauses", []):
            with st.expander(f"{clause['clause_no']} · {clause.get('title', '演示条文')}", expanded=True):
                st.write(clause["text"])
    with right:
        st.subheader(plan_data.get("filename", "演示方案"))
        st.caption(plan_data.get("disclaimer", PUBLIC_MATERIAL_NOTICE))
        for unit in plan_data.get("units", []):
            st.markdown(f"**第 {unit.get('locator', {}).get('line_start', '?')} 行**")
            st.write(unit["text"])
    st.caption("界面只展示本仓库自编的小型材料，不加载原项目语料或本机历史记录。")

with qa_tab:
    st.subheader("资料问答")
    question = st.text_input("问题", value="虚构演示条文中，支撑间距的上限是多少？")
    if st.button("运行问答", type="primary", key="run_qa"):
        try:
            runtime = build_runtime(mode)
            result = run_standard_question(runtime, question)
            payload = result_to_dict(result)
            st.session_state["qa_payload"] = payload
            st.session_state["qa_mode"] = mode
        except Exception as exc:
            st.error(f"运行失败（{type(exc).__name__}）。请检查依赖或演示配置。")
    if "qa_payload" in st.session_state:
        payload = st.session_state["qa_payload"]
        st.markdown(payload.get("text", "本轮没有返回文字。"))
        citations = payload.get("citations", [])
        if citations:
            st.markdown("**程序记录的引用**")
            for citation in citations:
                st.code(citation["evidence_id"], language=None)
        st.download_button(
            "下载问答 JSON",
            data=json.dumps({"execution_mode": st.session_state.get("qa_mode"), "result": payload},
                            ensure_ascii=False, indent=2),
            file_name="standard_answer.json",
            mime="application/json",
            key="download_qa",
        )

with plan_tab:
    st.subheader("方案核查")
    plan_data = _read_json(DEFAULT_PLAN)
    initial_text = plan_data.get("units", [{}])[0].get("text", "")
    plan_text = st.text_area("演示方案原文（可修改）", value=initial_text, height=150)
    if st.button("运行方案演示", type="primary", key="run_plan"):
        try:
            runtime = build_runtime(mode)
            document = make_plan_document(plan_data, text=plan_text)
            result = run_plan_review(runtime, document)
            report = render_plan_report(result)
            st.session_state["plan_report"] = report
            st.session_state["plan_payload"] = result_to_dict(result)
            st.session_state["plan_mode"] = mode
        except Exception as exc:
            st.error(f"运行失败（{type(exc).__name__}）。请检查依赖或演示配置。")
    if "plan_report" in st.session_state:
        st.markdown(st.session_state["plan_report"])
        st.download_button(
            "下载 Markdown 报告",
            data=st.session_state["plan_report"],
            file_name="report.md",
            mime="text/markdown",
            key="download_report",
        )
        st.download_button(
            "下载方案结果 JSON",
            data=json.dumps({"execution_mode": st.session_state.get("plan_mode"),
                             "result": st.session_state["plan_payload"]}, ensure_ascii=False, indent=2),
            file_name="plan_review.json",
            mime="application/json",
            key="download_plan_json",
        )