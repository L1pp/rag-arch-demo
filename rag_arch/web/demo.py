"""A cancellable four-stage simulation, with no project data or model calls."""
from concurrent.futures import CancelledError


STAGES = (
    ("preparing", "模拟：准备任务"),
    ("reading", "模拟：读取资料"),
    ("analyzing", "模拟：分析问题"),
    ("summarizing", "模拟：整理结果"),
)


def run_demo(request, progress, cancel_event):
    delay = request["duration_seconds"] / len(STAGES)
    for stage, detail in STAGES:
        if cancel_event.is_set():
            raise CancelledError("演示任务已取消。")
        progress(stage, detail)
        if cancel_event.wait(delay):
            raise CancelledError("演示任务已取消。")
    if request["fail_at_end"]:
        raise RuntimeError("模拟失败：用于检查界面错误状态，没有调用真实模型。")
    return {"text": "演示任务已完成。此结果仅用于验证后台运行和界面切换，未读取真实资料或调用模型。\n\n"
                    + "本次输入：" + request["prompt"], "simulation": True}
