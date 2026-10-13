"""The small public contract shared by the web UI and task service."""
from typing import Literal

from pydantic import BaseModel, Field, field_validator

AssistantId = Literal["chat", "standards", "plan"]
TaskStatus = Literal["queued", "running", "cancelling", "completed", "failed", "cancelled"]


class TaskResult(BaseModel):
    text: str
    simulation: bool = True
    model: str | None = None
    web_sources: list[dict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    native_tool_usage: dict = Field(default_factory=dict)
    audit_log: dict | None = None
    token_usage: dict = Field(default_factory=dict)
    citations: list[dict] = Field(default_factory=list)
    catalog_matches: list[dict] = Field(default_factory=list)
    status: str | None = None
    termination_reason: str | None = None
    answer_scope: str | None = None
    corpus_id: str | None = None
    grounded: bool = False
    tool_calls: int = 0
    model_calls: int = 0
    files: list[dict] = Field(default_factory=list)
    file_mode: str | None = None
    uses_embeddings: bool = False
    document: dict | None = None
    report: dict | None = None
    report_markdown: str | None = None
    page_sources: list[dict] = Field(default_factory=list)
    citation_check: dict | None = None


class TaskInputs(BaseModel):
    attachment_ids: list[str] = Field(default_factory=list)
    registered_plan_id: str | None = None
    plan_action: Literal["review", "question"] = "question"
    use_embeddings: bool = False
    max_steps: int | None = Field(default=None, ge=3, le=80)
    max_tool_calls: int | None = Field(default=None, ge=1, le=200)


class TaskSnapshot(TaskInputs):
    task_id: str
    assistant_id: AssistantId
    request_id: str
    prompt: str
    status: TaskStatus
    stage: str
    detail: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    elapsed_seconds: float
    duration_seconds: float | None
    updated_seq: int
    result: TaskResult | None = None
    error: str | None = None
    simulation: bool = True


class AssistantSnapshot(BaseModel):
    assistant_id: AssistantId
    title: str
    active_task_id: str | None
    tasks: list[TaskSnapshot]
    execution_mode: Literal["model", "simulation"] = "simulation"


class WorkspaceSnapshot(BaseModel):
    workspace_id: str
    server_instance_id: str
    cursor: int
    mode: Literal["infrastructure_demo", "chat_preview"] = "infrastructure_demo"
    model_name: str | None = None
    assistants: list[AssistantSnapshot]


class WorkspaceCommand(BaseModel):
    workspace_id: str


class SubmitTask(WorkspaceCommand, TaskInputs):
    assistant_id: AssistantId
    request_id: str = Field(min_length=1, max_length=128)
    prompt: str = Field(min_length=1, max_length=20000)
    duration_seconds: float = Field(default=5, ge=1, le=120)
    fail_at_end: bool = False

    @field_validator("prompt")
    @classmethod
    def nonempty_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("请输入消息内容。")
        return value.strip()
