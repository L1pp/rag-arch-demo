"""Only read this project's .env and explicit RAG_* process settings."""
from dataclasses import dataclass, field
from pathlib import Path
import os
from urllib.parse import urlparse

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_REQUEST_TIMEOUT_SECONDS = 1200.0
DEFAULT_CHAT_TIMEOUT_SECONDS = DEFAULT_REQUEST_TIMEOUT_SECONDS
# Fallback for the project's current Qwen model, not a validation ceiling.
# Explicit .env values must match the selected model/API's output capability.
DEFAULT_CHAT_MAX_TOKENS = 131072
DEFAULT_CHAT_THINKING_BUDGET = 4096
DEFAULT_CHAT_CONTEXT_BUDGET = 1000000
DEFAULT_PLAN_TIMEOUT_SECONDS = DEFAULT_REQUEST_TIMEOUT_SECONDS
DEFAULT_PLAN_MAX_TOKENS = 131072
DEFAULT_PLAN_THINKING_BUDGET = 8192
DEFAULT_PLAN_CONTEXT_BUDGET = 1000000
DEFAULT_PLAN_MAX_STEPS = 50
DEFAULT_PLAN_MAX_TOOL_CALLS = 150


def is_qwen38_model(model: str) -> bool:
    return isinstance(model, str) and model.lower().startswith("qwen3.8")


def validate_chat_budgets(max_tokens, thinking_budget) -> None:
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("max_tokens须为正整数；上限按所选模型/API能力配置。")
    if thinking_budget is not None and (
            type(thinking_budget) is not int or thinking_budget < 1):
        raise ValueError("thinking_budget须为正整数或None；上限按所选模型/API能力配置。")
    if thinking_budget is not None and thinking_budget >= max_tokens:
        raise ValueError("thinking_budget必须小于max_tokens，避免耗尽可见输出预算。")


def validate_chat_context_budget(context_budget, max_tokens) -> None:
    if type(context_budget) is not int or context_budget < 1:
        raise ValueError("chat_context_budget_tokens须为正整数。")
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("chat_max_tokens须为正整数。")
    if context_budget <= max_tokens:
        raise ValueError("chat_context_budget_tokens必须大于chat_max_tokens。")


def resolve_chat_budgets(settings, *, max_tokens=None, thinking_budget=None) -> tuple[int, int | None]:
    resolved_max_tokens = getattr(settings, "chat_max_tokens", DEFAULT_CHAT_MAX_TOKENS) if max_tokens is None else max_tokens
    resolved_thinking_budget = (getattr(settings, "chat_thinking_budget", None)
                                if thinking_budget is None else thinking_budget)
    validate_chat_budgets(resolved_max_tokens, resolved_thinking_budget)
    return resolved_max_tokens, resolved_thinking_budget


@dataclass(frozen=True)
class Settings:
    chat_model: str = ""
    chat_base_url: str = ""
    chat_api_key: str = field(default="", repr=False)
    embedding_model: str = ""
    embedding_base_url: str = ""
    embedding_api_key: str = field(default="", repr=False)
    timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    # Direct Settings(...) callers retain the historical shared timeout.  The
    # project .env path supplies the longer chat-specific timeout below.
    chat_timeout_seconds: float | None = None
    chat_max_tokens: int = DEFAULT_CHAT_MAX_TOKENS
    chat_thinking_budget: int | None = None
    chat_context_budget_tokens: int = DEFAULT_CHAT_CONTEXT_BUDGET
    chat_web_search_enabled: bool = True
    chat_web_extractor_enabled: bool = True
    chat_code_interpreter_enabled: bool = True
    chat_api_protocol: str = "auto"
    plan_timeout_seconds: float = DEFAULT_PLAN_TIMEOUT_SECONDS
    plan_max_tokens: int = DEFAULT_PLAN_MAX_TOKENS
    plan_thinking_budget: int | None = None
    plan_context_budget_tokens: int = DEFAULT_PLAN_CONTEXT_BUDGET
    plan_context_summary_ratio: float = 0.9
    embedding_batch_size: int = 20
    embedding_dimensions: int | None = None
    reranker_provider: str = "bailian"
    reranker_model: str = "qwen3-rerank"
    reranker_base_url: str = ""
    reranker_api_key: str = field(default="", repr=False)
    reranker_default_enabled: bool = False

    @property
    def chat_ready(self) -> bool:
        return all((self.chat_model, self.chat_base_url, self.chat_api_key))

    @property
    def effective_chat_timeout_seconds(self) -> float:
        return self.timeout_seconds if self.chat_timeout_seconds is None else self.chat_timeout_seconds

    @property
    def embedding_ready(self) -> bool:
        return all((self.embedding_model, self.embedding_base_url, self.embedding_api_key))

    @property
    def reranker_ready(self) -> bool:
        if self.reranker_provider == "local":
            from .local_reranking import LOCAL_MODEL, validate_local_base_url
            try:
                validate_local_base_url(self.reranker_base_url or "http://127.0.0.1:8766")
            except ValueError:
                return False
            return self.reranker_model == LOCAL_MODEL
        if self.reranker_provider != "bailian":
            return False
        if not self.reranker_model:
            return False
        if self.reranker_base_url or self.reranker_api_key:
            return bool(self.reranker_base_url and self.reranker_api_key)
        host = urlparse(self.embedding_base_url).hostname or ""
        return bool(self.embedding_api_key and host.endswith(".maas.aliyuncs.com"))

    def missing_fields(self, kind: str) -> list[str]:
        prefix = "chat" if kind == "chat" else "embedding"
        return [f"RAG_{prefix.upper()}_{part.upper()}" for part in ("model", "base_url", "api_key")
                if not getattr(self, f"{prefix}_{part}")]


def load_settings(env_path: Path | None = None, environ: dict[str, str] | None = None,
                  *, use_tokenplan: bool | None = None) -> Settings:
    """Load effective providers; the optional override selects only chat transport.

    chat_* describes the selected transport, not necessarily raw RAG_CHAT_*.
    Use use_tokenplan=False for an ordinary-endpoint check. Embeddings and
    reranking always retain their separately configured URLs and credentials.
    This override never modifies .env or the process environment.
    """
    values = dict(dotenv_values(env_path or ROOT / ".env", interpolate=False))
    values.update({k: v for k, v in (os.environ if environ is None else environ).items()
                   if k.startswith("RAG_")})
    def get(name: str, default: str = "") -> str:
        return (values.get("RAG_" + name) or default).strip()

    protocol = get("CHAT_API_PROTOCOL", "auto").lower()
    if protocol not in {"auto", "responses", "chat_completions"}:
        raise ValueError("RAG_CHAT_API_PROTOCOL 仅支持 auto、responses、chat_completions。")
    web_extractor = get("CHAT_WEB_EXTRACTOR_ENABLED", "true").lower()
    code_interpreter = get("CHAT_CODE_INTERPRETER_ENABLED", "true").lower()
    if web_extractor not in {"true", "false"} or code_interpreter not in {"true", "false"}:
        raise ValueError("原生工具能力开关仅支持 true 或 false。")
    web_search = get("CHAT_WEB_SEARCH_ENABLED", "true").lower()
    if web_search not in {"true", "false"}:
        raise ValueError("RAG_CHAT_WEB_SEARCH_ENABLED 仅支持 true 或 false。")
    tokenplan_option = get("USE_TOKENPLAN", "true").lower()
    if tokenplan_option not in {"true", "false"}:
        raise ValueError("RAG_USE_TOKENPLAN 仅支持 true 或 false。")
    if use_tokenplan is not None and type(use_tokenplan) is not bool:
        raise ValueError("use_tokenplan 须为布尔值或 None。")
    tokenplan_enabled = tokenplan_option == "true" if use_tokenplan is None else use_tokenplan
    tokenplan_base_url = get("TOKENPLAN_BASE_URL")
    tokenplan_api_key = get("TOKENPLAN_API_KEY")
    if tokenplan_enabled and bool(tokenplan_base_url) != bool(tokenplan_api_key):
        missing = ("RAG_TOKENPLAN_BASE_URL" if not tokenplan_base_url
                   else "RAG_TOKENPLAN_API_KEY")
        raise ValueError(
            "RAG_TOKENPLAN_BASE_URL 与 RAG_TOKENPLAN_API_KEY 必须同时填写；"
            f"当前缺少 {missing}，不会与 RAG_CHAT_* 混用。"
        )

    if use_tokenplan is True and not tokenplan_base_url:
        raise ValueError("明确选择TokenPlan时须填写 RAG_TOKENPLAN_BASE_URL 与 RAG_TOKENPLAN_API_KEY。")
    chat_base_url = (tokenplan_base_url if tokenplan_enabled else "") or get("CHAT_BASE_URL")
    chat_api_key = (tokenplan_api_key if tokenplan_enabled else "") or get("CHAT_API_KEY")
    chat_model = get("CHAT_MODEL")
    try:
        timeout = float(get("TIMEOUT_SECONDS", str(DEFAULT_REQUEST_TIMEOUT_SECONDS)))
        chat_timeout = float(get("CHAT_TIMEOUT_SECONDS", str(DEFAULT_CHAT_TIMEOUT_SECONDS)))
        chat_max_tokens = int(get("CHAT_MAX_TOKENS", str(DEFAULT_CHAT_MAX_TOKENS)))
        chat_context_budget = int(get("CHAT_CONTEXT_BUDGET_TOKENS", str(DEFAULT_CHAT_CONTEXT_BUDGET)))
        thinking_text = get("CHAT_THINKING_BUDGET")
        chat_thinking_budget = (int(thinking_text) if thinking_text
                                else DEFAULT_CHAT_THINKING_BUDGET if is_qwen38_model(chat_model) else None)
        plan_timeout = float(get("PLAN_TIMEOUT_SECONDS", str(DEFAULT_PLAN_TIMEOUT_SECONDS)))
        plan_max_tokens = int(get("PLAN_MAX_TOKENS", str(DEFAULT_PLAN_MAX_TOKENS)))
        plan_context_budget = int(get("PLAN_CONTEXT_BUDGET_TOKENS", str(DEFAULT_PLAN_CONTEXT_BUDGET)))
        plan_summary_ratio = float(get("PLAN_CONTEXT_SUMMARY_RATIO", "0.9"))
        plan_thinking_text = get("PLAN_THINKING_BUDGET")
        plan_thinking_budget = (int(plan_thinking_text) if plan_thinking_text
                                else DEFAULT_PLAN_THINKING_BUDGET if is_qwen38_model(chat_model) else None)
        batch = int(get("EMBEDDING_BATCH_SIZE", "20"))
        dimensions = int(get("EMBEDDING_DIMENSIONS")) if get("EMBEDDING_DIMENSIONS") else None
    except ValueError as exc:
        raise ValueError("超时、聊天输出/上下文预算、方案输出/上下文预算及摘要比例、思考预算、向量批量和向量维度必须是数字；维度可留空使用模型默认值。") from exc
    if (not 1 <= timeout <= 1200 or not 1 <= chat_timeout <= 1200
            or not 1 <= plan_timeout <= 1200 or not 1 <= batch <= 128):
        raise ValueError("通用、聊天和整份方案请求超时须为 1–1200 秒，向量批量须为 1–128。")
    try:
        validate_chat_budgets(chat_max_tokens, chat_thinking_budget)
    except ValueError as exc:
        raise ValueError(f"RAG_CHAT_MAX_TOKENS/RAG_CHAT_THINKING_BUDGET 配置无效：{exc}") from exc
    try:
        validate_chat_context_budget(chat_context_budget, chat_max_tokens)
    except ValueError as exc:
        raise ValueError(f"RAG_CHAT_CONTEXT_BUDGET_TOKENS 配置无效：{exc}") from exc
    try:
        validate_chat_budgets(plan_max_tokens, plan_thinking_budget)
    except ValueError as exc:
        raise ValueError(f"RAG_PLAN_MAX_TOKENS/RAG_PLAN_THINKING_BUDGET 配置无效：{exc}") from exc
    if plan_context_budget < 8000 or plan_context_budget <= plan_max_tokens:
        raise ValueError("RAG_PLAN_CONTEXT_BUDGET_TOKENS 须为至少8000的整数，且大于 RAG_PLAN_MAX_TOKENS。")
    if not 0 <= plan_summary_ratio < 1:
        raise ValueError("RAG_PLAN_CONTEXT_SUMMARY_RATIO 须在[0,1)内；0关闭滚动摘要。")
    if dimensions is not None and dimensions <= 0:
        raise ValueError("RAG_EMBEDDING_DIMENSIONS 须为正整数或留空。")
    for name, url in (("CHAT_BASE_URL", chat_base_url),
                      ("TOKENPLAN_BASE_URL", tokenplan_base_url),
                      ("EMBEDDING_BASE_URL", get("EMBEDDING_BASE_URL")),
                      ("RERANKER_BASE_URL", get("RERANKER_BASE_URL"))):
        if url:
            parsed = urlparse(url)
            if parsed.scheme not in {"https", "http"} or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError(f"RAG_{name} 须为不含账号密码的 HTTP(S) 接口根地址。")
    reranker_provider = get("RERANKER_PROVIDER", "bailian").lower()
    if reranker_provider not in {"bailian", "local"}:
        raise ValueError("RAG_RERANKER_PROVIDER 仅支持 bailian 或 local。")
    reranker_base_url = get("RERANKER_BASE_URL")
    reranker_default = get("RERANKER_DEFAULT_ENABLED", "false").lower()
    if reranker_default not in {"true", "false"}:
        raise ValueError("RAG_RERANKER_DEFAULT_ENABLED 仅支持 true 或 false。")
    if reranker_provider == "local":
        from .local_reranking import LOCAL_MODEL, validate_local_base_url
        reranker_model = get("RERANKER_MODEL", LOCAL_MODEL)
        if reranker_model != LOCAL_MODEL:
            raise ValueError(f"本地 RAG_RERANKER_MODEL 当前须为 {LOCAL_MODEL}；不会自动切换模型。")
        reranker_base_url = validate_local_base_url(reranker_base_url or "http://127.0.0.1:8766")
    else:
        reranker_model = get("RERANKER_MODEL", "qwen3-rerank")
        from .reranking import SUPPORTED_RERANK_MODELS
        if reranker_model not in SUPPORTED_RERANK_MODELS:
            supported = "、".join(SUPPORTED_RERANK_MODELS)
            raise ValueError(f"RAG_RERANKER_MODEL 不受支持：{reranker_model}；可选模型：{supported}。")
    return Settings(
        chat_model=chat_model, chat_base_url=chat_base_url, chat_api_key=chat_api_key,
        embedding_model=get("EMBEDDING_MODEL"), embedding_base_url=get("EMBEDDING_BASE_URL"),
        embedding_api_key=get("EMBEDDING_API_KEY"), timeout_seconds=timeout,
        chat_timeout_seconds=chat_timeout, chat_max_tokens=chat_max_tokens,
        chat_thinking_budget=chat_thinking_budget,
        chat_web_search_enabled=web_search == "true",
        chat_web_extractor_enabled=web_extractor == "true",
        chat_code_interpreter_enabled=code_interpreter == "true", chat_api_protocol=protocol,
        chat_context_budget_tokens=chat_context_budget, embedding_batch_size=batch,
        plan_timeout_seconds=plan_timeout, plan_max_tokens=plan_max_tokens,
        plan_thinking_budget=plan_thinking_budget,
        plan_context_budget_tokens=plan_context_budget,
        plan_context_summary_ratio=plan_summary_ratio,
        embedding_dimensions=dimensions,
        reranker_provider=reranker_provider, reranker_model=reranker_model,
        reranker_base_url=reranker_base_url,
        reranker_api_key=get("RERANKER_API_KEY") if reranker_provider == "bailian" else "",
        reranker_default_enabled=reranker_default == "true",
    )
