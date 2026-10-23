"""Construct separately configured chat/embedding clients without API calls.

Adapted from the course dating assistant's create_llm and day59 RAG embedding
factory; vendor names, credentials and model choices remain project settings.
"""
from .config import Settings, resolve_chat_budgets
from .runtime_audit import attach_chat_audit
from .web_search import (WebSearchChat, web_search_extra_body, create_web_search_provider,
                         select_chat_protocol, create_responses_chat)
TONGYI_FLASH = "tongyi-embedding-vision-flash-2026-03-06"
QWEN_TEXT = "qwen3.7-text-embedding"
QWEN_FLASH_TEXT = "qwen3.7-text-embedding-flash"


def _new_owned_async_http_client(timeout):
    import httpx
    return httpx.AsyncClient(timeout=timeout, follow_redirects=False)


def _mark_owned_async_http_client(model, client):
    """Bind ownership to the concrete model instance that created a client."""
    target = model
    seen = set()
    while id(target) not in seen:
        seen.add(id(target))
        values = getattr(target, "__dict__", {})
        child = values.get("_model") if isinstance(values, dict) else None
        if child is None:
            break
        target = child
    values = getattr(target, "__dict__", None)
    if not isinstance(values, dict):
        raise TypeError("async model must expose instance state for client ownership")
    values["_rag_owned_async_http_client"] = client


def _dispose_unattached_async_client(client):
    """Close a newly-created client if model construction fails before use."""
    import asyncio

    async def close_safely():
        try:
            await client.aclose()
        except Exception:
            # Preserve the constructor's original exception.
            return None

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(close_safely())
    else:
        loop.create_task(close_safely())


def create_chat_model(settings: Settings, *, max_tokens: int | None = None,
                      thinking_budget: int | None = None, native_web_search: bool = False,
                      http_async_client=None):
    if not settings.chat_ready:
        return None
    if select_chat_protocol(settings) == "responses":
        output_budget, thinking = resolve_chat_budgets(settings, max_tokens=max_tokens, thinking_budget=thinking_budget)
        return create_responses_chat(settings, max_tokens=output_budget, thinking_budget=thinking,
                                     http_async_client=http_async_client)
    cls = WebSearchChat
    if settings.chat_model.lower().startswith(("qwen3.8-flash", "qwen3.8-max")):
        from .tool_chat import Qwen38Chat
        cls = Qwen38Chat
    resolved_max_tokens, resolved_thinking_budget = resolve_chat_budgets(
        settings, max_tokens=max_tokens, thinking_budget=thinking_budget)
    kwargs = dict(
        model=settings.chat_model,
        api_key=settings.chat_api_key,
        base_url=settings.chat_base_url.rstrip("/"),
        timeout=settings.effective_chat_timeout_seconds,
        max_tokens=resolved_max_tokens,
        max_retries=0,
        use_responses_api=False,
        stream_usage=False,
    )
    if http_async_client is not None:
        kwargs["http_async_client"] = http_async_client
    extra_body = web_search_extra_body(settings) if native_web_search else {}
    if resolved_thinking_budget is not None:
        extra_body["thinking_budget"] = resolved_thinking_budget
    if extra_body:
        kwargs["extra_body"] = extra_body
    return attach_chat_audit(cls(**kwargs))


def create_async_chat_model(settings: Settings, *, max_tokens: int | None = None,
                            thinking_budget: int | None = None, native_web_search: bool = False,
                            http_async_client=None):
    """Construct a chat model with an owned per-model async transport by default."""
    if not settings.chat_ready:
        return None
    protocol = select_chat_protocol(settings)
    _, resolved_thinking = resolve_chat_budgets(
        settings, max_tokens=max_tokens, thinking_budget=thinking_budget)
    if (protocol == "responses" and resolved_thinking is not None
            and not settings.chat_model.lower().startswith("qwen3.8")):
        raise ValueError("此模型Responses不支持thinking_budget；请移除该专属参数，使用模型默认思考配置。")
    owns_client = http_async_client is None
    client = (http_async_client if not owns_client else
              _new_owned_async_http_client(settings.effective_chat_timeout_seconds))
    try:
        model = create_chat_model(settings, max_tokens=max_tokens, thinking_budget=thinking_budget,
                                  native_web_search=native_web_search, http_async_client=client)
        if owns_client:
            _mark_owned_async_http_client(model, client)
    except BaseException:
        if owns_client:
            _dispose_unattached_async_client(client)
        raise
    return model


async def async_close_model(model):
    """Close model-owned async clients and wrapper-owned async resources.

    Explicitly injected httpx clients are borrowed and remain open. The helper
    also accepts TerminalOutput wrappers and NativeWebSearchProvider instances
    used by async runners.
    """
    seen = set()

    async def close_one(value):
        if value is None or id(value) in seen:
            return
        seen.add(id(value))
        is_wrapper = False
        for name in ("_model", "_owner"):
            child = getattr(value, name, None)
            if child is not None and child is not value:
                await close_one(child)
                is_wrapper = True
        if is_wrapper:
            return

        # NativeWebSearchProvider's lazy client is owned by that provider only
        # when it constructed the client itself.
        provider_client = getattr(value, "async_client", None)
        if (provider_client is not None
                and getattr(value, "_owns_async_client", False)
                and callable(getattr(provider_client, "aclose", None))):
            await provider_client.aclose()
            value.async_client = None
            value._owns_async_client = False

        model_values = getattr(value, "__dict__", {})
        owned_model_async_http = (model_values.get("_rag_owned_async_http_client")
                                  if isinstance(model_values, dict) else None)
        if owned_model_async_http is not None:
            await owned_model_async_http.aclose()
            model_values.pop("_rag_owned_async_http_client", None)

        root_async = getattr(value, "root_async_client", None)
        owns_responses_client = bool(getattr(value, "_owns_root_async_client", False))
        if root_async is not None and owns_responses_client:
            closer = getattr(root_async, "aclose", None) or getattr(root_async, "close", None)
            if callable(closer):
                result = closer()
                if hasattr(result, "__await__"):
                    await result
            if owns_responses_client:
                value.root_async_client = None
                value._owns_root_async_client = False

    await close_one(model)


def create_embeddings(settings: Settings):
    if not settings.embedding_ready:
        return None
    if settings.embedding_model == TONGYI_FLASH:
        raise ValueError("Tongyi flash 快照需要原生多模态接口，当前尚未接入；当前选择为 qwen3.7-text-embedding。")

    dimensions = settings.embedding_dimensions
    batch_size = settings.embedding_batch_size
    if settings.embedding_model == QWEN_FLASH_TEXT:
        dimensions = dimensions or 1024
        if dimensions not in {256, 512, 768, 1024}:
            raise ValueError("qwen3.7-text-embedding-flash 维度须为 256、512、768 或 1024。")
        batch_size = min(batch_size, 20)
    elif settings.embedding_model == QWEN_TEXT:
        dimensions = dimensions or 1024
        if dimensions not in {256, 512, 768, 1024, 1536, 2048, 2560}:
            raise ValueError("qwen3.7-text-embedding 维度须为 256、512、768、1024、1536、2048 或 2560。")
        batch_size = min(batch_size, 20)
    elif settings.embedding_model == "text-embedding-v4":
        # DashScope v4 accepts at most ten texts per synchronous request.
        batch_size = min(batch_size, 10)
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_base_url.rstrip("/"),
        request_timeout=settings.timeout_seconds,
        max_retries=0,
        chunk_size=batch_size,
        dimensions=dimensions,
        model_kwargs={"encoding_format": "float"},
        check_embedding_ctx_length=False,
    )


def create_reranker(settings: Settings):
    """Construct the selected local/cloud provider without network requests."""
    if not settings.reranker_ready:
        return None
    if settings.reranker_provider == "local":
        from .local_reranking import LocalReranker
        return LocalReranker(settings.reranker_base_url or "http://127.0.0.1:8766",
                             model=settings.reranker_model, timeout=settings.timeout_seconds)
    from .reranking import BailianReranker

    client = BailianReranker(
        base_url=settings.reranker_base_url or settings.embedding_base_url,
        api_key=settings.reranker_api_key or settings.embedding_api_key,
        model=settings.reranker_model, timeout=settings.timeout_seconds,
    )
    client.provider = "bailian"
    return client
