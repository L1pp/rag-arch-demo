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


def create_chat_model(settings: Settings, *, max_tokens: int | None = None,
                      thinking_budget: int | None = None, native_web_search: bool = False):
    if not settings.chat_ready:
        return None
    if select_chat_protocol(settings) == "responses":
        output_budget, thinking = resolve_chat_budgets(settings, max_tokens=max_tokens, thinking_budget=thinking_budget)
        return create_responses_chat(settings, max_tokens=output_budget, thinking_budget=thinking)
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
    extra_body = web_search_extra_body(settings) if native_web_search else {}
    if resolved_thinking_budget is not None:
        extra_body["thinking_budget"] = resolved_thinking_budget
    if extra_body:
        kwargs["extra_body"] = extra_body
    return attach_chat_audit(cls(**kwargs))


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
