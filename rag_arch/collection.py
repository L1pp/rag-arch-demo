"""One in-memory upload collection, with no global index or disk persistence."""
from dataclasses import dataclass
from uuid import uuid4

from .loaders import _upload_name_kind, load_document, validate_upload_sizes
from .models import Chunk, LoadedDocument
from .plan_review_context import estimate_tokens
from .retrieval import LocalIndex
from .splitting import split_document

DIRECT_TEXT_MAX_TOKENS = 200_000
# Explicit legacy experiments can retain the former byte-based boundary.
DIRECT_TEXT_MAX_BYTES = 10 * 1024

@dataclass
class UploadCollection:
    collection_id: str
    documents: list[LoadedDocument]
    chunks: list[Chunk]
    index: LocalIndex | None
    uses_embeddings: bool
    mode: str = "retrieval"
    text_bytes: int = 0
    text_tokens: int = 0


def extracted_text_tokens(documents) -> int:
    """Estimate all extracted units across the selection, not upload bytes."""
    return sum(estimate_tokens(unit.text) for document in documents for unit in document.units)


def build_collection(files: list[tuple[str, bytes]], embeddings=None,
                     direct_text_max_bytes: int | None = None, *,
                     direct_text_max_tokens: int | None = None,
                     pdf_backend=None, documents=None) -> UploadCollection:
    """Token policy prefers complete text; the byte policy is legacy opt-in.

    Omitting both limits retains the low-level retrieval-only API used by
    existing experiments. Production chat callers pass the token limit.
    """
    if not files:
        raise ValueError("请先上传文件。")
    if direct_text_max_bytes is not None and direct_text_max_tokens is not None:
        raise ValueError("全文直读门槛只能指定direct_text_max_tokens或direct_text_max_bytes之一。")
    for name, limit in (("direct_text_max_tokens", direct_text_max_tokens),
                        ("direct_text_max_bytes", direct_text_max_bytes)):
        if limit is not None and (type(limit) is not int or limit < 0):
            raise ValueError(f"{name}须为不小于0的整数或None。")
    validate_upload_sizes(len(data) for _, data in files)
    collection_id = uuid4().hex
    if documents is None:
        documents = [(
            load_document(name, data, pdf_backend=pdf_backend)
            if pdf_backend is not None else load_document(name, data)
        ) for name, data in files]
    else:
        documents = list(documents)
        if len(documents) != len(files):
            raise ValueError("已解析文档数量与上传文件不一致。")
        for (filename, data), document in zip(files, documents):
            name, kind = _upload_name_kind(filename, data)
            if (not isinstance(document, LoadedDocument)
                    or document.filename != name or document.source_kind != kind):
                raise ValueError("已解析文档与上传文件不一致。")
    text_bytes = sum(len(unit.text.encode("utf-8")) for doc in documents for unit in doc.units)
    text_tokens = extracted_text_tokens(documents)
    direct = (text_tokens <= direct_text_max_tokens if direct_text_max_tokens is not None
              else embeddings is None and direct_text_max_bytes is not None
              and text_bytes <= direct_text_max_bytes)
    if direct:
        # Keep every extracted unit once, with its source; no index or ranking.
        chunks = [Chunk(doc.doc_id, doc.filename, f"{doc.doc_id}:full:{i}",
                        unit.text, dict(unit.locator), doc.source_kind, collection_id)
                  for doc in documents for i, unit in enumerate(doc.units, 1) if unit.text.strip()]
        return UploadCollection(collection_id, documents, chunks, None, False, "direct", text_bytes, text_tokens)
    chunks = [chunk for doc in documents for chunk in split_document(doc, collection_id)]
    return UploadCollection(collection_id, documents, chunks, LocalIndex(chunks, embeddings),
                            embeddings is not None, "retrieval", text_bytes, text_tokens)
