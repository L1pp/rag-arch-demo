"""Small shared records; uploaded documents need no engineering clause schema."""
from dataclasses import dataclass, field


@dataclass
class TextUnit:
    text: str
    locator: dict[str, object]


@dataclass
class LoadedDocument:
    doc_id: str
    filename: str
    source_kind: str
    units: list[TextUnit]
    warnings: list[str]
    unread_pages: list[int]
    page_count: int | None = None


@dataclass
class Chunk:
    doc_id: str
    filename: str
    chunk_id: str
    text: str
    locator: dict[str, object]
    source_kind: str
    collection_id: str


@dataclass
class SearchHit:
    chunk: Chunk
    score: float


@dataclass
class Answer:
    text: str
    hits: list[SearchHit]
    warnings: list[str]
    grounded: bool
    citation_check: dict | None = None
    audit_log: dict | None = None
    page_sources: list[dict] = field(default_factory=list)
    web_sources: list[dict] = field(default_factory=list)
    native_tool_observations: list[dict] = field(default_factory=list)
    native_tool_usage: dict = field(default_factory=dict)
    catalog_matches: list[dict] = field(default_factory=list)
    answer_scope: str = "evidence"
    tool_calls: int = 0


@dataclass
class ConversationTurn:
    question: str
    answer: Answer
