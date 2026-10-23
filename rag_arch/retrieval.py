# 模块说明：面向"用户上传文本"的轻量级内存检索实现，与标准规范基准测试解耦。
# 保留了课程 FlashRAG 检索器中"记录(record) / 检索(search)"分离的设计；
# 中文分词、参数校验以及可选的余弦 / RRF 融合属于本项目的改造。
# 上传文本默认在内存中建索引；公共规范可使用经校验的只读 BM25 数组包。
"""Small in-memory retrieval for uploaded text, independent of the standards benchmark.

The course FlashRAG retriever's record/search separation is retained here. Chinese
tokenization, validation and optional cosine/RRF fusion are project adaptations.
Uploaded text uses in-memory statistics. Public standards may attach a validated
read-only BM25 array package; pickle is never loaded.
"""
from collections import Counter  # 计数器：统计文档内的词频、以及全局的文档频率
from collections.abc import Sequence
import math  # 数学函数：BM25 平滑 IDF 中用到 log1p
import re  # 正则表达式：用于把文本切分成词元
import unicodedata  # Unicode 规范化（NFKC），统一全角/半角等不同写法

import numpy as np  # 数值计算：向量归一化、点积与有限值校验

from .async_support import run_local
from .models import Chunk, SearchHit  # 项目内数据模型：文本块 Chunk、检索命中 SearchHit
from .reranking import RerankError


# 词元切分正则，按顺序匹配四类片段：
# 1) 中日韩汉字区间（含扩展A区 \u3400-\u4dbf 与基本区 \u4e00-\u9fff）；
# 2) 英文单词或下划线标识符 [a-z_]+；
# 3) 完整数字（可带正负号、小数、百分号或千分号），保证 1.8 不会被拆成 1 和 8；
# 4) 比较运算符（>= <= != 优先于单字符 > < =），避免被拆开。
_TOKENS = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+|[a-z_]+|[-+]?\d+(?:\.\d+)?(?:%|‰)?|>=|<=|!=|[><=]")

_LAZY_METADATA_FIELDS = {
    "chunk_id", "std_no", "std_id", "evidence_kind", "document_part",
    "excluded_from_current_retrieval", "import_part_id",
}


class _LazyChunkRows(Sequence):
    """Validated ID/text view that decodes one embedding row on access."""

    is_lazy_chunk_rows = True

    def __init__(self, chunks, row_ids, metadata, id_text_rows, unique_text_count,
                 error_type):
        self._chunks = chunks
        self.row_ids = row_ids
        self.metadata = metadata
        self._id_text_rows = id_text_rows
        self.unique_text_count = unique_text_count
        self._error_type = error_type

    def __len__(self):
        return len(self.row_ids)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("lazy chunk row index must be an integer")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if (not isinstance(self.metadata[index], dict)
                or self.metadata[index].get("chunk_id") != self.row_ids[index]):
            raise self._error_type("lazy chunk ID metadata changed after validation")
        try:
            row = self._id_text_rows[index]
        except Exception as exc:
            raise self._error_type("lazy chunk id/text row could not be read") from exc
        if (not isinstance(row, (tuple, list)) or len(row) != 2
                or row[0] != self.row_ids[index]
                or not isinstance(row[1], str) or not row[1].strip()):
            raise self._error_type("lazy chunk id/text row is invalid or out of order")
        return row[0], row[1]

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]


def _validated_lazy_chunk_sequence(chunks, error_type=ValueError):
    """Recognize only the explicit LazyChunkSequence protocol.

    This validates small row IDs and metadata without indexing ``chunks`` or
    decoding any row text. An unmarked ordinary sequence returns ``None`` and
    keeps its historical path.
    """
    try:
        marker = getattr(chunks, "is_lazy_chunk_sequence", False)
    except Exception as exc:
        raise error_type("lazy chunk marker could not be read") from exc
    if marker is not True:
        return None

    def invalid(message):
        raise error_type(message)

    if (isinstance(chunks, (str, bytes)) or not isinstance(chunks, Sequence)):
        invalid("marked lazy chunks must implement Sequence")
    try:
        count = len(chunks)
        row_ids = chunks.row_ids
        metadata = chunks.metadata
        id_text_rows = chunks.id_text_rows
        unique_text_count = chunks.unique_text_count
        rows_count = len(id_text_rows)
    except Exception as exc:
        raise error_type("marked lazy chunks are missing sequence metadata") from exc
    if (type(count) is not int or count < 0
            or not isinstance(row_ids, list) or len(row_ids) != count
            or not isinstance(metadata, Sequence) or len(metadata) != count
            or not isinstance(id_text_rows, Sequence) or rows_count != count
            or type(unique_text_count) is not int
            or not 0 <= unique_text_count <= count):
        invalid("marked lazy chunk row counts or unique text count are inconsistent")

    seen_ids = set()
    for index in range(count):
        chunk_id = row_ids[index]
        row_metadata = metadata[index]
        if not isinstance(chunk_id, str) or not chunk_id or chunk_id in seen_ids:
            invalid("marked lazy chunk IDs must be unique, nonempty strings")
        seen_ids.add(chunk_id)
        if (not isinstance(row_metadata, dict)
                or not _LAZY_METADATA_FIELDS.issubset(row_metadata)
                or row_metadata.get("chunk_id") != chunk_id
                or any(field in row_metadata for field in ("text", "content", "markdown", "body"))):
            invalid("marked lazy chunk metadata is invalid or contains payload text")
        if (row_metadata.get("std_no") is not None
                and not isinstance(row_metadata["std_no"], str)):
            invalid("lazy metadata std_no must be a string or None")
        if (row_metadata.get("std_id") is not None
                and not isinstance(row_metadata["std_id"], str)):
            invalid("lazy metadata std_id must be a string or None")
        if (not isinstance(row_metadata.get("evidence_kind"), str)
                or not isinstance(row_metadata.get("document_part"), str)
                or type(row_metadata.get("excluded_from_current_retrieval")) is not bool
                or (row_metadata.get("import_part_id") is not None
                    and not isinstance(row_metadata["import_part_id"], str))):
            invalid("marked lazy chunk metadata fields have invalid types")
    return _LazyChunkRows(chunks, row_ids, metadata, id_text_rows,
                          unique_text_count, error_type)


def tokenize(text: str) -> list[str]:
    """Use Chinese characters/bigrams, words and intact numbers/comparators.

    This deliberately avoids English whitespace-only splitting and punctuation
    stripping: 1.8, 18, <= and >= must remain distinguishable.
    """
    # 先做 NFKC 规范化（全角转半角等）并统一转小写，消除写法差异带来的匹配失败
    value = unicodedata.normalize("NFKC", text).lower()
    # 把 Unicode 数学符号 ≤ ≥ ≠ 统一替换为 ASCII 形式，与上面的正则保持一致
    value = value.replace("≤", "<=").replace("≥", ">=").replace("≠", "!=")
    tokens: list[str] = []  # 存放最终输出的词元列表
    for token in _TOKENS.findall(value):  # 逐段遍历正则切分出的原始词元
        if "\u3400" <= token[0] <= "\u9fff":  # 判断该词元是否以汉字开头，即中文连续片段
            tokens.extend(token)  # 中文片段保留全部单字（unigram）
            tokens.extend(token[i:i + 2] for i in range(len(token) - 1))  # 再补充相邻两字组成的二元组（bigram），补充短语信息
        else:  # 非中文词元（英文单词、数字、比较符）
            tokens.append(token)  # 原样保留，确保 1.8 / 18 / <= 之间仍然可区分
    return tokens  # 返回最终词元列表


class LocalIndex:
    """BM25 locally; add normalized cosine retrieval and RRF when configured.

    Hybrid scores are reciprocal-rank scores, not similarity probabilities.
    Remote embedding errors are raised; they never silently switch to BM25.
    ``prepared_vectors`` may provide a pre-normalized, read-only float32/float64
    matrix. It is validated in row blocks and attached without coercion or copy;
    when supplied, ``embeddings`` is used only for query vectors.
    """

    def __init__(self, chunks: Sequence[Chunk], embeddings=None, *,
                 candidate_pool: int | None = None, rrf_constant: int = 60,
                 reranker=None, rerank_error_handler=None,
                 prepared_vectors: np.ndarray | None = None,
                 bm25_directory=None):
        lazy_rows = _validated_lazy_chunk_sequence(chunks)
        if lazy_rows is None:
            # Keep the established eager behavior for ordinary uploaded chunks.
            self.chunks = [chunk for chunk in chunks if chunk.text.strip()]
            self._lazy_rows = None
        else:
            # The marked corpus has already applied LocalIndex's whitespace
            # rule. Preserve its exact order and defer Chunk/text decoding.
            self.chunks = chunks
            self._lazy_rows = lazy_rows
        self.embeddings = embeddings  # 向量模型（可选）；为 None 时退化为纯 BM25 检索
        if (candidate_pool is not None
                and (isinstance(candidate_pool, bool) or not isinstance(candidate_pool, int)
                     or candidate_pool < 1)):
            raise ValueError("candidate_pool 必须为正整数或 None。")
        if (isinstance(rrf_constant, bool) or not isinstance(rrf_constant, int)
                or rrf_constant < 0):
            raise ValueError("rrf_constant 必须为非负整数。")
        self.candidate_pool = candidate_pool
        self.rrf_constant = rrf_constant
        self.reranker = reranker
        self.rerank_error_handler = rerank_error_handler
        self._bm25_store = None
        self._owns_bm25_store = False
        if bm25_directory is None:
            source_texts = (chunk.text for chunk in self.chunks) if self._lazy_rows is None else (
                text for _, text in self._lazy_rows)
            self._counts = [Counter(tokenize(text)) for text in source_texts]
            self._lengths = [sum(count.values()) for count in self._counts]
            self._average_length = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
            self._document_frequency = Counter(term for count in self._counts for term in count)
        else:
            from .binary_bm25 import BinaryBM25Store
            self._bm25_store = BinaryBM25Store.open(bm25_directory, chunks=self.chunks)
            self._owns_bm25_store = True
            # Do not retain a second Python Counter/length/DF representation.
            self._counts = self._lengths = self._document_frequency = None
            self._average_length = self._bm25_store.average_length
        self._vectors: np.ndarray | None = None  # 文档向量矩阵（已归一化），初始为空，按需构建
        try:
            self._prepare_vectors(embeddings, prepared_vectors)
        except BaseException:
            self.close_bm25()
            raise

    def _prepare_vectors(self, embeddings, prepared_vectors):
        if prepared_vectors is not None:
            # Prepared matrices are already normalized and may be backed by a
            # read-only mmap. Validate in bounded row blocks and retain the
            # original object so attaching a large cache does not duplicate it.
            self.bind_prepared_vectors(prepared_vectors, embeddings)
        elif embeddings is not None and self.chunks:  # 仅在配置了向量模型且存在有效文档时才计算向量
            try:
                # 批量请求所有文档的向量表示
                raw = embeddings.embed_documents([chunk.text for chunk in self.chunks])
            except Exception as exc:
                # 向量服务异常不做静默降级：直接抛错，避免悄悄退回 BM25 导致结果口径不一致
                raise RuntimeError("文档向量调用失败；未切换为其他检索方式，请检查向量服务配置。") from exc
            self._vectors = self._document_vectors(raw)  # 校验并 L2 归一化文档向量

    @classmethod
    async def acreate(cls, chunks: Sequence[Chunk], embeddings=None, *,
                      candidate_pool: int | None = None, rrf_constant: int = 60,
                      reranker=None, rerank_error_handler=None,
                      prepared_vectors: np.ndarray | None = None,
                      bm25_directory=None) -> "LocalIndex":
        """Build local statistics off-loop; use async document embedding only if needed."""
        created = []

        def construct():
            index = cls(chunks, None, candidate_pool=candidate_pool,
                        rrf_constant=rrf_constant, reranker=reranker,
                        rerank_error_handler=rerank_error_handler,
                        bm25_directory=bm25_directory)
            created.append(index)
            return index

        try:
            index = await run_local(construct)
            await index._aprepare_vectors(embeddings, prepared_vectors)
            return index
        except BaseException:
            if created:
                await run_local(created[0].close_bm25)
            raise

    async def _aprepare_vectors(self, embeddings, prepared_vectors):
        if prepared_vectors is not None:
            # Validate the known provider width and the matrix together, then
            # publish both in one assignment after all checks have succeeded.
            await run_local(self.bind_prepared_vectors, prepared_vectors, embeddings)
            return
        self.embeddings = embeddings
        if embeddings is None or not self.chunks:
            return
        embed_documents = getattr(embeddings, "aembed_documents", None)
        if not callable(embed_documents):
            raise RuntimeError("异步索引构建要求向量客户端实现 aembed_documents。")
        try:
            raw = await embed_documents([chunk.text for chunk in self.chunks])
        except Exception as exc:
            raise RuntimeError("文档向量调用失败；未切换为其他检索方式，请检查向量服务配置。") from exc
        self._vectors = await run_local(self._document_vectors, raw)

    def close_bm25(self):
        """Close an owned mapping after callers have drained this index.

        Task-local shallow copies borrow the mapping; their resource owner is
        responsible for closing it. Memory-only upload indexes need no cleanup.
        """
        if self._owns_bm25_store and self._bm25_store is not None:
            self._bm25_store.close()

    def __copy__(self):
        # Every task-local shallow copy borrows immutable lexical data. Closing
        # a task must never invalidate another task's or the owner's mapping.
        borrowed = type(self).__new__(type(self))
        borrowed.__dict__.update(self.__dict__)
        borrowed._owns_bm25_store = False
        return borrowed

    @property
    def mode(self) -> str:
        # 对外暴露当前检索模式：配置了向量即为 BM25 + 稠密 + RRF 融合，否则仅 BM25
        return "bm25_dense_rrf" if self.embeddings is not None else "bm25"

    def _document_vectors(self, raw) -> np.ndarray:
        try:
            matrix = np.asarray(raw, dtype=np.float64)  # 将原始返回转换为 float64 矩阵，便于数值校验
        except (TypeError, ValueError) as exc:  # 传入的不是数值型数据
            raise ValueError("文档向量须为数值组成的二维数组。") from exc
        # 校验：必须是二维；行数等于文档数；特征维度不能为 0
        if matrix.ndim != 2 or matrix.shape[0] != len(self.chunks) or matrix.shape[1] == 0:
            raise ValueError("文档向量的条数或维度不正确。")
        if not np.isfinite(matrix).all():  # 禁止 NaN / Inf 等非有限数值
            raise ValueError("文档向量含非有限数值。")
        with np.errstate(over="ignore", invalid="ignore"):  # 抑制极端值下的溢出告警，统一由下方校验判断
            norms = np.linalg.norm(matrix, axis=1)  # 逐行求 L2 范数
        if not np.isfinite(norms).all() or np.any(norms == 0):  # 存在零向量或范数不可用
            raise ValueError("文档向量含零向量或无法归一化的数值。")
        return matrix / norms[:, None]  # 按行做 L2 归一化；归一化后点积即等价于余弦相似度

    def bind_prepared_vectors(self, matrix: np.ndarray, embeddings=None) -> None:
        """Attach one validated read-only matrix before publishing this index.

        This startup-only operation does not rebuild BM25 state. If validation
        fails, neither the query provider nor vector reference is changed.
        """
        if self._vectors is not None:
            raise ValueError("文档向量已绑定，不能重复绑定预建矩阵。")
        validated = self._validate_prepared_vectors(matrix, embeddings)
        self.embeddings, self._vectors = embeddings, validated

    def _validate_prepared_vectors(self, matrix: np.ndarray, embeddings=None) -> np.ndarray:
        """Validate a normalized, read-only matrix without converting or copying it."""
        if not isinstance(matrix, np.ndarray):
            raise ValueError("预建文档向量须为只读 NumPy 数组或内存映射数组。")
        if matrix.flags.writeable:
            raise ValueError("预建文档向量必须为只读数组。")
        if matrix.dtype not in (np.dtype(np.float32), np.dtype(np.float64)) or not matrix.dtype.isnative:
            raise ValueError("预建文档向量 dtype 仅支持原生字节序 float32 或 float64。")
        if (matrix.ndim != 2 or matrix.shape[0] != len(self.chunks)
                or matrix.shape[1] == 0):
            raise ValueError("预建文档向量的条数或维度不正确。")
        self._validate_prepared_dimensions(matrix, embeddings)

        # Keep temporary finite masks and norm vectors bounded independently
        # of corpus size. No full-size float64 coercion, norm, or mask is made.
        row_bytes = max(1, int(matrix.shape[1]) * matrix.dtype.itemsize)
        rows_per_block = max(1, (8 * 1024 * 1024) // row_bytes)
        tolerance = 1e-12 if matrix.dtype == np.dtype(np.float64) else 1e-6
        for start in range(0, matrix.shape[0], rows_per_block):
            block = matrix[start:start + rows_per_block]
            if not np.isfinite(block).all():
                raise ValueError("预建文档向量含非有限数值。")
            with np.errstate(over="ignore", invalid="ignore"):
                norms = np.linalg.norm(block, axis=1)
            if not np.isfinite(norms).all() or np.any(norms == 0):
                raise ValueError("预建文档向量含零向量或无法归一化的数值。")
            if not np.allclose(norms, 1.0, rtol=tolerance, atol=tolerance):
                raise ValueError("预建文档向量必须预先按行归一化。")
        return matrix

    def _validate_prepared_dimensions(self, matrix: np.ndarray, embeddings=None) -> None:
        """Check known query-model width without rescanning matrix contents."""
        expected_dimensions = getattr(embeddings, "dimensions", None)
        if expected_dimensions is not None:
            if (isinstance(expected_dimensions, (bool, np.bool_))
                    or not isinstance(expected_dimensions, (int, np.integer))
                    or int(expected_dimensions) != matrix.shape[1]):
                raise ValueError("预建文档向量维度与查询向量模型不一致。")

    def _bm25_scores(self, query: str) -> list[float]:
        if self._bm25_store is not None:
            return self._bm25_store.scores(query)
        scores = [0.0] * len(self.chunks)  # 初始化每个文档的 BM25 得分为 0
        if not self._average_length:  # 没有有效文档（平均长度为 0）时直接返回全零
            return scores
        n = len(self.chunks)  # 文档总数，用于计算 IDF
        k1, b = 1.5, 0.75  # BM25 经典超参数：k1 控制词频饱和速度，b 控制文档长度归一化强度
        for term in set(tokenize(query)):  # 查询词元先去重再遍历，避免重复词被重复累加
            frequency = self._document_frequency.get(term, 0)  # 该词元出现在多少个文档中
            if not frequency:  # 词表外的词元对检索无贡献，直接跳过
                continue
            idf = math.log1p((n - frequency + 0.5) / (frequency + 0.5))  # 平滑 IDF；log1p 保证非负且数值稳定
            for i, counts in enumerate(self._counts):  # 遍历所有文档，累加该词元的得分贡献
                tf = counts.get(term, 0)  # 词元在当前文档中的词频
                if tf:  # 该文档不含此词则无需计算
                    # 长度归一化分母 = 词频饱和项 + 文档长度相对平均长度的惩罚项
                    denominator = tf + k1 * (1 - b + b * self._lengths[i] / self._average_length)
                    scores[i] += idf * tf * (k1 + 1) / denominator  # 标准 BM25 单项得分累加
        return scores  # 返回全部文档的 BM25 得分

    def _cosine_scores(self, query: str) -> list[float]:
        try:
            raw = self.embeddings.embed_query(query)  # 调用向量模型获取查询向量
        except Exception as exc:
            raise RuntimeError("查询向量调用失败；未切换为其他检索方式，请检查向量服务配置。") from exc  # 同样不做静默降级
        return self._cosine_scores_from_vector(raw)

    def _cosine_scores_from_vector(self, raw) -> list[float]:
        """Validate and rank a returned vector without making a provider call."""
        try:
            vector = np.asarray(raw, dtype=np.float64)  # 转换为 float64 数组再做校验
        except (TypeError, ValueError) as exc:
            raise ValueError("查询向量须为一维数值数组。") from exc
        if vector.ndim != 1 or vector.shape[0] != self._vectors.shape[1]:  # 必须是一维，且维度与文档向量列数一致
            raise ValueError("查询向量维度与文档索引不一致。")
        if not np.isfinite(vector).all():  # 禁止 NaN / Inf
            raise ValueError("查询向量含非有限数值。")
        with np.errstate(over="ignore", invalid="ignore"):  # 抑制浮点溢出/非法运算告警，交由下方判断
            norm = np.linalg.norm(vector)  # 查询向量的 L2 范数
        if not np.isfinite(norm) or norm == 0:  # 零向量无法进行余弦计算
            raise ValueError("查询向量为零向量或无法归一化。")
        # 文档向量已归一化，故与"单位化后的查询向量"做点积即为余弦相似度；tolist 转回 Python 列表
        return (self._vectors @ (vector / norm)).tolist()

    @staticmethod
    def _positive_order(scores: list[float], limit: int) -> list[int]:
        # 非正的余弦匹配不提供有效稠密证据；
        # 而 BM25 始终是混合检索中一个正常且显式的组成部分，因此统一只取正值。
        # 排序规则：分数降序；分数相同时按下标升序，保证结果稳定、可复现；最后截取前 limit 个。
        return sorted((i for i, score in enumerate(scores) if score > 0),
                      key=lambda i: (-scores[i], i))[:limit]

    def _dense_candidates(self, bm25, dense, top_k):
        candidate_count = min(len(self.chunks), self.candidate_pool or max(20, top_k * 4))
        scores: dict[int, float] = {}
        for ranked in (self._positive_order(bm25, candidate_count),
                       self._positive_order(dense, candidate_count)):
            for rank, i in enumerate(ranked, 1):
                scores[i] = scores.get(i, 0.0) + 1.0 / (self.rrf_constant + rank)
        order = sorted(scores, key=lambda i: (-scores[i], i))[:candidate_count]
        return [SearchHit(self.chunks[i], scores[i]) for i in order]

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        # 参数校验：注意 bool 是 int 的子类，需先单独排除 True/False
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k 必须为正整数。")
        if not isinstance(query, str):  # 查询必须为字符串
            raise TypeError("检索问题必须为文本。")
        query = query.strip()  # 去掉首尾空白，避免空白字符影响分词
        if not query or not self.chunks:  # 空查询或索引中没有文档时，直接返回空结果
            return []
        top_k = min(top_k, len(self.chunks))  # 返回条数不能超过文档总数
        bm25 = self._bm25_scores(query)  # 计算 BM25 得分；混合模式下它仍作为其中一路召回
        if self.embeddings is None:  # 纯 BM25 模式
            order = self._positive_order(bm25, top_k)  # 按 BM25 正得分排序，取前 top_k 个下标
            return [SearchHit(self.chunks[i], float(bm25[i])) for i in order]  # 组装检索命中结果并返回

        dense = self._cosine_scores(query)  # 计算稠密余弦相似度得分
        # 默认保留原有宽候选策略；上传附件可显式固定候选池，便于稳定复现。
        candidates = self._dense_candidates(bm25, dense, top_k)
        if self.reranker is None or not candidates:
            return candidates[:top_k]
        try:
            ranked = self.reranker.rerank(
                query, [hit.chunk.text for hit in candidates],
                top_n=len(candidates))
        except RerankError as exc:
            if self.rerank_error_handler is not None:
                self.rerank_error_handler({"code": exc.code, "status_code": exc.status_code})
            # Keep the original RRF order and its score semantics on service errors.
            return candidates[:top_k]
        positions = [item.index for item in ranked]
        if (len(positions) != len(candidates)
                or set(positions) != set(range(len(candidates)))
                or any(not math.isfinite(float(item.score)) for item in ranked)):
            raise ValueError("重排返回的候选数、位置或得分不完整。")
        return [SearchHit(candidates[item.index].chunk, item.score) for item in ranked[:top_k]]

    async def asearch(self, query: str, top_k: int = 5) -> list[SearchHit]:
        """Async retrieval with bounded CPU work and native provider awaits."""
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k 必须为正整数。")
        if not isinstance(query, str):
            raise TypeError("检索问题必须为文本。")
        query = query.strip()
        if not query or not self.chunks:
            return []
        top_k = min(top_k, len(self.chunks))
        bm25 = await run_local(self._bm25_scores, query)
        if self.embeddings is None:
            order = await run_local(self._positive_order, bm25, top_k)
            return [SearchHit(self.chunks[i], float(bm25[i])) for i in order]

        embed_query = getattr(self.embeddings, "aembed_query", None)
        if not callable(embed_query):
            raise RuntimeError("异步检索要求向量客户端实现 aembed_query。")
        try:
            raw = await embed_query(query)
        except Exception as exc:
            raise RuntimeError("查询向量调用失败；未切换为其他检索方式，请检查向量服务配置。") from exc
        dense = await run_local(self._cosine_scores_from_vector, raw)
        candidates = await run_local(self._dense_candidates, bm25, dense, top_k)
        if self.reranker is None or not candidates:
            return candidates[:top_k]
        arerank = getattr(self.reranker, "arerank", None)
        if not callable(arerank):
            raise RuntimeError("异步检索要求重排客户端实现 arerank。")
        try:
            ranked = await arerank(query, [hit.chunk.text for hit in candidates],
                                   top_n=len(candidates))
        except RerankError as exc:
            if self.rerank_error_handler is not None:
                await run_local(self.rerank_error_handler,
                                {"code": exc.code, "status_code": exc.status_code})
            return candidates[:top_k]
        positions = [item.index for item in ranked]
        if (len(positions) != len(candidates)
                or set(positions) != set(range(len(candidates)))
                or any(not math.isfinite(float(item.score)) for item in ranked)):
            raise ValueError("重排返回的候选数、位置或得分不完整。")
        return [SearchHit(candidates[item.index].chunk, item.score) for item in ranked[:top_k]]
