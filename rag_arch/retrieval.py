# 模块说明：面向"用户上传文本"的轻量级内存检索实现，与标准规范基准测试解耦。
# 保留了课程 FlashRAG 检索器中"记录(record) / 检索(search)"分离的设计；
# 中文分词、参数校验以及可选的余弦 / RRF 融合属于本项目的改造。
# 全程不加载任何序列化索引或 pickle 文件。
"""Small in-memory retrieval for uploaded text, independent of the standards benchmark.

The course FlashRAG retriever's record/search separation is retained here. Chinese
tokenization, validation and optional cosine/RRF fusion are project adaptations.
No serialized index or pickle is loaded.
"""
from collections import Counter  # 计数器：统计文档内的词频、以及全局的文档频率
import math  # 数学函数：BM25 平滑 IDF 中用到 log1p
import re  # 正则表达式：用于把文本切分成词元
import unicodedata  # Unicode 规范化（NFKC），统一全角/半角等不同写法

import numpy as np  # 数值计算：向量归一化、点积与有限值校验

from .models import Chunk, SearchHit  # 项目内数据模型：文本块 Chunk、检索命中 SearchHit
from .reranking import RerankError


# 词元切分正则，按顺序匹配四类片段：
# 1) 中日韩汉字区间（含扩展A区 \u3400-\u4dbf 与基本区 \u4e00-\u9fff）；
# 2) 英文单词或下划线标识符 [a-z_]+；
# 3) 完整数字（可带正负号、小数、百分号或千分号），保证 1.8 不会被拆成 1 和 8；
# 4) 比较运算符（>= <= != 优先于单字符 > < =），避免被拆开。
_TOKENS = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+|[a-z_]+|[-+]?\d+(?:\.\d+)?(?:%|‰)?|>=|<=|!=|[><=]")


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
    """

    def __init__(self, chunks: list[Chunk], embeddings=None, *,
                 candidate_pool: int | None = None, rrf_constant: int = 60,
                 reranker=None, rerank_error_handler=None):
        # 过滤掉纯空白文本块，只保留有实际内容的 chunk，避免污染词频与平均长度
        self.chunks = [chunk for chunk in chunks if chunk.text.strip()]
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
        self._counts = [Counter(tokenize(chunk.text)) for chunk in self.chunks]  # 每个文档的词频统计（词 -> 出现次数）
        self._lengths = [sum(count.values()) for count in self._counts]  # 每个文档的词元总数，即文档长度
        self._average_length = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0  # 平均文档长度；无文档时为 0
        self._document_frequency = Counter(term for count in self._counts for term in count)  # 文档频率：每个词出现在多少个不同文档中
        self._vectors: np.ndarray | None = None  # 文档向量矩阵（已归一化），初始为空，按需构建
        if embeddings is not None and self.chunks:  # 仅在配置了向量模型且存在有效文档时才计算向量
            try:
                # 批量请求所有文档的向量表示
                raw = embeddings.embed_documents([chunk.text for chunk in self.chunks])
            except Exception as exc:
                # 向量服务异常不做静默降级：直接抛错，避免悄悄退回 BM25 导致结果口径不一致
                raise RuntimeError("文档向量调用失败；未切换为其他检索方式，请检查向量服务配置。") from exc
            self._vectors = self._document_vectors(raw)  # 校验并 L2 归一化文档向量

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

    def _bm25_scores(self, query: str) -> list[float]:
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
        candidate_count = min(len(self.chunks), self.candidate_pool or max(20, top_k * 4))
        scores: dict[int, float] = {}  # 文档下标 -> 融合后的 RRF 分数
        for ranked in (self._positive_order(bm25, candidate_count),  # 第一路：BM25 召回队列
                       self._positive_order(dense, candidate_count)):  # 第二路：稠密向量召回队列
            for rank, i in enumerate(ranked, 1):  # 名次从 1 开始计数
                scores[i] = scores.get(i, 0.0) + 1.0 / (self.rrf_constant + rank)
        order = sorted(scores, key=lambda i: (-scores[i], i))[:candidate_count]
        candidates = [SearchHit(self.chunks[i], scores[i]) for i in order]
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
