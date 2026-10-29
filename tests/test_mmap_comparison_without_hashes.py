"""Direct comparison helpers do not rely on content digests."""
from types import SimpleNamespace
import unittest

from scripts import verify_bm25_mmap as bm25
from scripts import verify_corpus_mmap as corpus


class _FakeIndex:
    def __init__(self, score=0.5):
        self.chunks = [SimpleNamespace(chunk_id="A", locator={"std_no": "S"}),
                       SimpleNamespace(chunk_id="B", locator={})]
        self.local = SimpleNamespace(_bm25_scores=lambda query: [score, 0.0])

    def retrieve(self, query, *, mode, top_k, filters=None):
        return SimpleNamespace(query=query, mode=mode, reranked=False, raw_hits=[],
                               context_hits=[], missing_evidence=[])


class MmapComparisonWithoutHashesTests(unittest.TestCase):
    def test_query_labels_are_ordinal_and_reset_for_spawned_workers(self):
        bm25._set_query_ids(["private first query", "private second query"])
        self.assertEqual(bm25._query_id("private first query"), "query-0001")
        self.assertEqual(bm25._query_id("private second query"), "query-0002")
        bm25._set_query_ids(["private first query", "private second query"])
        self.assertEqual(bm25._query_id("private second query"), "query-0002")

    def test_bm25_signature_compares_actual_scores_and_order(self):
        first = bm25._signature(_FakeIndex(), "q")
        second = bm25._signature(_FakeIndex(score=0.6), "q")
        self.assertEqual(first["bm25_top30_ids"], ["A"])
        self.assertEqual(first["score_array"][0], float(0.5).hex())
        self.assertNotEqual(first, second)

    def test_corpus_semantic_snapshot_compares_actual_values(self):
        bm25._set_query_ids(["q"])
        first = corpus._semantic_snapshot(_FakeIndex(), ["q"])
        second = corpus._semantic_snapshot(_FakeIndex(score=0.6), ["q"])
        self.assertEqual(first["bm25_score_arrays"]["query-0001"][0], float(0.5).hex())
        self.assertNotEqual(first, second)
        self.assertIn("query-0001:filter-1", first["filtered_rrf_expansion"])


if __name__ == "__main__":
    unittest.main()
