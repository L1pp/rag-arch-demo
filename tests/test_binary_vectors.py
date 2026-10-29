from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from rag_arch.binary_vectors import (
    BinaryVectorStore,
    BinaryVectorStoreError,
    convert_core_vectors,
    read_core_embedding_identity,
)
import rag_arch.binary_vectors as binary_vectors
from rag_arch.retrieval import LocalIndex


IDENTITY = {
    "base_url": "https://vectors.example/v1",
    "model": "test-embed-v1",
    "dimensions": 3,
}
FIRST = [0.12345678901234568, 0.2, 0.3]
SECOND = [1.0, -2.0, 4.0]
OLD = [0.5, 0.5, 0.5]


def _chunk(chunk_id: str, text: str):
    return SimpleNamespace(chunk_id=chunk_id, text=text)


def _payload(*, queries=None, identity=None, documents=None, dimensions=3):
    return {
        "schema_version": 1,
        "identity": IDENTITY if identity is None else identity,
        "vector_dimensions": dimensions,
        "documents": {
            "第一条规范": FIRST,
            "历史旧文本": OLD,
            "第二条规范": SECOND,
        } if documents is None else documents,
        "queries": {"过期查询": OLD} if queries is None else queries,
    }


class BinaryVectorStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "core_embeddings.json"
        self.output = self.root / "core_vectors"
        self.chunks = [
            _chunk("std:second", "第二条规范"),
            _chunk("std:first", "第一条规范"),
            _chunk("std:second-copy", "第二条规范"),
            _chunk("std:blank", " \n "),
        ]
        self.write_source(_payload())

    def tearDown(self):
        self.temp.cleanup()

    def write_source(self, payload):
        self.source.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False),
                               encoding="utf-8")

    def convert(self, dtype="float64"):
        return convert_core_vectors(self.source, self.output, identity=IDENTITY,
                                    chunks=self.chunks, dtype=dtype)

    def open_store(self):
        return BinaryVectorStore.open(self.output, identity=IDENTITY, chunks=self.chunks)

    def test_round_trip_is_readonly_mmap_exactly_normalized_and_current_order(self):
        version = self.convert()
        store = self.open_store()
        try:
            expected_raw = np.asarray([SECOND, FIRST, SECOND], dtype=np.float64)
            expected = LocalIndex(self.chunks)._document_vectors(expected_raw)
            self.assertIsInstance(store.vectors, np.memmap)
            self.assertFalse(store.vectors.flags.writeable)
            self.assertEqual((3, 3), store.vectors.shape)
            self.assertEqual("float64", str(store.vectors.dtype))
            np.testing.assert_array_equal(expected, store.vectors)
            self.assertEqual({"std:second": 0, "std:first": 1, "std:second-copy": 2},
                             store.row_by_chunk_id)
            self.assertEqual(["std:second", "std:first", "std:second-copy"],
                             store.manifest["row_ids"])
            metadata = (version / "metadata.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("过期查询", metadata)
            self.assertNotIn("历史旧文本", metadata)
            self.assertEqual(3 * 3 * 8, store.manifest["vector_bytes"])
            self.assertNotIn("vectors_sha256", store.manifest)
            self.assertEqual((version / "vectors.npy").stat().st_size,
                             store.manifest["vectors_file_bytes"])
            self.assertEqual(3, store.manifest["source_document_records_scanned"])
            self.assertEqual(1, store.manifest["source_query_records_scanned"])
        finally:
            store.close()
            store.close()

    def test_identity_and_exact_chunk_order_text_and_ids_are_bound(self):
        self.convert()
        with self.assertRaises(BinaryVectorStoreError):
            BinaryVectorStore.open(self.output, identity={**IDENTITY, "model": "other"},
                                   chunks=self.chunks)
        with self.assertRaises(BinaryVectorStoreError):
            BinaryVectorStore.open(self.output, identity=IDENTITY,
                                   chunks=[self.chunks[1], self.chunks[0], *self.chunks[2:]])
        with self.assertRaises(BinaryVectorStoreError):
            BinaryVectorStore.open(self.output, identity=IDENTITY,
                                   chunks=[_chunk("std:second", "改过的规范"), *self.chunks[1:]])
        with self.assertRaises(BinaryVectorStoreError):
            BinaryVectorStore.open(self.output, identity=IDENTITY,
                                   chunks=[_chunk("different-id", "第二条规范"), *self.chunks[1:]])
        with self.assertRaises(BinaryVectorStoreError):
            BinaryVectorStore.open(self.output, identity=IDENTITY,
                                   chunks=[self.chunks[0], _chunk("std:second", "第一条规范"),
                                           *self.chunks[2:]])

    def test_missing_current_embedding_text_does_not_publish(self):
        self.chunks.append(_chunk("std:new", "新规范当前语料"))
        with self.assertRaises(BinaryVectorStoreError):
            self.convert()
        self.assertFalse((self.output / "CURRENT").exists())
        self.assertEqual([], list((self.output / "versions").glob("v1-*")))

    def test_duplicate_keys_truncation_and_unknown_query_vector_fail_closed(self):
        invalid_sources = [
            '{"schema_version":1,"schema_version":1,"identity":{},"vector_dimensions":3,"documents":{},"queries":{}}',
            '{"schema_version":1,"identity":{"base_url":"https://vectors.example/v1","model":"test-embed-v1","dimensions":3,"model":"other"},"vector_dimensions":3,"documents":{},"queries":{}}',
            json.dumps(_payload())[:-5],
        ]
        for raw in invalid_sources:
            with self.subTest(raw=raw[:40]):
                self.source.write_text(raw, encoding="utf-8")
                with self.assertRaises(BinaryVectorStoreError):
                    self.convert()
                self.assertFalse((self.output / "CURRENT").exists())

        self.write_source(_payload(queries={"stale": [0.0, 0.0, 0.0]}))
        with self.assertRaises(BinaryVectorStoreError):
            self.convert()
        self.assertFalse((self.output / "CURRENT").exists())

    def test_bad_documents_even_when_not_needed_fail_closed(self):
        for bad_vector in ([0.0, 0.0, 0.0], [1.0, 2.0]):
            with self.subTest(vector=bad_vector):
                self.write_source(_payload(documents={
                    "第一条规范": FIRST,
                    "历史旧文本": bad_vector,
                    "第二条规范": SECOND,
                }))
                with self.assertRaises(BinaryVectorStoreError):
                    self.convert()
                self.assertFalse((self.output / "CURRENT").exists())
        invalid_number = (json.dumps({"schema_version": 1, "identity": IDENTITY,
                                      "vector_dimensions": 3, "documents": {}, "queries": {}},
                                     separators=(",", ":"))
                          .replace('"documents":{}',
                                   '"documents":{"历史旧文本":[1,1e400,1]}'))
        self.source.write_text(invalid_number, encoding="utf-8")
        with self.assertRaises(BinaryVectorStoreError):
            self.convert()
        self.assertFalse((self.output / "CURRENT").exists())

    def test_float64_rounding_matches_json_loads_and_localindex(self):
        raw = ('{"schema_version":1,"identity":' + json.dumps(IDENTITY)
               + ',"vector_dimensions":3,"documents":{"第二条规范":'
               + '[0.123456789012345678901,0.200000000000000011,0.3],'
               + '"第一条规范":[1,2,3]},"queries":{}}')
        self.source.write_text(raw, encoding="utf-8")
        expected_raw = json.loads(raw)["documents"]
        chunks = [_chunk("second", "第二条规范"), _chunk("first", "第一条规范")]
        convert_core_vectors(self.source, self.output, identity=IDENTITY, chunks=chunks)
        store = BinaryVectorStore.open(self.output, identity=IDENTITY, chunks=chunks)
        try:
            expected = LocalIndex(chunks)._document_vectors(
                np.asarray([expected_raw[chunk.text] for chunk in chunks], dtype=np.float64))
            np.testing.assert_array_equal(expected, store.vectors)
        finally:
            store.close()

    def test_float32_is_explicit_and_old_versions_remain_mapped(self):
        old_version = self.convert()
        old_store = self.open_store()
        try:
            float32_version = self.convert(dtype="float32")
            new_store = self.open_store()
            try:
                self.assertNotEqual(old_version, float32_version)
                self.assertTrue(old_version.exists())
                self.assertEqual("float32", str(new_store.vectors.dtype))
                self.assertFalse(new_store.vectors.flags.writeable)
                self.assertTrue(np.allclose(np.linalg.norm(new_store.vectors, axis=1), 1.0,
                                            atol=5e-6, rtol=5e-6))
                query = np.asarray([0.25, -0.5, 1.0], dtype=np.float64)
                old_scores = old_store.vectors @ query
                new_scores = new_store.vectors @ query
                np.testing.assert_allclose(new_scores, old_scores, rtol=5e-8, atol=5e-8)
                self.assertEqual(np.argsort(-old_scores).tolist(), np.argsort(-new_scores).tolist())
                self.assertTrue(np.isfinite(old_store.vectors).all())
            finally:
                new_store.close()
        finally:
            old_store.close()

    def test_converted_vectors_open(self):
        self.convert()
        with self.open_store() as store:
            self.assertEqual((3, 3), store.vectors.shape)

    def test_known_width_rejects_oversized_vector_without_reading_the_tail(self):
        prefix = ('{"schema_version":1,"identity":'
                  + json.dumps(IDENTITY, separators=(",", ":"))
                  + ',"vector_dimensions":3,"documents":{"第一条规范":[0.1,0.2,0.3,')
        # A large syntactically valid tail must not be materialized into a
        # Python list after the fourth value already exceeds the known width.
        self.source.write_text(prefix + ",".join("1" for _ in range(100_000))
                               + ']},"queries":{}}', encoding="utf-8")
        original_parse = binary_vectors.ijson.basic_parse
        number_events = []

        def observed_parse(*args, **kwargs):
            for event in original_parse(*args, **kwargs):
                if event[0] == "number":
                    number_events.append(event[1])
                yield event

        with patch("rag_arch.binary_vectors.ijson.basic_parse", side_effect=observed_parse):
            with self.assertRaisesRegex(BinaryVectorStoreError, "超过已知embedding维度"):
                self.convert()
        self.assertEqual([1, 3, 3, 0.1, 0.2, 0.3, 1], number_events)
        self.assertFalse((self.output / "CURRENT").exists())

    def test_failed_reconversion_keeps_the_previous_current_pointer(self):
        first_version = self.convert()
        current_before = (self.output / "CURRENT").read_text(encoding="ascii")
        self.source.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
        with self.assertRaises(BinaryVectorStoreError):
            self.convert()
        self.assertEqual(current_before, (self.output / "CURRENT").read_text(encoding="ascii"))
        self.assertTrue(first_version.is_dir())

    def test_cli_does_not_need_settings_or_construct_a_provider(self):
        from scripts import convert_core_vectors as cli

        with patch.object(cli, "load_corpus", return_value=object()), \
                patch.object(cli, "EvidenceIndex",
                             return_value=SimpleNamespace(chunks=self.chunks)), \
                redirect_stdout(StringIO()):
            self.assertEqual(0, cli.main(["--source", str(self.source),
                                         "--output", str(self.output)]))
        self.assertTrue((self.output / "CURRENT").is_file())
        self.source.unlink()
        with self.open_store() as store:
            self.assertEqual((3, 3), store.vectors.shape)

    def test_corrupt_metadata_extra_row_and_invalid_current_are_rejected(self):
        version = self.convert()
        metadata = version / "metadata.jsonl"
        with metadata.open("a", encoding="utf-8") as handle:
            handle.write('{"chunk_id":"extra","text":"extra"}\n')
        manifest_path = version / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["metadata_bytes"] = metadata.stat().st_size
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(BinaryVectorStoreError):
            self.open_store()
        (self.output / "CURRENT").write_text("..\\outside\n", encoding="ascii")
        with self.assertRaises(BinaryVectorStoreError):
            self.open_store()

    def test_nonfinite_binary_matrix_is_rejected_and_mapping_is_released(self):
        version = self.convert()
        matrix = np.load(version / "vectors.npy", mmap_mode="r+", allow_pickle=False)
        matrix[0, 0] = np.nan
        matrix.flush()
        del matrix
        with self.assertRaises(BinaryVectorStoreError):
            self.open_store()
        (version / "vectors.npy").replace(version / "renamed.npy")

    def test_close_can_be_retried_after_exported_memoryview_is_released(self):
        self.convert()
        store = self.open_store()
        exported = memoryview(store.vectors._mmap)
        try:
            with self.assertRaises(BufferError):
                store.close()
            self.assertFalse(store._closed)
        finally:
            exported.release()
        store.close()
        store.close()
        self.assertTrue(store._closed)

    def test_output_junction_is_rejected_before_creating_version_files(self):
        if not hasattr(Path, "is_junction"):
            self.skipTest("当前Python未提供Path.is_junction")
        with patch.object(Path, "is_junction", return_value=True):
            with self.assertRaises(BinaryVectorStoreError):
                self.convert()
        self.assertFalse(self.output.exists())

    def test_failure_cleanup_does_not_remove_redirected_temporary_directory(self):
        original_check = binary_vectors._is_symlink_or_junction

        def redirected(path):
            return path.name.startswith(".tmp-") or original_check(path)

        with (patch.object(binary_vectors, "_scan_source",
                           side_effect=BinaryVectorStoreError("模拟构建失败")),
              patch.object(binary_vectors, "_is_symlink_or_junction", side_effect=redirected),
              patch.object(binary_vectors.shutil, "rmtree") as remove):
            with self.assertRaisesRegex(BinaryVectorStoreError, "模拟构建失败"):
                self.convert()
            remove.assert_not_called()

    def test_failure_cleanup_removes_only_verified_temporary_directory(self):
        with (patch.object(binary_vectors, "_scan_source",
                           side_effect=BinaryVectorStoreError("模拟构建失败")),
              patch.object(binary_vectors.shutil, "rmtree") as remove):
            with self.assertRaisesRegex(BinaryVectorStoreError, "模拟构建失败"):
                self.convert()
            remove.assert_called_once()
            target = remove.call_args.args[0]
            self.assertEqual((self.output / "versions").resolve(), target.resolve().parent)
            self.assertRegex(target.name, r"^\.tmp-[0-9a-f]{32}$")

    def test_legacy_manifest_digest_is_ignored(self):
        version = self.convert()
        manifest_path = version / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["vectors_sha256"] = "legacy-unchecked"
        manifest["source_stat_before"] = {"legacy": "ignored"}
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        store = self.open_store()
        try:
            self.assertEqual((3, 3), store.vectors.shape)
        finally:
            store.close()

    def test_identity_reader_streams_expected_cache_identity(self):
        self.assertEqual(IDENTITY, read_core_embedding_identity(self.source))


if __name__ == "__main__":
    unittest.main()
