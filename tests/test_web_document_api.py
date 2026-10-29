"""Upload limits, document-bound tasks, original pages and report downloads."""
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
import pymupdf

from rag_arch.standards import StandardCorpus
from rag_arch.web.app import create_app
from rag_arch.web.standards_resources import StandardsResources
from rag_arch.web.task_service import TaskService


class DocumentAPITests(unittest.TestCase):
    def setUp(self):
        # The synthetic registered-plan IDs are test-only and have no source files
        # or identifiers from the internal project.
        registration = patch("rag_arch.web.documents.REGISTERED_PLAN_IDS",
                             ("EXAMPLE_A", "EXAMPLE_B", "EXAMPLE_C"))
        registration.start()
        self.addCleanup(registration.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        processed = self.root / "data/processed/plans_v0"
        raw = self.root / "data/raw/plans"
        processed.mkdir(parents=True)
        raw.mkdir(parents=True)
        rows, units = [], []
        for pid in ("EXAMPLE_A", "EXAMPLE_B", "EXAMPLE_C", "EXAMPLE_X"):
            name = f"{pid}_中文样例.pdf"
            with pymupdf.open() as pdf:
                pdf.new_page().insert_text((30, 30), pid + " sample")
                data = pdf.tobytes()
            (raw / name).write_bytes(data)
            rows.append({"plan_id": pid, "filename": name, "page_count": 1})
            units.append({"plan_id": pid, "unit_id": pid + "/u1", "text": "示例方案正文",
                          "source_path": "data/raw/plans/" + name, "locator": {"page": 1}})
        (processed / "summary.json").write_text(json.dumps({"plans": rows}), encoding="utf-8")
        (processed / "text_units.jsonl").write_text("\n".join(json.dumps(u) for u in units), encoding="utf-8")
        self.pdf = data
        self.requests = []

        def runner(request, progress, cancel):
            self.requests.append(request)
            if request["assistant_id"] == "plan":
                return {"text": "方案回应", "report_markdown": "# 真实执行器报告\n\n样例",
                        "report_history": {"answer": "供下一轮使用"}}
            return {"text": "聊天回应", "hits": [{"internal": "保留给历史，前端不用"}]}

        self.service = TaskService(runner=runner, archive_root=None,
                                   execution_modes={name: "model" for name in ("chat", "standards", "plan")})
        resources = StandardsResources(self.root, corpus=StandardCorpus([], []))
        self.client = TestClient(create_app(self.service, root=self.root, resources=resources, auth_enabled=False))
        self.client.__enter__()
        self.ws = self.service.workspace_id

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.tmp.cleanup()

    def upload(self, name="小文件.txt", content=b"hello"):
        return self.client.post("/api/uploads", params={"workspace_id": self.ws, "filename": name}, content=content)

    def send(self, **kwargs):
        return self.client.post("/api/tasks", json={"workspace_id": self.ws, "assistant_id": "chat",
            "request_id": uuid4().hex, "prompt": "问题", **kwargs})

    def done(self, response):
        self.assertEqual(response.status_code, 202, response.text)
        tid = response.json()["task_id"]
        for _ in range(100):
            task = self.service.get_task(tid)
            if task["status"] in ("completed", "failed", "cancelled"):
                return task
            time.sleep(.005)
        self.fail("controlled task did not finish")

    def test_raw_upload_identity_public_metadata_and_download(self):
        response = self.upload()
        self.assertEqual(response.status_code, 201)
        asset = response.json()
        self.assertEqual(asset["filename"], "小文件.txt")
        self.assertNotIn("path", asset)
        self.assertEqual(len(self.client.get("/api/uploads", params={"workspace_id": self.ws}).json()["uploads"]), 1)
        original = self.client.get(f"/api/uploads/{asset['asset_id']}/download", params={"workspace_id": self.ws})
        self.assertEqual(original.content, b"hello")
        self.assertEqual(self.client.get("/api/uploads", params={"workspace_id": "old"}).status_code, 409)

    def test_ten_mb_limit_for_headers_stream_and_combined_selection(self):
        self.assertEqual(self.upload(content=b"x", name="x.txt").status_code, 201)
        too_big = self.client.post("/api/uploads", params={"workspace_id": self.ws, "filename": "x.txt"},
                                  content=b"x", headers={"Content-Length": str(11 * 1024 * 1024)})
        self.assertEqual(too_big.status_code, 413)
        streamed = self.upload(content=iter([b"x" * (6 * 1024 * 1024)] * 2))
        self.assertEqual(streamed.status_code, 413)
        a = self.upload(content=b"x" * (6 * 1024 * 1024)).json()["asset_id"]
        b = self.upload(content=b"x" * (6 * 1024 * 1024)).json()["asset_id"]
        self.assertEqual(self.send(attachment_ids=[a, b]).status_code, 422)
        self.assertEqual(self.upload(content=b"").status_code, 422)
        self.assertEqual(self.upload(name="run.exe").status_code, 422)

    def test_optional_upload_and_history_isolation_by_file_selection(self):
        a = self.upload("a.txt").json()["asset_id"]
        b = self.upload("b.txt").json()["asset_id"]
        self.done(self.send(attachment_ids=[a]))
        self.done(self.send(attachment_ids=[b]))
        self.done(self.send(attachment_ids=[a]))
        self.assertEqual([len(r["history"]) for r in self.requests], [0, 0, 1])
        self.assertIn("hits", self.requests[2]["history"][0]["result"])
        task = self.done(self.send())
        self.assertEqual(self.requests[-1]["history"], [])
        self.assertNotIn("hits", task["result"])

    def test_plan_selection_and_report_download(self):
        listing = self.client.get("/api/plans").json()
        self.assertEqual([row["plan_id"] for row in listing["plans"]], ["EXAMPLE_A", "EXAMPLE_B", "EXAMPLE_C"])
        self.assertTrue(all("中文" in row["label"] for row in listing["plans"]))
        self.assertEqual(self.send(assistant_id="plan").status_code, 422)
        self.assertEqual(self.send(assistant_id="plan", registered_plan_id="EXAMPLE_X").status_code, 422)
        task = self.done(self.send(assistant_id="plan", registered_plan_id="EXAMPLE_B", plan_action="review"))
        self.assertEqual(task["registered_plan_id"], "EXAMPLE_B")
        self.assertNotIn("report_history", task["result"])
        report = self.client.get(f"/api/tasks/{task['task_id']}/download",
                                 params={"workspace_id": self.ws, "format": "md"})
        self.assertTrue(report.text.startswith("# 真实执行器报告"))
        self.assertIn("attachment;", report.headers["content-disposition"])

    def test_uploaded_and_registered_page_view_does_not_create_task(self):
        asset = self.upload("页面.pdf", self.pdf).json()["asset_id"]
        before = self.service.snapshot()["cursor"]
        for params in ({"asset": asset, "workspace_id": self.ws}, {"plan_id": "EXAMPLE_B"}):
            response = self.client.get("/api/documents/pages", params=params)
            self.assertEqual(response.status_code, 200, response.text)
            image = self.client.get(response.json()["pages"][0]["image_url"])
            self.assertTrue(image.content.startswith(b"\x89PNG"))
            bad = self.client.get("/api/documents/pages", params={**params, "page": 999})
            self.assertEqual(bad.status_code, 422)
            self.assertIn("当前页面暂不支持", bad.text)
        self.assertEqual(self.service.snapshot()["cursor"], before)
        self.assertEqual(self.client.get("/api/documents/pages", params={"asset": asset, "workspace_id": "old"}).status_code, 409)
        self.assertEqual(self.client.get("/api/documents/pages", params={"plan_id": "EXAMPLE_X"}).status_code, 404)


if __name__ == "__main__":
    unittest.main()
