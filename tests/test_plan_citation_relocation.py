from __future__ import annotations

import unittest

from rag_arch.plan_documents import PlanDocument
from rag_arch.plan_review_agent import validate_plan_quotes


QUOTE = "支撑构件间距不得超过1.5米"


def make_document(second_page_text: str = f"第二页：{QUOTE}。") -> PlanDocument:
    return PlanDocument(
        "demo-plan",
        "plan.txt",
        [
            {"unit_id": "page-1", "text": "第一页：方案概述。", "locator": {"page": 1}},
            {"unit_id": "page-2", "text": second_page_text, "locator": {"page": 2}},
        ],
        source_kind="txt",
    )


class PlanCitationRelocationTests(unittest.TestCase):
    def test_relocated_quote_uses_actual_anchor_and_preserves_model_anchor(self):
        document = make_document()

        citations, issues = validate_plan_quotes(
            [{"source_ref": "S001", "quote": QUOTE}],
            document,
            {"page-1", "page-2"},
            required=True,
        )

        self.assertEqual(issues, [])
        self.assertEqual(len(citations), 1)
        citation = citations[0]
        self.assertEqual(citation["unit_id"], "page-2")
        self.assertEqual(citation["source_ref"], "S002")
        self.assertEqual(citation["model_source_ref"], "S001")
        self.assertEqual(citation["locator"], {"page": 2})
        self.assertTrue(citation["location_verified"])
        self.assertEqual(citation["resolution_basis"], "unique_quote_in_provided_units")

    def test_unknown_anchor_with_unique_quote_relocates_to_actual_anchor(self):
        document = make_document()

        citations, issues = validate_plan_quotes(
            [{"source_ref": "S999", "quote": QUOTE}],
            document,
            {"page-1", "page-2"},
        )

        self.assertEqual(issues, [])
        self.assertEqual(len(citations), 1)
        citation = citations[0]
        self.assertEqual(citation["unit_id"], "page-2")
        self.assertEqual(citation["source_ref"], "S002")
        self.assertEqual(citation["model_source_ref"], "S999")
        self.assertEqual(citation["locator"], {"page": 2})
        self.assertTrue(citation["location_verified"])
        self.assertEqual(citation["resolution_basis"], "unique_quote_in_provided_units")

    def test_quote_found_under_original_anchor_keeps_normal_resolution(self):
        document = make_document()

        citations, issues = validate_plan_quotes(
            [{"unit_id": "page-2", "source_ref": "S002", "quote": QUOTE}],
            document,
            {"page-1", "page-2"},
            required=True,
        )

        self.assertEqual(issues, [])
        self.assertEqual(len(citations), 1)
        citation = citations[0]
        self.assertEqual(citation["unit_id"], "page-2")
        self.assertEqual(citation["source_ref"], "S002")
        self.assertEqual(citation["locator"], {"page": 2})
        self.assertTrue(citation["location_verified"])
        self.assertNotIn("resolution_basis", citation)
        self.assertNotIn("model_source_ref", citation)

    def test_duplicate_quote_remains_unverified(self):
        document = make_document(f"第二页：{QUOTE}，再次出现：{QUOTE}。")

        citations, issues = validate_plan_quotes(
            [{"unit_id": "missing-unit", "source_ref": "S999", "quote": QUOTE}],
            document,
            {"page-1", "page-2"},
        )

        self.assertTrue(issues)
        self.assertEqual(len(citations), 1)
        self.assertFalse(citations[0]["location_verified"])
        self.assertEqual(citations[0]["resolution_basis"], "unverified_duplicate_quote")

    def test_quote_absent_from_document_remains_unverified(self):
        document = make_document()

        citations, issues = validate_plan_quotes(
            [{"unit_id": "missing-unit", "quote": "不在方案中的句子"}],
            document,
            {"page-1", "page-2"},
        )

        self.assertTrue(issues)
        self.assertEqual(len(citations), 1)
        self.assertFalse(citations[0]["location_verified"])
        self.assertEqual(citations[0]["resolution_basis"], "quote_not_located")


if __name__ == "__main__":
    unittest.main()
