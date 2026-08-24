import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.stage6_semantic_assist_runtime import (
    build_prompt,
    merge_missing_lines,
    read_live_stage5_lines,
    validate_result,
)


class Stage6SemanticAssistRuntimeTest(unittest.TestCase):
    def test_reads_canonical_live_stage5_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            path = run_dir / "L5" / "page_transcription.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "lines": [
                            {
                                "line_id": "line_002",
                                "reading_order": 2,
                                "devanagari_text": "द्वितीय",
                                "review_required": True,
                            },
                            {
                                "line_id": "line_001",
                                "reading_order": 1,
                                "raw_text": "प्रथम",
                                "review_required": True,
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            rows = read_live_stage5_lines(run_dir)
            self.assertEqual(
                [row["line_id"] for row in rows],
                ["line_001", "line_002"],
            )
            self.assertEqual(rows[0]["observed_htr"], "प्रथम")
            self.assertEqual(rows[1]["observed_htr"], "द्वितीय")

    def test_prompt_preserves_candidate_guardrails(self):
        prompt = build_prompt(
            [
                {
                    "line_id": "line_001",
                    "reading_order": 1,
                    "observed_htr": "श्रीगणेशायनमः",
                }
            ]
        )
        self.assertIn("NOT ground truth", prompt)
        self.assertIn("UNCALIBRATED", prompt)
        normalized_prompt = " ".join(prompt.split())
        self.assertIn(
            "Do NOT claim visual confirmation",
            normalized_prompt,
        )
        self.assertIn("SCHOLAR-ASSIST PACKET", prompt)

    def test_validation_counts_candidates(self):
        expected = [
            {"line_id": "line_001"},
            {"line_id": "line_002"},
        ]
        result = {
            "lines": [
                {
                    "line_id": "line_001",
                    "normalized_sanskrit_candidate": "एक",
                    "english_translation_candidate": "one",
                    "scholar_review_priority": "high",
                },
                {
                    "line_id": "line_002",
                    "normalized_sanskrit_candidate": "द्वि",
                    "english_translation_candidate": "two",
                    "scholar_review_priority": "medium",
                },
            ],
            "page_level": {
                "english_translation_candidate": "one two"
            },
        }
        v = validate_result(result, expected)
        self.assertEqual(v["status"], "ok")
        self.assertEqual(v["returned_lines"], 2)
        self.assertEqual(v["normalized_candidate_lines"], 2)
        self.assertEqual(v["translated_candidate_lines"], 2)
        self.assertTrue(v["page_translation_available"])

    def test_merge_missing_lines_preserves_order(self):
        expected = [
            {"line_id": "line_001"},
            {"line_id": "line_002"},
        ]
        base = {"lines": [{"line_id": "line_001"}]}
        continuation = {"lines": [{"line_id": "line_002"}]}
        merged = merge_missing_lines(base, continuation, expected)
        self.assertEqual(
            [row["line_id"] for row in merged["lines"]],
            ["line_001", "line_002"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
