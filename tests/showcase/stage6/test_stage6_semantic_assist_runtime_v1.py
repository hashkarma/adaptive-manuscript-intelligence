import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.stage6_semantic_assist_runtime import (
    build_line_batch_prompt,
    build_page_prompt,
    read_live_stage5_lines,
    split_batches,
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

    def test_split_batches_11_lines(self):
        rows = [
            {"line_id": f"line_{i:03d}"}
            for i in range(1, 12)
        ]
        batches = split_batches(rows, batch_size=4)
        self.assertEqual(
            [len(batch) for batch in batches],
            [4, 4, 3],
        )

    def test_line_prompt_preserves_guardrails(self):
        prompt = build_line_batch_prompt(
            [
                {
                    "line_id": "line_001",
                    "observed_htr": "श्रीगणेशायनमः",
                }
            ]
        )
        normalized = " ".join(prompt.split())
        self.assertIn("NOT ground truth", normalized)
        self.assertIn("UNCALIBRATED", normalized)
        self.assertIn("Do NOT claim visual confirmation", normalized)
        self.assertIn(
            "AI CANDIDATE — SCHOLAR VALIDATION PENDING",
            normalized,
        )

    def test_page_prompt_preserves_unverified_translation(self):
        prompt = build_page_prompt(
            [
                {
                    "line_id": "line_001",
                    "observed_htr": "x",
                    "normalized_sanskrit_candidate": "y",
                    "english_translation_candidate": "z",
                }
            ]
        )
        self.assertIn("NOT scholar verified", prompt)
        self.assertIn("ground_truth_available", prompt)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
