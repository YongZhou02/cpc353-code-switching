"""Corruption tests for the CPC353 KWIC validator."""

from __future__ import annotations

import csv
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analysis.validation.validate_kwic import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SOURCE,
    SCHEMAS,
    ValidationError,
    ValidationPaths,
    classify_full_review_difference,
    validate,
)


class ValidatorCorruptionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.source = root / "source.csv"
        self.outputs = root / "outputs"
        self.evidence = root / "evidence"
        self.outputs.mkdir()
        shutil.copy2(DEFAULT_SOURCE, self.source)
        for filename in SCHEMAS:
            shutil.copy2(DEFAULT_OUTPUT_DIR / filename, self.outputs / filename)
        self.paths = ValidationPaths(self.source, self.outputs, self.evidence)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def read_output(self, filename: str) -> pd.DataFrame:
        return pd.read_csv(self.outputs / filename, dtype=str, encoding="utf-8-sig").fillna("")

    def write_output(self, filename: str, frame: pd.DataFrame) -> None:
        frame.to_csv(self.outputs / filename, index=False, encoding="utf-8-sig")

    def assert_validation_fails(self, expected_text: str) -> None:
        with self.assertRaisesRegex(ValidationError, expected_text):
            validate(self.paths)

    def test_baseline_passes(self) -> None:
        summary, review_table, pattern_table = validate(self.paths)
        self.assertEqual(summary["kwic_rows"], summary["expected_kwic_rows"])
        self.assertEqual(len(review_table), 27)
        self.assertGreaterEqual(len(pattern_table), 4)

    def test_corrupted_keyword_value_fails(self) -> None:
        frame = self.read_output("04_kwic_concordance.csv")
        frame.loc[0, "Keyword"] = "corrupted-keyword"
        self.write_output("04_kwic_concordance.csv", frame)
        self.assert_validation_fails("Keyword .* Node .* disagree")

    def test_corrupted_review_label_fails(self) -> None:
        frame = self.read_output("04_kwic_concordance.csv")
        frame.loc[0, "Review Switching Type"] = "Tag Switching"
        self.write_output("04_kwic_concordance.csv", frame)
        self.assert_validation_fails("does not match source label")

    def test_missing_kwic_row_fails(self) -> None:
        frame = self.read_output("04_kwic_concordance.csv").iloc[:-1].copy()
        self.write_output("04_kwic_concordance.csv", frame)
        self.assert_validation_fails("missing 1 expected match")

    def test_duplicate_kwic_row_fails(self) -> None:
        frame = self.read_output("04_kwic_concordance.csv")
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        self.write_output("04_kwic_concordance.csv", frame)
        self.assert_validation_fails("duplicate match")

    def test_mismatched_keyword_frequency_fails(self) -> None:
        frame = self.read_output("05_keyword_frequency.csv")
        frame.loc[0, "Occurrences"] = str(int(frame.loc[0, "Occurrences"]) + 1)
        self.write_output("05_keyword_frequency.csv", frame)
        self.assert_validation_fails("Statistic mismatch in 05_keyword_frequency.csv")

    def test_duplicate_review_id_fails(self) -> None:
        with self.source.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
        first_review = next(row for row in rows if row and row[0] == "REV-001")
        rows.append(first_review)
        with self.source.open("w", encoding="utf-8-sig", newline="") as handle:
            csv.writer(handle).writerows(rows)
        self.assert_validation_fails("Duplicate Review IDs")

    def test_real_full_review_content_change_fails(self) -> None:
        frame = self.read_output("04_kwic_concordance.csv")
        frame.loc[0, "Full Review"] = str(frame.loc[0, "Full Review"]) + " CORRUPTED"
        self.write_output("04_kwic_concordance.csv", frame)
        self.assert_validation_fails("real content differences")

    def test_line_endings_are_harmless_but_content_is_not(self) -> None:
        self.assertEqual(
            classify_full_review_difference("first\r\nsecond", "first\nsecond"),
            "line-ending-only",
        )
        self.assertEqual(
            classify_full_review_difference("first\nsecond", "first\nchanged"),
            "real-content-mismatch",
        )


if __name__ == "__main__":
    unittest.main()
