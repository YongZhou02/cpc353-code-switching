"""Rigorous, read-only validation of the CPC353 KWIC analysis outputs.

Run from the repository root:
    python analysis/validation/validate_kwic.py

The validator independently recomputes exported statistics and KWIC matches.
It writes only to analysis/validation/evidence.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TypedDict

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "data" / "CPC353_dataset_v3 - Self-Collected Data.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "CPC353_kwic_outputs"
DEFAULT_EVIDENCE_DIR = Path(__file__).resolve().parent / "evidence"
SWITCH_TYPES = ("Intra-sentential", "Inter-sentential", "Tag Switching")
DEFAULT_KEYWORDS = (
    "sedap", "makan", "murah", "pedas", "lepak", "kedai",
    "best", "good", "nice", "service", "food", "delicious",
)
context_chars = 70

SCHEMAS = {
    "01_language_distribution.csv": [
        "Language Classification", "Review Count", "Percentage of All Reviews",
    ],
    "02_switching_type_distribution.csv": [
        "Switching Type", "Review Count", "Percentage of Code-Switched Reviews",
    ],
    "03_switching_type_combinations.csv": [
        "Label Combination", "Review Count", "Percentage of Code-Switched Reviews",
    ],
    "04_kwic_concordance.csv": [
        "Review ID", "Keyword", "Left Context", "Node", "Right Context",
        "Review Switching Type", "Full Review", "Manual Interpretation",
        "Exclude Reason (if any)",
    ],
    "05_keyword_frequency.csv": [
        "Keyword", "Occurrences", "Reviews Containing Keyword",
        "Percentage of Code-Switched Reviews",
    ],
}


class ValidationError(AssertionError):
    """Raised when an exported result disagrees with source-derived evidence."""


class AuditSummary(TypedDict):
    source_reviews: int
    code_switched_reviews: int
    kwic_rows: int
    expected_kwic_rows: int
    kwic_review_ids: int
    keyword_node_mismatches: int
    review_label_mismatches: int
    non_code_switched_rows: int
    missing_matches: int
    duplicate_matches: int
    unexpected_matches: int
    substring_only_matches: int
    exact_full_review_matches: int
    line_ending_only_differences: int
    trailing_whitespace_only_differences: int
    real_content_mismatches: int
    type_counts: dict[str, int]
    type_memberships: int
    combination_counts: dict[str, int]
    language_counts: dict[str, int]
    keyword_occurrences: dict[str, int]
    keyword_review_counts: dict[str, int]


@dataclass(frozen=True)
class ValidationPaths:
    source: Path = DEFAULT_SOURCE
    output_dir: Path = DEFAULT_OUTPUT_DIR
    evidence_dir: Path = DEFAULT_EVIDENCE_DIR


@dataclass(frozen=True)
class MatchKey:
    review_id: str
    keyword: str
    start: int
    end: int


# Original 27 examples, revised to follow the team's loanword policy.
KWIC_SELECTIONS = [
    ("REV-002", "sedap", 1, "Clear code-switching evidence", "English noun phrase is followed by Malay 'semuanya sedap' within one clause."),
    ("REV-017", "sedap", 1, "Clear code-switching evidence", "English invitation frame is followed by Malay evaluative phrase 'paling sedap'."),
    ("REV-033", "makan", 1, "Uncertain", "The clause is mixed, but 'best' is excluded as a common loanword; other English material still suggests mixing."),
    ("REV-045", "makan", 1, "Clear code-switching evidence", "Malay predicate 'boleh makan' occurs inside an English clause."),
    ("REV-065", "murah", 1, "Uncertain", "Mostly Malay evaluative words linked by 'and', which the team treats as a common loanword; insufficient alone."),
    ("REV-066", "murah", 1, "Clear code-switching evidence", "Malay 'Sangat murah' follows a longer English clause within the same segment."),
    ("REV-082", "pedas", 1, "Clear code-switching evidence", "Malay adjective 'pedas' fills the English construction 'not only ... but'."),
    ("REV-126", "pedas", 1, "Clear code-switching evidence", "English comment contains Malay 'tak tahan pedas', followed by English 'anywhere'."),
    ("REV-037", "lepak", 1, "Clear code-switching evidence", "English frame contains Malay 'utk lepak' and 'nyanyi'; evidence does not depend on common 'and'."),
    ("REV-074", "lepak", 1, "Clear code-switching evidence", "English frame 'Nice place to' is completed by Malay verbs 'lepak' and 'makan'."),
    ("REV-051", "service", 1, "Common loanword / uncertain", "'service' is excluded as a common loanword, so it is not evidence by itself."),
    ("REV-103", "nice", 1, "Clear code-switching evidence", "A Malay opening ('Harga ... berbaloi') is followed by an extended English evaluation."),
    ("REV-069", "murah", 1, "Uncertain", "Malay adjectives linked only by common loanword 'and'; do not force a classification."),
    ("REV-180", "pedas", 1, "Clear code-switching evidence", "English copular frame 'It's' takes the Malay adjective 'pedas'."),
    ("REV-001", "makan", 1, "Monolingual occurrence", "Immediate segment is Malay; the review switches to English elsewhere."),
    ("REV-020", "sedap", 1, "Monolingual occurrence", "Standalone Malay phrase; review-level label must not classify this hit."),
    ("REV-002", "food", 1, "Monolingual occurrence", "Immediate sentence is English; the review is mixed elsewhere."),
    ("REV-018", "good", 1, "Monolingual occurrence", "Immediate sentence is English; 'good' is not a switching event."),
    ("REV-020", "service", 1, "Common loanword / monolingual", "Immediate sentence is English and 'service' is excluded as a common loanword."),
    ("REV-017", "pedas", 1, "Food name", "'Asam Pedas Johor' is a specific dish name and is excluded under team policy."),
    ("REV-030", "pedas", 1, "Food name", "'assam pedas' names a dish; this hit is not sufficient evidence of switching."),
    ("REV-105", "kedai", 1, "Borrowing / uncertain", "'kedai' may be insertion or established Malaysian usage; retain as uncertain."),
    ("REV-007", "makan", 1, "Borrowing / uncertain", "'makan-makan' may be Malay insertion or an established Malaysian-English expression."),
    ("REV-055", "lepak", 1, "Metalinguistic / uncertain", "The writer discusses the word 'lepak' itself rather than simply using it."),
    ("REV-161", "lepak", 1, "Borrowing / uncertain", "'lepak' is followed by its English gloss 'hanging out'; borrowing status is debatable."),
    ("REV-088", "kedai", 1, "Borrowing / uncertain", "'kedai' fills an English noun slot but is common in Malaysian English."),
]

# Exact corpus excerpts, kept separate from keyword-frequency evidence.
PATTERN_EXAMPLES = [
    ("REV-044", "Inter-sentential switching", "Sotong keras macam batu , ayam keras macam sotong…. The food is just so so and doesn’t bring any woa effect after waited 130 mint of long queue … first and last.", "A Malay sentence is followed by a separate English sentence."),
    ("REV-123", "Inter-sentential switching", "Excellent service.Makanan sedap.Reasonable price.", "English, Malay, and English occur in successive sentence-like units."),
    ("REV-037", "Tag switching", "But overall, okay lah.", "Malay discourse particle 'lah' is attached to an otherwise English expression."),
    ("REV-097", "Tag switching", "the food so delicious and yummy good..good must try the friedmeat,soupnya juga sedap,kankongnya yummy trylah", "Malay particle '-lah' is attached to English 'try'; the rest of the review is also mixed."),
    ("REV-142", "Not English-Malay tag switching (exclude as Tag evidence)", "but sometimes u need a good food for 'makan besar' right ?", "The final English 'right?' is an English tag attached to English material; it does not demonstrate English-Malay tag switching."),
]


def normalize_line_endings(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def normalize_space(text: str) -> str:
    return " ".join(normalize_line_endings(text).split())


def parse_labels(value: str) -> tuple[str, ...]:
    values = {item.strip().casefold() for item in re.split(r"[,;/|]", value) if item.strip()}
    if not values:
        raise ValidationError("Code-switched review has no switching label")
    unknown = values - {label.casefold() for label in SWITCH_TYPES}
    if unknown:
        raise ValidationError(f"Unknown switching label(s): {sorted(unknown)}")
    return tuple(label for label in SWITCH_TYPES if label.casefold() in values)


def find_header_row(csv_path: Path) -> int:
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.reader(handle)):
            if row and row[0].strip() == "Review ID":
                return row_number
    raise ValidationError(f"Cannot find 'Review ID' header in {csv_path}")


def load_source(source_path: Path) -> pd.DataFrame:
    data = pd.read_csv(source_path, skiprows=find_header_row(source_path), dtype=str, encoding="utf-8-sig").fillna("")
    required = {"Review ID", "Review Text", "Language Classification", "Code-Switching Type"}
    missing = required - set(data.columns)
    if missing:
        raise ValidationError(f"Source dataset is missing columns: {sorted(missing)}")
    ids = data["Review ID"].astype(str).str.strip()
    reviews = data.loc[ids.str.fullmatch(r"REV-\d+")].copy()
    review_ids = reviews["Review ID"].astype(str).str.strip()
    duplicate_mask = review_ids.duplicated(keep=False)
    if bool(duplicate_mask.any()):
        duplicates = sorted(set(review_ids.loc[duplicate_mask].tolist()))
        raise ValidationError(f"Duplicate Review IDs in source dataset: {duplicates}")
    if reviews.empty:
        raise ValidationError("Source dataset contains no valid REV-xxx reviews")
    if bool(reviews["Review Text"].astype(str).eq("").any()):
        raise ValidationError("Source dataset contains blank Review Text values")
    return reviews


def read_output(output_dir: Path, filename: str) -> pd.DataFrame:
    path = output_dir / filename
    if not path.is_file():
        raise ValidationError(f"Missing output file: {path}")
    frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
    actual_columns = list(frame.columns)
    if actual_columns != SCHEMAS[filename]:
        raise ValidationError(f"Schema mismatch in {filename}: expected {SCHEMAS[filename]}, found {actual_columns}")
    return frame


def source_lookups(source: pd.DataFrame) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    ids = source["Review ID"].astype(str).str.strip().tolist()
    texts = source["Review Text"].astype(str).tolist()
    languages = source["Language Classification"].astype(str).str.strip().tolist()
    labels = source["Code-Switching Type"].astype(str).str.strip().tolist()
    return dict(zip(ids, texts)), dict(zip(ids, languages)), dict(zip(ids, labels))


def code_switched_reviews(source: pd.DataFrame) -> pd.DataFrame:
    language = source["Language Classification"].astype(str).str.strip()
    return source.loc[language.str.casefold().eq("code-switched")].copy()


def expected_matches(source: pd.DataFrame, keywords: Iterable[str]) -> list[MatchKey]:
    expected: list[MatchKey] = []
    for _, review in code_switched_reviews(source).iterrows():
        review_id = str(review["Review ID"]).strip()
        text = str(review["Review Text"])
        for keyword in keywords:
            pattern = re.compile(rf"(?<!\w){re.escape(keyword)}(?!\w)", re.IGNORECASE)
            for match in pattern.finditer(text):
                expected.append(MatchKey(review_id, keyword.casefold(), match.start(), match.end()))
    return expected


def locate_output_match(row: pd.Series, source_text: str) -> tuple[re.Match[str] | None, bool]:
    node = str(row["Node"])
    left_expected = str(row["Left Context"])
    right_expected = str(row["Right Context"])
    candidates: list[re.Match[str]] = []
    for match in re.finditer(re.escape(node), source_text, re.IGNORECASE):
        left = normalize_space(source_text[max(0, match.start() - context_chars):match.start()])
        right = normalize_space(source_text[match.end():match.end() + context_chars])
        if left == left_expected and right == right_expected:
            candidates.append(match)
    if len(candidates) != 1:
        return None, False
    match = candidates[0]
    before = source_text[match.start() - 1:match.start()]
    after = source_text[match.end():match.end() + 1]
    complete_word = not (
        (before and (before.isalnum() or before == "_"))
        or (after and (after.isalnum() or after == "_"))
    )
    return match, complete_word


def classify_full_review_difference(exported: str, source: str) -> str:
    if exported == source:
        return "exact"
    exported_lf = normalize_line_endings(exported)
    source_lf = normalize_line_endings(source)
    if exported_lf == source_lf:
        return "line-ending-only"
    if exported_lf.rstrip() == source_lf.rstrip():
        return "trailing-whitespace-only"
    return "real-content-mismatch"


def compare_table(filename: str, actual: pd.DataFrame, expected: pd.DataFrame, key_column: str) -> None:
    errors: list[str] = []
    actual_keys = actual[key_column].astype(str).tolist()
    expected_keys = expected[key_column].astype(str).tolist()
    if len(actual_keys) != len(set(actual_keys)):
        errors.append(f"duplicate keys in exported {key_column}")
    if set(actual_keys) != set(expected_keys):
        errors.append(
            f"keys differ; missing={sorted(set(expected_keys) - set(actual_keys))}, "
            f"unexpected={sorted(set(actual_keys) - set(expected_keys))}"
        )
    actual_indexed = actual.set_index(key_column)
    expected_indexed = expected.set_index(key_column)
    for key in sorted(set(actual_keys) & set(expected_keys)):
        for column in expected.columns:
            if column == key_column:
                continue
            actual_value = actual_indexed.at[key, column]
            expected_value = expected_indexed.at[key, column]
            if column.startswith("Percentage"):
                try:
                    agrees = abs(float(str(actual_value)) - float(str(expected_value))) < 0.005
                except (TypeError, ValueError):
                    agrees = False
            else:
                agrees = str(actual_value) == str(expected_value)
            if not agrees:
                errors.append(
                    f"{key!r} {column}: exported={actual_value!r}, recomputed={expected_value!r}"
                )
    if errors:
        raise ValidationError(f"Statistic mismatch in {filename}: " + "; ".join(errors))


def recompute_statistics(
    source: pd.DataFrame,
    expected: list[MatchKey],
    keywords: tuple[str, ...],
) -> tuple[dict[str, pd.DataFrame], dict[str, int], dict[str, int]]:
    total = len(source)
    cs = code_switched_reviews(source)
    cs_count = len(cs)
    language_counts = source["Language Classification"].astype(str).str.strip().value_counts()
    language = pd.DataFrame({
        "Language Classification": language_counts.index.astype(str),
        "Review Count": language_counts.astype(int).tolist(),
    })
    language["Percentage of All Reviews"] = (
        100 * language["Review Count"].astype(int) / total
    ).round(2)

    parsed = cs["Code-Switching Type"].astype(str).map(parse_labels)
    type_counts = {label: sum(label in labels for labels in parsed) for label in SWITCH_TYPES}
    switching = pd.DataFrame({
        "Switching Type": SWITCH_TYPES,
        "Review Count": [type_counts[label] for label in SWITCH_TYPES],
    })
    switching["Percentage of Code-Switched Reviews"] = (
        100 * switching["Review Count"].astype(int) / cs_count
    ).round(2)

    combination_counts: Counter[str] = Counter(
        " + ".join(parse_labels(str(value)))
        for value in cs["Code-Switching Type"].tolist()
    )
    combinations = pd.DataFrame([
        {
            "Label Combination": combination,
            "Review Count": count,
            "Percentage of Code-Switched Reviews": round(100 * count / cs_count, 2),
        }
        for combination, count in combination_counts.items()
    ])

    occurrence_counts = Counter(match.keyword for match in expected)
    review_sets: dict[str, set[str]] = {keyword.casefold(): set() for keyword in keywords}
    for match in expected:
        review_sets.setdefault(match.keyword, set()).add(match.review_id)
    keyword_frequency = pd.DataFrame([
        {
            "Keyword": keyword.casefold(),
            "Occurrences": occurrence_counts[keyword.casefold()],
            "Reviews Containing Keyword": len(review_sets[keyword.casefold()]),
            "Percentage of Code-Switched Reviews": round(
                100 * len(review_sets[keyword.casefold()]) / cs_count, 2
            ),
        }
        for keyword in keywords
    ])
    return (
        {
            "01_language_distribution.csv": language,
            "02_switching_type_distribution.csv": switching,
            "03_switching_type_combinations.csv": combinations,
            "05_keyword_frequency.csv": keyword_frequency,
        },
        type_counts,
        dict(combination_counts),
    )


def validate(
    paths: ValidationPaths,
    keywords: tuple[str, ...] = DEFAULT_KEYWORDS,
) -> tuple[AuditSummary, pd.DataFrame, pd.DataFrame]:
    source = load_source(paths.source)
    outputs = {filename: read_output(paths.output_dir, filename) for filename in SCHEMAS}
    kwic = outputs["04_kwic_concordance.csv"]
    texts, languages, labels = source_lookups(source)
    cs = code_switched_reviews(source)
    expected = expected_matches(source, keywords)

    recomputed, type_counts, combination_counts = recompute_statistics(source, expected, keywords)
    key_columns = {
        "01_language_distribution.csv": "Language Classification",
        "02_switching_type_distribution.csv": "Switching Type",
        "03_switching_type_combinations.csv": "Label Combination",
        "05_keyword_frequency.csv": "Keyword",
    }
    for filename, expected_table in recomputed.items():
        compare_table(filename, outputs[filename], expected_table, key_columns[filename])

    errors: list[str] = []
    actual_matches: list[MatchKey] = []
    keyword_node_mismatches = 0
    review_label_mismatches = 0
    non_code_switched_rows = 0
    substring_only = 0
    difference_counts: Counter[str] = Counter()

    for row_number, (_, row) in enumerate(kwic.iterrows(), start=2):
        review_id = str(row["Review ID"]).strip()
        keyword = str(row["Keyword"])
        node = str(row["Node"])
        if review_id not in texts:
            errors.append(f"KWIC row {row_number}: unknown Review ID {review_id!r}")
            continue
        if languages[review_id].casefold() != "code-switched":
            non_code_switched_rows += 1
            errors.append(f"KWIC row {row_number}: {review_id} is not Code-switched")
        if keyword.casefold() != node.casefold():
            keyword_node_mismatches += 1
            errors.append(f"KWIC row {row_number}: Keyword {keyword!r} and Node {node!r} disagree")
        exported_label = str(row["Review Switching Type"]).strip()
        if exported_label != labels[review_id]:
            review_label_mismatches += 1
            errors.append(
                f"KWIC row {row_number}: label {exported_label!r} does not match "
                f"source label {labels[review_id]!r} for {review_id}"
            )

        source_text = texts[review_id]
        match, complete_word = locate_output_match(row, source_text)
        if match is None:
            errors.append(f"KWIC row {row_number}: node/context cannot be uniquely located in {review_id}")
            continue
        if not complete_word:
            substring_only += 1
            errors.append(f"KWIC row {row_number}: {node!r} is a substring-only match in {review_id}")
        actual_matches.append(MatchKey(review_id, keyword.casefold(), match.start(), match.end()))
        difference_counts[classify_full_review_difference(str(row["Full Review"]), source_text)] += 1

    expected_counter = Counter(expected)
    actual_counter = Counter(actual_matches)
    missing_counter = expected_counter - actual_counter
    extra_counter = actual_counter - expected_counter
    duplicate_count = sum(
        max(0, actual_counter[key] - expected_counter.get(key, 0))
        for key in actual_counter if key in expected_counter
    )
    unexpected_count = sum(count for key, count in extra_counter.items() if key not in expected_counter)
    missing_count = sum(missing_counter.values())
    if missing_count:
        errors.append(f"KWIC extraction is missing {missing_count} expected match(es): {list(missing_counter.items())[:5]}")
    if duplicate_count:
        errors.append(f"KWIC extraction contains {duplicate_count} duplicate match(es)")
    if unexpected_count:
        errors.append(f"KWIC extraction contains {unexpected_count} unexpected match(es): {list(extra_counter.items())[:5]}")
    if difference_counts["real-content-mismatch"]:
        errors.append(
            f"{difference_counts['real-content-mismatch']} KWIC Full Review value(s) contain real content differences"
        )
    if errors:
        raise ValidationError("Validation failed:\n- " + "\n- ".join(errors))

    review_table = build_kwic_evidence(kwic, texts, keywords)
    pattern_table = build_pattern_evidence(source, labels)
    occurrence_counts = Counter(match.keyword for match in expected)
    review_sets: dict[str, set[str]] = {}
    for match in expected:
        review_sets.setdefault(match.keyword, set()).add(match.review_id)
    language_counts = source["Language Classification"].astype(str).str.strip().value_counts()
    summary: AuditSummary = {
        "source_reviews": len(source),
        "code_switched_reviews": len(cs),
        "kwic_rows": len(kwic),
        "expected_kwic_rows": len(expected),
        "kwic_review_ids": int(kwic["Review ID"].nunique()),
        "keyword_node_mismatches": keyword_node_mismatches,
        "review_label_mismatches": review_label_mismatches,
        "non_code_switched_rows": non_code_switched_rows,
        "missing_matches": missing_count,
        "duplicate_matches": duplicate_count,
        "unexpected_matches": unexpected_count,
        "substring_only_matches": substring_only,
        "exact_full_review_matches": difference_counts["exact"],
        "line_ending_only_differences": difference_counts["line-ending-only"],
        "trailing_whitespace_only_differences": difference_counts["trailing-whitespace-only"],
        "real_content_mismatches": difference_counts["real-content-mismatch"],
        "type_counts": type_counts,
        "type_memberships": sum(type_counts.values()),
        "combination_counts": combination_counts,
        "language_counts": {str(k): int(str(v)) for k, v in language_counts.items()},
        "keyword_occurrences": dict(occurrence_counts),
        "keyword_review_counts": {k: len(v) for k, v in review_sets.items()},
    }
    return summary, review_table, pattern_table


def relevant_segment(text: str, match: re.Match[str]) -> str:
    left_boundaries = [text.rfind(mark, 0, match.start()) for mark in ".!?\n"]
    start = max(left_boundaries + [-1]) + 1
    right_boundaries = [
        position
        for mark in ".!?\n"
        for position in [text.find(mark, match.end())]
        if position != -1
    ]
    end = min(right_boundaries) + 1 if right_boundaries else len(text)
    return normalize_space(text[start:end])


def build_kwic_evidence(kwic: pd.DataFrame, texts: dict[str, str], allowed_keywords: tuple[str, ...]) -> pd.DataFrame:
    review_ids = kwic["Review ID"].astype(str)
    keywords = kwic["Keyword"].astype(str)
    rows: list[dict[str, str]] = []
    for review_id, keyword, occurrence, category, interpretation in KWIC_SELECTIONS:
        if keyword.casefold() not in allowed_keywords:
            continue
        mask = review_ids.eq(review_id) & keywords.str.casefold().eq(keyword.casefold())
        candidates = kwic.loc[mask].reset_index(drop=True)
        if len(candidates) < occurrence:
            raise ValidationError(f"Selected evidence is missing: {review_id} {keyword} #{occurrence}")
        row = candidates.iloc[occurrence - 1]
        source_text = texts[review_id]
        match, _ = locate_output_match(row, source_text)
        if match is None:
            raise ValidationError(f"Cannot locate selected evidence: {review_id} {keyword}")
        rows.append({
            "Review ID": review_id,
            "Evidence source": "KWIC occurrence",
            "Relevant sentence or segment": relevant_segment(source_text, match),
            "Keyword": str(row["Node"]),
            "Surrounding context": f"{row['Left Context']} [{row['Node']}] {row['Right Context']}",
            "Existing review-level label": str(row["Review Switching Type"]),
            "Occurrence-level category": category,
            "Suggested interpretation": interpretation,
        })
    return pd.DataFrame(rows, columns=[
        "Review ID", "Evidence source", "Relevant sentence or segment", "Keyword",
        "Surrounding context", "Existing review-level label",
        "Occurrence-level category", "Suggested interpretation",
    ])


def build_pattern_evidence(source: pd.DataFrame, labels: dict[str, str]) -> pd.DataFrame:
    texts, _, _ = source_lookups(source)
    rows: list[dict[str, str]] = []
    for review_id, pattern, excerpt, interpretation in PATTERN_EXAMPLES:
        if excerpt not in texts[review_id]:
            raise ValidationError(f"Pattern excerpt for {review_id} is not an exact source substring")
        rows.append({
            "Review ID": review_id,
            "Evidence source": "Corpus excerpt (not a KWIC node)",
            "Relevant sentence or segment": excerpt,
            "Keyword": "",
            "Surrounding context": excerpt,
            "Existing review-level label": labels[review_id],
            "Occurrence-level category": pattern,
            "Suggested interpretation": interpretation,
        })
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False)


def write_evidence(
    paths: ValidationPaths,
    summary: AuditSummary,
    review_table: pd.DataFrame,
    pattern_table: pd.DataFrame,
) -> None:
    paths.evidence_dir.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([review_table, pattern_table], ignore_index=True)
    combined.to_csv(
        paths.evidence_dir / "revised_qualitative_evidence.csv",
        index=False,
        encoding="utf-8-sig",
    )
    clear_mask = review_table["Occurrence-level category"].eq(
        "Clear code-switching evidence"
    )
    clear = review_table.loc[clear_mask].copy()
    qualified = review_table.loc[~clear_mask].copy()
    type_counts = summary["type_counts"]
    report = [
        "# CPC353 KWIC Validation Evidence",
        "",
        "## Validation result",
        "",
        "**All implemented checks passed for the current source and exported outputs.** This covers schema, independent statistic reconciliation, row-level labels, complete-word matching, and extraction completeness.",
        "",
        "## Match correctness",
        "",
        f"- Exported KWIC rows checked: {summary['kwic_rows']}",
        f"- Keyword/Node disagreements: {summary['keyword_node_mismatches']}",
        f"- Review-level label disagreements: {summary['review_label_mismatches']}",
        f"- Rows from non-Code-switched reviews: {summary['non_code_switched_rows']}",
        f"- Substring-only matches: {summary['substring_only_matches']}",
        f"- Exact Full Review matches: {summary['exact_full_review_matches']}",
        f"- LF/CRLF-only differences: {summary['line_ending_only_differences']}",
        f"- Trailing-whitespace-only differences: {summary['trailing_whitespace_only_differences']}",
        f"- Real content mismatches: {summary['real_content_mismatches']}",
        "",
        "## Extraction completeness",
        "",
        f"- Independently expected KWIC matches: {summary['expected_kwic_rows']}",
        f"- Missing matches: {summary['missing_matches']}",
        f"- Duplicate matches: {summary['duplicate_matches']}",
        f"- Unexpected matches: {summary['unexpected_matches']}",
        "",
        "## Recomputed statistics",
        "",
        f"- Language counts: {summary['language_counts']}",
        f"- Code-switched reviews: {summary['code_switched_reviews']} of {summary['source_reviews']}",
        f"- Switching-type counts: {summary['type_counts']}",
        f"- Multi-label memberships: {summary['type_memberships']}",
        f"- Mutually exclusive label combinations: {summary['combination_counts']}",
        f"- Keyword occurrences: {summary['keyword_occurrences']}",
        f"- Distinct reviews by keyword: {summary['keyword_review_counts']}",
        "",
        "The four exported distribution/frequency CSVs were compared field by field against these independently recomputed values, including percentage denominators.",
        "",
        "## Clear occurrence-level code-switching evidence",
        "",
        markdown_table(clear),
        "",
        "## Monolingual, food-name, borrowing, and uncertain KWIC cases",
        "",
        markdown_table(qualified),
        "",
        "## Additional sentence-boundary and tag evidence",
        "",
        "These are exact corpus excerpts, not extra KWIC nodes and not frequency evidence.",
        "",
        markdown_table(pattern_table),
        "",
        "## Evidence-based findings",
        "",
        f"1. **Intra-sentential is the most common review-level annotation:** {type_counts['Intra-sentential']} of {summary['code_switched_reviews']} code-switched reviews ({100 * type_counts['Intra-sentential'] / summary['code_switched_reviews']:.2f}%). Clear illustrations include REV-002, REV-045, REV-082, REV-126, and REV-180. The frequency claim comes from the full corpus, not the purposive examples.",
        "",
        "2. **Inter-sentential switching is present as a sentence-boundary pattern.** REV-044 moves from a Malay sentence to an English sentence; REV-123 moves English-Malay-English across successive units. These examples demonstrate form, not prevalence.",
        "",
        "3. **KWIC frequency is not switching frequency.** REV-001, REV-018, and REV-020 contain monolingual hits; REV-017 and REV-030 contain dish names; REV-007, REV-051, REV-088, and REV-161 involve loanword or borrowing questions.",
        "",
        "## Method statement",
        "",
        "The validator independently re-extracted complete-word matches from each Code-switched review and reconciled them with the KWIC export. It checked row correctness separately from extraction completeness. Review-level annotations describe whole reviews; individual occurrences were interpreted from context. The team's policy excludes proper food names and common loanwords such as 'and', 'service', 'best', 'order', and 'menu' as standalone switching evidence.",
    ]
    (paths.evidence_dir / "kwic_validation_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--keywords", nargs="+", default=list(DEFAULT_KEYWORDS))
    parser.add_argument("--context-chars", type=int, default=70)
    return parser.parse_args()


def main() -> None:
    global context_chars
    args = parse_args()
    if args.context_chars < 10:
        raise SystemExit("--context-chars must be at least 10")
    context_chars = args.context_chars
    paths = ValidationPaths(args.source, args.output_dir, args.evidence_dir)
    keywords = tuple(dict.fromkeys(str(word).casefold() for word in args.keywords))
    try:
        summary, review_table, pattern_table = validate(paths, keywords)
        write_evidence(paths, summary, review_table, pattern_table)
    except ValidationError as error:
        raise SystemExit(str(error)) from error
    print(f"Validated {summary['kwic_rows']} exported KWIC rows.")
    print(f"Independently expected {summary['expected_kwic_rows']} KWIC matches.")
    print("All implemented checks passed.")
    print(f"Evidence written to: {paths.evidence_dir.resolve()}")


if __name__ == "__main__":
    main()
