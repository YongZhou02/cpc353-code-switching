"""KWIC analysis for English-Malay code-switching in food reviews.

The program treats the team's manual labels as the source of truth.  It does
not automatically relabel proper nouns, dish names, or borrowed food terms.
Instead, it combines review-level multi-label statistics with concordance
lines that can be interpreted in context.

Example:
    python analysis/kwic_analysis.py \
        "data/CPC353_dataset_v3 - Self-Collected Data.csv"
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import pandas as pd


SWITCH_TYPES = ("Intra-sentential", "Inter-sentential", "Tag Switching")
DEFAULT_KEYWORDS = (
    "sedap", "makan", "murah", "pedas", "lepak", "kedai",
    "best", "good", "nice", "service", "food", "delicious",
)

# Small bilingual stop-list for exploratory lexical frequency.  It is not a
# language detector; its purpose is only to remove very common function words.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
    "from", "had", "has", "have", "he", "her", "his", "i", "in", "is",
    "it", "its", "me", "my", "of", "on", "or", "our", "she", "so",
    "that", "the", "their", "them", "they", "this", "to", "very", "was",
    "we", "were", "with", "you", "your", "ada", "adalah", "aku", "apa",
    "dan", "dari", "dengan", "dia", "di", "ini", "itu", "je", "juga",
    "kat", "ke", "lagi", "lah", "macam", "nak", "nya", "pun", "sangat",
    "saya", "tak", "takde", "tidak", "tu", "untuk", "yg", "yang",
}

TOKEN_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*", re.UNICODE)


def find_header_row(csv_path: Path) -> int:
    """Return the zero-based row containing the real CSV header."""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.reader(handle)):
            if row and row[0].strip() == "Review ID":
                return row_number
    raise ValueError("Cannot find the 'Review ID' header row.")


def load_reviews(csv_path: Path) -> pd.DataFrame:
    """Load and validate review rows without changing the team's labels."""
    data = pd.read_csv(
        csv_path,
        skiprows=find_header_row(csv_path),
        dtype=str,
        encoding="utf-8-sig",
    ).fillna("")

    required = {
        "Review ID", "Review Text", "Language Classification",
        "Code-Switching Type",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    review_ids = data["Review ID"].astype(str)
    valid_rows = review_ids.str.fullmatch(r"REV-\d+")
    data = data.loc[valid_rows].copy()
    for column in required:
        values = data[column].astype(str)
        data.loc[:, column] = values if column == "Review Text" else values.str.strip()

    if data.empty:
        raise ValueError("No valid REV-xxx review rows were found.")
    review_ids = data["Review ID"]
    duplicate_rows = review_ids.duplicated()
    if bool(duplicate_rows.any()):
        duplicates = data.loc[duplicate_rows, "Review ID"].tolist()
        raise ValueError(f"Duplicate Review IDs: {duplicates}")
    blank_rows = data["Review Text"].eq("")
    if bool(blank_rows.any()):
        blank_ids = data.loc[blank_rows, "Review ID"].tolist()
        raise ValueError(f"Blank review text: {blank_ids}")
    return data


def parse_switch_types(value: str) -> tuple[str, ...]:
    """Parse a comma/semicolon/slash separated multi-label cell."""
    labels = re.split(r"\s*(?:,|;|/|\|)\s*", value.strip()) if value.strip() else []
    canonical = {label.casefold(): label for label in SWITCH_TYPES}
    found: set[str] = set()
    for label in labels:
        matched = canonical.get(label.casefold())
        if matched is None:
            raise ValueError(f"Unknown switching type: {label!r}")
        found.add(matched)
    if not found:
        raise ValueError("Code-switched review has no recognised switching type")
    # Canonical ordering prevents equivalent combinations such as
    # "Intra + Inter" and "Inter + Intra" from becoming separate rows.
    return tuple(label for label in SWITCH_TYPES if label in found)


def clean_context(text: str) -> str:
    return " ".join(text.split())


def language_distribution(data: pd.DataFrame) -> pd.DataFrame:
    result = (
        data["Language Classification"]
        .value_counts(dropna=False)
        .rename_axis("Language Classification")
        .reset_index(name="Review Count")
    )
    result["Percentage of All Reviews"] = (
        100 * result["Review Count"] / len(data)
    ).round(2)
    return result


def switching_statistics(code_switched: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    parsed: list[tuple[str, ...]] = [
        parse_switch_types(str(value))
        for value in code_switched["Code-Switching Type"].tolist()
    ]
    counts: Counter[str] = Counter(label for labels in parsed for label in labels)
    denominator = len(code_switched)

    distribution = pd.DataFrame({
        "Switching Type": SWITCH_TYPES,
        "Review Count": [counts[label] for label in SWITCH_TYPES],
    })
    distribution["Percentage of Code-Switched Reviews"] = (
        100 * distribution["Review Count"] / denominator
    ).round(2) if denominator else 0.0

    combination_names = pd.Series(
        [" + ".join(labels) for labels in parsed], dtype=str
    )
    combinations = (
        combination_names.value_counts()
        .rename_axis("Label Combination")
        .reset_index(name="Review Count")
    )
    combinations["Percentage of Code-Switched Reviews"] = (
        100 * combinations["Review Count"] / denominator
    ).round(2) if denominator else 0.0
    return distribution, combinations


def build_kwic(
    code_switched: pd.DataFrame,
    keywords: tuple[str, ...],
    context_chars: int,
) -> pd.DataFrame:
    """Create one concordance row per complete-word keyword occurrence."""
    rows: list[dict[str, str]] = []
    for _, review in code_switched.iterrows():
        text = str(review["Review Text"])
        for keyword in keywords:
            pattern = re.compile(rf"(?<!\w){re.escape(keyword)}(?!\w)", re.IGNORECASE)
            for match in pattern.finditer(text):
                rows.append({
                    "Review ID": str(review["Review ID"]),
                    "Keyword": match.group(0),
                    "Left Context": clean_context(
                        text[max(0, match.start() - context_chars):match.start()]
                    ),
                    "Node": match.group(0),
                    "Right Context": clean_context(
                        text[match.end():match.end() + context_chars]
                    ),
                    "Review Switching Type": str(review["Code-Switching Type"]),
                    "Full Review": text,
                    "Manual Interpretation": "",
                    "Exclude Reason (if any)": "",
                })
    return pd.DataFrame(rows, columns=[
        "Review ID", "Keyword", "Left Context", "Node", "Right Context",
        "Review Switching Type", "Full Review", "Manual Interpretation",
        "Exclude Reason (if any)",
    ])


def keyword_summary(kwic: pd.DataFrame, code_switched_count: int) -> pd.DataFrame:
    if kwic.empty:
        return pd.DataFrame(columns=[
            "Keyword", "Occurrences", "Reviews Containing Keyword",
            "Percentage of Code-Switched Reviews",
        ])
    summary = (
        kwic.assign(Keyword=kwic["Keyword"].str.casefold())
        .groupby("Keyword", as_index=False)
        .agg(
            Occurrences=("Review ID", "size"),
            Reviews_Containing_Keyword=("Review ID", "nunique"),
        )
    )
    summary = summary.sort_values("Occurrences", ascending=False)
    summary = summary.rename(columns={
        "Reviews_Containing_Keyword": "Reviews Containing Keyword"
    })
    summary["Percentage of Code-Switched Reviews"] = (
        100 * summary["Reviews Containing Keyword"] / code_switched_count
    ).round(2) if code_switched_count else 0.0
    return summary


def lexical_frequency(code_switched: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Return exploratory content-word frequencies from code-switched reviews."""
    review_sets: dict[str, set[str]] = {}
    occurrences: Counter[str] = Counter()
    for _, review in code_switched.iterrows():
        review_text = str(review["Review Text"])
        review_id = str(review["Review ID"])
        tokens = [token.casefold() for token in TOKEN_RE.findall(review_text)]
        tokens = [t for t in tokens if len(t) > 1 and t not in STOPWORDS]
        occurrences.update(tokens)
        for token in set(tokens):
            review_sets.setdefault(token, set()).add(review_id)

    rows = [
        {
            "Token": token,
            "Occurrences": count,
            "Reviews Containing Token": len(review_sets[token]),
        }
        for token, count in occurrences.most_common(top_n)
    ]
    return pd.DataFrame(rows)


def write_report(
    output_path: Path,
    data: pd.DataFrame,
    code_switched: pd.DataFrame,
    language: pd.DataFrame,
    switching: pd.DataFrame,
    combinations: pd.DataFrame,
    keyword_counts: pd.DataFrame,
) -> None:
    top_type = switching.sort_values("Review Count", ascending=False).iloc[0]
    top_type_name = str(top_type["Switching Type"])
    top_type_count = int(top_type["Review Count"])
    top_type_percentage = float(top_type["Percentage of Code-Switched Reviews"])
    lines = [
        "# English-Malay Code-Switching KWIC Analysis",
        "",
        "## Scope and annotation assumptions",
        "",
        "The analysis accepts the team's manual labels as the source of truth. "
        "Switching types are multi-label, so one review may contribute to more "
        "than one type and percentages do not sum to 100%. Proper nouns, brand "
        "names, restaurant names, and specific dish names were excluded during "
        "team annotation where possible; general food words were treated as "
        "ordinary lexical items.",
        "",
        "## Corpus overview",
        "",
        f"- Total reviews: {len(data)}",
        f"- Code-switched reviews: {len(code_switched)}",
        "",
        language.to_markdown(index=False),
        "",
        "## Code-switching patterns",
        "",
        switching.to_markdown(index=False),
        "",
        f"The most common annotated pattern is **{top_type_name}**, "
        f"found in {top_type_count} code-switched reviews "
        f"({top_type_percentage:.2f}%).",
        "",
        "## Multi-label combinations",
        "",
        combinations.to_markdown(index=False),
        "",
        "## KWIC keyword summary",
        "",
        keyword_counts.to_markdown(index=False) if not keyword_counts.empty else "No keyword matches.",
        "",
        "## Interpretation guidance",
        "",
        "A keyword occurrence is not automatically a code-switching event. "
        "Interpret each concordance line using its left and right context. "
        "Record whether it is an intra-sentential switch, inter-sentential "
        "switch, tag switch, general food term, proper name, borrowing, "
        "monolingual use, or uncertain case.",
        "",
        "## Research-question answer",
        "",
        f"The preliminary review-level result indicates that "
        f"{top_type_name.lower()} switching is the dominant "
        "English-Malay code-switching pattern in this food-review corpus. "
        "The final discussion should support this distribution with selected "
        "KWIC examples and explain where the language boundary occurs.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def analyse(
    data: pd.DataFrame,
    output_dir: Path,
    keywords: tuple[str, ...],
    context_chars: int,
    top_n: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    language_values = data["Language Classification"].astype(str)
    code_switched_mask = language_values.str.casefold().eq("code-switched")
    code_switched = data.loc[code_switched_mask].copy()

    language = language_distribution(data)
    switching, combinations = switching_statistics(code_switched)
    kwic = build_kwic(code_switched, keywords, context_chars)
    keywords_table = keyword_summary(kwic, len(code_switched))
    tokens = lexical_frequency(code_switched, top_n)

    tables = {
        "01_language_distribution.csv": language,
        "02_switching_type_distribution.csv": switching,
        "03_switching_type_combinations.csv": combinations,
        "04_kwic_concordance.csv": kwic,
        "05_keyword_frequency.csv": keywords_table,
        "06_content_word_frequency.csv": tokens,
    }
    for filename, table in tables.items():
        table.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")

    write_report(
        output_dir / "07_analysis_summary.md",
        data,
        code_switched,
        language,
        switching,
        combinations,
        keywords_table,
    )

    print(f"Total reviews: {len(data)}")
    print(f"Code-switched reviews: {len(code_switched)}")
    print("\nSwitching types (multi-label):")
    print(switching.to_string(index=False))
    print("\nMost common label combinations:")
    print(combinations.head(10).to_string(index=False))
    print(f"\nKWIC occurrences: {len(kwic)}")
    print(f"Created 7 analysis files in: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="CSV containing the review dataset")
    parser.add_argument(
        "--output-dir", type=Path,
        help="Output folder (default: <CSV folder>/CPC353_kwic_outputs)",
    )
    parser.add_argument(
        "--keywords", nargs="+", default=list(DEFAULT_KEYWORDS),
        help="Complete-word keywords used as KWIC nodes",
    )
    parser.add_argument(
        "--context-chars", type=int, default=70,
        help="Maximum characters shown on each side of the KWIC node",
    )
    parser.add_argument(
        "--top-n", type=int, default=50,
        help="Number of exploratory content-word frequencies to save",
    )
    args = parser.parse_args()
    if not args.csv.is_file():
        parser.error(f"File not found: {args.csv}")
    if args.context_chars < 10:
        parser.error("--context-chars must be at least 10")
    if args.top_n < 1:
        parser.error("--top-n must be at least 1")
    if not args.keywords or any(not str(value).strip() for value in args.keywords):
        parser.error("--keywords must contain nonempty keywords")
    return args


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.csv.parent / "CPC353_kwic_outputs"
    data = load_reviews(args.csv)
    analyse(
        data,
        output_dir,
        tuple(dict.fromkeys(keyword.casefold() for keyword in args.keywords)),
        args.context_chars,
        args.top_n,
    )


if __name__ == "__main__":
    main()
