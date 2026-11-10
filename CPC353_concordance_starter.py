"""CPC353 Assignment 1: simple corpus statistics + concordance (KWIC).

How to run:
  pip install pandas
  python CPC353_concordance_starter.py "Self-Collected Data.csv"

The input CSV must be the Google Sheets export of the Self-Collected Data tab.
It may contain up to three title/description rows above the column headers.
"""

import argparse
import csv
import re
from pathlib import Path

import pandas as pd

KEYWORDS = ["sedap", "makan", "murah", "pedas", "lepak", "kedai"]
CONTEXT_CHARS = 55
CATEGORIES = ["Intra-sentential", "Inter-sentential", "Tag Switching"]


def load_reviews(csv_path: Path) -> pd.DataFrame:
    # Identify the real header so both Google Sheet exports and cleaned CSVs work.
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        header_offset = next(
            (i for i, row in enumerate(csv.reader(handle))
             if row and row[0].strip() == "Review ID"),
            None,
        )
    if header_offset is None:
        raise ValueError("Cannot find the 'Review ID' header in the CSV.")

    data = pd.read_csv(csv_path, skiprows=header_offset, dtype=str, encoding="utf-8-sig")
    mandatory = {"Review ID", "Review Text", "Language Classification", "Code-Switching Type"}
    missing = mandatory - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # Ignore trailing blank rows; only real review IDs are part of the corpus.
    data = data[data["Review ID"].fillna("").str.fullmatch(r"REV-\d+")].copy()
    for field in mandatory:
        data[field] = data[field].fillna("").str.strip()

    if data["Review ID"].duplicated().any():
        raise ValueError("Duplicate Review IDs found; confirm the final data with your teammate.")
    if (data["Review Text"] == "").any():
        raise ValueError("Blank Review Text found; confirm the final data with your teammate.")
    return data


def run_analysis(data: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(data)
    code_switched = data[data["Language Classification"] == "Code-switched"].copy()
    cs_count = len(code_switched)

    # Descriptive statistics: number of reviews, not number of switching events.
    language = data["Language Classification"].value_counts().rename_axis("Language")
    language = language.reset_index(name="Review Count")
    language["Percentage of All Reviews"] = (100 * language["Review Count"] / total).round(2)
    language.to_csv(output_dir / "01_language_distribution.csv", index=False, encoding="utf-8-sig")

    # One review can be counted under multiple switching types.
    split_types = code_switched["Code-Switching Type"].str.split(r"\s*,\s*").explode().str.strip()
    counts = split_types.value_counts()
    switching = pd.DataFrame({
        "Switching Type": CATEGORIES,
        "Review Count": [int(counts.get(category, 0)) for category in CATEGORIES],
    })
    switching["Percentage of Code-Switched Reviews"] = (
        100 * switching["Review Count"] / cs_count
    ).round(2) if cs_count else 0.0
    switching.to_csv(output_dir / "02_switching_type_distribution.csv", index=False, encoding="utf-8-sig")

    # Concordance: literal complete-word matches; preserve the original text.
    kwic_rows = []
    for _, review in code_switched.iterrows():
        full_text = review["Review Text"]
        for keyword in KEYWORDS:
            pattern = rf"(?<!\w){re.escape(keyword)}(?!\w)"
            for match in re.finditer(pattern, full_text, flags=re.IGNORECASE):
                kwic_rows.append({
                    "Review ID": review["Review ID"],
                    "Keyword": match.group(),
                    "Left Context": " ".join(full_text[max(0, match.start()-CONTEXT_CHARS):match.start()].split()),
                    "Right Context": " ".join(full_text[match.end():match.end()+CONTEXT_CHARS].split()),
                    "Review Switching Type": review["Code-Switching Type"],
                    "Manual KWIC Interpretation": "",  # YOU fill: intra/inter/monolingual/loanword/uncertain
                })

    kwic = pd.DataFrame(kwic_rows, columns=[
        "Review ID", "Keyword", "Left Context", "Right Context",
        "Review Switching Type", "Manual KWIC Interpretation",
    ])
    kwic.to_csv(output_dir / "03_KWIC_for_manual_review.csv", index=False, encoding="utf-8-sig")

    if len(kwic):
        keyword_summary = (kwic.assign(Keyword=kwic["Keyword"].str.lower())
                           .groupby("Keyword", as_index=False)
                           .agg(Occurrences=("Review ID", "size"),
                                Reviews_Containing_Keyword=("Review ID", "nunique"))
                           .sort_values("Occurrences", ascending=False))
    else:
        keyword_summary = pd.DataFrame(columns=["Keyword", "Occurrences", "Reviews_Containing_Keyword"])
    keyword_summary.to_csv(output_dir / "04_keyword_frequency.csv", index=False, encoding="utf-8-sig")

    print(f"Total reviews: {total}; code-switched reviews: {cs_count}")
    print("\nLanguage distribution:\n", language.to_string(index=False))
    print("\nSwitching types (multi-label):\n", switching.to_string(index=False))
    print("\nKeyword concordance summary:\n", keyword_summary.to_string(index=False))
    print(f"\nCreated four CSV files in: {output_dir.resolve()}")
    print("IMPORTANT: A keyword occurrence is NOT automatically a code-switching event.")
    print("Review the KWIC lines manually using your team's loanword and food-name rules.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Google Sheets CSV exported from Self-Collected Data")
    args = parser.parse_args()
    if not args.csv.is_file():
        parser.error(f"File not found: {args.csv}")
    data = load_reviews(args.csv)
    run_analysis(data, args.csv.parent / "CPC353_analysis_outputs")


if __name__ == "__main__":
    main()
