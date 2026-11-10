# Analysis

`kwic_analysis.py` performs review-level multi-label statistics and keyword-in-
context (KWIC) analysis for the English-Malay food-review corpus.

Run it from the repository root:

```bash
python analysis/kwic_analysis.py "data/CPC353_dataset_v3 - Self-Collected Data.csv"
```

Use different KWIC nodes when needed:

```bash
python analysis/kwic_analysis.py "data/CPC353_dataset_v3 - Self-Collected Data.csv" \
  --keywords sedap makan murah pedas best good service
```

The default output folder is `data/CPC353_kwic_outputs/`. It contains:

1. language distribution;
2. switching-type distribution;
3. multi-label switching combinations;
4. KWIC concordance lines for manual interpretation;
5. keyword frequencies;
6. exploratory content-word frequencies; and
7. a Markdown analysis summary.

Important interpretation rules:

- A review may have more than one switching type, so percentages do not sum to
  100%.
- A keyword occurrence is not automatically a switching event. Read both sides
  of the KWIC line before interpreting it.
- The program accepts the team's labels. It does not automatically relabel
  proper nouns, restaurant names, dish names, or borrowed food terms.
- Each review is processed separately, so contexts never cross review
  boundaries.

## Validate existing outputs

Run the independent validator from the repository root:

```bash
python analysis/validation/validate_kwic.py
```

It checks the exported schemas and statistics, re-extracts all expected KWIC
matches from the source reviews, and writes revised evidence to
`analysis/validation/evidence/`. It does not modify the dataset or the original
analysis outputs.

Run the corruption tests with:

```bash
python -m unittest discover -s analysis/validation/tests -v
```
