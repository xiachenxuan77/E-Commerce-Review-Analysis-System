"""
statistics.py
Reads aspect_results.csv and generates:
  - aspect_distribution.csv  (aspect frequency counts)
  - aspect_sentiment_table.csv (positive / neutral / negative counts per aspect)
"""

import os
import pandas as pd
from collections import defaultdict

BASE_DIR             = os.path.dirname(__file__)
INPUT_CSV            = os.path.join(BASE_DIR, "outputs", "aspect_results.csv")
DISTRIBUTION_CSV     = os.path.join(BASE_DIR, "outputs", "aspect_distribution.csv")
SENTIMENT_TABLE_CSV  = os.path.join(BASE_DIR, "outputs", "aspect_sentiment_table.csv")


def compute_statistics(
    input_csv: str = INPUT_CSV,
    distribution_csv: str = DISTRIBUTION_CSV,
    sentiment_table_csv: str = SENTIMENT_TABLE_CSV,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute aspect distribution and sentiment breakdown.

    Returns
    -------
    (distribution_df, sentiment_table_df)
    """
    print(f"[statistics] Reading {input_csv} ...")
    df = pd.read_csv(input_csv, dtype=str).fillna("")

    freq:      dict[str, int]             = defaultdict(int)
    sentiment: dict[str, dict[str, int]]  = defaultdict(lambda: {"Positive": 0, "Neutral": 0, "Negative": 0})

    for _, row in df.iterrows():
        aspects_raw   = row.get("aspects", "")
        sentiment_raw = row.get("aspect_sentiment", "")

        if not aspects_raw:
            continue

        aspects = [a.strip() for a in aspects_raw.split(";") if a.strip()]

        # Build aspect→sentiment map for this row
        asp_sent: dict[str, str] = {}
        for pair in sentiment_raw.split(";"):
            if ":" in pair:
                asp, sent = pair.split(":", 1)
                asp_sent[asp.strip()] = sent.strip()

        for aspect in aspects:
            freq[aspect] += 1
            sent_label = asp_sent.get(aspect, "Neutral")
            if sent_label in sentiment[aspect]:
                sentiment[aspect][sent_label] += 1
            else:
                sentiment[aspect]["Neutral"] += 1

    # ── aspect_distribution.csv ─────────────────────────────────────────────
    dist_df = (
        pd.DataFrame(
            [{"aspect": k, "frequency": v} for k, v in freq.items()]
        )
        .sort_values("frequency", ascending=False)
        .reset_index(drop=True)
    )
    dist_df.to_csv(distribution_csv, index=False)
    print(f"[statistics] Saved aspect_distribution.csv  ({len(dist_df)} aspects)")

    # ── aspect_sentiment_table.csv ───────────────────────────────────────────
    sent_rows = []
    for aspect, counts in sentiment.items():
        sent_rows.append({
            "aspect":   aspect,
            "Positive": counts["Positive"],
            "Neutral":  counts["Neutral"],
            "Negative": counts["Negative"],
        })
    sent_df = (
        pd.DataFrame(sent_rows)
        .sort_values("aspect")
        .reset_index(drop=True)
    )
    sent_df.to_csv(sentiment_table_csv, index=False)
    print(f"[statistics] Saved aspect_sentiment_table.csv")

    print("\n── Aspect Distribution (top 10) ──────────────────")
    print(dist_df.head(10).to_string(index=False))
    print("\n── Sentiment Breakdown ───────────────────────────")
    print(sent_df.to_string(index=False))

    return dist_df, sent_df


if __name__ == "__main__":
    compute_statistics()
