"""
absb.statistics
=================

Aggregates the per-review ABSA results into corpus-level statistics:
    - aspect_distribution.csv      (aspect, frequency)
    - aspect_sentiment_table.csv   (aspect, Positive, Neutral, Negative)
"""

import csv
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from absb import config


def compute_aspect_distribution(results: List[Dict]) -> List[Tuple[str, int]]:
    """Count how many reviews mention each aspect, sorted by frequency desc."""
    counter: Counter = Counter()
    for r in results:
        for aspect in r["aspects"]:
            counter[aspect] += 1

    return sorted(counter.items(), key=lambda kv: kv[1], reverse=True)


def compute_aspect_sentiment_table(results: List[Dict]) -> Dict[str, Dict[str, int]]:
    """Count Positive / Neutral / Negative occurrences per aspect."""
    table: Dict[str, Dict[str, int]] = defaultdict(lambda: {"Positive": 0, "Neutral": 0, "Negative": 0})

    for r in results:
        for aspect, sentiment in r["aspect_sentiment"].items():
            if sentiment in table[aspect]:
                table[aspect][sentiment] += 1

    return dict(table)


def save_aspect_distribution(distribution: List[Tuple[str, int]], output_path=None) -> str:
    if output_path is None:
        output_path = config.ASPECT_DISTRIBUTION_FILE

    config.ensure_output_dir()

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["aspect", "frequency"])
        for aspect, freq in distribution:
            writer.writerow([aspect, freq])

    return output_path


def save_aspect_sentiment_table(table: Dict[str, Dict[str, int]], output_path=None) -> str:
    if output_path is None:
        output_path = config.ASPECT_SENTIMENT_TABLE_FILE

    config.ensure_output_dir()

    # Order by total mentions, descending, for readability.
    ordered_aspects = sorted(table.items(), key=lambda kv: sum(kv[1].values()), reverse=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["aspect", "Positive", "Neutral", "Negative"])
        for aspect, counts in ordered_aspects:
            writer.writerow([aspect, counts.get("Positive", 0), counts.get("Neutral", 0), counts.get("Negative", 0)])

    return output_path


def generate_statistics(results: List[Dict]):
    """
    Compute and persist both statistics output files.

    Returns
    -------
    (distribution, sentiment_table)
        distribution     : list[(aspect, frequency)]
        sentiment_table   : dict[aspect, {"Positive": int, "Neutral": int, "Negative": int}]
    """
    print("[statistics] Computing aspect distribution and sentiment table...")

    distribution = compute_aspect_distribution(results)
    sentiment_table = compute_aspect_sentiment_table(results)

    dist_path = save_aspect_distribution(distribution)
    table_path = save_aspect_sentiment_table(sentiment_table)

    print(f"[statistics] Saved: {dist_path}")
    print(f"[statistics] Saved: {table_path}")

    return distribution, sentiment_table


if __name__ == "__main__":
    from absb.main import process_corpus

    results = process_corpus()
    generate_statistics(results)
