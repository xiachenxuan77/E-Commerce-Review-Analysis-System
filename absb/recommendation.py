"""
absb.recommendation
=====================

Generates business recommendations from the aspect sentiment table.

For each aspect, a "negative_ratio" is computed:
    negative_ratio = Negative / (Positive + Neutral + Negative)

That ratio is then mapped to a recommendation using simple, transparent
rule-based thresholds (no ML).
"""

import csv
from typing import Dict, List, Tuple

from absb import config

# -----------------------------------------------------------------------
# Aspect-specific recommendation templates, used when an aspect's negative
# ratio crosses the HIGH threshold.
# -----------------------------------------------------------------------
_HIGH_NEGATIVE_RECOMMENDATIONS = {
    "Quality": "High negative sentiment on Quality. Recommendation: Strengthen quality control and review supplier/material standards.",
    "Price": "High negative sentiment on Price. Recommendation: Reassess pricing strategy or better communicate the value proposition.",
    "Delivery": "High negative sentiment on Delivery. Recommendation: Improve logistics efficiency and shipment tracking.",
    "Packaging": "High negative sentiment on Packaging. Recommendation: Upgrade packaging materials to reduce damage during transit.",
    "Customer Service": "High negative sentiment on Customer Service. Recommendation: Provide additional support staff training and faster response times.",
    "Usability": "High negative sentiment on Usability. Recommendation: Simplify product instructions and improve ease of use.",
    "Size and Fit": "High negative sentiment on Size and Fit. Recommendation: Provide clearer sizing charts and fit guidance.",
    "Design and Appearance": "High negative sentiment on Design and Appearance. Recommendation: Gather design feedback and consider aesthetic revisions.",
    "Comfort": "High negative sentiment on Comfort. Recommendation: Re-evaluate materials and ergonomics for improved comfort.",
    "Taste and Flavor": "High negative sentiment on Taste and Flavor. Recommendation: Review recipe/formulation based on customer taste feedback.",
}

_MODERATE_NEGATIVE_TEMPLATE = (
    "Moderate negative sentiment on {aspect}. Recommendation: Monitor customer "
    "feedback on {aspect} closely and address recurring complaints."
)

_LOW_NEGATIVE_TEMPLATE = (
    "Low negative sentiment on {aspect}. Recommendation: Maintain current standards for {aspect}."
)

_HIGH_THRESHOLD = 0.40
_MODERATE_THRESHOLD = 0.20


def _build_recommendation(aspect: str, negative_ratio: float) -> str:
    if negative_ratio >= _HIGH_THRESHOLD:
        return _HIGH_NEGATIVE_RECOMMENDATIONS.get(
            aspect,
            f"High negative sentiment on {aspect}. Recommendation: Investigate and address recurring negative feedback.",
        )
    if negative_ratio >= _MODERATE_THRESHOLD:
        return _MODERATE_NEGATIVE_TEMPLATE.format(aspect=aspect)
    return _LOW_NEGATIVE_TEMPLATE.format(aspect=aspect)


def compute_recommendations(sentiment_table: Dict[str, Dict[str, int]]) -> List[Tuple[str, float, str]]:
    """
    Returns a list of (aspect, negative_ratio, recommendation) sorted by
    negative_ratio descending (most problematic aspects first).
    """
    rows: List[Tuple[str, float, str]] = []

    for aspect, counts in sentiment_table.items():
        total = counts.get("Positive", 0) + counts.get("Neutral", 0) + counts.get("Negative", 0)
        negative_ratio = (counts.get("Negative", 0) / total) if total > 0 else 0.0
        recommendation = _build_recommendation(aspect, negative_ratio)
        rows.append((aspect, round(negative_ratio, 4), recommendation))

    rows.sort(key=lambda r: r[1], reverse=True)
    return rows


def save_recommendations(rows: List[Tuple[str, float, str]], output_path=None) -> str:
    if output_path is None:
        output_path = config.RECOMMENDATIONS_FILE

    config.ensure_output_dir()

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["aspect", "negative_ratio", "recommendation"])
        for aspect, ratio, rec in rows:
            writer.writerow([aspect, ratio, rec])

    return output_path


def generate_recommendations(sentiment_table: Dict[str, Dict[str, int]]) -> str:
    print("[recommendation] Generating business recommendations...")
    rows = compute_recommendations(sentiment_table)
    output_path = save_recommendations(rows)
    print(f"[recommendation] Saved: {output_path}")
    return output_path


if __name__ == "__main__":
    from absb.main import process_corpus
    from absb.statistics import compute_aspect_sentiment_table

    results = process_corpus()
    table = compute_aspect_sentiment_table(results)
    generate_recommendations(table)
