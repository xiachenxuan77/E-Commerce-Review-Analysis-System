"""
recommendation.py
Reads aspect_sentiment_table.csv, computes negative ratios,
and outputs recommendations.csv for aspects with high negative rates.
"""

import os
import pandas as pd

BASE_DIR            = os.path.dirname(__file__)
SENTIMENT_TABLE_CSV = os.path.join(BASE_DIR, "outputs", "aspect_sentiment_table.csv")
RECOMMENDATIONS_CSV = os.path.join(BASE_DIR, "outputs", "recommendations.csv")

# Negative-ratio threshold above which we generate a recommendation
NEGATIVE_THRESHOLD = 0.10   # 10 %

RECOMMENDATION_TEMPLATES: dict[str, str] = {
    "Battery":          "Improve battery capacity and optimize charging performance.",
    "Display":          "Enhance screen quality, brightness, and touch responsiveness.",
    "Price":            "Consider discounts, promotional campaigns, or better value bundles.",
    "Packaging":        "Improve packaging quality and protection to prevent transit damage.",
    "Delivery":         "Optimize logistics, reduce shipping time, and improve tracking updates.",
    "Customer Service": "Strengthen customer support response times and return/refund policies.",
    "Quality":          "Tighten manufacturing quality control and use higher-grade materials.",
    "Performance":      "Optimize software/hardware performance and reduce lag or freezes.",
    "Design":           "Revisit product aesthetics based on customer style preferences.",
    "Size":             "Offer size variants or clearer dimension information in listings.",
}

GENERIC_TEMPLATE = "Investigate customer complaints about {aspect} and implement targeted improvements."


def generate_recommendations(
    sentiment_table_csv: str = SENTIMENT_TABLE_CSV,
    recommendations_csv: str = RECOMMENDATIONS_CSV,
    threshold: float = NEGATIVE_THRESHOLD,
) -> pd.DataFrame:
    """
    Generate business recommendations for aspects with high negative sentiment.

    Parameters
    ----------
    sentiment_table_csv : path to aspect_sentiment_table.csv
    recommendations_csv : output path
    threshold           : minimum negative ratio (0-1) to trigger a recommendation

    Returns
    -------
    pd.DataFrame with columns: aspect, negative_ratio, recommendation
    """
    print(f"[recommendation] Reading {sentiment_table_csv} ...")
    df = pd.read_csv(sentiment_table_csv)

    records = []
    for _, row in df.iterrows():
        aspect   = row["aspect"]
        pos      = int(row.get("Positive", 0))
        neu      = int(row.get("Neutral",  0))
        neg      = int(row.get("Negative", 0))
        total    = pos + neu + neg

        if total == 0:
            continue

        neg_ratio = neg / total
        if neg_ratio >= threshold:
            rec = RECOMMENDATION_TEMPLATES.get(
                aspect,
                GENERIC_TEMPLATE.format(aspect=aspect)
            )
            records.append({
                "aspect":         aspect,
                "negative_ratio": f"{neg_ratio:.1%}",
                "recommendation": rec,
            })

    rec_df = pd.DataFrame(records)
    if not rec_df.empty:
        rec_df["_sort_key"] = rec_df["negative_ratio"].str.rstrip("%").astype(float)
        rec_df = (
            rec_df
            .sort_values("_sort_key", ascending=False)
            .drop(columns=["_sort_key"])
            .reset_index(drop=True)
        )

    rec_df.to_csv(recommendations_csv, index=False)
    print(f"[recommendation] Saved {len(rec_df)} recommendations → {recommendations_csv}")
    print()
    print(rec_df.to_string(index=False))

    return rec_df


if __name__ == "__main__":
    generate_recommendations()
