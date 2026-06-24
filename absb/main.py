"""
absb.main
==========

Reads the review corpus, runs single_review_analysis() over every row,
and writes absb/outputs/aspect_results.csv.
"""

import csv
from typing import Dict, List, Optional

import pandas as pd

from absb import config
from absb.aspect_dictionary import get_aspect_dictionary
from absb.aspect_sentiment import get_analyzer
from absb.single_review_analysis import analyze_single_review


def load_corpus(input_path: Optional[str] = None) -> pd.DataFrame:
    """Load the review corpus and validate required columns."""
    if input_path is None:
        input_path = config.INPUT_FILE

    df = pd.read_csv(input_path)

    required = (config.REVIEW_ID_COLUMN, config.REVIEW_TEXT_COLUMN)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input corpus '{input_path}' is missing required column(s): {missing}. "
            f"Expected at least: {list(required)}"
        )

    df[config.REVIEW_TEXT_COLUMN] = df[config.REVIEW_TEXT_COLUMN].fillna("").astype(str)
    return df


def process_corpus(input_path: Optional[str] = None, verbose: bool = True) -> List[Dict]:
    """
    Run aspect extraction + aspect-level sentiment classification on every
    review in the corpus.

    Returns
    -------
    list[dict]
        One structured result (see single_review_analysis.analyze_single_review)
        per review, in corpus order.
    """
    df = load_corpus(input_path)

    aspect_dict = get_aspect_dictionary()
    analyzer = get_analyzer()

    results: List[Dict] = []
    total = len(df)

    id_col = config.REVIEW_ID_COLUMN
    text_col = config.REVIEW_TEXT_COLUMN

    for i, row in enumerate(df.itertuples(index=False), start=1):
        review_id = getattr(row, id_col)
        original_review = getattr(row, text_col)

        result = analyze_single_review(
            review_id=review_id,
            original_review=original_review,
            aspect_dict=aspect_dict,
            analyzer=analyzer,
        )
        results.append(result)

        if verbose and i % 10000 == 0:
            print(f"  [main] Processed {i}/{total} reviews...")

    if verbose:
        print(f"  [main] Processed {total}/{total} reviews.")

    return results


def save_results(results: List[Dict], output_path: Optional[str] = None) -> str:
    """
    Write absb/outputs/aspect_results.csv with columns:
        review_id, original_review, aspects, aspect_sentiment, keywords
    """
    if output_path is None:
        output_path = config.ASPECT_RESULTS_FILE

    config.ensure_output_dir()

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["review_id", "original_review", "aspects", "aspect_sentiment", "keywords"])

        for r in results:
            aspects_str = ";".join(r["aspects"])
            sentiment_str = ";".join(f"{aspect}:{label}" for aspect, label in r["aspect_sentiment"].items())
            keywords_str = ";".join(r["keywords"])

            writer.writerow([r["review_id"], r["original_review"], aspects_str, sentiment_str, keywords_str])

    return output_path


def run() -> List[Dict]:
    """Process the full corpus and persist aspect_results.csv. Returns the raw results list."""
    print("[main] Loading corpus and running aspect extraction + sentiment classification...")
    results = process_corpus()
    output_path = save_results(results)
    print(f"[main] Saved: {output_path}")
    return results


if __name__ == "__main__":
    run()
