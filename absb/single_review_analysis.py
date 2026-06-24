"""
absb.single_review_analysis
=============================

Runs the full single-review ABSA pipeline:
    aspect extraction (dictionary matching) -> aspect-level sentiment (VADER)

and returns one structured result dictionary per review.
"""

from typing import Dict, Optional

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from absb.aspect_extraction import extract_aspects_and_keywords
from absb.aspect_sentiment import get_analyzer, predict_aspect_sentiment


def analyze_single_review(
    review_id,
    original_review: str,
    aspect_dict: Optional[Dict] = None,
    analyzer: Optional[SentimentIntensityAnalyzer] = None,
) -> Dict[str, object]:
    """
    Run aspect extraction + aspect-level sentiment analysis on one review.

    Parameters
    ----------
    review_id : Any
        The review's identifier (passed through unchanged).
    original_review : str
        The raw review text.
    aspect_dict : dict, optional
        Aspect -> keyword-list mapping. If not provided, the default
        dictionary from absb.aspect_dictionary is loaded.
    analyzer : SentimentIntensityAnalyzer, optional
        Reused VADER analyzer instance (recommended for batch processing
        so a new analyzer isn't constructed per review).

    Returns
    -------
    dict with:
        "review_id"        : Any
        "original_review"  : str
        "aspects"          : list[str]
        "aspect_sentiment" : dict[str, str]
        "keywords"         : list[str]
    """
    if analyzer is None:
        analyzer = get_analyzer()

    extraction = extract_aspects_and_keywords(original_review, aspect_dict)
    aspects = extraction["aspects"]
    keywords = extraction["keywords"]
    aspect_keyword_map = extraction["aspect_keyword_map"]

    aspect_sentiment = predict_aspect_sentiment(original_review, aspect_keyword_map, analyzer)

    return {
        "review_id": review_id,
        "original_review": original_review,
        "aspects": aspects,
        "aspect_sentiment": aspect_sentiment,
        "keywords": keywords,
    }


if __name__ == "__main__":
    result = analyze_single_review(1, "The quality is good but delivery was slow.")
    print(result)
