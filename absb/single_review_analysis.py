"""
single_review_analysis.py
End-to-end analysis of a single review: extraction → sentiment → final output.
"""

from aspect_extraction import extract_aspects
from aspect_sentiment import predict_aspect_sentiment
import re

def clean_html(text):
    return re.sub(r'<.*?>', '', str(text))

def analyze_review(review: str, method: str = "vader") -> dict:
    review = clean_html(review)
    """
    Full ABSA pipeline for a single review.

    Parameters
    ----------
    review : str   – raw review text
    method : str   – sentiment method: "vader" (default) or "distilbert"

    Returns
    -------
    {
        "aspects":          ["Quality", "Delivery"],
        "aspect_sentiment": {"Quality": "Positive", "Delivery": "Negative"},
        "keywords":         ["quality", "delivery", "slow"]
    }

    Example
    -------
    >>> result = analyze_review("The quality is good but delivery was slow.")
    >>> result["aspects"]
    ['Quality', 'Delivery']
    >>> result["aspect_sentiment"]
    {'Quality': 'Positive', 'Delivery': 'Negative'}
    """
    extraction = extract_aspects(review)
    aspects = extraction["aspects"]
    keywords = extraction["keywords"]

    aspect_sentiment = predict_aspect_sentiment(
        review, aspects, keywords=keywords, method=method
    )

    return {
        "aspects": aspects,
        "aspect_sentiment": aspect_sentiment,
        "keywords": keywords,
    }


if __name__ == "__main__":
    import json

    samples = [
        "The quality is good but delivery was slow.",
        "Packaging was damaged and seller did not reply.",
        "Great battery life and beautiful display. Price is a bit high though.",
        "Fast shipping! The screen looks amazing.",
    ]

    for s in samples:
        result = analyze_review(s)
        print("Review:", s)
        print(json.dumps(result, indent=2))
        print()
