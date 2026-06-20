"""
aspect_sentiment.py
Predicts sentiment per aspect using VADER (fast, no GPU needed).
Optionally falls back to DistilBERT for higher accuracy.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import List, Optional

_vader = SentimentIntensityAnalyzer()

# ── Optional DistilBERT (loaded lazily to avoid slow startup) ──────────────
_distilbert_pipeline = None


def _get_distilbert():
    global _distilbert_pipeline
    if _distilbert_pipeline is None:
        from transformers import pipeline
        _distilbert_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
            max_length=512,
        )
    return _distilbert_pipeline


# ── Helpers ────────────────────────────────────────────────────────────────

def _vader_label(compound: float) -> str:
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    return "Neutral"


def _build_aspect_context(review: str, aspect: str, keywords: list[str]) -> str:
    """
    Build a short context string for the aspect.
    Tries to extract the sentence(s) containing the aspect keywords;
    falls back to the full review.
    """
    import re
    sentences = re.split(r"[.!?]+", review)
    relevant = []
    kw_lower = {k.lower() for k in keywords}
    for sent in sentences:
        sent_lower = sent.lower()
        if any(kw in sent_lower for kw in kw_lower):
            relevant.append(sent.strip())
    return " ".join(relevant) if relevant else review


# ── Public API ─────────────────────────────────────────────────────────────

def predict_aspect_sentiment(
    review: str,
    aspects: List[str],
    keywords: Optional[List[str]] = None,
    method: str = "vader",
) -> dict:
    """
    Predict sentiment for each aspect in a review.

    Parameters
    ----------
    review   : str        – original review text
    aspects  : list[str]  – aspect names to score (from extract_aspects)
    keywords : list[str]  – matched keywords (used to isolate context)
    method   : str        – "vader" (default, fast) or "distilbert" (slower, more accurate)

    Returns
    -------
    dict mapping aspect -> "Positive" | "Neutral" | "Negative"

    Example
    -------
    >>> predict_aspect_sentiment(
    ...     "The quality is good but delivery was slow.",
    ...     ["Quality", "Delivery"]
    ... )
    {'Quality': 'Positive', 'Delivery': 'Negative'}
    """
    if not aspects:
        return {}

    from aspect_dictionary import get_aspect_dictionary
    aspect_dict = get_aspect_dictionary()

    results = {}

    for aspect in aspects:
        aspect_keywords = aspect_dict.get(aspect, [])
        context = _build_aspect_context(review, aspect, aspect_keywords + (keywords or []))

        if method == "distilbert":
            pipe = _get_distilbert()
            out = pipe(context)[0]
            label = out["label"].capitalize()
            # DistilBERT returns POSITIVE / NEGATIVE only; map to our labels
            if label not in ("Positive", "Negative"):
                label = "Neutral"
            results[aspect] = label
        else:
            scores = _vader.polarity_scores(context)
            results[aspect] = _vader_label(scores["compound"])

    return results


if __name__ == "__main__":
    review = "The quality is good but delivery was slow."
    aspects = ["Quality", "Delivery"]
    print(predict_aspect_sentiment(review, aspects))
