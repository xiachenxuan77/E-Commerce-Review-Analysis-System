"""
absb.aspect_sentiment
========================

Aspect-level sentiment classification using VADER (Valence Aware
Dictionary and sEntiment Reasoner).

No BERT / RoBERTa / DeBERTa / DistilBERT is used here.

Strategy
--------
A single review can mention several aspects with *different* sentiment
("The quality is good but delivery was slow."). Running VADER once on
the whole review would blend those opposite sentiments together, so
instead:

1. The review is split into clauses (on sentence punctuation and on
   contrastive conjunctions such as "but", "however", "although", ...).
2. For each aspect, we find the clause(s) that actually contain one of
   that aspect's matched keywords and run VADER only on those clauses.
3. If an aspect's keywords don't fall cleanly into a single clause
   (e.g. no clause boundaries detected), VADER falls back to scoring
   the entire review for that aspect.
4. The averaged VADER "compound" score is converted into
   Positive / Neutral / Negative using the standard VADER thresholds.
"""

import re
from typing import Dict, List, Optional

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# -----------------------------------------------------------------------
# Clause splitting
# -----------------------------------------------------------------------
_CLAUSE_SPLIT_PATTERN = re.compile(
    r"[.!?;]+|\b(?:but|however|although|though|yet|whereas|except that)\b",
    re.IGNORECASE,
)

# Standard VADER compound-score thresholds
_POSITIVE_THRESHOLD = 0.05
_NEGATIVE_THRESHOLD = -0.05

# Keyword-in-clause pattern cache (reused across all reviews)
_KEYWORD_PATTERN_CACHE: Dict[str, "re.Pattern"] = {}

# -----------------------------------------------------------------------
# Domain lexicon augmentation.
# VADER's general-purpose lexicon under-scores some single-word terms
# that are strongly polarized in an e-commerce review context (e.g.
# "slow" delivery, "damaged" packaging). This extends VADER's existing
# lexicon with a small set of domain-relevant words -- it is still pure
# VADER (a dictionary lookup), just with a richer dictionary. Scores
# follow VADER's standard -4..+4 convention.
# -----------------------------------------------------------------------
_DOMAIN_LEXICON_UPDATES: Dict[str, float] = {
    "slow": -1.9,
    "late": -1.8,
    "delayed": -1.8,
    "delay": -1.6,
    "damaged": -2.5,
    "broken": -2.5,
    "defective": -2.8,
    "flimsy": -2.0,
    "overpriced": -2.0,
    "pricey": -0.8,
    "uncomfortable": -1.8,
    "fast": 1.8,
    "sturdy": 1.8,
    "durable": 1.8,
    "comfortable": 2.0,
    "useful": 1.8,
    "convenient": 1.6,
    "cozy": 1.8,
    "stylish": 1.6,
    "affordable": 1.4,
}


def get_analyzer() -> SentimentIntensityAnalyzer:
    """Create a new VADER SentimentIntensityAnalyzer with domain lexicon updates."""
    analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon.update(_DOMAIN_LEXICON_UPDATES)
    return analyzer


def _get_keyword_pattern(keyword: str) -> "re.Pattern":
    key = keyword.lower().strip()
    pattern = _KEYWORD_PATTERN_CACHE.get(key)
    if pattern is None:
        pattern = re.compile(r"\b" + re.escape(key) + r"\b")
        _KEYWORD_PATTERN_CACHE[key] = pattern
    return pattern


def _keyword_in_text(keyword: str, lower_text: str) -> bool:
    return bool(_get_keyword_pattern(keyword).search(lower_text))


def _split_into_clauses(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    raw_parts = _CLAUSE_SPLIT_PATTERN.split(text)
    clauses = [c.strip() for c in raw_parts if c and c.strip()]
    return clauses if clauses else [text.strip()]


def _classify_compound(compound: float) -> str:
    if compound >= _POSITIVE_THRESHOLD:
        return "Positive"
    if compound <= _NEGATIVE_THRESHOLD:
        return "Negative"
    return "Neutral"


def predict_aspect_sentiment(
    text: str,
    aspect_keyword_map: Dict[str, List[str]],
    analyzer: Optional[SentimentIntensityAnalyzer] = None,
) -> Dict[str, str]:
    """
    Predict sentiment polarity (Positive/Neutral/Negative) for each aspect
    found in a review.

    Parameters
    ----------
    text : str
        The raw review text.
    aspect_keyword_map : dict
        aspect -> list of matched keywords, as produced by
        absb.aspect_extraction.extract_aspects_and_keywords().
    analyzer : SentimentIntensityAnalyzer, optional
        Reused analyzer instance (recommended for batch processing).

    Returns
    -------
    dict[str, str]
        e.g. {"Quality": "Positive", "Delivery": "Negative"}
    """
    if analyzer is None:
        analyzer = get_analyzer()

    aspect_sentiment: Dict[str, str] = {}

    if not aspect_keyword_map:
        return aspect_sentiment

    clauses = _split_into_clauses(text)

    if not clauses:
        for aspect in aspect_keyword_map:
            aspect_sentiment[aspect] = "Neutral"
        return aspect_sentiment

    lower_clauses = [c.lower() for c in clauses]
    clause_score_cache: Dict[int, float] = {}
    whole_review_score: Optional[float] = None

    def clause_compound(idx: int) -> float:
        if idx not in clause_score_cache:
            clause_score_cache[idx] = analyzer.polarity_scores(clauses[idx])["compound"]
        return clause_score_cache[idx]

    for aspect, keyword_list in aspect_keyword_map.items():
        matched_indices = [
            i for i, lower_clause in enumerate(lower_clauses)
            if any(_keyword_in_text(kw, lower_clause) for kw in keyword_list)
        ]

        if matched_indices:
            scores = [clause_compound(i) for i in matched_indices]
            avg_compound = sum(scores) / len(scores)
        else:
            # Fallback: aspect keyword matched the full text but not any
            # individual clause (rare) -- score the whole review instead.
            if whole_review_score is None:
                safe_text = text if isinstance(text, str) else ""
                whole_review_score = analyzer.polarity_scores(safe_text)["compound"]
            avg_compound = whole_review_score

        aspect_sentiment[aspect] = _classify_compound(avg_compound)

    return aspect_sentiment


if __name__ == "__main__":
    from absb.aspect_extraction import extract_aspects_and_keywords

    sample = "The quality is good but delivery was slow."
    extraction = extract_aspects_and_keywords(sample)
    result = predict_aspect_sentiment(sample, extraction["aspect_keyword_map"])
    print(result)
