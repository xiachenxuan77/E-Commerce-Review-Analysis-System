"""
absb.aspect_extraction
========================

Rule-based aspect & keyword extraction using dictionary matching.

No BERT / Transformer / LLM / Topic Modeling is used here -- aspects are
detected purely by matching keywords from absb.aspect_dictionary against
the (lower-cased) review text using whole-word / whole-phrase regex
matching.
"""

import re
from typing import Dict, List, Optional

from absb.aspect_dictionary import get_aspect_dictionary

# -----------------------------------------------------------------------
# Compiled-pattern cache.
# Keywords repeat across tens of thousands of reviews, so we compile each
# keyword's regex pattern once and reuse it for every review.
# -----------------------------------------------------------------------
_PATTERN_CACHE: Dict[str, "re.Pattern"] = {}


def _get_pattern(keyword: str) -> "re.Pattern":
    """Return (and cache) a compiled whole-word/whole-phrase regex for a keyword."""
    key = keyword.lower().strip()
    pattern = _PATTERN_CACHE.get(key)
    if pattern is None:
        pattern = re.compile(r"\b" + re.escape(key) + r"\b")
        _PATTERN_CACHE[key] = pattern
    return pattern


def _normalize(text) -> str:
    """Lower-case and guard against non-string / NaN input."""
    if not isinstance(text, str):
        return ""
    return text.lower()


def extract_aspects_and_keywords(
    text: str,
    aspect_dict: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, object]:
    """
    Perform dictionary-based aspect & keyword extraction on a single review.

    Parameters
    ----------
    text : str
        The raw review text (``original_review``).
    aspect_dict : dict, optional
        Aspect -> keyword-list mapping. Defaults to
        absb.aspect_dictionary.get_aspect_dictionary().

    Returns
    -------
    dict with:
        "aspects"           : list[str]            distinct aspects found
        "keywords"          : list[str]             distinct matched keywords
        "aspect_keyword_map": dict[str, list[str]]  aspect -> matched keywords
                              (used internally by absb.aspect_sentiment to
                              locate which part of the review to score)
    """
    if aspect_dict is None:
        aspect_dict = get_aspect_dictionary()

    normalized_text = _normalize(text)

    aspects: List[str] = []
    keywords: List[str] = []
    aspect_keyword_map: Dict[str, List[str]] = {}

    if not normalized_text:
        return {"aspects": aspects, "keywords": keywords, "aspect_keyword_map": aspect_keyword_map}

    for aspect, keyword_list in aspect_dict.items():
        matched_keywords = []
        for keyword in keyword_list:
            if _get_pattern(keyword).search(normalized_text):
                matched_keywords.append(keyword)

        if matched_keywords:
            aspects.append(aspect)
            aspect_keyword_map[aspect] = matched_keywords
            for kw in matched_keywords:
                if kw not in keywords:
                    keywords.append(kw)

    return {"aspects": aspects, "keywords": keywords, "aspect_keyword_map": aspect_keyword_map}


if __name__ == "__main__":
    sample = "The quality is good but delivery was slow."
    result = extract_aspects_and_keywords(sample)
    print(result)
