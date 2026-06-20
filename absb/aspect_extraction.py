"""
aspect_extraction.py
Extracts aspects and keywords from a review using the aspect dictionary.
"""

import re
from aspect_dictionary import get_aspect_dictionary


def _tokenize(text: str) -> list[str]:
    """Lowercase and split text into word tokens."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def extract_aspects(review: str) -> dict:
    """
    Extract aspects and matched keywords from a review.

    Parameters
    ----------
    review : str
        Raw review text.

    Returns
    -------
    dict with keys:
        "aspects"  : list of matched aspect names
        "keywords" : list of matched keyword tokens (deduplicated, order-preserved)

    Example
    -------
    >>> result = extract_aspects("The quality is good but delivery was slow.")
    >>> result["aspects"]
    ['Quality', 'Delivery']
    >>> result["keywords"]
    ['quality', 'delivery', 'slow']
    """
    aspect_dict = get_aspect_dictionary()
    tokens = _tokenize(review)
    token_set = set(tokens)

    matched_aspects = []
    matched_keywords = []
    seen_keywords: set[str] = set()

    for aspect, keywords in aspect_dict.items():
        for kw in keywords:
            kw_lower = kw.lower()
            # Support multi-word keywords
            if " " in kw_lower:
                if kw_lower in review.lower():
                    if aspect not in matched_aspects:
                        matched_aspects.append(aspect)
                    if kw_lower not in seen_keywords:
                        matched_keywords.append(kw_lower)
                        seen_keywords.add(kw_lower)
            else:
                if kw_lower in token_set:
                    if aspect not in matched_aspects:
                        matched_aspects.append(aspect)
                    if kw_lower not in seen_keywords:
                        matched_keywords.append(kw_lower)
                        seen_keywords.add(kw_lower)

    # Also add any review tokens that were matched as keywords
    # (preserves original token order in review)
    ordered_keywords: list[str] = []
    seen_ordered: set[str] = set()
    for token in tokens:
        if token in seen_keywords and token not in seen_ordered:
            ordered_keywords.append(token)
            seen_ordered.add(token)
    # Append multi-word matches not captured above
    for kw in matched_keywords:
        if " " in kw and kw not in seen_ordered:
            ordered_keywords.append(kw)
            seen_ordered.add(kw)

    return {
        "aspects": matched_aspects,
        "keywords": ordered_keywords,
    }


if __name__ == "__main__":
    sample = "The quality is good but delivery was slow."
    result = extract_aspects(sample)
    print("Review:", sample)
    print("Aspects:", result["aspects"])
    print("Keywords:", result["keywords"])
