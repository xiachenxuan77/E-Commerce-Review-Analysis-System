"""
absb.aspect_dictionary
========================

Manually defined aspect -> keyword dictionary used for rule-based
(dictionary matching) aspect extraction, plus helpers to access and
export it.

NOTE: This is intentionally a plain Python dictionary (no ML / no
topic modeling) per the project requirements.
"""

import json
from copy import deepcopy
from typing import Dict, List, Optional

from absb import config

# -----------------------------------------------------------------------
# Aspect keyword dictionary
# -----------------------------------------------------------------------
# Each aspect maps to a list of keywords / short phrases. Keywords are
# matched case-insensitively against review text using whole-word /
# whole-phrase matching (see absb.aspect_extraction).
# -----------------------------------------------------------------------
ASPECT_KEYWORDS: Dict[str, List[str]] = {
    "Quality": [
        "quality", "durable", "durability", "material", "sturdy",
        "sturdiness", "well made", "well-made", "solid", "flimsy",
        "defective", "broke", "broken", "cheaply made", "poorly made",
    ],
    "Price": [
        "price", "cheap", "expensive", "cost", "value", "worth",
        "overpriced", "affordable", "pricey", "discount", "bargain",
    ],
    "Delivery": [
        "delivery", "shipping", "shipment", "courier", "arrived",
        "late", "delayed", "fast shipping", "on time", "tracking",
        "arrive",
    ],
    "Packaging": [
        "packaging", "package", "box", "wrapped", "wrapping",
        "damaged", "crushed", "dented", "seal", "boxed",
    ],
    "Customer Service": [
        "customer service", "support", "refund", "return", "replacement",
        "response", "complaint", "warranty", "contacted", "customer care",
    ],
    "Usability": [
        "easy to use", "useful", "functionality", "user friendly",
        "user-friendly", "instructions", "setup", "works well",
        "easy to install", "convenient", "easy to clean",
    ],
    "Size and Fit": [
        "size", "fit", "fits", "tight", "loose", "small", "large",
        "length", "width", "sizing", "oversized",
    ],
    "Design and Appearance": [
        "design", "look", "looks", "color", "colour", "style",
        "appearance", "beautiful", "cute", "stylish", "elegant",
    ],
    "Comfort": [
        "comfortable", "comfort", "soft", "cozy", "cosy",
        "uncomfortable", "ergonomic", "snug",
    ],
    "Taste and Flavor": [
        "taste", "tastes", "flavor", "flavour", "delicious", "smell",
        "aroma", "bitter", "sweet", "fresh",
    ],
}


def get_aspect_dictionary() -> Dict[str, List[str]]:
    """Return a deep copy of the aspect -> keyword dictionary."""
    return deepcopy(ASPECT_KEYWORDS)


def save_aspect_dictionary_json(output_path: Optional[str] = None) -> str:
    """
    Export the aspect dictionary to a JSON file.

    Parameters
    ----------
    output_path : str, optional
        Destination path. Defaults to absb/outputs/aspect_dictionary.json
        (config.ASPECT_DICTIONARY_FILE).

    Returns
    -------
    str
        The path the dictionary was written to.
    """
    if output_path is None:
        output_path = config.ASPECT_DICTIONARY_FILE

    config.ensure_output_dir()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ASPECT_KEYWORDS, f, indent=4, ensure_ascii=False)

    return output_path


if __name__ == "__main__":
    saved_path = save_aspect_dictionary_json()
    print(f"Aspect dictionary exported to: {saved_path}")
