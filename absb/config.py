"""
absb.config
============

Centralized configuration for the ABSA package.

Every file path / folder location used anywhere else in the ``absb``
package MUST be imported from this module. No other module should
hardcode a path.
"""

import os

# -----------------------------------------------------------------------
# Base locations
# -----------------------------------------------------------------------
# This file lives at: <project_root>/absb/config.py
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../absb
PROJECT_ROOT = os.path.dirname(PACKAGE_DIR)                       # .../E-Commerce-Review-Analysis-System

# -----------------------------------------------------------------------
# Input
# -----------------------------------------------------------------------
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
INPUT_FILE = os.path.join(DATA_DIR, "final_corpus.csv")

# Required columns in the input corpus
REVIEW_ID_COLUMN = "review_id"
REVIEW_TEXT_COLUMN = "original_review"

# -----------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------
OUTPUT_DIR = os.path.join(PACKAGE_DIR, "outputs")

ASPECT_RESULTS_FILE = os.path.join(OUTPUT_DIR, "aspect_results.csv")
ASPECT_DISTRIBUTION_FILE = os.path.join(OUTPUT_DIR, "aspect_distribution.csv")
ASPECT_SENTIMENT_TABLE_FILE = os.path.join(OUTPUT_DIR, "aspect_sentiment_table.csv")
RECOMMENDATIONS_FILE = os.path.join(OUTPUT_DIR, "recommendations.csv")
ASPECT_DICTIONARY_FILE = os.path.join(OUTPUT_DIR, "aspect_dictionary.json")


def ensure_output_dir() -> None:
    """Create the absb/outputs/ directory if it doesn't already exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
