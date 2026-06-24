"""
absb
====

Rule-based Aspect-Based Sentiment Analysis (ABSA) package for the
E-Commerce Review Analysis System.

Pipeline:
    Review Corpus
        -> Aspect Extraction        (dictionary matching)
        -> Aspect Sentiment         (VADER)
        -> Statistics Generation
        -> Recommendation Generation
        -> Output Files (absb/outputs/)

Run the full pipeline with:
    python -m absb.run_pipeline
"""

__all__ = [
    "config",
    "aspect_dictionary",
    "aspect_extraction",
    "aspect_sentiment",
    "single_review_analysis",
    "main",
    "statistics",
    "recommendation",
    "run_pipeline",
]
