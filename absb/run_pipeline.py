"""
absb.run_pipeline
====================

Single entry point for the full ABSA pipeline.

Run with:
    python -m absb.run_pipeline

Pipeline order:
    main            -> aspect_results.csv
    statistics      -> aspect_distribution.csv, aspect_sentiment_table.csv
    recommendation  -> recommendations.csv
    aspect_dictionary -> aspect_dictionary.json

All five output files are written to absb/outputs/.
"""

import time

from absb import config
from absb.aspect_dictionary import save_aspect_dictionary_json
from absb.main import run as run_main
from absb.recommendation import generate_recommendations
from absb.statistics import generate_statistics


def run_pipeline() -> None:
    start = time.time()

    config.ensure_output_dir()

    print("=" * 60)
    print("Running ABSA Pipeline")
    print("=" * 60)

    # 1. Aspect extraction (dictionary matching) + aspect-level sentiment (VADER)
    results = run_main()

    # 2. Statistics: aspect distribution + aspect sentiment table
    distribution, sentiment_table = generate_statistics(results)

    # 3. Business recommendations
    generate_recommendations(sentiment_table)

    # 4. Export the aspect dictionary used for extraction
    dict_path = save_aspect_dictionary_json()
    print(f"[aspect_dictionary] Saved: {dict_path}")

    elapsed = time.time() - start
    print("=" * 60)
    print(f"Pipeline complete in {elapsed:.2f}s.")
    print(f"Outputs written to: {config.OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
