"""
main.py
Processes final_corpus.csv row by row through the ABSA pipeline
and writes aspect_results.csv.
"""

import os
import csv
import time
import pandas as pd
from single_review_analysis import analyze_review

BASE_DIR   = os.path.dirname(__file__)
INPUT_CSV  = os.path.join(BASE_DIR, "data", "final_corpus.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "outputs", "aspect_results.csv")

BATCH_SIZE  = 5_000   # flush to disk every N rows
METHOD      = "vader" # "vader" (fast) or "distilbert" (slower, GPU-optional)
MAX_ROWS    = 50    # set an int to limit rows for testing, e.g. 10_000


def _fmt_list(lst: list) -> str:
    return ";".join(str(x) for x in lst)


def _fmt_dict(d: dict) -> str:
    return ";".join(f"{k}:{v}" for k, v in d.items())


def run(input_csv: str = INPUT_CSV, output_csv: str = OUTPUT_CSV, method: str = METHOD):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    print(f"[main] Reading {input_csv} ...")
    df = pd.read_csv(input_csv, nrows=MAX_ROWS)
    total = len(df)
    print(f"[main] Total reviews to process: {total:,}")

    fieldnames = [
        "review_id", "original_review",
        "aspects", "aspect_sentiment", "keywords",
    ]

    start = time.time()
    processed = 0
    buffer = []

    with open(output_csv, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for _, row in df.iterrows():
            review_id      = row.get("review_id", processed + 1)
            original_review = str(row.get("original_review", ""))

            result = analyze_review(original_review, method=method)

            buffer.append({
                "review_id":       review_id,
                "original_review": original_review,
                "aspects":         _fmt_list(result["aspects"]),
                "aspect_sentiment": _fmt_dict(result["aspect_sentiment"]),
                "keywords":        _fmt_list(result["keywords"]),
            })

            processed += 1

            if len(buffer) >= BATCH_SIZE:
                writer.writerows(buffer)
                buffer.clear()
                elapsed = time.time() - start
                rate = processed / elapsed
                eta  = (total - processed) / rate if rate > 0 else 0
                print(
                    f"  [{processed:>7,}/{total:,}]  "
                    f"{rate:,.0f} rev/s  ETA {eta/60:.1f} min"
                )

        if buffer:
            writer.writerows(buffer)

    elapsed = time.time() - start
    print(f"\n[main] Done! {processed:,} reviews in {elapsed:.1f}s → {output_csv}")


if __name__ == "__main__":
    run()
