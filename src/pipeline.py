from pathlib import Path

import pandas as pd
from tqdm import tqdm

from extract import categorize_transaction
from ingest import load_bank_statement

CONFIDENCE_THRESHOLD = 0.75
OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent / "outputs" / "categorized_transactions.csv"
)


def process_statement(csv_path: str | Path) -> list[dict]:
    transactions = load_bank_statement(csv_path)
    results = []

    for txn in tqdm(transactions, desc="Categorizing transactions"):
        try:
            ai_result = categorize_transaction(
                txn["description"], txn["amount"], txn["type"]
            )
            row = {**txn, **ai_result}
        except Exception as e:
            row = {
                **txn,
                "category": "Other",
                "confidence": 0.0,
                "reasoning": f"Categorization failed: {e}",
            }
        results.append(row)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)

    total = len(results)
    auto_categorized = sum(
        1 for r in results if r.get("confidence", 0) >= CONFIDENCE_THRESHOLD
    )
    flagged = sum(
        1 for r in results if r.get("confidence", 0) < CONFIDENCE_THRESHOLD
    )

    print(f"{total} transactions processed")
    print(f"{auto_categorized} auto-categorized (confidence >= 0.75)")
    print(f"{flagged} flagged for review (confidence < 0.75)")

    return results


if __name__ == "__main__":
    sample_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "sample_statements"
        / "extracted_from_pdf.csv"
    )
    process_statement(sample_path)
