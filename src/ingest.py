from pathlib import Path

import pandas as pd


def _resolve_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError(f"Missing required column. Expected one of: {candidates}")


def load_bank_statement(file_path: str | Path) -> list[dict]:
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.lower().str.strip()

    date_col = _resolve_column(
        df, ["date", "transaction date", "posting date", "trans date"]
    )
    desc_col = _resolve_column(
        df, ["description", "desc", "memo", "details", "narrative"]
    )

    has_debit_credit = "debit" in df.columns and "credit" in df.columns
    has_amount = "amount" in df.columns
    has_type = "type" in df.columns

    if not has_debit_credit and not has_amount:
        raise ValueError("CSV must have either 'amount' or 'debit'/'credit' columns")

    transactions = []

    for _, row in df.iterrows():
        date = str(row[date_col]).strip()
        description = str(row[desc_col]).strip()

        if has_debit_credit:
            debit = pd.to_numeric(row["debit"], errors="coerce")
            credit = pd.to_numeric(row["credit"], errors="coerce")
            debit = 0.0 if pd.isna(debit) else float(debit)
            credit = 0.0 if pd.isna(credit) else float(credit)

            if debit > 0:
                amount, txn_type = debit, "debit"
            elif credit > 0:
                amount, txn_type = credit, "credit"
            else:
                continue
        else:
            raw_amount = pd.to_numeric(row["amount"], errors="coerce")
            if pd.isna(raw_amount):
                continue

            amount = float(raw_amount)
            if has_type:
                txn_type = str(row["type"]).strip().lower()
                amount = abs(amount)
            elif amount < 0:
                txn_type, amount = "debit", abs(amount)
            else:
                txn_type, amount = "credit", amount

        txn = {
            "date": date,
            "description": description,
            "amount": amount,
            "type": txn_type,
        }
        # Pass through any extra columns like year
        for col in df.columns:
            if col not in ["date", "description", "amount", "type",
                          date_col, desc_col, "debit", "credit"] and col not in txn:
                val = row[col]
                if not pd.isna(val):
                    txn[col] = val
        transactions.append(txn)

    print(f"Loaded {len(transactions)} transactions")
    return transactions


if __name__ == "__main__":
    sample_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "sample_statements"
        / "test_statement.csv"
    )
    transactions = load_bank_statement(sample_path)
    for txn in transactions[:3]:
        print(txn)
