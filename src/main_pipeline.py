import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from extract import categorize_transaction
from ingest import load_bank_statement
from invoice_extractor import extract_invoice_data
from receipt_extractor import extract_receipt_data

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

STATEMENT_PATH = DATA_DIR / "sample_statements" / "extracted_from_pdf.csv"
INVOICES_DIR = DATA_DIR / "sample_invoices"
RECEIPTS_DIR = DATA_DIR / "sample_receipts"

DEFAULT_PERIOD_LABEL = "Unknown period"
AMOUNT_TOLERANCE = 0.01


def _to_amount(value) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = re.sub(r"[^\d.\-]", "", str(value))
    try:
        return float(cleaned) if cleaned else 0.0
    except ValueError:
        return 0.0


def _normalize_status(status) -> str:
    return "paid" if str(status).lower().strip() == "paid" else "unpaid"


def _format_money(amount: float) -> str:
    sign = "-" if amount < 0 else ""
    return f"{sign}${abs(amount):,.2f}"


def _amounts_match(a: float, b: float) -> bool:
    return abs(a - b) <= AMOUNT_TOLERANCE


def _get_period_label(transactions: list[dict]) -> str:
    dates = [t.get("date", "") for t in transactions if t.get("date")]
    if not dates:
        return "Unknown Period"

    # Extract year from transactions if available
    years = []
    for t in transactions:
        year = t.get("year", "")
        if year and str(year) not in ("", "nan", "None"):
            years.append(str(int(float(str(year)))))

    # Also search for year in date strings themselves
    if not years:
        for d in dates:
            year_match = re.search(r'\b(20\d{2}|19\d{2})\b', str(d))
            if year_match:
                years.append(year_match.group(1))

    year_suffix = f" {years[0]}" if years else ""
    return f"{dates[0]} {year_suffix.strip()} to {dates[-1]} {year_suffix.strip()}"


def _categorize_transactions(transactions: list[dict]) -> list[dict]:
    categorized = []
    for txn in tqdm(transactions, desc="Step 1: Categorizing transactions"):
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
        categorized.append(row)
    return categorized


def _process_invoices(folder: Path) -> list[dict]:
    pdf_files = sorted(folder.glob("*.pdf")) if folder.is_dir() else []
    invoices = []

    for pdf_path in tqdm(pdf_files, desc="Step 2: Extracting invoices"):
        data = extract_invoice_data(pdf_path)
        if "error" in data:
            continue
        invoices.append(
            {
                "source_file": pdf_path.name,
                "invoice_number": data.get("invoice_number", ""),
                "invoice_date": data.get("invoice_date", ""),
                "due_date": data.get("due_date", ""),
                "vendor_name": data.get("vendor_name", ""),
                "client_name": data.get("client_name", ""),
                "total_amount": _to_amount(data.get("total_amount")),
                "status": _normalize_status(data.get("status", "unpaid")),
            }
        )
    return invoices


def _process_receipts(folder: Path) -> list[dict]:
    if not folder.is_dir():
        return []

    image_files = sorted(
        p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    receipts = []

    for image_path in tqdm(image_files, desc="Step 3: Extracting receipts"):
        data = extract_receipt_data(image_path)
        if "error" in data:
            continue
        receipts.append(
            {
                "source_file": image_path.name,
                "merchant_name": data.get("merchant_name", ""),
                "date": data.get("date", ""),
                "subtotal": _to_amount(data.get("subtotal")),
                "tax": _to_amount(data.get("tax")),
                "total": _to_amount(data.get("total")),
                "category": data.get("category", "Other"),
                "items": data.get("items", []),
            }
        )
    return receipts


def _reconcile(transactions: list[dict], invoices: list[dict]) -> list[dict]:
    used_invoice_indices: set[int] = set()
    report = []

    for txn in transactions:
        matched_invoice = None
        match_index = None

        for idx, invoice in enumerate(invoices):
            if idx in used_invoice_indices:
                continue
            if _amounts_match(txn["amount"], invoice["total_amount"]):
                matched_invoice = invoice
                match_index = idx
                break

        if match_index is not None:
            used_invoice_indices.add(match_index)

        report.append(
            {
                "date": txn["date"],
                "description": txn["description"],
                "amount": txn["amount"],
                "type": txn["type"],
                "category": txn.get("category", "Other"),
                "reconciliation_status": "reconciled" if matched_invoice else "unreconciled",
                "matched_invoice_number": (
                    matched_invoice["invoice_number"] if matched_invoice else ""
                ),
                "matched_vendor": (
                    matched_invoice["vendor_name"] if matched_invoice else ""
                ),
            }
        )

    return report


def _build_expense_ledger(expenses: list[dict]) -> list[dict]:
    ledger = []
    for txn in expenses:
        ledger.append(
            {
                "date": txn["date"],
                "description": txn["description"],
                "amount": txn["amount"],
                "category": txn.get("category", "Other"),
                "confidence": txn.get("confidence", 0.0),
            }
        )
    return sorted(ledger, key=lambda r: (r["category"], r["date"]))


def _expenses_by_category(expenses: list[dict]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for txn in expenses:
        totals[txn.get("category", "Other")] += txn["amount"]
    return dict(sorted(totals.items(), key=lambda x: x[0]))


def _print_summary(
    total_income: float,
    expense_totals: dict[str, float],
    total_expenses: float,
    net_profit: float,
    unpaid_total: float,
    matched_count: int,
    unmatched_count: int,
    period_label: str,
) -> None:
    width = 32
    print("\n" + "=" * width)
    print("LEDGERAI - BOOKS OF ACCOUNTS")
    print(f"Period: {period_label}")
    print("=" * width)
    print("INCOME")
    print(f"Total Revenue:        {_format_money(total_income):>12}")
    print()
    print("EXPENSES")
    for category, amount in expense_totals.items():
        label = f"{category}:"
        print(f"{label:<22}{_format_money(amount):>12}")
    print(f"{'Total Expenses:':<22}{_format_money(total_expenses):>12}")
    print()
    print(f"{'NET PROFIT:':<22}{_format_money(net_profit):>12}")
    print()
    print("ACCOUNTS RECEIVABLE")
    print(f"Unpaid invoices:      {_format_money(unpaid_total):>12}")
    print()
    print("RECONCILIATION")
    print(f"Matched:              {matched_count:>12} transactions")
    print(f"Unmatched:            {unmatched_count:>12} transactions")
    print("=" * width + "\n")


def run_full_pipeline(
    bank_statement_path: str | Path | None = None,
    invoices_folder: str | Path | None = None,
    receipts_folder: str | Path | None = None,
) -> dict:
    statement_path = Path(bank_statement_path or STATEMENT_PATH)
    invoices_dir = Path(invoices_folder or INVOICES_DIR)
    receipts_dir = Path(receipts_folder or RECEIPTS_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # STEP 1 - Bank statement
    print("\n--- STEP 1: Processing bank statement ---")
    try:
        raw_transactions = load_bank_statement(statement_path)
    except Exception as e:
        print(f"Failed to load bank statement: {e}")
        raw_transactions = []

    period_label = _get_period_label(raw_transactions)
    print(f"Statement period: {period_label}")

    categorized = _categorize_transactions(raw_transactions) if raw_transactions else []
    income = [t for t in categorized if t["type"] == "credit"]
    expenses = [t for t in categorized if t["type"] == "debit"]

    # STEP 2 - Invoices
    print("\n--- STEP 2: Processing invoices ---")
    invoices = _process_invoices(invoices_dir)
    paid_invoices = [i for i in invoices if i["status"] == "paid"]
    unpaid_invoices = [i for i in invoices if i["status"] == "unpaid"]

    # STEP 3 - Receipts
    print("\n--- STEP 3: Processing receipts ---")
    receipts = _process_receipts(receipts_dir)

    # STEP 4 - Reconciliation
    print("\n--- STEP 4: Reconciliation ---")
    reconciliation = _reconcile(categorized, invoices)
    matched_count = sum(1 for r in reconciliation if r["reconciliation_status"] == "reconciled")
    unmatched_count = len(reconciliation) - matched_count

    # STEP 5 - P&L
    print("\n--- STEP 5: Generating P&L ---")
    total_income = sum(t["amount"] for t in income)
    expense_totals = _expenses_by_category(expenses)
    total_expenses = sum(expense_totals.values())
    net_profit = total_income - total_expenses
    unpaid_total = sum(i["total_amount"] for i in unpaid_invoices)
    receipts_total = sum(r["total"] for r in receipts)

    expense_ledger = _build_expense_ledger(expenses)

    pl_rows = [
        {"line_item": "Total Revenue", "amount": total_income},
        *[
            {"line_item": f"Expense - {cat}", "amount": amt}
            for cat, amt in expense_totals.items()
        ],
        {"line_item": "Total Expenses", "amount": total_expenses},
        {"line_item": "Net Profit", "amount": net_profit},
        {"line_item": "Accounts Receivable (Unpaid Invoices)", "amount": unpaid_total},
        {"line_item": "Total Receipts", "amount": receipts_total},
    ]

    # STEP 6 - Save outputs
    print("\n--- STEP 6: Saving outputs ---")
    pd.DataFrame(categorized).to_csv(OUTPUT_DIR / "cash_book.csv", index=False)
    pd.DataFrame(unpaid_invoices).to_csv(
        OUTPUT_DIR / "accounts_receivable.csv", index=False
    )
    pd.DataFrame(expense_ledger).to_csv(OUTPUT_DIR / "expense_ledger.csv", index=False)
    pd.DataFrame(reconciliation).to_csv(
        OUTPUT_DIR / "reconciliation_report.csv", index=False
    )
    pd.DataFrame(pl_rows).to_csv(OUTPUT_DIR / "pl_statement.csv", index=False)

    print(f"Saved 5 reports to {OUTPUT_DIR}")

    # STEP 7 - Summary
    _print_summary(
        total_income=total_income,
        expense_totals=expense_totals,
        total_expenses=total_expenses,
        net_profit=net_profit,
        unpaid_total=unpaid_total,
        matched_count=matched_count,
        unmatched_count=unmatched_count,
        period_label=period_label,
    )

    return {
        "summary": {
            "total_revenue": total_income,
            "total_expenses": total_expenses,
            "net_profit": net_profit,
            "accounts_receivable": unpaid_total,
            "transactions_processed": len(categorized),
            "invoices_processed": len(invoices),
            "receipts_processed": len(receipts),
            "matched_transactions": matched_count,
            "unmatched_transactions": unmatched_count,
            "period_label": period_label,
        },
        "pl_statement": pl_rows,
        "cash_book": categorized,
        "accounts_receivable": unpaid_invoices,
        "expense_ledger": expense_ledger,
        "reconciliation": reconciliation,
        "receipts": receipts,
        "transactions": categorized,
        "income": income,
        "expenses": expenses,
        "invoices": invoices,
        "paid_invoices": paid_invoices,
        "unpaid_invoices": unpaid_invoices,
        "pl": {
            "total_income": total_income,
            "expense_totals": expense_totals,
            "total_expenses": total_expenses,
            "net_profit": net_profit,
            "unpaid_total": unpaid_total,
            "receipts_total": receipts_total,
        },
    }


if __name__ == "__main__":
    run_full_pipeline()