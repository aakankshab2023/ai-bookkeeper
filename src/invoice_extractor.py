import json
import os
import re
from pathlib import Path

import pandas as pd
import pdfplumber
from dotenv import load_dotenv
from groq import Groq
from tqdm import tqdm

from prompts import INVOICE_EXTRACTION_PROMPT

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

CSV_COLUMNS = [
    "invoice_number",
    "invoice_date",
    "due_date",
    "vendor_name",
    "client_name",
    "total_amount",
    "status",
]
OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent / "outputs" / "extracted_invoices.csv"
)


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    pages: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())

    if not pages:
        raise ValueError(f"No text could be extracted from: {path}")

    return "\n\n".join(pages)


def _parse_json_response(raw: str) -> dict:
    text = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence_match:
        text = fence_match.group(1).strip()
    return json.loads(text)


def extract_invoice_data(pdf_path: str | Path) -> dict:
    """Extract structured invoice fields from a PDF using pdfplumber + Groq."""
    try:
        text = extract_text_from_pdf(pdf_path)
    except (FileNotFoundError, ValueError) as e:
        return {"error": str(e)}

    user_message = f"Invoice text:\n\n{text}"

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": INVOICE_EXTRACTION_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        raw = response.choices[0].message.content or ""
    except Exception as e:
        return {"error": f"Groq API call failed: {e}"}

    try:
        return _parse_json_response(raw)
    except json.JSONDecodeError as e:
        return {
            "error": f"Failed to parse JSON from model response: {e}",
            "raw_response": raw,
        }


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


def _to_csv_row(data: dict) -> dict | None:
    if "error" in data:
        return None
    return {
        "invoice_number": data.get("invoice_number", ""),
        "invoice_date": data.get("invoice_date", ""),
        "due_date": data.get("due_date", ""),
        "vendor_name": data.get("vendor_name", ""),
        "client_name": data.get("client_name", ""),
        "total_amount": _to_amount(data.get("total_amount")),
        "status": _normalize_status(data.get("status", "unpaid")),
    }


def process_all_invoices(folder_path: str | Path) -> list[dict]:
    folder = Path(folder_path)
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    pdf_files = sorted(folder.glob("*.pdf"))
    rows: list[dict] = []

    for pdf_path in tqdm(pdf_files, desc="Extracting invoices"):
        data = extract_invoice_data(pdf_path)
        row = _to_csv_row(data)
        if row:
            rows.append(row)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(OUTPUT_PATH, index=False)

    total_processed = len(pdf_files)
    unpaid = [r for r in rows if r["status"] == "unpaid"]
    paid = [r for r in rows if r["status"] == "paid"]
    unpaid_total = sum(r["total_amount"] for r in unpaid)

    print(f"{total_processed} invoices processed")
    print(f"{len(unpaid)} unpaid invoices totaling ${unpaid_total:,.2f}")
    print(f"{len(paid)} paid invoices")

    return rows


if __name__ == "__main__":
    sample_folder = (
        Path(__file__).resolve().parent.parent / "data" / "sample_invoices"
    )
    process_all_invoices(sample_folder)
