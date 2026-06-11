"""Extract bank transactions from PDF statements using a two-pass Groq pipeline."""

import base64
import io
import json
import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from pdf2image import convert_from_path
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
TEXT_MODEL = "llama-3.3-70b-versatile"

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_STATEMENTS_DIR = BASE_DIR / "data" / "sample_statements"
DUMMY_PDF_PATH = SAMPLE_STATEMENTS_DIR / "dummy_statement.pdf"
OUTPUT_CSV_PATH = SAMPLE_STATEMENTS_DIR / "extracted_from_pdf.csv"

TRANSCRIBE_PROMPT = """
Transcribe ALL text from this bank statement exactly as you see it, preserving the layout with spaces and line breaks. Include every word, number, symbol, and date — especially the statement period dates in the header which contain the year.
"""

PARSE_TRANSACTIONS_PROMPT = """You are an expert accountant. Below is raw text from a bank statement. Extract EVERY transaction.

Rules:
- If multiple transactions share one date row, extract each separately with the same date
- If a row has no date, use the most recent date
- Skip: balance brought forward, totals, headers, opening/closing balance rows
- Money out/debit = money leaving account
- Money in/credit = money entering account

CRITICAL RULE about missing dates:
When a row has no date, it ALWAYS belongs to the most recent date above it. Example:
| 24 February | Anytown Jewelers | 150.00 out |
|             | Direct Deposit   | 25.00 in   |
Both get date 24 February. Extract BOTH.
Never skip a dateless row - it is always a real transaction sharing the date above it.

Return ONLY valid JSON array:
[{{"date":"...","description":"...","amount":0.00,"type":"debit or credit"}}]

RAW TEXT:
{raw_text}"""


def _parse_json_array(raw: str) -> list[dict]:
    text = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence_match:
        text = fence_match.group(1).strip()

    candidates = [text]
    array_match = re.search(r"\[[\s\S]*\]", text)
    if array_match:
        candidates.append(array_match.group(0))

    parsed = None
    last_error: json.JSONDecodeError | None = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            break
        except json.JSONDecodeError as e:
            last_error = e

    if parsed is None:
        raise last_error or json.JSONDecodeError("No JSON array found", raw, 0)

    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict) and "transactions" in parsed:
        return parsed["transactions"]
    if isinstance(parsed, dict) and "error" in parsed:
        raise ValueError(parsed["error"])
    raise ValueError("Expected a JSON array of transactions")


def _parse_amount(value) -> float | None:
    try:
        if isinstance(value, (int, float)):
            return abs(float(value))
        cleaned = re.sub(r"[^\d.]", "", str(value))
        return abs(float(cleaned)) if cleaned else None
    except (TypeError, ValueError):
        return None


def _date_match_key(date_str: str) -> str:
    date_str = str(date_str).strip()
    mm_dd = re.match(r"(\d{1,2})/(\d{1,2})", date_str)
    if mm_dd:
        return f"{int(mm_dd.group(1)):02d}/{int(mm_dd.group(2)):02d}"
    iso = re.match(r"(\d{4})-(\d{2})-(\d{2})", date_str)
    if iso:
        return f"{int(iso.group(2)):02d}/{int(iso.group(3)):02d}"
    return date_str.lower()


def _amount_match_key(amount: float) -> float:
    return round(amount, 2)


def _detail_match_key(date_str: str, amount: float) -> tuple[str, float]:
    return (_date_match_key(date_str), _amount_match_key(amount))


def _pil_image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def pdf_to_images(pdf_path: str | Path) -> list[Image.Image]:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    try:
        return convert_from_path(
            str(path),
            fmt="png",
            dpi=200,
            poppler_path=r"C:\Release-26.02.0-0\poppler-26.02.0\Library\bin",
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to convert PDF to images: {e}. "
            "On Windows, install Poppler and add it to PATH "
            "(https://github.com/oschwartz10612/poppler-windows/releases)."
        ) from e


def transcribe_page(image: Image.Image, page_num: int) -> str:
    data_uri = _pil_image_to_base64(image)

    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": TRANSCRIBE_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_uri},
                        },
                    ],
                }
            ],
        )
        text = (response.choices[0].message.content or "").strip()
        print(f"  Page {page_num}: transcribed {len(text)} characters")
        return text
    except Exception as e:
        print(f"  Page {page_num}: transcription error - {e}")
        return ""


def parse_transactions_from_text(raw_text: str) -> list[dict]:
    if not raw_text.strip():
        return []

    prompt = PARSE_TRANSACTIONS_PROMPT.format(raw_text=raw_text)

    try:
        response = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content or ""
    except Exception as e:
        print(f"  Parse error (API): {e}")
        return []

    try:
        items = _parse_json_array(raw)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  Parse error (JSON): {e}")
        print(f"  Raw AI response:\n{raw}")
        return []

    transactions: list[dict] = []
    for item in items:
        normalized = _normalize_transaction(item)
        if normalized and normalized["description"]:
            transactions.append(normalized)

    return transactions


def extract_transactions_from_page(image: Image.Image, page_num: int) -> list[dict]:
    page_text = transcribe_page(image, page_num)
    return parse_transactions_from_text(page_text)


def extract_page_data(
    image: Image.Image, page_num: int
) -> tuple[list[dict], list[dict]]:
    transactions = extract_transactions_from_page(image, page_num)
    if transactions:
        print(f"  Page {page_num}: {len(transactions)} transaction(s)")
    else:
        print(f"  Page {page_num}: no transactions found")
    return transactions, []


def _normalize_transaction(txn: dict) -> dict | None:
    amount = _parse_amount(txn.get("amount"))
    if amount is None:
        return None

    txn_type = str(txn.get("type", "")).lower().strip()
    if txn_type not in {"debit", "credit"}:
        txn_type = "debit" if float(txn.get("amount", 0)) < 0 else "credit"

    return {
        "date": str(txn.get("date", "")).strip(),
        "description": str(txn.get("description", "")).strip(),
        "amount": amount,
        "type": txn_type,
    }


def _normalize_merchant_detail(item: dict) -> dict | None:
    amount = _parse_amount(item.get("amount"))
    merchant_detail = str(
        item.get("merchant_detail") or item.get("description", "")
    ).strip()
    if amount is None or not merchant_detail:
        return None

    return {
        "date": str(item.get("date", "")).strip(),
        "amount": amount,
        "merchant_detail": merchant_detail,
    }


def _merge_merchant_details(
    transactions: list[dict], merchant_details: list[dict]
) -> tuple[list[dict], int]:
    detail_map: dict[tuple[str, float], str] = {}
    for detail in merchant_details:
        key = _detail_match_key(detail["date"], detail["amount"])
        existing = detail_map.get(key)
        if not existing or len(detail["merchant_detail"]) > len(existing):
            detail_map[key] = detail["merchant_detail"]

    enriched = 0
    merged: list[dict] = []
    for txn in transactions:
        key = _detail_match_key(txn["date"], txn["amount"])
        if key in detail_map:
            txn = {**txn, "description": detail_map[key]}
            enriched += 1
        merged.append(txn)

    return merged, enriched


def _deduplicate_transactions(transactions: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    unique: list[dict] = []

    for txn in transactions:
        key = (txn["date"], round(txn["amount"], 2), txn["type"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(txn)

    return unique


def extract_bank_statement_from_pdf(
    pdf_path: str | Path,
    output_csv: str | Path | None = None,
) -> list[dict]:
    csv_path = Path(output_csv) if output_csv else OUTPUT_CSV_PATH
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Converting PDF to images: {pdf_path}")
    pages = pdf_to_images(pdf_path)
    print(f"Pass 1: Transcribing {len(pages)} page(s)...")

    page_texts: list[str] = []
    for page_num, page_image in enumerate(pages, start=1):
        page_text = transcribe_page(page_image, page_num)
        if page_text:
            page_texts.append(f"--- PAGE {page_num} ---\n{page_text}")

    combined_text = "\n\n".join(page_texts)
    print(f"\nPass 2: Parsing transactions from {len(combined_text)} characters...")

    # Extract year from statement text generically
    year_match = re.search(r'\b(20\d{2}|19\d{2})\b', combined_text)
    statement_year = year_match.group(1) if year_match else None

    all_transactions = parse_transactions_from_text(combined_text)

    before_dedup = len(all_transactions)
    all_transactions = _deduplicate_transactions(all_transactions)
    duplicates_removed = before_dedup - len(all_transactions)

    # Add year to every transaction
    if statement_year:
        for t in all_transactions:
            t['year'] = statement_year

    pd.DataFrame(all_transactions).to_csv(csv_path, index=False)

    debits = sum(1 for t in all_transactions if t["type"] == "debit")
    credits = sum(1 for t in all_transactions if t["type"] == "credit")
    total_debits = sum(t["amount"] for t in all_transactions if t["type"] == "debit")
    total_credits = sum(t["amount"] for t in all_transactions if t["type"] == "credit")

    print("\n--- Extraction Summary ---")
    print(f"Pages processed:     {len(pages)}")
    if statement_year:
        print(f"Year detected:       {statement_year}")
    print(f"Transactions found:  {len(all_transactions)}")
    if duplicates_removed:
        print(f"Duplicates removed:  {duplicates_removed}")
    print(f"  Debits:            {debits} (${total_debits:,.2f})")
    print(f"  Credits:           {credits} (${total_credits:,.2f})")
    print(f"Saved to:            {csv_path}")

    return all_transactions


def create_dummy_statement_pdf(output_path: str | Path | None = None) -> Path:
    path = Path(output_path) if output_path else DUMMY_PDF_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()
    story = [
        Paragraph(
            "<b>DESIGNS BY LENA - Business Checking</b>",
            styles["Heading2"],
        ),
        Spacer(1, 0.1 * inch),
        Paragraph("Statement Period: January 2024", styles["Normal"]),
        Spacer(1, 0.25 * inch),
        Paragraph("<b>Transaction History</b>", styles["Heading3"]),
        Spacer(1, 0.1 * inch),
    ]

    rows = [
        ["Date", "Description", "Amount", "Type"],
        ["2024-01-05", "ADOBE *CREATIVE CLOUD", "$54.99", "Debit"],
        ["2024-01-10", "STRIPE TRANSFER CLIENT PAYMENT", "$3,500.00", "Credit"],
        ["2024-01-15", "AWS AMAZON WEB SERVICES", "$89.00", "Debit"],
        ["2024-01-18", "CHIPOTLE ONLINE ORDER", "$24.18", "Debit"],
        ["2024-01-22", "FIGMA INC SUBSCRIPTION", "$15.00", "Debit"],
        ["2024-01-26", "ZELLE DEPOSIT MERIDIAN HEALTH", "$2,200.00", "Credit"],
    ]

    table = Table(rows, colWidths=[1.1 * inch, 3.0 * inch, 1.0 * inch, 0.8 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(table)
    doc.build(story)
    return path


if __name__ == "__main__":
    pdf_path = BASE_DIR / "data" / "sample_statements" / "Bank Statement Example Final.pdf"
    output_csv = BASE_DIR / "data" / "sample_statements" / "extracted_from_pdf.csv"

    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
    else:
        print(f"Input PDF:  {pdf_path}")
        print(f"Output CSV: {output_csv}")
        extract_bank_statement_from_pdf(pdf_path, output_csv)