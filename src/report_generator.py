"""Generate a professional LedgerAI PDF report from pipeline outputs."""

from datetime import date
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
REPORT_PATH = OUTPUT_DIR / "LedgerAI_Report.pdf"

PERIOD_LABEL = "January - March 2024"

# Palette
SLATE_900 = colors.HexColor("#0f172a")
SLATE_700 = colors.HexColor("#334155")
SLATE_500 = colors.HexColor("#64748b")
SLATE_200 = colors.HexColor("#e2e8f0")
SLATE_50 = colors.HexColor("#f8fafc")
GREEN = colors.HexColor("#15803d")
GREEN_BG = colors.HexColor("#dcfce7")
RED = colors.HexColor("#b91c1c")
RED_BG = colors.HexColor("#fee2e2")
BLUE = colors.HexColor("#1d4ed8")
YELLOW_BG = colors.HexColor("#fef9c3")
INDIGO = colors.HexColor("#4338ca")


def _money(value) -> str:
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"${amount:,.2f}"


def _read_csv(name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _section_title(text: str, styles) -> Paragraph:
    style = ParagraphStyle(
        "SectionTitle",
        parent=styles["Heading2"],
        fontSize=16,
        textColor=SLATE_900,
        spaceAfter=12,
        spaceBefore=4,
    )
    return Paragraph(text, style)


def _build_cover(styles) -> list:
    title = ParagraphStyle(
        "CoverTitle",
        parent=styles["Title"],
        fontSize=36,
        textColor=INDIGO,
        alignment=TA_CENTER,
        spaceAfter=8,
        fontName="Helvetica-Bold",
    )
    subtitle = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontSize=14,
        textColor=SLATE_700,
        alignment=TA_CENTER,
        spaceAfter=6,
    )
    meta = ParagraphStyle(
        "CoverMeta",
        parent=styles["Normal"],
        fontSize=11,
        textColor=SLATE_500,
        alignment=TA_CENTER,
        spaceAfter=4,
    )

    today = date.today().strftime("%B %d, %Y")
    divider = Table([[""]], colWidths=[5 * inch], rowHeights=[2])
    divider.setStyle(
        TableStyle([("LINEBELOW", (0, 0), (-1, -1), 1.5, INDIGO)])
    )

    return [
        Spacer(1, 2.2 * inch),
        Paragraph("LedgerAI", title),
        Spacer(1, 0.15 * inch),
        Paragraph("AI-Generated Books of Accounts", subtitle),
        Spacer(1, 0.35 * inch),
        Paragraph(f"Period: {PERIOD_LABEL}", meta),
        Paragraph(f"Generated on: {today}", meta),
        Spacer(1, 0.4 * inch),
        divider,
    ]


def _build_pl_page(pl_df: pd.DataFrame, styles) -> list:
    story = [_section_title("Profit &amp; Loss Statement", styles), Spacer(1, 0.1 * inch)]

    if pl_df.empty:
        story.append(Paragraph("No P&amp;L data available.", styles["Normal"]))
        return story

    rows = [["Item", "Amount"]]
    row_styles: list[tuple] = []  # (row_index, style_key)

    income_rows: list[int] = []
    expense_rows: list[int] = []
    net_profit_row: int | None = None

    for _, record in pl_df.iterrows():
        item = str(record.get("line_item", ""))
        amount = _money(record.get("amount", 0))
        row_idx = len(rows)
        rows.append([item, amount])

        if item == "Total Revenue":
            income_rows.append(row_idx)
        elif item.startswith("Expense -") or item == "Total Expenses":
            expense_rows.append(row_idx)
        elif item == "Net Profit":
            net_profit_row = row_idx

    table = Table(rows, colWidths=[4.2 * inch, 1.8 * inch], repeatRows=1)
    style_commands = [
        ("BACKGROUND", (0, 0), (-1, 0), SLATE_900),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (1, 1), (1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, SLATE_200),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, SLATE_50]),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]

    for idx in income_rows:
        style_commands.extend(
            [
                ("TEXTCOLOR", (0, idx), (-1, idx), GREEN),
                ("BACKGROUND", (0, idx), (-1, idx), GREEN_BG),
                ("FONTNAME", (0, idx), (-1, idx), "Helvetica-Bold"),
            ]
        )

    for idx in expense_rows:
        style_commands.extend(
            [
                ("TEXTCOLOR", (0, idx), (-1, idx), RED),
                ("BACKGROUND", (0, idx), (-1, idx), RED_BG),
            ]
        )

    if net_profit_row is not None:
        style_commands.extend(
            [
                ("TEXTCOLOR", (0, net_profit_row), (-1, net_profit_row), BLUE),
                ("FONTNAME", (0, net_profit_row), (-1, net_profit_row), "Helvetica-Bold"),
                ("FONTSIZE", (0, net_profit_row), (-1, net_profit_row), 12),
                ("LINEABOVE", (0, net_profit_row), (-1, net_profit_row), 1.5, SLATE_500),
                ("BACKGROUND", (0, net_profit_row), (-1, net_profit_row), colors.HexColor("#eff6ff")),
            ]
        )

    table.setStyle(TableStyle(style_commands))
    story.append(table)
    return story


def _truncate(text: str, max_len: int = 42) -> str:
    text = str(text)
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def _build_cash_book_page(cash_df: pd.DataFrame, styles) -> list:
    story = [
        _section_title("Cash Book - All Transactions", styles),
        Spacer(1, 0.08 * inch),
    ]

    if cash_df.empty:
        story.append(Paragraph("No transaction data available.", styles["Normal"]))
        return story

    rows = [["Date", "Description", "Amount", "Category", "Type"]]
    debit_rows: list[int] = []
    credit_rows: list[int] = []

    for _, record in cash_df.iterrows():
        txn_type = str(record.get("type", "")).lower()
        row_idx = len(rows)
        rows.append(
            [
                str(record.get("date", "")),
                _truncate(record.get("description", "")),
                _money(record.get("amount", 0)),
                str(record.get("category", "")),
                txn_type,
            ]
        )
        if txn_type == "credit":
            credit_rows.append(row_idx)
        elif txn_type == "debit":
            debit_rows.append(row_idx)

    col_widths = [0.85 * inch, 2.55 * inch, 0.85 * inch, 0.95 * inch, 0.6 * inch]
    table = Table(rows, colWidths=col_widths, repeatRows=1)
    style_commands = [
        ("BACKGROUND", (0, 0), (-1, 0), SLATE_900),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (2, 1), (2, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.25, SLATE_200),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, SLATE_50]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]

    for idx in credit_rows:
        style_commands.append(("TEXTCOLOR", (2, idx), (2, idx), GREEN))
    for idx in debit_rows:
        style_commands.append(("TEXTCOLOR", (2, idx), (2, idx), RED))

    table.setStyle(TableStyle(style_commands))
    story.append(table)
    return story


def _build_ar_page(ar_df: pd.DataFrame, styles) -> list:
    story = [_section_title("Accounts Receivable", styles), Spacer(1, 0.1 * inch)]

    if ar_df.empty:
        story.append(Paragraph("No outstanding receivables.", styles["Normal"]))
        return story

    rows = [["Invoice #", "Vendor", "Amount", "Due Date", "Status"]]
    unpaid_rows: list[int] = []

    for _, record in ar_df.iterrows():
        status = str(record.get("status", "")).lower()
        row_idx = len(rows)
        rows.append(
            [
                str(record.get("invoice_number", "")),
                _truncate(record.get("vendor_name", ""), 28),
                _money(record.get("total_amount", 0)),
                str(record.get("due_date", "")),
                status,
            ]
        )
        if status == "unpaid":
            unpaid_rows.append(row_idx)

    col_widths = [1.2 * inch, 1.8 * inch, 1.0 * inch, 1.0 * inch, 0.8 * inch]
    table = Table(rows, colWidths=col_widths, repeatRows=1)
    style_commands = [
        ("BACKGROUND", (0, 0), (-1, 0), SLATE_900),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (2, 1), (2, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, SLATE_200),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, SLATE_50]),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]

    for idx in unpaid_rows:
        style_commands.append(("BACKGROUND", (0, idx), (-1, idx), YELLOW_BG))

    table.setStyle(TableStyle(style_commands))
    story.append(table)
    return story


def generate_report(output_path: str | Path | None = None) -> Path:
    """Build LedgerAI PDF report from outputs CSV files."""
    out_path = Path(output_path) if output_path else REPORT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pl_df = _read_csv("pl_statement.csv")
    cash_df = _read_csv("cash_book.csv")
    ar_df = _read_csv("accounts_receivable.csv")

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title="LedgerAI Report",
        author="LedgerAI",
    )

    story: list = []
    story.extend(_build_cover(styles))
    story.append(PageBreak())
    story.extend(_build_pl_page(pl_df, styles))
    story.append(PageBreak())
    story.extend(_build_cash_book_page(cash_df, styles))
    story.append(PageBreak())
    story.extend(_build_ar_page(ar_df, styles))

    doc.build(story)
    return out_path


if __name__ == "__main__":
    path = generate_report()
    print(f"Report saved to {path}")
