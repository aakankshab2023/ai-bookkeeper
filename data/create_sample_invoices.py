"""Generate sample invoice PDFs for development and testing."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUTPUT_DIR = Path(__file__).resolve().parent / "sample_invoices"

INVOICES = [
    {
        "filename": "invoice_acme_design_2500.pdf",
        "from_name": "Acme Design Co",
        "from_address": "1420 Market St, Suite 200\nSan Francisco, CA 94102",
        "to_name": "LedgerAI Client",
        "to_address": "88 Pine Street\nSeattle, WA 98101",
        "invoice_number": "INV-2024-0015",
        "date": "2024-01-15",
        "due_date": "2024-02-15",
        "status": "Unpaid",
        "line_items": [
            ("Brand Design", 1500.00),
            ("Logo Design", 1000.00),
        ],
    },
    {
        "filename": "invoice_aws_89.pdf",
        "from_name": "AWS",
        "from_address": "Amazon Web Services, Inc.\n410 Terry Ave N\nSeattle, WA 98109",
        "to_name": "LedgerAI Client",
        "to_address": "88 Pine Street\nSeattle, WA 98101",
        "invoice_number": "AWS-INV-202401-8842",
        "date": "2024-01-01",
        "due_date": "2024-01-30",
        "status": "Paid",
        "line_items": [
            ("Cloud Hosting", 89.00),
        ],
    },
    {
        "filename": "invoice_freelancer_john_750.pdf",
        "from_name": "Freelancer John",
        "from_address": "john@freelancewriter.io\nPortland, OR 97201",
        "to_name": "LedgerAI Client",
        "to_address": "88 Pine Street\nSeattle, WA 98101",
        "invoice_number": "FJ-2024-0120",
        "date": "2024-01-20",
        "due_date": "2024-02-20",
        "status": "Unpaid",
        "line_items": [
            ("Content Writing", 750.00),
        ],
    },
]


def _money(amount: float) -> str:
    return f"${amount:,.2f}"


def build_invoice_pdf(invoice: dict, output_path: Path) -> None:
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "InvoiceTitle",
        parent=styles["Heading1"],
        fontSize=22,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=4,
    )
    label_style = ParagraphStyle(
        "Label",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=2,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#334155"),
    )
    right_style = ParagraphStyle(
        "Right",
        parent=body_style,
        alignment=TA_RIGHT,
    )

    total = sum(amount for _, amount in invoice["line_items"])
    status = invoice["status"]
    status_color = (
        colors.HexColor("#15803d") if status == "Paid" else colors.HexColor("#b45309")
    )

    story = []

    story.append(Paragraph("INVOICE", title_style))
    story.append(Spacer(1, 0.15 * inch))

    meta_data = [
        ["Invoice #:", invoice["invoice_number"]],
        ["Date:", invoice["date"]],
        ["Due Date:", invoice["due_date"]],
        ["Status:", status],
    ]
    meta_table = Table(meta_data, colWidths=[1.1 * inch, 2.2 * inch], hAlign="RIGHT")
    meta_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#64748b")),
                ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor("#1e293b")),
                ("TEXTCOLOR", (1, 3), (1, 3), status_color),
                ("FONTNAME", (1, 3), (1, 3), "Helvetica-Bold"),
                ("ALIGN", (0, 0), (-1, -1), "RIGHT"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )

    from_block = [
        Paragraph("From", label_style),
        Spacer(1, 4),
        Paragraph(f"<b>{invoice['from_name']}</b>", body_style),
        Spacer(1, 6),
        Paragraph(invoice["from_address"].replace("\n", "<br/>"), body_style),
    ]
    header_row = Table(
        [[from_block, meta_table]],
        colWidths=[3.5 * inch, 3 * inch],
    )
    header_row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(header_row)
    story.append(Spacer(1, 0.35 * inch))

    story.append(Paragraph("Bill To", label_style))
    story.append(Paragraph(f"<b>{invoice['to_name']}</b>", body_style))
    story.append(
        Paragraph(invoice["to_address"].replace("\n", "<br/>"), body_style)
    )
    story.append(Spacer(1, 0.35 * inch))

    line_rows = [["Description", "Amount"]]
    for desc, amount in invoice["line_items"]:
        line_rows.append([desc, _money(amount)])
    line_rows.append(["", ""])
    line_rows.append(["Total", _money(total)])

    line_table = Table(
        line_rows,
        colWidths=[4.5 * inch, 1.75 * inch],
        repeatRows=1,
    )
    line_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#475569")),
                ("TEXTCOLOR", (0, 1), (-1, -2), colors.HexColor("#334155")),
                ("ALIGN", (1, 0), (1, -1), "RIGHT"),
                ("LINEBELOW", (0, 0), (-1, 0), 1, colors.HexColor("#cbd5e1")),
                ("LINEBELOW", (0, 1), (-1, -3), 0.5, colors.HexColor("#e2e8f0")),
                ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, -1), (-1, -1), 12),
                ("LINEABOVE", (0, -1), (-1, -1), 1, colors.HexColor("#94a3b8")),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story.append(line_table)
    story.append(Spacer(1, 0.5 * inch))

    if status == "Unpaid":
        story.append(
            Paragraph(
                "Payment is due by the date shown above. "
                "Please remit payment via ACH or wire transfer.",
                ParagraphStyle(
                    "Note",
                    parent=body_style,
                    fontSize=9,
                    textColor=colors.HexColor("#64748b"),
                ),
            )
        )
    else:
        story.append(
            Paragraph(
                "<b>PAID</b> — Thank you for your payment.",
                ParagraphStyle(
                    "PaidNote",
                    parent=body_style,
                    fontSize=10,
                    textColor=colors.HexColor("#15803d"),
                ),
            )
        )

    doc.build(story)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for invoice in INVOICES:
        path = OUTPUT_DIR / invoice["filename"]
        build_invoice_pdf(invoice, path)
        total = sum(a for _, a in invoice["line_items"])
        print(f"Created {path.name} — {_money(total)} ({invoice['status']})")

    print(f"\n{len(INVOICES)} invoices saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
