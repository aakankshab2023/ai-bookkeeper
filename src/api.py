import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from main_pipeline import run_full_pipeline
from report_generator import generate_report

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
REPORT_PATH = OUTPUT_DIR / "LedgerAI_Report.pdf"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
(UPLOADS_DIR / "invoices").mkdir(parents=True, exist_ok=True)
(UPLOADS_DIR / "receipts").mkdir(parents=True, exist_ok=True)

app = FastAPI(title="LedgerAI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _save_upload(upload: UploadFile, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    content = await upload.read()
    dest.write_bytes(content)


def _clear_upload_dir(folder: Path) -> None:
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)


@app.get("/")
async def root():
    return {"status": "LedgerAI API running"}


@app.get("/health")
async def health():
    return {"status": "healthy", "model": "llama-3.3-70b-versatile"}


@app.post("/process")
async def process(
    bank_statement: UploadFile = File(...),
    invoices: Optional[List[UploadFile]] = File(default=None),
    receipts: Optional[List[UploadFile]] = File(default=None),
):
    if invoices is None:
        invoices = []
    else:
        invoices = [f for f in invoices if f.filename]

    if receipts is None:
        receipts = []
    else:
        receipts = [f for f in receipts if f.filename]

    if not bank_statement.filename:
        raise HTTPException(status_code=400, detail="bank_statement file is required")

    invoices_dir = UPLOADS_DIR / "invoices"
    receipts_dir = UPLOADS_DIR / "receipts"
    _clear_upload_dir(invoices_dir)
    _clear_upload_dir(receipts_dir)

    statement_path = UPLOADS_DIR / (bank_statement.filename or "bank_statement.csv")

    try:
        await _save_upload(bank_statement, statement_path)

        for invoice in invoices:
            if invoice.filename:
                await _save_upload(invoice, invoices_dir / invoice.filename)

        for receipt in receipts:
            if receipt.filename:
                await _save_upload(receipt, receipts_dir / receipt.filename)

        # Handle PDF bank statements
        if statement_path.suffix.lower() == ".pdf":
            from pdf_bank_extractor import extract_bank_statement_from_pdf
            csv_path = UPLOADS_DIR / "extracted_bank_statement.csv"
            extract_bank_statement_from_pdf(statement_path, csv_path)
            statement_path = csv_path

        result = run_full_pipeline(
            bank_statement_path=statement_path,
            invoices_folder=invoices_dir,
            receipts_folder=receipts_dir,
        )

        pdf_ready = False
        try:
            from report_generator import generate_report
            generate_report()
            pdf_ready = REPORT_PATH.exists()
        except Exception as e:
            print(f"PDF generation failed: {e}")

        return JSONResponse(
            content={
                "status": "success",
                "summary": result["summary"],
                "pl_statement": result["pl_statement"],
                "cash_book": result["cash_book"],
                "accounts_receivable": result["accounts_receivable"],
                "pdf_ready": pdf_ready,
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}") from e


@app.get("/download/report")
async def download_report():
    if not REPORT_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Report not found. Run POST /process first.",
        )
    return FileResponse(
        path=str(REPORT_PATH),
        filename="LedgerAI_Report.pdf",
        media_type="application/pdf",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)