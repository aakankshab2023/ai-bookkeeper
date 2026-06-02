import base64
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from PIL import Image, ImageDraw, ImageFont

from prompts import RECEIPT_EXTRACTION_PROMPT

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png"}

SAMPLE_RECEIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "sample_receipts"
    / "test_receipt.jpg"
)


def _parse_json_response(raw: str) -> dict:
    text = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence_match:
        text = fence_match.group(1).strip()
    return json.loads(text)


def image_to_base64(image_path: str | Path) -> tuple[str, str]:
    """Read image file and return (base64_string, mime_type)."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported image type '{suffix}'. "
            f"Use one of: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )

    mime_type = "image/png" if suffix == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return encoded, mime_type


def extract_receipt_data(image_path: str | Path) -> dict:
    """Extract structured receipt fields from an image using Groq vision."""
    try:
        b64_image, mime_type = image_to_base64(image_path)
    except (FileNotFoundError, ValueError) as e:
        return {"error": str(e)}

    data_uri = f"data:{mime_type};base64,{b64_image}"

    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "system", "content": RECEIPT_EXTRACTION_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract all receipt data from this image.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": data_uri},
                        },
                    ],
                },
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


def create_test_receipt(output_path: str | Path | None = None) -> Path:
    """Create a sample Chipotle receipt image for testing."""
    path = Path(output_path) if output_path else SAMPLE_RECEIPT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGB", (400, 480), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    lines = [
        "CHIPOTLE MEXICAN GRILL",
        "123 Main Street",
        "Date: 2024-01-18",
        "",
        "Burrito Bowl           $12.50",
        "Chips & Guac            $4.25",
        "",
        "Subtotal               $16.75",
        "Tax                     $1.35",
        "TOTAL                  $18.10",
        "",
        "Thank you for your visit!",
    ]

    y = 24
    for line in lines:
        draw.text((24, y), line, fill="black", font=font)
        y += 28

    save_path = path.with_suffix(".jpg") if path.suffix.lower() != ".jpg" else path
    img.save(save_path, format="JPEG", quality=95)
    return save_path


if __name__ == "__main__":
    print(f"Creating sample receipt at {SAMPLE_RECEIPT_PATH}...")
    receipt_path = create_test_receipt(SAMPLE_RECEIPT_PATH)

    print("Running extraction...")
    result = extract_receipt_data(receipt_path)
    print(json.dumps(result, indent=2))
