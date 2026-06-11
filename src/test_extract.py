import sys
import os
import base64
import io
from pathlib import Path

sys.path.insert(0, 'src')

from groq import Groq
from pdf2image import convert_from_path
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv('GROQ_API_KEY'))

pdf_path = Path('data/sample_statements/Bank Statement Example Final.pdf')
print(f"File exists: {pdf_path.exists()}")
print(f"Full path: {pdf_path.resolve()}")

pages = convert_from_path(
    str(pdf_path.resolve()),
    poppler_path=r'C:\Release-26.02.0-0\poppler-26.02.0\Library\bin'
)

print(f"Pages found: {len(pages)}")

img = pages[0]
buf = io.BytesIO()
img.save(buf, format='PNG')
b64 = base64.b64encode(buf.getvalue()).decode()

response = client.chat.completions.create(
    model='meta-llama/llama-4-scout-17b-16e-instruct',
    messages=[{
        'role': 'user',
        'content': [
            {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64}'}},
            {'type': 'text', 'text': 'Transcribe ALL text from this bank statement exactly as you see it, preserving layout with spaces and line breaks. Include every single word and number.'}
        ]
    }]
)
print(response.choices[0].message.content)