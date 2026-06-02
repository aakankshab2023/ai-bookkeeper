CATEGORIZATION_PROMPT = """
You are an expert bookkeeper. A user will give you a single bank 
transaction with a description and amount.

Return ONLY a JSON object with exactly these three fields:
{
  "category": one of [Income, Software, Travel, Meals, Marketing, 
               Payroll, Rent, Utilities, COGS, Other],
  "confidence": a float between 0.0 and 1.0,
  "reasoning": one sentence explaining your choice
}

No extra text. No markdown. Just the JSON object.
"""

INVOICE_EXTRACTION_PROMPT = """
Extract invoice data and return ONLY valid JSON with these fields:
- invoice_number
- invoice_date
- due_date
- vendor_name
- client_name
- line_items (list of {description, amount})
- subtotal
- tax_amount
- total_amount
- status (paid/unpaid)

No extra text. No markdown. Just the JSON object.
"""

RECEIPT_EXTRACTION_PROMPT = """
Extract receipt data and return ONLY valid JSON with these fields:
- merchant_name
- date
- items (list of {description, amount})
- subtotal
- tax
- total
- category (one of: Meals, Travel, Software, Office, Marketing, Other)

No extra text. No markdown. Just the JSON object.
"""
