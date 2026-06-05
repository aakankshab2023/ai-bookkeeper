CATEGORIZATION_PROMPT = """
You are an expert bookkeeper. Categorize a bank transaction.

Return ONLY this exact JSON format, nothing else:
{"category": "X", "confidence": 0.0, "reasoning": "Y"}

Rules:
- category MUST be exactly one of these words only:
  Income, Software, Travel, Meals, Marketing, Payroll, 
  Rent, Utilities, COGS, Office, Other
- Use Income ONLY for credit transactions (money in)
- Use Office for office supplies, equipment, stationery
- Use Other for expenses that fit nowhere else
- NEVER put explanations in the category field
- NEVER use Income for debit/outgoing transactions
- confidence is a number between 0.0 and 1.0
- reasoning is ONE short sentence maximum

Example output:
{"category": "Software", "confidence": 0.97, "reasoning": "Adobe Creative Cloud is a software subscription"}
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
