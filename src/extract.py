import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def categorize_transaction(description: str, amount: float, txn_type: str) -> dict:
    from prompts import CATEGORIZATION_PROMPT

    user_message = (
        f"Transaction: {description}, Amount: ${amount}, "
        f"Type: {txn_type} (debit = money OUT, credit = money IN)"
    )
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": CATEGORIZATION_PROMPT},
            {"role": "user", "content": user_message}
        ]
    )
    
    result = response.choices[0].message.content
    return json.loads(result)

if __name__ == "__main__":
    test = categorize_transaction("Adobe Creative Cloud", 54.99, "debit")
    print(test)