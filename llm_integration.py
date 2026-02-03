import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=API_KEY)

def ask_llm(question, context_chunks):
    prompt = "Answer based strictly on the following context:\n"
    for chunk in context_chunks:
        prompt += f"- {chunk}\n"
    prompt += f"\nQuestion: {question}\nAnswer:"

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.5
    )

    return response.choices[0].message.content