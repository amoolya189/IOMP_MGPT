import os
import time
from google import genai

# ---------------- GEMINI CLIENT ----------------
_gemini_client = None


def get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY")
        )
    return _gemini_client


# ---------------- GEMINI ANSWER ----------------
def gemini_answer(question, previous_answer=""):
    """
    Generates a STRICT Mahabharata-only Gemini response.
    Out-of-domain questions are politely rejected
    without breaking conversation context.
    """

    system_rules = """
You are a STRICT Mahabharata-only expert assistant.

ABSOLUTE RULES:
- Answer ONLY questions related to the Mahabharata.
- If a question is NOT related to the Mahabharata:
  - Do NOT answer it.
  - Reply with a brief redirection back to the Mahabharata.
- Do NOT introduce modern geography, politics, science, or general knowledge.
- Do NOT ask clarifying questions.
- Maintain continuity with the last valid Mahabharata topic.
"""

    if previous_answer:
        prompt = f"""
{system_rules}

Previous valid Mahabharata answer:
{previous_answer}

User question:
{question}

Instructions:
- If Mahabharata-related, answer clearly and structurally.
- If NOT, gently redirect without changing the topic.
"""
    else:
        prompt = f"""
{system_rules}

User question:
{question}

Instructions:
- Answer ONLY if related to the Mahabharata.
- Otherwise, redirect politely.
"""

    retries = 5

    for attempt in range(retries):
        try:
            client = get_gemini_client()

            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt
            )

            return response.text.strip()

        except Exception as e:
            error_message = str(e)

            print(f"Gemini error: {error_message}")

            # Retry only for temporary overload errors
            if "503" in error_message or "UNAVAILABLE" in error_message:
                if attempt < retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                return (
                    "The Mahabharata knowledge service is busy right now. "
                    "Please try again in a few moments."
                )

            return f"Gemini error: {error_message}"