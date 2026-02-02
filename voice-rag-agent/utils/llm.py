from groq import Groq

client = Groq(api_key="YOUR_GROQ_API_KEY")

def call_llm(query, context):
    prompt = f"""
You are a technical support voice assistant.
Use the manual context.

Context:
{context}

User Question:
{query}

Answer in short spoken style sentences.
"""

    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_completion_tokens=1024,
        stream=True
    )

    for chunk in completion:
        content = chunk.choices[0].delta.content
        if content:
            yield content
