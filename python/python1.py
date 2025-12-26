from groq import Groq
import os

os.environ["GROQ_API_KEY"]   # Use the new one!

client = Groq()
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Hello! Tell me a fun fact."}]
)
print(response.choices[0].message.content)  