from openai import OpenAI
import os
# NVIDIA_API_KEY = "nvapi-"
from dotenv import load_dotenv
load_dotenv(".env")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
client = OpenAI(
    api_key=NVIDIA_API_KEY,
    base_url="https://integrate.api.nvidia.com/v1",
)

response = client.chat.completions.create(
    model="z-ai/glm5",  # <-- EXACT from your list
    messages=[
        {"role": "user", "content": "Say hello in one sentence."}
    ],
    temperature=0.2,
)

print(response.choices[0].message.content)