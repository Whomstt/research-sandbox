# References: https://docs.mistral.ai/getting-started/quickstart/

import os
from mistralai import Mistral
from dotenv import load_dotenv


load_dotenv()

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-small-2503"

client = Mistral(api_key=api_key)

chat_response = client.chat.complete(
    model=model,
    messages=[
        {
            "role": "user",
            "content": "What is the best champagne mock wine?",
        },
    ],
)
print(chat_response.choices[0].message.content)
