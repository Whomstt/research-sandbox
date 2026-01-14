# References: https://docs.mistral.ai/getting-started/quickstart/

import os
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-embed"

client = Mistral(api_key=api_key)

embeddings_response = client.embeddings.create(
    model=model, inputs=["Embed this", "Embed that"]
)

print(embeddings_response)
