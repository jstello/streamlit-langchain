import os 
from openai import AzureOpenAI


client = AzureOpenAI(
    api_version="2023-07-01-preview",
    azure_endpoint="https://ai-proxy.lab.epam.com",
    api_key=os.environ["DIAL_API_KEY"],
)

response = client.chat.completions.create(
    model="gpt-4-1106-preview",
    messages=[
        {
            "role": "user",
            "content": "Hello!",
        }
    ]
)

print(response.choices[0].message.content)