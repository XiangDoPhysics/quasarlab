import os
from anthropic import Anthropic

# The client automatically picks up the ANTHROPIC_API_KEY environment variable
# If you set the environment variable in your terminal, you can just do:
client = Anthropic()

# Alternatively, you can pass the API key directly (less secure):
# client = Anthropic(api_key="your-api-key-here")

message = client.messages.create(
    model="claude-4-6-sonnet", # Use a suitable model
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "How do I connect the Anthropic API with Python?"}
    ]
)

print(message.content[0].text)
