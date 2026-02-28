import anthropic
from nbconvert import export

# Make sure to set your API key in the environment variable before running:
export ANTHROPIC_API_KEY = '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'

client = anthropic.Anthropic()

message = client.messages.create(
    model = 'claude-opus-4-6',
    max_tokens = 1000,
    messages = [
        {
            "role": "user",
            "content": "What should I search for to find the latest developments in renewable energy?"
        }
    ]
)
print(message.content)