from strands import Agent, tool
from strands.models.ollama import OllamaModel

# Ollama
ollama_model = OllamaModel(
  model_id="gpt-oss:20b",
  host="http://localhost:11434"

)

@tool
def word_count(text: str) -> int:
    """Count words in text."""
    return len(text.split())

agent = Agent(model=ollama_model, tools=[word_count])
response = agent("How many words are in this sentence?")