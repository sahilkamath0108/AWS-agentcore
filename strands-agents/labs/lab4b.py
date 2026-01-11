
from strands import Agent, tool
from ddgs import DDGS
from ddgs.exceptions import RatelimitException
import logging
from strands.models.ollama import OllamaModel

# Configure logging
logging.getLogger("strands").setLevel(logging.INFO)

@tool
def websearch(keywords: str, region: str = "us-en", max_results: int | None = None) -> str:
    """Search the web to get updated information."""
    try:
        results = DDGS().text(keywords, region=region, max_results=max_results)
        return results if results else "No results found."
    except RatelimitException:
        return "RatelimitException: Please try again after a short delay."
    except Exception as e:
        return f"Exception: {e}"


ollama_model = OllamaModel(
  model_id="gpt-oss:20b",
  host="http://localhost:11434"

)

recipe_agent = Agent(
    model=ollama_model,
    system_prompt="""You are a cooking assistant. Find and suggest recipes as requested by the user.""", 
    tools=[websearch], # Import the websearch tool we created above
)

response = recipe_agent("Suggest a recipe with chicken and broccoli.")

print(f"Metrics : {response.metrics}") # Optional, but recommended.