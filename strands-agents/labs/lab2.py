from strands import Agent
from strands.models.ollama import OllamaModel
from strands_tools import file_read, file_write, http_request

# Configure the Ollama model
ollama_model = OllamaModel(
    model_id="gpt-oss:20b",
    host="http://localhost:11434",
    params={
        "max_tokens": 2048,
        "temperature": 0.7,
        "top_p": 0.9,
        "stream": True,
    },
)


system_prompt = """
Weather Information
    - You can also make HTTP requests to the National Weather Service API.
    - Process and display weather forecast data for locations in the United States.
"""

# - When retrieving weather information, first get coordinates using https://api.weather.gov/points/{latitude},{longitude},  or
#         https://api.weather.gov/points/{zipcode}, then use the returned forecast URL. You can make additional http requests as well.

# Create the agent with tools
local_agent = Agent(
    system_prompt=system_prompt, # Define a system Prompt
    model=ollama_model,
    tools=[http_request],  # Add your custom tools here
)


local_agent("what is the weather in new york?")
