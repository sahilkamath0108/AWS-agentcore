from strands import Agent
from strands.models.ollama import OllamaModel
from strands_tools import file_read, file_write


ollama_model = OllamaModel(
    model_id="llama3.2:latest",
    host="http://localhost:11434",
    params={
        "max_tokens": 2048,
        "temperature": 0.7,
        "top_p": 0.9,
        "stream": True,
    },
)



system_prompt = "You are a an agent which can read and write files to current directory"

# Create the agent with tools
local_agent = Agent(
    system_prompt=system_prompt, # Define a system Prompt
    model=ollama_model,
    tools=[file_read, file_write],  # Add your custom tools here
)


local_agent("can you create a test123.md file, with the content about current temperature?")
