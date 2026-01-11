from strands import Agent, tool
from strands.models.ollama import OllamaModel

# New Imports
# from strands.tools.mcp import MCPClient
# from mcp import stdio_client, StdioServerParameters
from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamablehttp_client


# Ollama
ollama_model = OllamaModel(
  model_id="gpt-oss:20b",
  host="http://localhost:11434"

)

streamable_http_mcp_client = MCPClient(
    lambda: streamablehttp_client(
        url="http://localhost:8123/mcp"
    ))

with streamable_http_mcp_client:
    tools = streamable_http_mcp_client.list_tools_sync()

    agent = Agent(
        system_prompt="Provide weather details using the available tools.",
        model=ollama_model, 
        tools=tools
        )
    agent("What is weather in New York?")