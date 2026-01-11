from strands import Agent, tool
from strands.models.ollama import OllamaModel

# New Imports
from strands.tools.mcp import MCPClient
from mcp import stdio_client, StdioServerParameters

# Ollama
ollama_model = OllamaModel(
  model_id="gpt-oss:20b",
  host="http://localhost:11434"

)

# For macOS/Linux:
# stdio_mcp_client = MCPClient(lambda: stdio_client(
#     StdioServerParameters(
#         command="uvx", 
#         args=["awslabs.aws-documentation-mcp-server@latest"]
#     )
# ))

# For Windows:
stdio_mcp_client = MCPClient(lambda: stdio_client(
    StdioServerParameters(
        command="uvx", 
        args=[
            "--from", 
            "awslabs.aws-documentation-mcp-server@latest", 
            "awslabs.aws-documentation-mcp-server.exe"
        ]
    )
))

# Create an agent with MCP tools
with stdio_mcp_client:
    tools = stdio_mcp_client.list_tools_sync()

    # Create an agent with these tools
    agent = Agent(
        system_prompt="You are an AWS documentation assistant. Use available tools to answer questions accurately.",
        model=ollama_model, 
        tools=tools
        )
    agent("What is AWS Lambda?")