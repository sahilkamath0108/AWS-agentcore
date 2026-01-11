# 🤖 Amazon Bedrock AgentCore FAQ Solution

This repository contains a set of reference implementations for intelligent FAQ agents using **Amazon Bedrock AgentCore**. The included agents demonstrate professional implementations for language model-powered question answering, retrieval-augmented generation, and conversation memory.

## 📚 Project Structure

This project includes three agent implementations with the following capabilities:

1. **`00_langgraph_agent.py`** – LangGraph agent for basic semantic FAQ answering powered by LangChain.
2. **`01_agentcore_runtime.py`** – AgentCore runtime agent with tool-based FAQ retrieval and query rephrasing.
3. **`02_agentcore_memory.py`** – Advanced AgentCore agent with persistent conversation memory and preference tracking.

Agents use a customizable Q&A dataset (`qna.csv`, replaceable with any compatible CSV) as the backend knowledge source for queries.

## 🛠️ Setup & Requirements

**System Requirements:**
- Python 3.13+
- Windows, macOS, or Linux
- [uv](https://docs.astral.sh/uv/getting-started/installation/) dependency manager

Check Python version:
```bash
python --version
```

Install uv:
```bash
pip install uv
```

### AWS & Cloud Prerequisites

- AWS account (with Amazon Bedrock access)
- AWS CLI credentials configured ([guide](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html))
- Supported AWS Region (e.g. `ap-southeast-2`, `us-east-1`)

### LLM API

- GROQ API Key [(get one)](https://console.groq.com)

## 📦 Installation & Configuration

**1. Clone this repo and enter the directory:**
```bash
cd agentcore-crash-course
```

**2. Install Python dependencies:**
```bash
uv sync
```

**3. Set up your `.env` file:**
```env
GROQ_API_KEY=your_groq_api_key_here
HF_API_KEY=your_huggingface_api_key_here
```

## ▶️ Running the Agents

**LangGraph (standalone, local test):**
```bash
python 00_langgraph_agent.py
```
- Launches an interactive FAQ answer agent (uses semantic document retrieval on your Q&A dataset)

**AgentCore Runtime (tool-powered, cloud-ready):**
```bash
agentcore configure -e 01_agentcore_runtime.py
agentcore launch --env GROQ_API_KEY=your_groq_api_key_here
agentcore invoke '{"prompt": "What is product warranty?"}'
```
- Supports deployment with tool definitions for query/response handling

**AgentCore With Memory (context + state-full):**
```bash
agentcore configure -e 02_agentcore_memory.py
agentcore launch --env GROQ_API_KEY=your_groq_api_key_here
agentcore invoke '{"prompt": "Update my preference and answer"}'
```
- Enables persistent context/history for stateful conversation solutions

## ⚙️ Troubleshooting

- **Python version error:** Ensure Python 3.13+ is installed.
- **Missing API key:** Make sure `.env` file is present, with your GROQ_API_KEY.
- **FAISS install fails:**
    ```bash
    uv pip install --upgrade faiss-cpu
    ```
- **AWS credentials not found:**
    ```bash
    aws configure
    ```

## 📚 Resources

- [Amazon Bedrock AgentCore](https://aws.amazon.com/bedrock/agentcore/)
- [Official AgentCore Documentation](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agentcore-get-started-toolkit.html)
- [AgentCore Sample Implementations](https://github.com/awslabs/amazon-bedrock-agentcore-samples)

---
© Codebasics Inc. All rights reserved.
