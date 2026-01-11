import csv
import os
from typing import List
from typing_extensions import TypedDict

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.agents import create_agent
from dotenv import load_dotenv

# Import AgentCore runtime
from bedrock_agentcore.runtime import BedrockAgentCoreApp
# Create the AgentCore app instance
app = BedrockAgentCoreApp()

_ = load_dotenv()

def custom_load_items(faq_path: str) -> List[Document]:
    dlist = []
    with open(faq_path, "r", encoding="utf-8") as rawcsv:
        csv_iter = csv.DictReader(rawcsv)
        for csv_row in csv_iter:
            question_part = csv_row["question"].strip()
            answer_part = csv_row["answer"].strip()
            dlist.append(Document(page_content=f"Q: {question_part}\nA: {answer_part}"))
    return dlist


docs = custom_load_items("./qna.csv")
embedding_util = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)
text_chunker = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
divided_docs = text_chunker.split_documents(docs)
doc_indexer = FAISS.from_documents(divided_docs, embedding_util)


@tool
def query_faq_index(question: str) -> str:
    """Search the FAQ knowledge base for relevant information.
    Use this tool when the user asks questions about products, services, or policies.
    
    Args:
        query: The search query to find relevant FAQ entries
        
    Returns:
        Relevant FAQ entries that might answer the question
    """
    results = doc_indexer.similarity_search(question, k=3)
    
    if not results:
        return "No relevant FAQ entries found."
    
    context = "\n\n---\n\n".join([
        f"FAQ Entry {i+1}:\n{doc.page_content}" 
        for i, doc in enumerate(results)
    ])
    
    return f"Found {len(results)} relevant FAQ entries:\n\n{context}"


@tool
def multiresult_faq_lookup(query: str, response_count: int = 5) -> str:
    """Search the FAQ knowledge base with more results for complex queries.
    Use this when the initial search doesn't provide enough information.
    
    Args:
        query: The search query
        num_results: Number of results to retrieve (default: 5)
        
    Returns:
        More comprehensive FAQ entries
    """
    results = doc_indexer.similarity_search(query, k=response_count)
    
    if not results:
        return "No relevant FAQ entries found."
    
    context = "\n\n---\n\n".join([
        f"FAQ Entry {i+1}:\n{doc.page_content}" 
        for i, doc in enumerate(results)
    ])
    
    return f"Found {len(results)} detailed FAQ entries:\n\n{context}"


@tool
def alt_perspective_query(user_query: str, aspect: str) -> str:
    """Reformulate the query to focus on a specific aspect.
    Use this when you need to search for a different angle of the question.
    
    Args:
        original_query: The original user question
        focus_aspect: The specific aspect to focus on (e.g., "pricing", "activation", "troubleshooting")
        
    Returns:
        A reformulated query focused on the specified aspect
    """
    reformulated = f"{aspect} details for {user_query}"
    results = doc_indexer.similarity_search(reformulated, k=3)
    
    if not results:
        return f"No results found for aspect: {focus_aspect}"
    
    context = "\n\n---\n\n".join([
        f"Entry {i+1}:\n{doc.page_content}" 
        for i, doc in enumerate(results)
    ])
    
    return f"Results for '{focus_aspect}' aspect:\n\n{context}"



tools = [query_faq_index, multiresult_faq_lookup, alt_perspective_query]

system_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

system_prompt = """You serve as a robust FAQ answering interface, utilizing a custom knowledge engine and advanced search tools.
- Start with a regular FAQ query.
- If the basic search yields nothing, broaden or rephrase to retrieve more entries.
- Explore alternative perspectives and topics for complex questions if relevant.
- Only present helpful, concise information—never checklist steps or direct tutorial lines.
- If nothing is found, state it simply.
"""

custom_agent = create_agent(
    model=system_llm,
    tools=tools,
    system_prompt=system_prompt
)


# AgentCore Entrypoint
@app.entrypoint
def query_agent_main(payload, context):
    """Custom handler for entrypoint access in production runtime"""
    print("Got job payload:", payload)
    print("Job context:", context)
    
    user_query = payload.get("prompt", "No input prompt provided.")
    result = custom_agent.invoke({"messages": [("human", user_query)]})
    print("Agent Output Result:", result)
    return {"result": result['messages'][-1].content}


if __name__ == "__main__":
    app.run()