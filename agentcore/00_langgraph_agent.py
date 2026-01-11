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

from dotenv import load_dotenv
from langchain.agents import create_agent

_ = load_dotenv()


def import_faq_items(csv_path: str) -> List[Document]:
    question_docs = []
    with open(csv_path, "r", encoding="utf-8") as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for csv_row in csv_reader:
            q_field = csv_row["question"].strip()
            a_field = csv_row["answer"].strip()
            question_docs.append(Document(page_content=f"Q: {q_field}\nA: {a_field}"))
    return question_docs


faq_documents = import_faq_items("./qna.csv")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)
text_chunker = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
doc_segments = text_chunker.split_documents(faq_documents)
vector_index = FAISS.from_documents(doc_segments, embedding_model)


@tool
def knowledge_search(query: str) -> str:
    """Search the FAQ knowledge base for relevant information.
    Use this tool when the user asks questions about products, services, or policies.
    
    Args:
        query: The search query to find relevant FAQ entries
        
    Returns:
        Relevant FAQ entries that might answer the question
    """
    results = vector_index.similarity_search(query, k=3)
    
    if not results:
        return "No relevant FAQ entries found."
    
    context = "\n\n---\n\n".join([
        f"FAQ Entry {i+1}:\n{doc.page_content}" 
        for i, doc in enumerate(results)
    ])
    
    return f"Found {len(results)} relevant FAQ entries:\n\n{context}"


@tool
def extended_knowledge_search(query: str, top_k: int = 5) -> str:
    """Search the FAQ knowledge base with more results for complex queries.
    Use this when the initial search doesn't provide enough information.
    
    Args:
        query: The search query
        num_results: Number of results to retrieve (default: 5)
        
    Returns:
        More comprehensive FAQ entries
    """
    results = vector_index.similarity_search(query, k=top_k)
    
    if not results:
        return "No relevant FAQ entries found."
    
    context = "\n\n---\n\n".join([
        f"FAQ Entry {i+1}:\n{doc.page_content}" 
        for i, doc in enumerate(results)
    ])
    
    return f"Found {len(results)} detailed FAQ entries:\n\n{context}"


@tool
def focused_query_search(base_query: str, focus_area: str) -> str:
    """Reformulate the query to focus on a specific aspect.
    Use this when you need to search for a different angle of the question.
    
    Args:
        original_query: The original user question
        focus_aspect: The specific aspect to focus on (e.g., "pricing", "activation", "troubleshooting")
        
    Returns:
        A reformulated query focused on the specified aspect
    """
    reformulated_query = f"{focus_area} related to {base_query}"
    results = vector_index.similarity_search(reformulated_query, k=3)
    
    if not results:
        return f"No results found for aspect: {focus_aspect}"
    
    context = "\n\n---\n\n".join([
        f"Entry {i+1}:\n{doc.page_content}" 
        for i, doc in enumerate(results)
    ])
    
    return f"Results for '{focus_aspect}' aspect:\n\n{context}"



toolkit = [knowledge_search, extended_knowledge_search, focused_query_search]

question_model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

system_brief = """You act as a data-driven answer engine, consulting the document index and associated utilities.

Use provided queries and keyword search to locate helpful answers.
- Try simple search first
- If the question is complex or uncertain, reformulate and search different angles
- If nothing relevant is found, communicate this precisely
- Never reference a 'tutorial' or step list
"""

answer_engine = create_agent(
    model=question_model,
    tools=toolkit,
    system_prompt=system_brief
)

if __name__ == "__main__":
    test_query = answer_engine.invoke({"messages": [("human", "How do I enable product X?")]})
    print(test_query['messages'][-1].content)  # Demo run output
