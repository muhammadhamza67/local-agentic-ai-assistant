from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from tavily import TavilyClient
import chromadb
from chromadb.utils import embedding_functions
import os

# --- LLM setup ---

llm = ChatOpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
    model="qwen2.5-vl-3b-instruct",
)

tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

# --- RAG setup: load and embed the university document on startup ---

DOCUMENT_PATH = "afut_university_info.txt"

with open(DOCUMENT_PATH, "r", encoding="utf-8") as f:
    full_text = f.read()

raw_chunks = [chunk.strip() for chunk in full_text.split("\n\n") if chunk.strip()]
print(f"Loaded document, split into {len(raw_chunks)} chunks.")

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

chroma_client = chromadb.Client()
try:
    chroma_client.delete_collection("university_docs")
except Exception:
    pass

collection = chroma_client.create_collection(
    name="university_docs",
    embedding_function=embedding_fn
)
collection.add(
    documents=raw_chunks,
    ids=[f"chunk_{i}" for i in range(len(raw_chunks))]
)
print(f"Stored {len(raw_chunks)} chunks in the vector database. Ready.")


# --- Tools ---

@tool
def web_search(query: str) -> str:
    """Search the web for current, real-time, or up-to-date information —
    including news, weather, prices, sports scores, or anything time-sensitive."""
    results = tavily.search(query=query, max_results=5, topic="news", search_depth="advanced")
    formatted = []
    for r in results.get("results", []):
        formatted.append(f"- {r['title']}: {r['content'][:300]}\n  Source: {r['url']}")
    return "\n".join(formatted) if formatted else "No results found."


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the exact result."""
    import re
    if not re.match(r'^[\d\s\.\+\-\*\/\(\)]+$', expression):
        return "Error: expression contains disallowed characters."
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error evaluating expression: {e}"


@tool
def search_documents(query: str) -> str:
    """Search Al-Fareed University of Technology's (AFUT) official information
    document — use this for ANY question about AFUT, including admissions,
    tuition fees, faculty, leadership, programs, campus facilities, or
    student societies. This is the authoritative source for AFUT-specific
    facts; do not use web_search for AFUT questions."""
    results = collection.query(query_texts=[query], n_results=3)
    chunks = results["documents"][0]
    if not chunks:
        return "No relevant information found in the university document."
    return "\n\n---\n\n".join(chunks)


# --- Agent setup ---

memory = MemorySaver()
agent = create_react_agent(llm, tools=[web_search, calculator, search_documents], checkpointer=memory)


def run_agent(user_message: str, session_id: str) -> dict:
    config = {"configurable": {"thread_id": session_id}}

    # Record how many messages exist BEFORE this call, so we can isolate
    # just the new messages this turn added (memory keeps growing the history).
    prior_state = agent.get_state(config)
    prior_count = len(prior_state.values.get("messages", [])) if prior_state.values else 0

    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_message}]},
        config=config
    )

    # Only look at messages added THIS turn, not the whole history
    new_messages = result["messages"][prior_count:]

    sources = []
    used_document = False

    for m in new_messages:
        if hasattr(m, "content") and isinstance(m.content, str):
            if "Source:" in m.content:
                for line in m.content.split("\n"):
                    if line.strip().startswith("Source:"):
                        url = line.replace("Source:", "").strip()
                        if url not in sources:
                            sources.append(url)
            if hasattr(m, "name") and m.name == "search_documents":
                used_document = True

    final_answer = result["messages"][-1].content

    return {
        "answer": final_answer,
        "sources": sources,
        "used_document": used_document
    }


# --- API ---

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


@app.post("/chat")
def chat(req: ChatRequest):
    return run_agent(req.message, req.session_id)


@app.get("/")
def health_check():
    return {"status": "RAG-enabled agent server is running"}