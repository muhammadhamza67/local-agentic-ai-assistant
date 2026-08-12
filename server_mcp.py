from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
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

# --- RAG setup (same as server_rag.py) ---

DOCUMENT_PATH = "afut_university_info.txt"
with open(DOCUMENT_PATH, "r", encoding="utf-8") as f:
    full_text = f.read()
raw_chunks = [chunk.strip() for chunk in full_text.split("\n\n") if chunk.strip()]

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
try:
    chroma_client.delete_collection("university_docs")
except Exception:
    pass
collection = chroma_client.create_collection(name="university_docs", embedding_function=embedding_fn)
collection.add(documents=raw_chunks, ids=[f"chunk_{i}" for i in range(len(raw_chunks))])
print(f"Stored {len(raw_chunks)} chunks in the vector database. Ready.")


# --- Regular tools (NOT via MCP) ---

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
def search_documents(query: str) -> str:
    """Search Al-Fareed University of Technology's (AFUT) official information
    document — use this for ANY question about AFUT, including admissions,
    tuition fees, faculty, leadership, programs, campus facilities, or
    student societies."""
    results = collection.query(query_texts=[query], n_results=3)
    chunks = results["documents"][0]
    if not chunks:
        return "No relevant information found in the university document."
    return "\n\n---\n\n".join(chunks)


# --- Global state, set up once at startup ---

agent = None
memory = MemorySaver()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once when the server starts. We connect to the MCP calculator
    server HERE (not at import time) because connecting is an async
    operation, and this is where FastAPI lets us safely run async setup."""
    global agent

    mcp_client = MultiServerMCPClient(
        {
            "calculator": {
                "command": "python",
                "args": ["mcp_calculator_server.py"],
                "transport": "stdio",
            }
        }
    )

    mcp_tools = await mcp_client.get_tools()
    print(f"Tools discovered from MCP server: {[t.name for t in mcp_tools]}")

    # Combine MCP-provided tools with our regular hardcoded tools —
    # the agent doesn't care or know the difference between them.
    all_tools = mcp_tools + [web_search, search_documents]

    agent = create_react_agent(llm, tools=all_tools, checkpointer=memory)

    yield  # server runs here

    # (optional cleanup could go here when the server shuts down)


async def run_agent(user_message: str, session_id: str) -> dict:
    config = {"configurable": {"thread_id": session_id}}

    prior_state = await agent.aget_state(config)
    prior_count = len(prior_state.values.get("messages", [])) if prior_state.values else 0

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": user_message}]},
        config=config
    )

    new_messages = result["messages"][prior_count:]

    print(f"\n=== New messages this turn ({len(new_messages)}) ===")
    for m in new_messages:
        msg_type = type(m).__name__
        tool_calls = getattr(m, "tool_calls", None)
        name = getattr(m, "name", None)
        print(f"[{msg_type}] name={name} tool_calls={tool_calls}")

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

    return {"answer": final_answer, "sources": sources, "used_document": used_document}


# --- API ---

app = FastAPI(lifespan=lifespan)

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
async def chat(req: ChatRequest):
    return await run_agent(req.message, req.session_id)


@app.get("/")
def health_check():
    return {"status": "MCP-enabled agent server is running"}