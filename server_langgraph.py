from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from tavily import TavilyClient
import os

# --- Setup (same as langgraph_agent.py) ---

llm = ChatOpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
    model="qwen2.5-vl-3b-instruct",
)

tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


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


# MemorySaver gives LangGraph built-in conversation memory, keyed by thread_id.
# This replaces the manual `conversations = {}` dictionary from server.py.
memory = MemorySaver()
agent = create_react_agent(llm, tools=[web_search, calculator], checkpointer=memory)


def run_agent(user_message: str, session_id: str) -> dict:
    """Runs the LangGraph agent and extracts an answer + sources,
    in the same shape the Flutter app already expects."""

    print(f"\n=== New request — session_id: '{session_id}' ===")
    config = {"configurable": {"thread_id": session_id}}

    # Check what LangGraph already has stored for this thread, BEFORE this call
    existing_state = agent.get_state(config)
    prior_message_count = len(existing_state.values.get("messages", [])) if existing_state.values else 0
    print(f"Messages already in memory for this session before this call: {prior_message_count}")

    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_message}]},
        config=config
    )

    print(f"Messages in memory AFTER this call: {len(result['messages'])}")

    sources = []
    search_rounds = 0

    for m in result["messages"]:
        # Tool result messages carry the raw search output — pull out URLs
        if hasattr(m, "content") and isinstance(m.content, str) and "Source:" in m.content:
            search_rounds += 1
            for line in m.content.split("\n"):
                if line.strip().startswith("Source:"):
                    url = line.replace("Source:", "").strip()
                    if url not in sources:
                        sources.append(url)

    final_answer = result["messages"][-1].content

    return {"answer": final_answer, "sources": sources, "search_rounds": search_rounds}


# --- API setup (identical shape to server.py) ---

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
    result = run_agent(req.message, req.session_id)
    return result


@app.get("/")
def health_check():
    return {"status": "LangGraph agent server is running"}