from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from tavily import TavilyClient
import os
import json
import re

# --- Setup clients (same as agent.py) ---

client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio"
)

tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

MODEL_NAME = "qwen2.5-vl-3b-instruct"  # match your LM Studio API Model Identifier

tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current, real-time, or up-to-date information on ANY topic — including news, weather, prices, sports scores, current events, or facts that may have changed recently. This is your ONLY way to get live information; there is no separate weather or news tool, so always use this one for anything time-sensitive.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression and return the exact result. Use this for any math calculation instead of computing it yourself, since you may make arithmetic mistakes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A mathematical expression to evaluate, e.g. '15 * 0.2' or '(45 + 30) / 3'"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]


def calculator(expression: str) -> str:
    """Safely evaluate a basic math expression — no arbitrary code execution allowed."""
    import re as regex
    # Only allow digits, basic operators, parentheses, decimal points, and spaces —
    # this blocks anything dangerous from being run.
    if not regex.match(r'^[\d\s\.\+\-\*\/\(\)]+$', expression):
        return "Error: expression contains disallowed characters."
    try:
        result = eval(expression)  # safe here ONLY because of the regex check above
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


def web_search(query: str) -> str:
    results = tavily.search(
        query=query,
        max_results=5,
        topic="news",
        search_depth="advanced"
    )
    formatted = []
    for r in results.get("results", []):
        formatted.append(f"- {r['title']}: {r['content'][:300]}\n  Source: {r['url']}")
    return "\n".join(formatted) if formatted else "No results found."


# --- Conversation memory ---
# Simple in-memory store: {session_id: [list of messages]}
# NOTE: this resets if the server restarts, and doesn't persist to disk —
# good enough for learning, not for a real product yet.
conversations = {}


def run_agent(user_message: str, session_id: str) -> dict:
    """Runs the full agentic loop: the model can search MULTIPLE times if it
    decides the first results weren't enough, up to a safety limit.
    Now also remembers prior messages in the same session."""

    # Get this session's history, or start a new one
    messages = conversations.get(session_id, [])
    messages.append({"role": "user", "content": user_message})

    sources = []
    MAX_ITERATIONS = 3  # safety limit so it can never loop forever

    for iteration in range(MAX_ITERATIONS):
        print(f"\n=== ROUND {iteration + 1} ===")

        # On the last allowed iteration, don't offer the tool anymore —
        # force the model to answer with whatever it has.
        offer_tools = tools if iteration < MAX_ITERATIONS - 1 else []

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=offer_tools
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            print(f"Model is satisfied — answering after {iteration} search round(s).")
            messages.append({"role": "assistant", "content": msg.content})
            conversations[session_id] = messages  # save updated history
            return {"answer": msg.content, "sources": sources, "search_rounds": iteration}

        # Model wants to use a tool (search or calculate) — run it and loop back
        messages.append(msg)
        for tool_call in msg.tool_calls:
            if tool_call.function.name == "web_search":
                args = json.loads(tool_call.function.arguments)
                query = args["query"]
                print(f"Model wants to search for: '{query}'")
                search_results = web_search(query)

                for line in search_results.split("\n"):
                    if line.strip().startswith("Source:"):
                        url = line.replace("Source:", "").strip()
                        if url not in sources:  # avoid duplicate sources across rounds
                            sources.append(url)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": search_results
                })

            elif tool_call.function.name == "calculator":
                args = json.loads(tool_call.function.arguments)
                expression = args["expression"]
                print(f"Model wants to calculate: '{expression}'")
                calc_result = calculator(expression)
                print(f"Calculator result: {calc_result}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": calc_result
                })

            else:
                # Model requested a tool that doesn't exist (hallucinated name).
                # We MUST still respond to every tool_call, or the conversation breaks.
                print(f"WARNING: Model requested unknown tool '{tool_call.function.name}' — telling it that tool doesn't exist.")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": f"Error: no tool named '{tool_call.function.name}' exists. There is no dedicated tool for that. Use 'web_search' instead — it can find weather, prices, scores, or any other current information."
                })

    # If we hit MAX_ITERATIONS without a clean final answer, this is a fallback —
    # shouldn't normally trigger since the last round has tools disabled.
    conversations[session_id] = messages
    return {"answer": "I wasn't able to find a confident answer after multiple searches.",
            "sources": sources, "search_rounds": MAX_ITERATIONS}


# --- API setup ---

app = FastAPI()

# Allow the Flutter app to call this from a phone/emulator/browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"  # Flutter can send a unique ID per device/user later


@app.post("/chat")
def chat(req: ChatRequest):
    result = run_agent(req.message, req.session_id)
    return result


@app.post("/reset")
def reset(req: ChatRequest):
    """Clears memory for a session — useful for a 'New Chat' button later."""
    conversations.pop(req.session_id, None)
    return {"status": "conversation cleared"}


@app.get("/")
def health_check():
    return {"status": "Agent server is running"}