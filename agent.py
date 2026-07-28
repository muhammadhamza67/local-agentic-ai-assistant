from openai import OpenAI
from tavily import TavilyClient
import os
import json

# --- Setup clients ---

# Local model running in LM Studio
client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio"  # LM Studio doesn't check this, but the library requires something
)

# Tavily web search client (reads key from environment variable)
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

# --- Define the tool the model is allowed to use ---

tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current, real-time information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


def sanitize_query(query: str) -> str:
    """Fix common small-model glitches: missing spaces around short words like
    'on', 'in', 'to', etc. Small/VL models sometimes glue words together."""
    import re
    fixes = {
        r"\bon(\w)": r"on \1",   # "onfusion" -> "on fusion"
        r"\bin(\w)": r"in \1",   # "inflation" is a real word, so this is imperfect —
                                  # good enough for now, refine if it causes new bugs
    }
    cleaned = query
    # Only apply the "on" fix for now — safest, least likely to break real words
    cleaned = re.sub(r"\bon(fusion|energy|climate|ai|technology)\b", r"on \1", cleaned)
    return cleaned.strip()


def web_search(query: str) -> str:
    """Actually call Tavily and return a short text summary of results."""
    query = sanitize_query(query)
    print(f"(sanitized query: '{query}')")
    results = tavily.search(
        query=query,
        max_results=5,
        topic="news",            # bias toward actual news articles, not generic web pages
        search_depth="advanced"  # better relevance ranking than the default "basic"
    )
    formatted = []
    for r in results.get("results", []):
        formatted.append(f"- {r['title']}: {r['content'][:300]}\n  Source: {r['url']}")
    return "\n".join(formatted) if formatted else "No results found."


# --- The conversation so far ---

messages = [
    {"role": "user", "content": "What's the latest news on fusion energy?"}
]

# IMPORTANT: change this to match the exact API Model Identifier
# shown in LM Studio's Developer/Local Server tab
MODEL_NAME = "qwen2.5-vl-3b-instruct"

# --- Step 1: Ask the model. It may respond directly, or ask to call a tool. ---

response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=messages,
    tools=tools
)

msg = response.choices[0].message
print("\n--- Model's first move ---")
print(msg)

# --- Step 2: If the model wants to use a tool, run it and loop back ---

if msg.tool_calls:
    # Add the assistant's tool-call message to the conversation
    messages.append(msg)

    for tool_call in msg.tool_calls:
        if tool_call.function.name == "web_search":
            args = json.loads(tool_call.function.arguments)
            query = args["query"]
            print(f"\n--- Model requested a search for: '{query}' ---")

            search_results = web_search(query)
            print("\n--- Search results returned to model ---")
            print(search_results)

            # Feed the tool's result back into the conversation
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": search_results
            })

    # Step 3: Ask the model again, now that it has real search results
    final_response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=tools
    )

    print("\n--- Final answer (grounded in search results) ---")
    print(final_response.choices[0].message.content)

else:
    # Model answered directly without needing to search
    print("\n--- Final answer (no search needed) ---")
    print(msg.content)
    #$env:TAVILY_API_KEY="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"
uvicorn server:app --host 0.0.0.0 --port 8000