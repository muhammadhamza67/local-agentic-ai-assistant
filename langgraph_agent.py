"""
LangGraph version of your agent — same behavior as agent.py / server.py,
but built using a framework instead of a hand-written loop.

Compare this file to agent.py side by side: notice what code DISAPPEARED
(the manual while-loop, the manual tool_calls checking) and what stayed
conceptually the same (you still define tools, still need a real function
behind each tool, the model still decides when to use them).
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from tavily import TavilyClient

# --- Same local model connection as before, just using LangChain's wrapper ---
llm = ChatOpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
    model="qwen2.5-vl-3b-instruct",
)

tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


# --- Same web_search logic as before, but as a LangChain @tool ---
# Notice: no manual JSON schema needed anymore — the docstring and type
# hints below ARE the tool definition. LangGraph reads them automatically.
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


# --- This one line replaces your entire hand-written loop from server.py ---
# create_react_agent builds the whole think -> act -> observe -> answer loop
# for you, including multi-round looping, automatically.
agent = create_react_agent(llm, tools=[web_search, calculator])


def ask_agent(question: str):
    """Run the LangGraph agent on a question and print each step."""
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})

    print("\n--- Full message history (every step of the loop) ---")
    for m in result["messages"]:
        role = m.type if hasattr(m, "type") else m.get("role", "?")
        content = m.content if hasattr(m, "content") else m.get("content", "")
        print(f"[{role}] {content}")

    print("\n--- Final answer ---")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    ask_agent("What's the current weather in Karachi?")
