"""
Your first multi-agent system.

Until now, you've had ONE agent choosing between TOOLS (web_search,
calculator, search_documents). This is different: now you have THREE
SEPARATE AGENTS, each with their own job, and a SUPERVISOR whose only
role is to decide which agent should act next.

The team:
- Researcher: has access to web_search. Its ONLY job is gathering raw
  information — it does not write a polished final answer.
- Writer: has NO tools. Its ONLY job is taking whatever research exists
  so far and turning it into a clean, well-organized summary.
- Supervisor: doesn't do any work itself. It looks at the conversation
  so far and decides: should the Researcher go next? The Writer? Or is
  the task actually done?

This pattern is called "supervisor orchestration" — one of the most
common real-world multi-agent architectures.
"""

from typing import Literal, TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from tavily import TavilyClient
import os

llm = ChatOpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
    model="qwen2.5-vl-3b-instruct",
)

tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


@tool
def web_search(query: str) -> str:
    """Search the web for current, real-time information."""
    results = tavily.search(query=query, max_results=5, topic="news", search_depth="advanced")
    formatted = []
    for r in results.get("results", []):
        formatted.append(f"- {r['title']}: {r['content'][:300]}")
    return "\n".join(formatted) if formatted else "No results found."


# --- Shared state ---
# Every agent in the team reads from and writes to this SAME state object.
# This is how they "communicate" — not by talking directly to each other,
# but by all seeing the same growing conversation history.

class TeamState(TypedDict):
    messages: Annotated[list, add_messages]
    next_agent: str
    writer_has_run: bool  # tracks whether the writer already produced a summary


# --- Researcher agent ---
# A normal single-tool agent, exactly like the ones you've already built.

researcher = create_react_agent(llm, tools=[web_search])


def researcher_node(state: TeamState) -> dict:
    print("\n>>> RESEARCHER is working...")

    researcher_instruction = {
        "role": "system",
        "content": (
            "You are a researcher. Use web_search to gather relevant facts. "
            "Report your RAW FINDINGS only — bullet points of facts and sources "
            "are fine. Do NOT write a polished summary or conclusion; that is "
            "the writer's job, not yours."
        )
    }

    result = researcher.invoke({"messages": [researcher_instruction] + state["messages"]})
    new_messages = result["messages"][len(state["messages"]) + 1:]  # skip the injected system message
    print(f">>> RESEARCHER produced {len(new_messages)} new message(s)")
    return {"messages": new_messages}


# --- Writer agent ---
# No tools at all — it can only read what's already in the conversation
# and write a summary. It CANNOT search the web itself.

def writer_node(state: TeamState) -> dict:
    print("\n>>> WRITER is working...")

    # Instead of handing the raw message objects (which include tool_calls
    # metadata that can confuse a small model), extract just the readable
    # TEXT content into a simple, clean prompt.
    original_question = state["messages"][0].content

    research_text = ""
    for m in state["messages"][1:]:
        content = getattr(m, "content", "")
        if content:  # skip empty messages (like the researcher's tool-call trigger message)
            research_text += content + "\n\n"

    writer_prompt = (
        f"The user asked: {original_question}\n\n"
        f"Here is the research that was gathered:\n{research_text}\n\n"
        f"Write a clean, well-organized final summary answering the user's "
        f"question, based on this research."
    )

    response = llm.invoke([{"role": "user", "content": writer_prompt}])
    print(">>> WRITER finished the summary")
    return {"messages": [response], "writer_has_run": True}


# --- Supervisor ---
# This is the coordinator. It doesn't do the work — it just decides who
# goes next, based on what's happened in the conversation so far.

def supervisor_node(state: TeamState) -> dict:
    print("\n>>> SUPERVISOR is deciding who goes next...")

    supervisor_prompt = (
        "You are a supervisor managing two workers: 'researcher' and 'writer'.\n"
        "- If the user's question needs information gathered from the web "
        "and no research has happened yet, respond with exactly: researcher\n"
        "- If research HAS already happened and it's time to write a final "
        "summary, respond with exactly: writer\n"
        "- If a final written summary already exists in the conversation, "
        "respond with exactly: FINISH\n"
        "Respond with ONLY one word: researcher, writer, or FINISH. Nothing else."
    )

    response = llm.invoke(
        [{"role": "system", "content": supervisor_prompt}] + state["messages"]
    )

    decision = response.content.strip().lower()
    print(f">>> SUPERVISOR decided: '{decision}'")

    if "researcher" in decision:
        return {"next_agent": "researcher"}
    elif "writer" in decision:
        return {"next_agent": "writer"}
    elif "finish" in decision:
        return {"next_agent": "FINISH"}
    else:
        # Ambiguous response. If the writer already produced a summary,
        # we're done — don't loop back to it again. Otherwise, check if
        # research has happened yet to decide what's still needed.
        if state.get("writer_has_run"):
            print(">>> (Ambiguous decision, but writer already ran — finishing)")
            return {"next_agent": "FINISH"}

        has_researched = any(
            getattr(m, "name", None) == "web_search" or
            (hasattr(m, "content") and "web_search" in str(getattr(m, "tool_calls", "")))
            for m in state["messages"]
        )
        fallback = "writer" if has_researched else "researcher"
        print(f">>> (Ambiguous decision, falling back to: {fallback})")
        return {"next_agent": fallback}


def route(state: TeamState) -> Literal["researcher", "writer", "__end__"]:
    """Reads the supervisor's decision and tells LangGraph which node to go to."""
    if state["next_agent"] == "FINISH":
        return END
    return state["next_agent"]


# --- Build the graph ---

graph = StateGraph(TeamState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("researcher", researcher_node)
graph.add_node("writer", writer_node)

graph.set_entry_point("supervisor")
graph.add_edge("researcher", "supervisor")
graph.add_edge("writer", "supervisor")
graph.add_conditional_edges("supervisor", route)

# Safety net: even with correct logic, ALWAYS cap total steps when building
# multi-agent systems. This is standard practice — it guarantees the graph
# can never run forever, no matter what bug might exist in the routing logic.
team = graph.compile()

if __name__ == "__main__":
    result = team.invoke(
        {
            "messages": [{"role": "user", "content": "Research the latest developments in fusion energy and give me a short summary."}],
            "next_agent": "",
            "writer_has_run": False
        },
        config={"recursion_limit": 10}  # hard safety cap on total steps
    )

    print("\n\n=== FINAL RESULT ===")
    print(result["messages"][-1].content)

    print("\n\n=== DEBUG: all messages ===")
    for i, m in enumerate(result["messages"]):
        content_preview = str(getattr(m, "content", "NO CONTENT ATTR"))[:200]
        print(f"[{i}] {type(m).__name__}: {content_preview!r}")