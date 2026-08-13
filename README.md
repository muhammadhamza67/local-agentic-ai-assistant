# Local Agentic AI Assistant

A working AI agent system that runs entirely free and local — covering the core building blocks of agentic AI: a hand-built agent loop, a LangGraph rebuild, RAG over a custom knowledge base, MCP (Model Context Protocol) for reusable tools, and a multi-agent researcher/writer/supervisor system. A Flutter app serves as the frontend.

## What it does

Unlike a regular chatbot that only generates text from memory, this agent can:

- **Decide for itself** which of three tools it needs: web search, a calculator, or a document search — or answer directly if it already knows something
- **Search the web** for current information (news, weather, prices, etc.) using the Tavily search API
- **Perform exact calculations** using a real calculator tool instead of guessing at arithmetic
- **Answer questions from a custom document** (a university information sheet) using RAG — Retrieval-Augmented Generation — instead of only relying on the web or its own training data
- **Loop multiple times** if the first attempt doesn't return good enough results
- **Remember conversation history**, so follow-up questions work correctly
- **Handle its own mistakes gracefully** — if it tries to call a tool that doesn't exist, it recovers instead of breaking
- **Use tools built as independent MCP servers** — the calculator runs as a separate process, connected to the agent via the Model Context Protocol, not hardcoded into the agent's own code
- **Coordinate multiple specialized agents** — a separate Researcher/Writer/Supervisor system where a supervisor agent routes work between a research agent (gathers facts) and a writer agent (synthesizes them into a summary)

## Two implementations, on purpose

This project intentionally includes **two versions of the same agent**:

1. **`agent.py` / `server.py`** — built entirely by hand, with a manual loop, manual tool-call handling, and a manual memory dictionary. Built first, to understand exactly what an "agentic loop" actually does under the hood.
2. **`server_langgraph.py` / `server_rag.py`** — the same agent rebuilt using **LangGraph**, a real agent framework. Built second, once the fundamentals were understood, to see what a framework abstracts away (multi-round looping, tool schemas, conversation memory) versus what stays conceptually the same either way.

Comparing the two is a genuinely useful way to understand what frameworks are actually doing for you.

## Architecture

```
Flutter App  →  FastAPI Server  →  Local LLM (via LM Studio)
                      ↓                    ↓
              Tavily Search          Decides: search? calculate?
              Calculator             check documents? just answer?
              ChromaDB (RAG)
```

- **`main.dart`** — Flutter frontend. Chat UI showing answers, sources, search rounds used, and whether the answer came from the custom document.
- **`server.py`** — Hand-built FastAPI backend with a manual agent loop.
- **`server_langgraph.py`** — LangGraph version of the same agent, with built-in memory via `MemorySaver`.
- **`server_rag.py`** — LangGraph version plus a third tool: semantic search over a custom document using ChromaDB and sentence embeddings.
- **`server_mcp.py`** — LangGraph agent that connects to a separate MCP server for its calculator tool, combined with `web_search` and `search_documents`.
- **`mcp_calculator_server.py`** — A standalone MCP server exposing the calculator as an independent, reusable service — runs as its own process, discoverable by any MCP-compatible client, not just this specific agent.
- **`mcp_agent_client.py`** — A minimal example showing an agent connecting to the MCP calculator server and automatically discovering its tools.
- **`multi_agent_team.py`** / **`server_multiagent.py`** — A multi-agent system with three roles: a Researcher (gathers raw facts via web search), a Writer (synthesizes those facts into a polished summary, no tools of its own), and a Supervisor (decides which agent acts next, with a hard recursion limit as a safety net against infinite loops).
- **`agent.py`** — Original standalone test script used to first prove out the agent loop.
- **`rag_test.py`** — Standalone script to test document retrieval quality in isolation, before wiring it into the full agent.
- **`afut_university_info.txt`** — Sample knowledge base document used to test RAG (a fictional university's info, used specifically because the model has no other way to know these facts except by retrieving them).

## How RAG works in this project

1. On startup, the document is split into chunks (by paragraph)
2. Each chunk is converted into an embedding (a vector representing its meaning) using a local `sentence-transformers` model
3. Chunks + embeddings are stored in ChromaDB, a local vector database
4. When the agent decides to use `search_documents`, the question itself is embedded the same way, and ChromaDB returns the most semantically similar chunks — even if the wording doesn't match the document exactly
5. Those chunks are given to the model as context to answer from

## Tech stack

- **LLM:** Qwen2.5-VL-3B-Instruct, running locally and for free via [LM Studio](https://lmstudio.ai)
- **Agent framework:** [LangGraph](https://github.com/langchain-ai/langgraph) (for the framework-based versions)
- **Search:** [Tavily API](https://tavily.com) (free tier)
- **Vector database:** [ChromaDB](https://www.trychroma.com/) (local, free)
- **Embeddings:** `sentence-transformers` (all-MiniLM-L6-v2, local, free)
- **Backend:** Python, FastAPI
- **Frontend:** Flutter

## Running it locally

1. Open LM Studio, load a model that supports tool calling, and start the local server (Developer tab)
2. Set your Tavily API key as an environment variable: `TAVILY_API_KEY`
3. Install dependencies: `pip install fastapi uvicorn openai tavily-python langgraph langchain-openai chromadb sentence-transformers mcp langchain-mcp-adapters`
4. Run whichever backend you want to test, e.g.: `uvicorn server_rag:app --host 0.0.0.0 --port 8000` (or `server_mcp:app`, `server_multiagent:app`)
5. In the Flutter project folder, run `flutter pub get` then `flutter run`

## What I learned building this

- The core "agentic loop" (think → decide → act → observe → answer) that underlies every AI agent
- Tool descriptions matter as much as the code behind them — vague descriptions cause wrong assumptions about what a tool can do
- Small local models will sometimes hallucinate tool names or misuse available tools (e.g. approximating math instead of using a calculator tool) — a real agent needs to handle these gracefully
- What a framework like LangGraph actually abstracts away (looping, tool schemas, memory) versus what stays a fundamental challenge either way (tool selection quality, reasoning limitations of small models)
- Adding conversation memory can introduce new bugs in code that assumed a stateless request — found and fixed exactly this issue with a metadata tagging bug
- RAG (embeddings + vector search) lets an agent answer accurately from a specific knowledge base, matching by meaning rather than exact keywords
- Multi-agent orchestration requires deliberately narrow agent roles — a "researcher" that also writes polished answers makes a separate "writer" agent redundant, so role boundaries have to be enforced explicitly in each agent's instructions
- Multi-agent systems need hard safety limits (like a max recursion count), since coordination logic bugs can create infinite loops in a way single-agent tool loops don't
- Passing raw framework message objects (with tool-call metadata) between agents can confuse smaller models — extracting plain text into a clean prompt is more reliable for agent-to-agent handoffs

## Known limitations

- Uses a small (3B parameter) local model, weaker at self-correction and precise reasoning than larger models
- **Tool-selection is inconsistent for the same question.** In direct testing, asking the identical question ("what's the weather in Karachi right now?") in separate fresh sessions produced different behavior: sometimes the model correctly called `web_search` (after occasionally first hallucinating a non-existent `get_weather` tool and self-correcting), and sometimes it skipped tool use entirely and answered directly without attempting any tool. This is a known characteristic of smaller local models — larger models (GPT-4, Claude, Gemini) are significantly more consistent at deciding when to use available tools.
- Conversation memory and the vector database are both in-memory only — lost on server restart
- Document chunking is simple (paragraph-based) — a larger or less-structured document would need a more robust chunking strategy

## Possible next steps

- Learn MCP (Model Context Protocol) for standardized tool connections
- Build a multi-agent system (multiple agents collaborating on a task)
- Add persistent storage for both conversation history and the vector database
- Support uploading new documents for RAG dynamically, instead of a hardcoded file