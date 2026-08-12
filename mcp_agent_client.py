"""
This version of the agent does NOT define calculator() itself at all.
Instead, it connects to mcp_calculator_server.py as a CLIENT, and pulls
in whatever tools that server exposes, automatically.

This is the real power of MCP: your agent code has zero knowledge of how
the calculator actually works internally — it just knows "connect to this
server, and use whatever tools it offers."
"""

import asyncio
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

llm = ChatOpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
    model="qwen2.5-vl-3b-instruct",
)


async def main():
    # This tells our agent: "connect to mcp_calculator_server.py by running
    # it as a subprocess, and talk to it using the stdio transport."
    client = MultiServerMCPClient(
        {
            "calculator": {
                "command": "python",
                "args": ["mcp_calculator_server.py"],
                "transport": "stdio",
            }
        }
    )

    # This automatically discovers whatever tools the MCP server exposes —
    # in this case, just calculator — without us defining it in this file.
    tools = await client.get_tools()

    print(f"Tools discovered from MCP server: {[t.name for t in tools]}")

    agent = create_react_agent(llm, tools=tools)

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What is 234 * 18, then add 500?"}]}
    )

    print("\n--- Final answer ---")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())