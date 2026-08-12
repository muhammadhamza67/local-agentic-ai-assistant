"""
Your first MCP server.

This is fundamentally different from how you built tools before. Previously,
`calculator` was a Python function living INSIDE server_rag.py, only usable
by that one agent. Here, it's a completely separate, standalone program that
speaks the MCP protocol — meaning ANY MCP-compatible application (your agent,
Claude Desktop, another developer's app) can connect to it and use it,
without needing your Python code at all.

Run this file directly to start the server:
    python mcp_calculator_server.py
"""

from mcp.server.fastmcp import FastMCP

# Create the MCP server, giving it a name that identifies it to whatever connects
mcp = FastMCP("calculator-server")


@mcp.tool()
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the exact result.
    Use this for any math calculation instead of computing it yourself."""
    import re
    if not re.match(r'^[\d\s\.\+\-\*\/\(\)]+$', expression):
        return "Error: expression contains disallowed characters."
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error evaluating expression: {e}"


if __name__ == "__main__":
    # This starts the server and makes it listen for connections.
    # "stdio" transport means it communicates over standard input/output —
    # the simplest way for a local app (like your agent) to talk to it.
    mcp.run(transport="stdio")