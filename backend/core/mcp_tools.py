"""MCP tool definitions for Claude Agent SDK.

Tools:
- write_memory: save a memory to SQLite + ChromaDB
- recall_memory: semantic search over memories
- web_search: Tavily API call
"""

from typing import Any, Optional

from .memory.manager import MemoryManager


def make_tools(memory_manager: MemoryManager, character_id: str, tavily_key: Optional[str] = None):
    """Return a list of tool dicts compatible with Claude API tool_use format."""

    async def write_memory(
        content: str,
        category: str = "general",
        contextual_importance: float = 0.5,
        semantic_importance: float = 0.5,
        identity_importance: float = 0.5,
        user_importance: float = 0.5,
    ) -> dict:
        """Save a memory about the current conversation or character state."""
        memory_id = memory_manager.write_memory(
            character_id=character_id,
            content=content,
            category=category,
            contextual_importance=contextual_importance,
            semantic_importance=semantic_importance,
            identity_importance=identity_importance,
            user_importance=user_importance,
        )
        return {"memory_id": memory_id, "status": "saved"}

    async def recall_memory(query: str, top_k: int = 5) -> dict:
        """Search stored memories semantically relevant to the query."""
        results = memory_manager.recall_memory(
            character_id=character_id,
            query=query,
            top_k=top_k,
        )
        return {"memories": results, "count": len(results)}

    async def web_search(query: str) -> dict:
        """Search the web using Tavily for current information."""
        if not tavily_key:
            return {"error": "Tavily API key not configured"}
        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=tavily_key)
            response = client.search(query, max_results=5)
            return {"results": response.get("results", [])}
        except Exception as e:
            return {"error": str(e)}

    # Tool schema definitions for Claude API
    tool_schemas = [
        {
            "name": "write_memory",
            "description": (
                "Save an important memory about this conversation, the user, or "
                "your own state. Use this when you encounter information worth "
                "remembering for future conversations."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The memory content to save",
                    },
                    "category": {
                        "type": "string",
                        "description": "Category: identity, user, semantic, contextual",
                        "enum": ["identity", "user", "semantic", "contextual"],
                    },
                    "contextual_importance": {
                        "type": "number",
                        "description": "How important is this in current context? (0.0-1.0)",
                    },
                    "semantic_importance": {
                        "type": "number",
                        "description": "How semantically meaningful is this? (0.0-1.0)",
                    },
                    "identity_importance": {
                        "type": "number",
                        "description": "How relevant to character identity? (0.0-1.0)",
                    },
                    "user_importance": {
                        "type": "number",
                        "description": "How important is this to the user? (0.0-1.0)",
                    },
                },
                "required": ["content"],
            },
        },
        {
            "name": "recall_memory",
            "description": (
                "Search your stored memories for information relevant to the current "
                "query. Use this at the start of conversations to recall relevant context."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in memories",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of memories to retrieve (default: 5)",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "web_search",
            "description": "Search the web for current information using Tavily.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                },
                "required": ["query"],
            },
        },
    ]

    tool_handlers = {
        "write_memory": write_memory,
        "recall_memory": recall_memory,
        "web_search": web_search,
    }

    return tool_schemas, tool_handlers
