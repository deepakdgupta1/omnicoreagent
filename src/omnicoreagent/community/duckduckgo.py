from typing import Any, Dict, Optional
from omnicoreagent.core.tools.local_tools_registry import Tool


class DuckDuckGoSearchTool:
    """DuckDuckGo Search Tool integration."""

    def __init__(self, fixed_max_results: Optional[int] = None):
        try:
            import duckduckgo_search  # noqa: F401
        except ImportError:
            raise ImportError(
                "Could not import `duckduckgo-search` python package. "
                "Please install it using `pip install duckduckgo-search`."
            )
        self.fixed_max_results = fixed_max_results

    def get_tool(self) -> Tool:
        return Tool(
            name="duckduckgo_search",
            description="Search the web using DuckDuckGo. Privacy-focused search.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results.",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
            function=self._search,
        )

    async def _search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Search DuckDuckGo."""
        from duckduckgo_search import DDGS

        try:
            limit = self.fixed_max_results or max_results
            # DDGS is synchronous
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=limit))

            formatted_results = []
            for result in results:
                formatted_results.append(
                    f"Title: {result.get('title')}\nURL: {result.get('href')}\nSnippet: {result.get('body')}\n"
                )

            return {
                "status": "success",
                "data": results,
                "message": "\n---\n".join(formatted_results)
                if formatted_results
                else "No results found.",
            }

        except Exception as e:
            return {
                "status": "error",
                "data": None,
                "message": f"Error searching DuckDuckGo: {str(e)}",
            }
