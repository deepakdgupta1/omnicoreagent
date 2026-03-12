import os
from typing import Optional, Dict, Any
from omnicoreagent.core.tools.local_tools_registry import Tool

try:
    from brave import Brave
except ImportError:
    Brave = None


class BraveSearchTool:
    """Brave Search Tool integration."""

    def __init__(self, api_key: Optional[str] = None):
        if Brave is None:
            raise ImportError(
                "Could not import `brave` python package. "
                "Please install it with `pip install brave-search`."
            )

        self.api_key = api_key or os.environ.get("BRAVE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "BRAVE_API_KEY is required. Please set the BRAVE_API_KEY environment variable."
            )

    def get_tool(self) -> Tool:
        return Tool(
            name="brave_search",
            description="Search the web using Brave Search.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search for.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "The maximum number of results to return.",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
            function=self._search,
        )

    async def _search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Search Brave."""

        try:
            brave = Brave(api_key=self.api_key)
            # brave library is synchronous
            search_results = brave.search(q=query, count=max_results)

            web_results = []
            formatted_results = []

            if hasattr(search_results, "web") and search_results.web:
                for result in search_results.web.results:
                    web_result = {
                        "title": result.title,
                        "url": str(result.url),
                        "description": result.description,
                    }
                    web_results.append(web_result)
                    formatted_results.append(
                        f"Title: {result.title}\nURL: {result.url}\nDescription: {result.description}\n"
                    )

            return {
                "status": "success",
                "data": web_results,
                "message": "\n---\n".join(formatted_results)
                if formatted_results
                else "No results found.",
            }
        except Exception as e:
            return {
                "status": "error",
                "data": None,
                "message": f"Error searching Brave: {str(e)}",
            }
