import os
from typing import Any, Dict, Optional
from omnicoreagent.core.tools.local_tools_registry import Tool

try:
    import serpapi
except ImportError:
    serpapi = None


class SerpApiGoogleSearch:
    """SerpApi Google Search Tool integration."""

    def __init__(self, api_key: Optional[str] = None):
        if serpapi is None:
            raise ImportError(
                "Could not import `google-search-results` python package. "
                "Please install it with `pip install google-search-results`."
            )

        self.api_key = api_key or os.environ.get("SERP_API_KEY")
        if not self.api_key:
            # We don't raise error here to avoid breaking import if enc var missing,
            # but _search will fail. A common pattern is to raise in __init__ if strictly required.
            pass

    def get_tool(self) -> Tool:
        return Tool(
            name="serpapi_google_search",
            description="Search Google using SerpApi.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results.",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
            function=self._search,
        )

    async def _search(self, query: str, num_results: int = 10) -> Dict[str, Any]:
        if not self.api_key:
            return {
                "status": "error",
                "data": None,
                "message": "SERP_API_KEY not provided.",
            }

        try:
            params = {"q": query, "api_key": self.api_key, "num": num_results}
            # serpapi client is synchronous
            search = serpapi.GoogleSearch(params)
            results = search.get_dict()

            organic_results = results.get("organic_results", [])
            formatted_results = []

            for result in organic_results:
                formatted_results.append(
                    f"Title: {result.get('title')}\nURL: {result.get('link')}\nSnippet: {result.get('snippet')}\n"
                )

            return {
                "status": "success",
                "data": organic_results,
                "message": "\n---\n".join(formatted_results)
                if formatted_results
                else "No results found.",
            }

        except Exception as e:
            return {
                "status": "error",
                "data": None,
                "message": f"Error searching SerpApi: {str(e)}",
            }
