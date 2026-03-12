import urllib.parse
from typing import Any, Dict, List, Optional

try:
    import httpx
except ImportError:
    httpx = None

from omnicoreagent.core.tools.local_tools_registry import Tool


class Searxng:
    def __init__(
        self,
        host: str,
        engines: Optional[List[str]] = None,
        fixed_max_results: Optional[int] = None,
    ):
        self.host = host
        self.host = host
        self.engines = engines or []
        self.fixed_max_results = fixed_max_results

        if httpx is None:
            raise ImportError(
                "Could not import `httpx` python package. "
                "Please install it with `pip install httpx`."
            )

    def get_tool(self) -> Tool:
        return Tool(
            name="searxng_search",
            description="Search the web using a SearXNG instance.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "category": {
                        "type": "string",
                        "description": "Category: general, images, it, map, music, news, science, videos",
                    },
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
            function=self._search,
        )

    async def _search(
        self, query: str, category: Optional[str] = None, max_results: int = 5
    ) -> Dict[str, Any]:
        encoded_query = urllib.parse.quote(query)
        url = f"{self.host}/search?format=json&q={encoded_query}"
        if self.engines:
            url += f"&engines={','.join(self.engines)}"
        if category:
            url += f"&categories={category}"

        try:
            resp = httpx.get(url).json()
            results = self.fixed_max_results or max_results
            resp["results"] = resp.get("results", [])[:results]
            return {
                "status": "success",
                "data": resp,
                "message": f"Found {len(resp['results'])} results",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


SearxngTools = Searxng
