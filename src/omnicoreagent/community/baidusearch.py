from typing import Any, Dict, List, Optional
from omnicoreagent.core.tools.local_tools_registry import Tool
from omnicoreagent.core.utils import log_debug

try:
    from baidusearch.baidusearch import search  # type: ignore
except ImportError:
    search = None

try:
    from pycountry import pycountry
except ImportError:
    pycountry = None


class BaiduSearchTools:
    """
    BaiduSearch is a toolkit for searching Baidu easily.
    """

    def __init__(
        self,
        fixed_max_results: Optional[int] = None,
        fixed_language: Optional[str] = None,
        headers: Optional[Any] = None,
        proxy: Optional[str] = None,
        timeout: Optional[int] = 10,
        debug: Optional[bool] = False,
    ):
        self.fixed_max_results = fixed_max_results
        self.fixed_language = fixed_language
        self.headers = headers
        self.proxy = proxy
        self.timeout = timeout
        self.proxy = proxy
        self.timeout = timeout
        self.debug = debug

        if search is None:
            raise ImportError(
                "Could not import `baidusearch` python package. "
                "Please install it with `pip install baidusearch`."
            )
        if pycountry is None:
            raise ImportError(
                "Could not import `pycountry` python package. "
                "Please install it with `pip install pycountry`."
            )

    def get_tool(self) -> Tool:
        return Tool(
            name="baidu_search",
            description="Execute Baidu search and return results.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keyword",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 5,
                    },
                    "language": {
                        "type": "string",
                        "description": "Search language",
                        "default": "zh",
                    },
                },
                "required": ["query"],
            },
            function=self._baidu_search,
        )

    async def _baidu_search(
        self, query: str, max_results: int = 5, language: str = "zh"
    ) -> Dict[str, Any]:
        """Execute Baidu search and return results"""

        max_results = self.fixed_max_results or max_results
        language = self.fixed_language or language

        if len(language) != 2:
            try:
                language = pycountry.languages.lookup(language).alpha_2
            except LookupError:
                language = "zh"

        log_debug(f"Searching Baidu [{language}] for: {query}")

        try:
            # baidusearch.search is synchronous
            results = baidusearch.baidusearch.search(
                keyword=query, num_results=max_results
            )

            res: List[Dict[str, str]] = []
            for idx, item in enumerate(results, 1):
                res.append(
                    {
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "abstract": item.get("abstract", ""),
                        "rank": str(idx),
                    }
                )

            return {
                "status": "success",
                "data": res,
                "message": f"Found {len(res)} results",
            }
        except Exception as e:
            return {
                "status": "error",
                "data": None,
                "message": f"Error searching Baidu: {str(e)}",
            }
