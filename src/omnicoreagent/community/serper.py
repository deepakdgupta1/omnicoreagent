from os import getenv
from typing import Any, Dict, Optional

try:
    import requests
except ImportError:
    requests = None

from omnicoreagent.core.tools.local_tools_registry import Tool
from omnicoreagent.core.utils import log_error, log_warning


class SerperTools:
    def __init__(
        self,
        api_key: Optional[str] = None,
        location: str = "us",
        language: str = "en",
        num_results: int = 10,
    ):
        self.api_key = api_key or getenv("SERPER_API_KEY")
        if requests is None:
            raise ImportError(
                "`requests` not installed. Please install using `pip install requests`"
            )
        if not self.api_key:
            log_warning("SERPER_API_KEY not set.")
        self.location = location
        self.language = language
        self.num_results = num_results

    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"X-API-KEY": self.api_key or "", "Content-Type": "application/json"}
        try:
            response = requests.post(
                f"https://google.serper.dev/{endpoint}", json=params, headers=headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            log_error(f"Serper request failed: {e}")
            return {"error": str(e)}

    def get_tool(self) -> Tool:
        return Tool(
            name="serper_search",
            description="Search Google using the Serper API.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "num_results": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
            function=self._search_web,
        )

    async def _search_web(
        self, query: str, num_results: Optional[int] = None
    ) -> Dict[str, Any]:
        if not self.api_key:
            return {
                "status": "error",
                "data": None,
                "message": "SERPER_API_KEY not set",
            }
        try:
            params = {
                "q": query,
                "gl": self.location,
                "hl": self.language,
                "num": num_results or self.num_results,
            }
            result = self._make_request("search", params)
            if "error" in result:
                return {"status": "error", "data": None, "message": result["error"]}
            return {"status": "success", "data": result, "message": "Search completed"}
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class SerperSearchNews(SerperTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="serper_search_news",
            description="Search Google News using the Serper API.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "num_results": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
            function=self._search_news,
        )

    async def _search_news(
        self, query: str, num_results: Optional[int] = None
    ) -> Dict[str, Any]:
        if not self.api_key:
            return {
                "status": "error",
                "data": None,
                "message": "SERPER_API_KEY not set",
            }
        try:
            params = {
                "q": query,
                "gl": self.location,
                "hl": self.language,
                "num": num_results or self.num_results,
            }
            result = self._make_request("news", params)
            if "error" in result:
                return {"status": "error", "data": None, "message": result["error"]}
            return {
                "status": "success",
                "data": result,
                "message": "News search completed",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class SerperScrapeWebpage(SerperTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="serper_scrape_webpage",
            description="Scrape content from a webpage using the Serper API.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "markdown": {"type": "boolean", "default": False},
                },
                "required": ["url"],
            },
            function=self._scrape_webpage,
        )

    async def _scrape_webpage(self, url: str, markdown: bool = False) -> Dict[str, Any]:
        if not self.api_key:
            return {
                "status": "error",
                "data": None,
                "message": "SERPER_API_KEY not set",
            }
        try:
            headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
            params: Dict[str, Any] = {"url": url}
            if markdown:
                params["format"] = "markdown"
            response = requests.post(
                "https://scrape.serper.dev", json=params, headers=headers
            )
            response.raise_for_status()
            return {
                "status": "success",
                "data": response.json(),
                "message": f"Scraped {url}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}
