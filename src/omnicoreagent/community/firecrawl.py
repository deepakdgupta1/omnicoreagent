from os import getenv
from typing import Any, Dict, List, Optional
from omnicoreagent.core.tools.local_tools_registry import Tool

try:
    from firecrawl import FirecrawlApp
except ImportError:
    FirecrawlApp = None


class FirecrawlBase:
    def __init__(self, api_key: Optional[str] = None):
        if FirecrawlApp is None:
            raise ImportError(
                "Could not import `firecrawl` python package. "
                "Please install it using `pip install firecrawl`."
            )
        self.api_key = api_key or getenv("FIRECRAWL_API_KEY")
        self.app = None
        if self.api_key:
            try:
                self.app = FirecrawlApp(api_key=self.api_key)
            except Exception:
                self.app = None


class FirecrawlScrape(FirecrawlBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="firecrawl_scrape",
            description="Scrape a website using Firecrawl.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "formats": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["url"],
            },
            function=self._scrape,
        )

    async def _scrape(
        self, url: str, formats: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        if not self.app:
            return {
                "status": "error",
                "data": None,
                "message": "Firecrawl API key not set",
            }

        try:
            params = {}
            if formats:
                params["formats"] = formats

            result = self.app.scrape_url(url, params=params)
            return {"status": "success", "data": result, "message": "Scrape successful"}
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class FirecrawlCrawl(FirecrawlBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="firecrawl_crawl",
            description="Crawl a website using Firecrawl.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "limit": {"type": "integer", "default": 10},
                    "formats": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["url"],
            },
            function=self._crawl,
        )

    async def _crawl(
        self, url: str, limit: int = 10, formats: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        if not self.app:
            return {
                "status": "error",
                "data": None,
                "message": "Firecrawl API key not set",
            }

        try:
            params = {"limit": limit}
            if formats:
                params["scrapeOptions"] = {"formats": formats}

            # Firecrawl crawl returns a job ID or waits. Assuming sync wait for simplicity or handling async if library supports.
            # The library typically returns a job object or result.
            # Using the simplified sync call pattern from previous implementation
            result = self.app.crawl_url(url, params=params)
            return {
                "status": "success",
                "data": result,
                "message": "Crawl initiated/completed",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class FirecrawlSearch(FirecrawlBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="firecrawl_search",
            description="Search the web using Firecrawl.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
            function=self._search,
        )

    async def _search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        if not self.app:
            return {
                "status": "error",
                "data": None,
                "message": "Firecrawl API key not set",
            }

        try:
            # Note: Method name might vary based on version, assuming search() exists as per previous file
            # Previous file used app.search(query, **params)
            result = self.app.search(query, params={"limit": limit})
            # Check if result has data/success attributes
            data = result if isinstance(result, dict) else result.__dict__
            return {"status": "success", "data": data, "message": "Search completed"}
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}
