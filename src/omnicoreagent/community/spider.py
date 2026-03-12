from typing import Any, Dict, Optional

from omnicoreagent.core.tools.local_tools_registry import Tool
from omnicoreagent.core.utils import logger

try:
    from spider import Spider as ExternalSpider
except ImportError:
    ExternalSpider = None


class SpiderTools:
    def __init__(
        self, max_results: Optional[int] = None, optional_params: Optional[dict] = None
    ):
        if ExternalSpider is None:
            raise ImportError(
                "`spider-client` not installed. Please install using `pip install spider-client`"
            )
        self.max_results = max_results
        self.optional_params = optional_params or {}

    def get_tool(self) -> Tool:
        return Tool(
            name="spider_search",
            description="Search the web using Spider.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
            function=self._search,
        )

    async def _search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        try:
            app = ExternalSpider()
            n = self.max_results or max_results
            options = {"fetch_page_content": False, "num": n, **self.optional_params}
            results = app.search(query, options)
            return {"status": "success", "data": results, "message": "Search completed"}
        except Exception as e:
            logger.error(f"Spider search failed: {e}")
            return {"status": "error", "data": None, "message": str(e)}


class SpiderScrape(SpiderTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="spider_scrape",
            description="Scrape content from a webpage using Spider.",
            inputSchema={
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
            function=self._scrape,
        )

    async def _scrape(self, url: str) -> Dict[str, Any]:
        try:
            app = ExternalSpider()
            options = {"return_format": "markdown", **self.optional_params}
            results = app.scrape_url(url, options)
            return {"status": "success", "data": results, "message": f"Scraped {url}"}
        except Exception as e:
            logger.error(f"Spider scrape failed: {e}")
            return {"status": "error", "data": None, "message": str(e)}


class SpiderCrawl(SpiderTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="spider_crawl",
            description="Crawl a website using Spider.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["url"],
            },
            function=self._crawl,
        )

    async def _crawl(self, url: str, limit: int = 10) -> Dict[str, Any]:
        try:
            app = ExternalSpider()
            options = {
                "return_format": "markdown",
                "limit": limit,
                **self.optional_params,
            }
            results = app.crawl_url(url, options)
            return {"status": "success", "data": results, "message": f"Crawled {url}"}
        except Exception as e:
            logger.error(f"Spider crawl failed: {e}")
            return {"status": "error", "data": None, "message": str(e)}
