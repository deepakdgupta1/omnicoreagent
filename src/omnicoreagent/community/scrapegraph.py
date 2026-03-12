from os import getenv
from typing import Any, Dict, Optional
from omnicoreagent.core.tools.local_tools_registry import Tool

try:
    from scrapegraph_py import Client
except ImportError:
    Client = None


class ScrapeGraphBase:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or getenv("SGAI_API_KEY")
        self.client = None
        if Client is None:
            raise ImportError(
                "scrapegraph-py not installed. Please install it using `pip install scrapegraph-py`."
            )
        elif self.api_key:
            try:
                self.client = Client(api_key=self.api_key)
            except Exception:
                pass


class ScrapeGraphSmartScraper(ScrapeGraphBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="scrapegraph_smartscraper",
            description="Extract structured data from a webpage using LLM.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "prompt": {"type": "string"},
                },
                "required": ["url", "prompt"],
            },
            function=self._smart_scrape,
        )

    async def _smart_scrape(self, url: str, prompt: str) -> Dict[str, Any]:
        if not self.client:
            return {
                "status": "error",
                "data": None,
                "message": "ScrapeGraph API key not set",
            }

        try:
            response = self.client.smartscraper(website_url=url, user_prompt=prompt)
            return {
                "status": "success",
                "data": response.get("result"),
                "message": "Smart scrap successful",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class ScrapeGraphMarkdownify(ScrapeGraphBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="scrapegraph_markdownify",
            description="Convert a webpage to markdown.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                },
                "required": ["url"],
            },
            function=self._markdownify,
        )

    async def _markdownify(self, url: str) -> Dict[str, Any]:
        if not self.client:
            return {
                "status": "error",
                "data": None,
                "message": "ScrapeGraph API key not set",
            }

        try:
            response = self.client.markdownify(website_url=url)
            return {
                "status": "success",
                "data": response.get("result"),
                "message": "Markdown conversion successful",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class ScrapeGraphSearch(ScrapeGraphBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="scrapegraph_search",
            description="Search the web and extract information.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                },
                "required": ["prompt"],
            },
            function=self._search,
        )

    async def _search(self, prompt: str) -> Dict[str, Any]:
        if not self.client:
            return {
                "status": "error",
                "data": None,
                "message": "ScrapeGraph API key not set",
            }

        try:
            response = self.client.searchscraper(user_prompt=prompt)
            return {
                "status": "success",
                "data": response.get("result"),
                "message": "Search successful",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}
