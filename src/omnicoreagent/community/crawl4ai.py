from typing import Any, Dict
from omnicoreagent.core.tools.local_tools_registry import Tool

try:
    from crawl4ai import AsyncWebCrawler
except ImportError:
    AsyncWebCrawler = None


class Crawl4AI(
    Tool
):  # Inherit from Tool or just implement get_tool? We implement get_tool standard.
    # Note: Previous standard uses class with get_tool.
    pass


class Crawl4AICrawl:
    def __init__(self):
        if AsyncWebCrawler is None:
            raise ImportError(
                "Could not import `crawl4ai` python package. "
                "Please install it using `pip install crawl4ai`."
            )

    def get_tool(self) -> Tool:
        return Tool(
            name="crawl4ai_crawl",
            description="Crawl a website using Crawl4AI.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                },
                "required": ["url"],
            },
            function=self._crawl,
        )

    async def _crawl(self, url: str) -> Dict[str, Any]:
        try:
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url=url)
                content = result.markdown if result.markdown else result.text
                return {
                    "status": "success",
                    "data": content,
                    "message": f"Crawled {len(content) if content else 0} chars",
                }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}
