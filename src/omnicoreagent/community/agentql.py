from os import getenv
from typing import Any, Dict, Optional
from omnicoreagent.core.tools.local_tools_registry import Tool

try:
    import agentql
    from playwright.sync_api import sync_playwright
except ImportError:
    agentql = None
    sync_playwright = None


class AgentQLBase:
    def __init__(self, api_key: Optional[str] = None):
        if agentql is None:
            raise ImportError(
                "Could not import `agentql` python package. "
                "Please install it using `pip install agentql playwright` and run `playwright install`."
            )
        self.api_key = api_key or getenv("AGENTQL_API_KEY")


class AgentQLScrapeWebsite(AgentQLBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="agentql_scrape_website",
            description="Scrape text content from a website using AgentQL.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                },
                "required": ["url"],
            },
            function=self._scrape,
        )

    async def _scrape(self, url: str) -> Dict[str, Any]:
        if not self.api_key:
            return {
                "status": "error",
                "data": None,
                "message": "AGENTQL_API_KEY not set",
            }
        if agentql is None:
            return {
                "status": "error",
                "data": None,
                "message": "agentql not installed. Please install it using `pip install agentql playwright` and run `playwright install`.",
            }

        query = """
        {
            text_content[]
        }
        """
        try:
            with (
                sync_playwright() as playwright,
                playwright.chromium.launch(headless=True) as browser,
            ):
                page = agentql.wrap(browser.new_page())
                page.goto(url)
                response = page.query_data(query)

                if isinstance(response, dict) and "text_content" in response:
                    text_items = [
                        item
                        for item in response["text_content"]
                        if item and item.strip()
                    ]
                    deduplicated = list(set(text_items))
                    content = " ".join(deduplicated)
                    return {
                        "status": "success",
                        "data": content,
                        "message": f"Scraped {len(content)} characters",
                    }
                return {
                    "status": "success",
                    "data": "",
                    "message": "No text content found",
                }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class AgentQLCustomQuery(AgentQLBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="agentql_custom_query",
            description="Scrape a website using a custom AgentQL query.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "query": {"type": "string"},
                },
                "required": ["url", "query"],
            },
            function=self._custom_scrape,
        )

    async def _custom_scrape(self, url: str, query: str) -> Dict[str, Any]:
        if not self.api_key:
            return {
                "status": "error",
                "data": None,
                "message": "AGENTQL_API_KEY not set",
            }
        if agentql is None:
            return {
                "status": "error",
                "data": None,
                "message": "agentql not installed. Please install it using `pip install agentql playwright` and run `playwright install`.",
            }

        try:
            with (
                sync_playwright() as playwright,
                playwright.chromium.launch(headless=True) as browser,
            ):
                page = agentql.wrap(browser.new_page())
                page.goto(url)
                response = page.query_data(query)
                return {
                    "status": "success",
                    "data": response,
                    "message": "Query executed successfully",
                }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}
