import re
from os import getenv
from typing import Any, Dict, Optional

try:
    import requests
except ImportError:
    requests = None
from omnicoreagent.core.tools.local_tools_registry import Tool


class ZendeskSearchArticles:
    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        company_name: Optional[str] = None,
    ):
        self.username = username or getenv("ZENDESK_USERNAME")
        self.password = password or getenv("ZENDESK_PASSWORD")
        self.company_name = company_name or getenv("ZENDESK_COMPANY_NAME")
        if requests is None:
            raise ImportError(
                "`requests` not installed. Please install using `pip install requests`"
            )

    def get_tool(self) -> Tool:
        return Tool(
            name="zendesk_search_articles",
            description="Search for articles in Zendesk Help Center.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
            function=self._search,
        )

    async def _search(self, query: str) -> Dict[str, Any]:
        if not self.username or not self.password or not self.company_name:
            return {
                "status": "error",
                "data": None,
                "message": "Missing Zendesk credentials or company name",
            }

        url = f"https://{self.company_name}.zendesk.com/api/v2/help_center/articles/search.json"
        try:
            response = requests.get(
                url, params={"query": query}, auth=(self.username, self.password)
            )
            response.raise_for_status()

            clean = re.compile("<.*?>")
            results = []
            for article in response.json().get("results", []):
                text_body = re.sub(clean, "", article.get("body", ""))
                results.append(
                    {
                        "id": article.get("id"),
                        "title": article.get("title"),
                        "url": article.get("html_url"),
                        "snippet": text_body[:200] + "..."
                        if len(text_body) > 200
                        else text_body,
                    }
                )

            return {
                "status": "success",
                "data": results,
                "message": f"Found {len(results)} articles",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}
