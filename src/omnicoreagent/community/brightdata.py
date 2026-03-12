import json
from os import getenv
from typing import Any, Dict, Optional

import requests
from omnicoreagent.core.tools.local_tools_registry import Tool
from omnicoreagent.core.utils import log_info


class BrightDataTools:
    def __init__(
        self,
        api_key: Optional[str] = None,
        serp_zone: str = "serp_api",
        web_unlocker_zone: str = "web_unlocker1",
        verbose: bool = False,
        timeout: int = 600,
    ):
        self.api_key = api_key or getenv("BRIGHT_DATA_API_KEY")
        self.verbose = verbose
        self.endpoint = "https://api.brightdata.com/request"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        self.web_unlocker_zone = getenv(
            "BRIGHT_DATA_WEB_UNLOCKER_ZONE", web_unlocker_zone
        )
        self.serp_zone = getenv("BRIGHT_DATA_SERP_ZONE", serp_zone)
        self.timeout = timeout

    def get_tool(self) -> Tool:
        # Defaults to search engine
        return Tool(
            name="brightdata_search",
            description="Search using Google, Bing, or Yandex via Bright Data.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "engine": {
                        "type": "string",
                        "enum": ["google", "bing", "yandex"],
                        "default": "google",
                    },
                    "num_results": {"type": "integer", "default": 10},
                    "language": {"type": "string"},
                    "country_code": {"type": "string"},
                },
                "required": ["query"],
            },
            function=self._search_engine,
        )

    async def _make_request(self, payload: Dict) -> str:
        if not self.api_key:
            raise ValueError("API Key not set")

        if self.verbose:
            log_info(f"[Bright Data] Request: {payload.get('url')}")

        try:
            # Sync request for now as BrightData doesn't have an official async lib
            response = requests.post(
                self.endpoint, headers=self.headers, data=json.dumps(payload)
            )
            if response.status_code != 200:
                raise Exception(
                    f"Failed to scrape: {response.status_code} - {response.text}"
                )
            return response.text
        except Exception as e:
            raise Exception(f"Request failed: {e}")

    async def _search_engine(
        self,
        query: str,
        engine: str = "google",
        num_results: int = 10,
        language: Optional[str] = None,
        country_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Search using Google, Bing, or Yandex and return results."""
        try:
            from urllib.parse import quote

            encoded_query = quote(query)

            base_urls = {
                "google": f"https://www.google.com/search?q={encoded_query}",
                "bing": f"https://www.bing.com/search?q={encoded_query}",
                "yandex": f"https://yandex.com/search/?text={encoded_query}",
            }

            if engine not in base_urls:
                return {"status": "error", "data": None, "message": "Invalid engine"}

            search_url = base_urls[engine]
            if engine == "google":
                params = []
                if language:
                    params.append(f"hl={language}")
                if country_code:
                    params.append(f"gl={country_code}")
                if num_results:
                    params.append(f"num={num_results}")
                if params:
                    search_url += "&" + "&".join(params)

            payload = {
                "url": search_url,
                "zone": self.serp_zone,
                "format": "raw",
                "data_format": "markdown",
            }
            content = await self._make_request(payload)
            return {
                "status": "success",
                "data": {"content": content},
                "message": "Search successful",
            }

        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class BrightDataScrape(BrightDataTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="brightdata_scrape_url",
            description="Scrape a webpage and return content in Markdown format.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                },
                "required": ["url"],
            },
            function=self._scrape_as_markdown,
        )

    async def _scrape_as_markdown(self, url: str) -> Dict[str, Any]:
        try:
            payload = {
                "url": url,
                "zone": self.web_unlocker_zone,
                "format": "raw",
                "data_format": "markdown",
            }
            content = await self._make_request(payload)
            return {
                "status": "success",
                "data": {"content": content},
                "message": "Scrape successful",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}
