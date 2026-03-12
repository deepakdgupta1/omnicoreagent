from os import getenv
from typing import Any, Dict, Optional

try:
    import httpx
except ImportError:
    httpx = None

from omnicoreagent.core.tools.local_tools_registry import Tool
from omnicoreagent.core.utils import logger


class JinaReadUrl:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://r.jina.ai/",
        max_content_length: int = 10000,
        timeout: Optional[int] = None,
    ):
        if httpx is None:
            raise ImportError(
                "`httpx` not installed. Please install using `pip install httpx`"
            )
        self.api_key = api_key or getenv("JINA_API_KEY")
        self.base_url = base_url
        self.max_content_length = max_content_length
        self.timeout = timeout

    def _get_headers(self) -> Dict[str, str]:
        headers = {
            "Accept": "application/json",
            "X-With-Links-Summary": "true",
            "X-With-Images-Summary": "true",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.timeout:
            headers["X-Timeout"] = str(self.timeout)
        return headers

    def _truncate_content(self, content: str) -> str:
        if len(content) > self.max_content_length:
            return content[: self.max_content_length] + "... (content truncated)"
        return content

    def get_tool(self) -> Tool:
        return Tool(
            name="jina_read_url",
            description="Read a URL and return its content using the Jina Reader API.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to read"},
                },
                "required": ["url"],
            },
            function=self._read_url,
        )

    async def _read_url(self, url: str) -> Dict[str, Any]:
        full_url = f"{self.base_url}{url}"
        try:
            response = httpx.get(full_url, headers=self._get_headers())
            response.raise_for_status()
            content = response.json()
            truncated = self._truncate_content(str(content))
            return {
                "status": "success",
                "data": truncated,
                "message": "URL content retrieved",
            }
        except Exception as e:
            logger.error(f"Error reading URL: {e}")
            return {"status": "error", "data": None, "message": str(e)}


class JinaSearchQuery(JinaReadUrl):
    def __init__(
        self,
        api_key: Optional[str] = None,
        search_url: str = "https://s.jina.ai/",
        max_content_length: int = 10000,
        timeout: Optional[int] = None,
        search_query_content: bool = True,
    ):
        super().__init__(
            api_key=api_key, max_content_length=max_content_length, timeout=timeout
        )
        self.search_url = search_url
        self.search_query_content = search_query_content

    def get_tool(self) -> Tool:
        return Tool(
            name="jina_search_query",
            description="Search the web using Jina Reader API.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
            function=self._search_query,
        )

    async def _search_query(self, query: str) -> Dict[str, Any]:
        headers = self._get_headers()
        if not self.search_query_content:
            headers["X-Respond-With"] = "no-content"

        body = {"q": query}
        try:
            response = httpx.post(self.search_url, headers=headers, json=body)
            response.raise_for_status()
            content = response.json()
            truncated = self._truncate_content(str(content))
            return {
                "status": "success",
                "data": truncated,
                "message": "Search results retrieved",
            }
        except Exception as e:
            logger.error(f"Error performing search: {e}")
            return {"status": "error", "data": None, "message": str(e)}
