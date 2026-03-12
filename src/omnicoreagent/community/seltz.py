from os import getenv
from typing import Any, Dict, Optional

from omnicoreagent.core.tools.local_tools_registry import Tool
from omnicoreagent.core.utils import logger

try:
    from seltz import Seltz
except ImportError:
    Seltz = None


class SeltzTools:
    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        insecure: bool = False,
        max_documents: int = 10,
    ):
        self.api_key = api_key or getenv("SELTZ_API_KEY")
        if not self.api_key:
            logger.error("SELTZ_API_KEY not set.")
        self.max_documents = max_documents
        self.client = None
        if Seltz is None:
            raise ImportError(
                "`seltz` not installed. Please install using `pip install seltz`"
            )  # Assuming package name
        if self.api_key:
            kwargs: Dict[str, Any] = {"api_key": self.api_key}
            if endpoint:
                kwargs["endpoint"] = endpoint
            if insecure:
                kwargs["insecure"] = insecure
            self.client = Seltz(**kwargs)

    def get_tool(self) -> Tool:
        return Tool(
            name="seltz_search",
            description="Search using the Seltz AI-powered search API.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_documents": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
            function=self._search,
        )

    async def _search(
        self, query: str, max_documents: Optional[int] = None
    ) -> Dict[str, Any]:
        if not self.client:
            return {"status": "error", "data": None, "message": "SELTZ_API_KEY not set"}
        try:
            response = self.client.search(
                query, max_documents=max_documents or self.max_documents
            )
            docs = []
            for doc in getattr(response, "documents", []) or []:
                d: Dict[str, Any] = {}
                if hasattr(doc, "url") and doc.url:
                    d["url"] = doc.url
                if hasattr(doc, "content") and doc.content:
                    d["content"] = doc.content
                if d:
                    docs.append(d)
            return {
                "status": "success",
                "data": docs,
                "message": f"Found {len(docs)} documents",
            }
        except Exception as e:
            logger.error(f"Seltz search failed: {e}")
            return {"status": "error", "data": None, "message": str(e)}
