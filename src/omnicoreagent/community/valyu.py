from os import getenv
from typing import Any, Dict, List, Optional

from omnicoreagent.core.tools.local_tools_registry import Tool
from omnicoreagent.core.utils import log_error

try:
    from valyu import Valyu
except ImportError:
    Valyu = None


class ValyuTools:
    def __init__(
        self,
        api_key: Optional[str] = None,
        text_length: int = 1000,
        max_results: int = 10,
        relevance_threshold: float = 0.5,
        max_price: float = 30.0,
        content_category: Optional[str] = None,
        search_start_date: Optional[str] = None,
        search_end_date: Optional[str] = None,
        sources: Optional[List[str]] = None,
        tool_call_mode: bool = False,
    ):
        self.api_key = api_key or getenv("VALYU_API_KEY")
        if not self.api_key:
            raise ValueError("VALYU_API_KEY not set.")

        if Valyu is None:
            raise ImportError(
                "`valyu` not installed. Please install using `pip install valyu`"
            )

        self.valyu = Valyu(api_key=self.api_key)
        self.text_length = text_length
        self.max_results = max_results
        self.relevance_threshold = relevance_threshold
        self.max_price = max_price
        self.content_category = content_category
        self.search_start_date = search_start_date
        self.search_end_date = search_end_date
        self.sources = sources
        self.tool_call_mode = tool_call_mode

    def _parse_results(self, results: List[Any]) -> List[Dict[str, Any]]:
        parsed = []
        for r in results:
            d: Dict[str, Any] = {}
            if hasattr(r, "url") and r.url:
                d["url"] = r.url
            if hasattr(r, "title") and r.title:
                d["title"] = r.title
            if hasattr(r, "source") and r.source:
                d["source"] = r.source
            if hasattr(r, "relevance_score"):
                d["relevance_score"] = r.relevance_score
            if hasattr(r, "content") and r.content:
                content = r.content
                if self.text_length and len(content) > self.text_length:
                    content = content[: self.text_length] + "..."
                d["content"] = content
            parsed.append(d)
        return parsed

    def _valyu_search(self, query: str, search_type: str, **kwargs) -> Dict[str, Any]:
        try:
            params = {
                "query": query,
                "search_type": search_type,
                "max_num_results": self.max_results,
                "is_tool_call": self.tool_call_mode,
                "relevance_threshold": self.relevance_threshold,
                "max_price": self.max_price,
            }
            if kwargs.get("sources") or self.sources:
                params["included_sources"] = kwargs.get("sources") or self.sources
            if kwargs.get("content_category") or self.content_category:
                params["category"] = (
                    kwargs.get("content_category") or self.content_category
                )
            if kwargs.get("start_date") or self.search_start_date:
                params["start_date"] = (
                    kwargs.get("start_date") or self.search_start_date
                )
            if kwargs.get("end_date") or self.search_end_date:
                params["end_date"] = kwargs.get("end_date") or self.search_end_date

            response = self.valyu.search(**params)
            if not response.success:
                return {
                    "status": "error",
                    "data": None,
                    "message": response.error or "Search failed",
                }
            data = self._parse_results(response.results or [])
            return {
                "status": "success",
                "data": data,
                "message": f"Found {len(data)} results",
            }
        except Exception as e:
            log_error(f"Valyu search failed: {e}")
            return {"status": "error", "data": None, "message": str(e)}

    def get_tool(self) -> Tool:
        return Tool(
            name="valyu_search_academic",
            description="Search academic sources (ArXiv, PubMed) using Valyu.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "YYYY-MM-DD"},
                },
                "required": ["query"],
            },
            function=self._search_academic,
        )

    async def _search_academic(
        self,
        query: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        sources = [
            "valyu/valyu-arxiv",
            "valyu/valyu-pubmed",
            "wiley/wiley-finance-papers",
            "wiley/wiley-finance-books",
        ]
        return self._valyu_search(
            query,
            "proprietary",
            sources=sources,
            start_date=start_date,
            end_date=end_date,
        )


class ValyuSearchWeb(ValyuTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="valyu_search_web",
            description="Search the web using Valyu.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "content_category": {"type": "string"},
                },
                "required": ["query"],
            },
            function=self._search_web,
        )

    async def _search_web(
        self,
        query: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        content_category: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._valyu_search(
            query,
            "web",
            content_category=content_category,
            start_date=start_date,
            end_date=end_date,
        )


class ValyuSearchPaper(ValyuTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="valyu_search_paper",
            description="Search within a specific ArXiv paper using Valyu.",
            inputSchema={
                "type": "object",
                "properties": {
                    "paper_url": {"type": "string"},
                    "query": {"type": "string"},
                },
                "required": ["paper_url", "query"],
            },
            function=self._search_paper,
        )

    async def _search_paper(self, paper_url: str, query: str) -> Dict[str, Any]:
        if not paper_url.startswith("https:/"):
            return {"status": "error", "data": None, "message": "Invalid paper URL"}
        return self._valyu_search(query, "proprietary", sources=[paper_url])
