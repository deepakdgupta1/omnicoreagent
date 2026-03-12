from os import getenv
from typing import Any, Dict, Optional

from omnicoreagent.core.tools.local_tools_registry import Tool
from omnicoreagent.core.utils import logger

try:
    from openbb import obb as openbb_app
except ImportError:
    openbb_app = None


class OpenBBBase:
    def __init__(
        self,
        obb: Optional[Any] = None,
        openbb_pat: Optional[str] = None,
        provider: str = "yfinance",
    ):
        if openbb_app is None:
            raise ImportError(
                "Could not import `openbb` python package. "
                "Please install it with `pip install openbb`."
            )
        self.obb = obb or openbb_app
        self.provider = provider
        if self.obb:
            try:
                pat = openbb_pat or getenv("OPENBB_PAT")
                if pat:
                    self.obb.account.login(pat=pat)
            except Exception as e:
                logger.error(f"Error logging into OpenBB: {e}")


class OpenBBGetStockPrice(OpenBBBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="openbb_get_stock_price",
            description="Get current stock price for symbol(s).",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                },
                "required": ["symbol"],
            },
            function=self._get_price,
        )

    async def _get_price(self, symbol: str) -> Dict[str, Any]:
        try:
            result = self.obb.equity.price.quote(
                symbol=symbol, provider=self.provider
            ).to_polars()
            data = result.to_dicts()
            return {
                "status": "success",
                "data": data,
                "message": f"Stock price for {symbol}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class OpenBBSearchCompany(OpenBBBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="openbb_search_company",
            description="Search for company ticker symbols.",
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
        try:
            result = self.obb.equity.search(query).to_polars()
            data = result.to_dicts()
            return {
                "status": "success",
                "data": data,
                "message": f"Search results for {query}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class OpenBBGetCompanyNews(OpenBBBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="openbb_get_company_news",
            description="Get company news.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "limit": {"type": "integer", "default": 5},
                },
                "required": ["symbol"],
            },
            function=self._get_news,
        )

    async def _get_news(self, symbol: str, limit: int = 5) -> Dict[str, Any]:
        try:
            result = self.obb.news.company(
                symbol=symbol, provider=self.provider, limit=limit
            ).to_polars()
            data = result.to_dicts()
            # Clean images if any
            for d in data:
                if "images" in d:
                    del d["images"]

            return {"status": "success", "data": data, "message": f"News for {symbol}"}
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}
