from os import getenv
from typing import Any, Dict, Optional
import httpx

from omnicoreagent.core.tools.local_tools_registry import Tool


class FinancialDatasetsBase:
    def __init__(self, api_key: Optional[str] = None):
        if httpx is None:
            raise ImportError(
                "Could not import `httpx` python package. "
                "Please install it using `pip install httpx`."
            )
        self.api_key = api_key or getenv("FINANCIAL_DATASETS_API_KEY")
        self.base_url = "https://api.financialdatasets.ai"

    async def _make_request(
        self, endpoint: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not self.api_key:
            return {"status": "error", "data": None, "message": "API key not set"}

        headers = {"X-API-KEY": self.api_key}
        url = f"{self.base_url}/{endpoint}"

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=headers, params=params)
                response.raise_for_status()
                return {
                    "status": "success",
                    "data": response.json(),
                    "message": "Request successful",
                }
            except Exception as e:
                return {
                    "status": "error",
                    "data": None,
                    "message": f"Request failed: {str(e)}",
                }


class FinancialDatasetsGetIncomeStatements(FinancialDatasetsBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="financial_datasets_income_statements",
            description="Get income statements for a ticker.",
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "period": {
                        "type": "string",
                        "enum": ["annual", "quarterly", "ttm"],
                        "default": "annual",
                    },
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["ticker"],
            },
            function=self._get_statements,
        )

    async def _get_statements(
        self, ticker: str, period: str = "annual", limit: int = 10
    ) -> Dict[str, Any]:
        return await self._make_request(
            "financials/income-statements",
            {"ticker": ticker, "period": period, "limit": limit},
        )


class FinancialDatasetsGetBalanceSheets(FinancialDatasetsBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="financial_datasets_balance_sheets",
            description="Get balance sheets for a ticker.",
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "period": {"type": "string", "default": "annual"},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["ticker"],
            },
            function=self._get_sheets,
        )

    async def _get_sheets(
        self, ticker: str, period: str = "annual", limit: int = 10
    ) -> Dict[str, Any]:
        return await self._make_request(
            "financials/balance-sheets",
            {"ticker": ticker, "period": period, "limit": limit},
        )


class FinancialDatasetsGetCashFlowStatements(FinancialDatasetsBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="financial_datasets_cash_flow",
            description="Get cash flow statements for a ticker.",
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "period": {"type": "string", "default": "annual"},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["ticker"],
            },
            function=self._get_flow,
        )

    async def _get_flow(
        self, ticker: str, period: str = "annual", limit: int = 10
    ) -> Dict[str, Any]:
        return await self._make_request(
            "financials/cash-flow-statements",
            {"ticker": ticker, "period": period, "limit": limit},
        )


class FinancialDatasetsGetStockPrices(FinancialDatasetsBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="financial_datasets_stock_prices",
            description="Get stock prices for a ticker.",
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "interval": {"type": "string", "default": "1d"},
                    "limit": {"type": "integer", "default": 100},
                },
                "required": ["ticker"],
            },
            function=self._get_prices,
        )

    async def _get_prices(
        self, ticker: str, interval: str = "1d", limit: int = 100
    ) -> Dict[str, Any]:
        return await self._make_request(
            "prices", {"ticker": ticker, "interval": interval, "limit": limit}
        )
