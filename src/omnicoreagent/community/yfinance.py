from typing import Any, Dict, Optional

from omnicoreagent.core.tools.local_tools_registry import Tool

try:
    import yfinance as yf
except ImportError:
    yf = None


class YFinanceBase:
    def __init__(self, session: Optional[Any] = None):
        if yf is None:
            raise ImportError(
                "Could not import `yfinance` python package. "
                "Please install it with `pip install yfinance`."
            )
        self.session = session


class YFinanceGetStockPrice(YFinanceBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="yfinance_get_stock_price",
            description="Get current stock price for a symbol.",
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
            stock = yf.Ticker(symbol, session=self.session)
            price = stock.info.get("regularMarketPrice", stock.info.get("currentPrice"))
            if price:
                return {
                    "status": "success",
                    "data": {"symbol": symbol, "price": price},
                    "message": f"Current price for {symbol}: {price}",
                }
            return {
                "status": "error",
                "data": None,
                "message": f"Could not fetch price for {symbol}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class YFinanceGetCompanyInfo(YFinanceBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="yfinance_get_company_info",
            description="Get company profile and overview.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                },
                "required": ["symbol"],
            },
            function=self._get_info,
        )

    async def _get_info(self, symbol: str) -> Dict[str, Any]:
        try:
            info = yf.Ticker(symbol, session=self.session).info
            if info:
                # Basic cleaning/filtering could happen here, but returning full info is also fine if standardized
                return {
                    "status": "success",
                    "data": info,
                    "message": f"Company info for {symbol}",
                }
            return {
                "status": "error",
                "data": None,
                "message": f"No info found for {symbol}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class YFinanceGetHistoricalPrices(YFinanceBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="yfinance_get_historical_prices",
            description="Get historical stock prices.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "period": {"type": "string", "default": "1mo"},
                    "interval": {"type": "string", "default": "1d"},
                },
                "required": ["symbol"],
            },
            function=self._get_history,
        )

    async def _get_history(
        self, symbol: str, period: str = "1mo", interval: str = "1d"
    ) -> Dict[str, Any]:
        try:
            stock = yf.Ticker(symbol, session=self.session)
            hist = stock.history(period=period, interval=interval)
            data = hist.reset_index().to_dict(orient="records")
            # Convert timestamps to str
            for d in data:
                if "Date" in d:
                    d["Date"] = str(d["Date"])

            return {
                "status": "success",
                "data": data,
                "message": f"Historical data for {symbol}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}
