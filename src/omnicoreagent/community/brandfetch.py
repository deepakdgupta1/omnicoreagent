from os import getenv
from typing import Any, Optional, Dict

try:
    import httpx
except ImportError:
    httpx = None

from omnicoreagent.core.tools.local_tools_registry import Tool


class BrandfetchTools:
    """
    Brandfetch API toolkit for retrieving brand data and searching brands.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        client_id: Optional[str] = None,
        base_url: str = "https://api.brandfetch.io/v2",
        timeout: Optional[float] = 20.0,
    ):
        self.api_key = api_key or getenv("BRANDFETCH_API_KEY")
        self.client_id = client_id or getenv("BRANDFETCH_CLIENT_ID")
        self.base_url = base_url
        self.timeout = httpx.Timeout(timeout)  # type: ignore
        self.search_url = f"{self.base_url}/search"
        self.search_url = f"{self.base_url}/search"
        self.brand_url = f"{self.base_url}/brands"

        if httpx is None:
            raise ImportError(
                "Could not import `httpx` python package. "
                "Please install it with `pip install httpx`."
            )

    def get_tool(self) -> Tool:
        # Default to search by identifier
        return Tool(
            name="brandfetch_search_by_identifier",
            description="Search for brand data by identifier (domain, brand id, isin, stock ticker).",
            inputSchema={
                "type": "object",
                "properties": {
                    "identifier": {
                        "type": "string",
                        "description": "Domain (nike.com), Brand ID (id_0dwKPKT), ISIN (US6541061031), Stock Ticker (NKE)",
                    },
                },
                "required": ["identifier"],
            },
            function=self._search_by_identifier,
        )

    async def _search_by_identifier(self, identifier: str) -> Dict[str, Any]:
        """
        Search for brand data by identifier.
        """
        if not self.api_key:
            return {"status": "error", "data": None, "message": "API key is required"}

        url = f"{self.brand_url}/{identifier}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, headers=headers)

                if response.status_code == 404:
                    return {
                        "status": "error",
                        "data": None,
                        "message": f"Brand not found for identifier: {identifier}",
                    }

                response.raise_for_status()
                return {
                    "status": "success",
                    "data": response.json(),
                    "message": "Brand data retrieved",
                }

        except Exception as e:
            return {
                "status": "error",
                "data": None,
                "message": f"Request failed: {str(e)}",
            }


class BrandfetchSearchByBrand:
    def __init__(self, client_id: Optional[str] = None, timeout: float = 20.0):
        self.client_id = client_id or getenv("BRANDFETCH_CLIENT_ID")
        self.base_url = "https://api.brandfetch.io/v2"
        self.search_url = f"{self.base_url}/search"
        self.timeout = httpx.Timeout(timeout)  # type: ignore

    def get_tool(self) -> Tool:
        return Tool(
            name="brandfetch_search_by_brand",
            description="Search for brands by name using the Brand Search API.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Brand name to search for",
                    },
                },
                "required": ["name"],
            },
            function=self._search_by_brand,
        )

    async def _search_by_brand(self, name: str) -> Dict[str, Any]:
        if not self.client_id:
            return {"status": "error", "data": None, "message": "Client ID is required"}

        url = f"{self.search_url}/{name}"
        params = {"c": self.client_id}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, params=params)

                if response.status_code == 404:
                    return {
                        "status": "error",
                        "data": None,
                        "message": f"No brands found for name: {name}",
                    }

                response.raise_for_status()
                return {
                    "status": "success",
                    "data": response.json(),
                    "message": "Brand search successful",
                }
        except Exception as e:
            return {
                "status": "error",
                "data": None,
                "message": f"Request failed: {str(e)}",
            }
