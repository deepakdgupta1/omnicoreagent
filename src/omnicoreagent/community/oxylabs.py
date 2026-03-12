from os import getenv
from typing import Any, Dict, List, Optional

from omnicoreagent.core.tools.local_tools_registry import Tool
from omnicoreagent.core.utils import log_error, log_info

try:
    from oxylabs import RealtimeClient
    from oxylabs.sources.response import Response
    from oxylabs.utils.types import render
except ImportError:
    RealtimeClient = None
    Response = None
    render = None


class OxylabsTools:
    def __init__(self, username: Optional[str] = None, password: Optional[str] = None):
        if RealtimeClient is None:
            raise ImportError(
                "Oxylabs SDK not found. Please install it with: pip install oxylabs"
            )
        self.username = username or getenv("OXYLABS_USERNAME")
        self.password = password or getenv("OXYLABS_PASSWORD")
        if not self.username or not self.password:
            raise ValueError("OXYLABS_USERNAME and OXYLABS_PASSWORD must be set.")
        self.client = RealtimeClient(self.username, self.password)

    def get_tool(self) -> Tool:
        return Tool(
            name="oxylabs_search_google",
            description="Search Google using Oxylabs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "domain_code": {"type": "string", "default": "com"},
                },
                "required": ["query"],
            },
            function=self._search_google,
        )

    def _extract_organic(self, response: Response) -> List[Dict[str, Any]]:
        results = []
        if response.results and len(response.results) > 0:
            result = response.results[0]
            if hasattr(result, "content_parsed") and result.content_parsed:
                content = result.content_parsed
                if hasattr(content, "results") and content.results:
                    raw = content.results.raw if hasattr(content.results, "raw") else {}
                    for item in raw.get("organic", []):
                        results.append(
                            {
                                "title": item.get("title", "").strip(),
                                "url": item.get("url", "").strip(),
                                "description": item.get("desc", "").strip(),
                                "position": item.get("pos", 0),
                            }
                        )
            if (
                not results
                and hasattr(result, "content")
                and isinstance(result.content, dict)
            ):
                for item in result.content.get("results", {}).get("organic", []):
                    results.append(
                        {
                            "title": item.get("title", "").strip(),
                            "url": item.get("url", "").strip(),
                            "description": item.get("desc", "").strip(),
                            "position": item.get("pos", 0),
                        }
                    )
        return results

    async def _search_google(
        self, query: str, domain_code: str = "com"
    ) -> Dict[str, Any]:
        try:
            if not query or not query.strip():
                return {
                    "status": "error",
                    "data": None,
                    "message": "Query cannot be empty",
                }
            response: Response = self.client.google.scrape_search(
                query=query.strip(), domain=domain_code, parse=True
            )
            results = self._extract_organic(response)
            log_info(f"Google search completed. Found {len(results)} results")
            return {
                "status": "success",
                "data": {"query": query, "results": results},
                "message": f"Found {len(results)} results",
            }
        except Exception as e:
            log_error(f"Google search failed: {e}")
            return {"status": "error", "data": None, "message": str(e)}


class OxylabsGetAmazonProduct(OxylabsTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="oxylabs_get_amazon_product",
            description="Get Amazon product details by ASIN using Oxylabs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "asin": {"type": "string"},
                    "domain_code": {"type": "string", "default": "com"},
                },
                "required": ["asin"],
            },
            function=self._get_amazon_product,
        )

    async def _get_amazon_product(
        self, asin: str, domain_code: str = "com"
    ) -> Dict[str, Any]:
        try:
            asin = asin.strip().upper()
            if len(asin) != 10 or not asin.isalnum():
                return {
                    "status": "error",
                    "data": None,
                    "message": f"Invalid ASIN: {asin}",
                }
            response: Response = self.client.amazon.scrape_product(
                query=asin, domain=domain_code, parse=True
            )
            product = {"found": False, "asin": asin}
            if response.results and len(response.results) > 0:
                result = response.results[0]
                content = getattr(result, "content", None) or {}
                if isinstance(content, dict):
                    product.update(
                        {
                            "found": True,
                            "title": content.get("title", "").strip(),
                            "price": content.get("price", 0),
                            "currency": content.get("currency", ""),
                            "rating": content.get("rating", 0),
                            "url": content.get("url", ""),
                        }
                    )
            return {
                "status": "success",
                "data": product,
                "message": "Product lookup complete",
            }
        except Exception as e:
            log_error(f"Amazon product lookup failed: {e}")
            return {"status": "error", "data": None, "message": str(e)}


class OxylabsSearchAmazon(OxylabsTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="oxylabs_search_amazon",
            description="Search Amazon products using Oxylabs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "domain_code": {"type": "string", "default": "com"},
                },
                "required": ["query"],
            },
            function=self._search_amazon,
        )

    async def _search_amazon(
        self, query: str, domain_code: str = "com"
    ) -> Dict[str, Any]:
        try:
            if not query or not query.strip():
                return {
                    "status": "error",
                    "data": None,
                    "message": "Query cannot be empty",
                }
            response: Response = self.client.amazon.scrape_search(
                query=query.strip(), domain=domain_code, parse=True
            )
            products = []
            if response.results and len(response.results) > 0:
                result = response.results[0]
                content = getattr(result, "content", None) or {}
                if isinstance(content, dict):
                    for item in content.get("results", {}).get("organic", []):
                        products.append(
                            {
                                "title": item.get("title", "").strip(),
                                "asin": item.get("asin", "").strip(),
                                "price": item.get("price", 0),
                                "url": item.get("url", "").strip(),
                            }
                        )
            return {
                "status": "success",
                "data": {"query": query, "products": products},
                "message": f"Found {len(products)} products",
            }
        except Exception as e:
            log_error(f"Amazon search failed: {e}")
            return {"status": "error", "data": None, "message": str(e)}


class OxylabsScrapeWebsite(OxylabsTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="oxylabs_scrape_website",
            description="Scrape website content using Oxylabs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "render_javascript": {"type": "boolean", "default": False},
                },
                "required": ["url"],
            },
            function=self._scrape_website,
        )

    async def _scrape_website(
        self, url: str, render_javascript: bool = False
    ) -> Dict[str, Any]:
        try:
            url = url.strip()
            if not url.startswith(("http://", "https://")):
                return {
                    "status": "error",
                    "data": None,
                    "message": f"Invalid URL: {url}",
                }
            response: Response = self.client.universal.scrape_url(
                url=url, render=render.HTML if render_javascript else None, parse=True
            )
            content_info: Dict[str, Any] = {"url": url}
            if response.results and len(response.results) > 0:
                result = response.results[0]
                content = str(result.content) if result.content else ""
                content_info["content_length"] = len(content)
                content_info["content_preview"] = content[:1000]
            return {
                "status": "success",
                "data": content_info,
                "message": f"Scraped {url}",
            }
        except Exception as e:
            log_error(f"Website scraping failed: {e}")
            return {"status": "error", "data": None, "message": str(e)}
