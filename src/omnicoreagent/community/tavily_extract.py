import os

try:
    import httpx
except ImportError:
    httpx = None
from typing import Any, Dict, Optional
from omnicoreagent.core.tools.local_tools_registry import Tool


class TavilyExtract:
    """Tavily Extract Tool integration."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Tavily Extract tool.

        Args:
            api_key: Tavily API key. If not provided, it will be read from TAVILY_API_KEY environment variable.
        """
        if httpx is None:
            raise ImportError(
                "`httpx` not installed. Please install using `pip install httpx`"
            )
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Tavily API key is required. Set TAVILY_API_KEY environment variable or pass it to constructor."
            )

    def get_tool(self) -> Tool:
        """Return the configured Tool object."""
        return Tool(
            name="tavily_extract",
            description="Extract content from URLs using Tavily. Get main content from web pages in clean format.",
            inputSchema={
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "string",
                        "description": "Comma-separated list of URLs to extract content from.",
                    },
                    "include_images": {
                        "type": "boolean",
                        "description": "Include images in extracted content.",
                        "default": False,
                    },
                },
                "required": ["urls"],
            },
            function=self._extract,
        )

    async def _extract(
        self,
        urls: str,
        include_images: bool = False,
    ) -> Dict[str, Any]:
        """Execute the extract request."""
        url = "https://api.tavily.com/extract"

        url_list = [u.strip() for u in urls.split(",") if u.strip()]

        payload = {
            "api_key": self.api_key,
            "urls": url_list,
            "include_images": include_images,
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    url, json=payload, timeout=60.0
                )  # Extraction might take longer
                response.raise_for_status()
                data = response.json()

                results = data.get("results", [])
                formatted_results = []
                for result in results:
                    # Handle both successful and failed extractions from Tavily side
                    if "failed_reason" in result and result["failed_reason"]:
                        formatted_results.append(
                            f"## {result.get('url')}\n**Extraction Failed**: {result.get('failed_reason')}\n"
                        )
                    else:
                        formatted_results.append(
                            f"## {result.get('url')}\n{result.get('raw_content')}\n"
                        )

                return {
                    "status": "success",
                    "data": data,
                    "message": "\n---\n".join(formatted_results)
                    if formatted_results
                    else "No content extracted.",
                }

            except httpx.HTTPStatusError as e:
                return {
                    "status": "error",
                    "data": None,
                    "message": f"HTTP error: {e.response.status_code} - {e.response.text}",
                }
            except Exception as e:
                return {
                    "status": "error",
                    "data": None,
                    "message": f"Extraction failed: {str(e)}",
                }
