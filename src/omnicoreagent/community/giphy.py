import uuid
from os import getenv
from typing import Any, Optional

import httpx
from omnicoreagent.core.tools.local_tools_registry import Tool
from omnicoreagent.core.utils import Image, logger


class GiphySearch:
    def __init__(
        self,
        api_key: Optional[str] = None,
        limit: int = 1,
    ):
        self.api_key = api_key or getenv("GIPHY_API_KEY")
        self.limit = limit
        if not self.api_key:
            logger.error("No Giphy API key provided")

    def get_tool(self) -> Tool:
        return Tool(
            name="giphy_search",
            description="Find a GIPHY gif.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A text description of the required gif.",
                    },
                    "limit": {"type": "integer", "default": 1},
                },
                "required": ["query"],
            },
            function=self._search_gifs,
        )

    async def _search_gifs(self, query: str, limit: Optional[int] = None) -> Any:
        base_url = "https://api.giphy.com/v1/gifs/search"
        params = {
            "api_key": self.api_key,
            "q": query,
            "limit": limit or self.limit,
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(base_url, params=params)
                response.raise_for_status()
                data = response.json()

            gif_urls = []
            image_artifacts = []

            for gif in data.get("data", []):
                images = gif.get("images", {})
                original_image = images.get("original")
                if not original_image:
                    continue

                media_id = str(uuid.uuid4())
                gif_url = original_image["url"]
                alt_text = gif.get("title", "") or query
                gif_urls.append(gif_url)

                # Create ImageArtifact for the GIF
                image_artifact = Image(
                    id=media_id, url=gif_url, alt_text=alt_text, revised_prompt=query
                )
                image_artifacts.append(image_artifact)

            if image_artifacts:
                return {
                    "status": "success",
                    "data": {"gif_urls": gif_urls, "images": str(image_artifacts)},
                    "message": f"Found {len(gif_urls)} GIF(s)",
                }
            else:
                return {"status": "success", "data": [], "message": "No gifs found"}

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error occurred: {e.response.status_code} - {e.response.text}"
            )
            return {
                "status": "error",
                "data": None,
                "message": f"HTTP error occurred: {e.response.status_code}",
            }
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            return {"status": "error", "data": None, "message": str(e)}
