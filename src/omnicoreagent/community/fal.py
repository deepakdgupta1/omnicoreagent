from os import getenv
from typing import Any, Dict, Optional

from omnicoreagent.core.tools.local_tools_registry import Tool
from omnicoreagent.core.utils import logger

try:
    import fal_client
except ImportError:
    fal_client = None


class FalBase:
    def __init__(self, api_key: Optional[str] = None):
        if fal_client is None:
            raise ImportError(
                "Could not import `fal-client` python package. "
                "Please install it using `pip install fal-client`."
            )
        self.api_key = api_key or getenv("FAL_API_KEY")
        self.seen_logs: set[str] = set()

    def on_queue_update(self, update):
        if isinstance(update, fal_client.InProgress) and update.logs:
            for log in update.logs:
                message = log["message"]
                if message not in self.seen_logs:
                    logger.info(message)
                    self.seen_logs.add(message)


class FalGenerateMedia(FalBase):
    def __init__(
        self, api_key: Optional[str] = None, model: str = "fal-ai/hunyuan-video"
    ):
        super().__init__(api_key)
        self.model = model

    def get_tool(self) -> Tool:
        return Tool(
            name="fal_generate_media",
            description="Generate media (image/video) using Fal AI.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                },
                "required": ["prompt"],
            },
            function=self._generate,
        )

    async def _generate(self, prompt: str) -> Dict[str, Any]:
        if not self.api_key:
            return {"status": "error", "data": None, "message": "FAL_API_KEY not set"}

        try:
            result = fal_client.subscribe(
                self.model,
                arguments={"prompt": prompt},
                with_logs=True,
                on_queue_update=self.on_queue_update,
            )

            media_data = {}
            if "image" in result:
                media_data["type"] = "image"
                media_data["url"] = result.get("image", {}).get("url", "")
            elif "video" in result:
                media_data["type"] = "video"
                media_data["url"] = result.get("video", {}).get("url", "")

            return {
                "status": "success",
                "data": media_data,
                "message": f"Media generated: {media_data.get('url')}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class FalImageToImage(FalBase):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "fal-ai/flux/dev/image-to-image",
    ):
        super().__init__(api_key)
        self.model = model

    def get_tool(self) -> Tool:
        return Tool(
            name="fal_image_to_image",
            description="Transform an image based on a prompt using Fal AI.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "image_url": {"type": "string"},
                },
                "required": ["prompt", "image_url"],
            },
            function=self._generate,
        )

    async def _generate(self, prompt: str, image_url: str) -> Dict[str, Any]:
        if not self.api_key:
            return {"status": "error", "data": None, "message": "FAL_API_KEY not set"}

        try:
            result = fal_client.subscribe(
                self.model,
                arguments={"image_url": image_url, "prompt": prompt},
                with_logs=True,
                on_queue_update=self.on_queue_update,
            )

            url = result.get("images", [{}])[0].get("url", "")
            return {
                "status": "success",
                "data": {"url": url},
                "message": f"Image generated: {url}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}
