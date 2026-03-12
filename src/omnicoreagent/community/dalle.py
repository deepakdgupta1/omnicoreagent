from os import getenv
from typing import Any, Dict, Optional
from omnicoreagent.core.tools.local_tools_registry import Tool

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class DalleBase:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or getenv("OPENAI_API_KEY")


class DalleCreateImage(DalleBase):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "dall-e-3",
        size: str = "1024x1024",
        quality: str = "standard",
        n: int = 1,
    ):
        if OpenAI is None:
            raise ImportError(
                "Could not import `openai` python package. "
                "Please install it using `pip install openai`."
            )
        super().__init__(api_key)
        self.model = model
        self.size = size
        self.quality = quality
        self.n = n

    def get_tool(self) -> Tool:
        return Tool(
            name="dalle_create_image",
            description="Generate an image for a prompt using DALL-E.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                },
                "required": ["prompt"],
            },
            function=self._create_image,
        )

    async def _create_image(self, prompt: str) -> Dict[str, Any]:
        if not self.api_key:
            return {
                "status": "error",
                "data": None,
                "message": "OPENAI_API_KEY not set",
            }

        try:
            client = OpenAI(api_key=self.api_key)
            response = client.images.generate(
                prompt=prompt,
                model=self.model,
                n=self.n,
                quality=self.quality,
                size=self.size,  # type: ignore
            )

            images = []
            if response.data:
                for img in response.data:
                    if img.url:
                        images.append(
                            {"url": img.url, "revised_prompt": img.revised_prompt}
                        )

            return {
                "status": "success",
                "data": images,
                "message": f"Generated {len(images)} images",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}
