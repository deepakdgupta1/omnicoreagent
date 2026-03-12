from __future__ import annotations

import os
from typing import Any, Dict, Optional
from uuid import uuid4

from omnicoreagent.core.tools.local_tools_registry import Tool
from omnicoreagent.core.utils import log_debug, logger

try:
    from google import genai
    from google.genai import types
    from PIL import Image as PILImage
except ImportError:
    genai = None
    types = None
    PILImage = None

ALLOWED_MODELS = ["gemini-2.5-flash-image"]
ALLOWED_RATIOS = [
    "1:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "4:5",
    "5:4",
    "9:16",
    "16:9",
    "21:9",
]


class NanoBananaImageGen:
    def __init__(
        self,
        model: str = "gemini-2.5-flash-image",
        aspect_ratio: str = "1:1",
        api_key: Optional[str] = None,
    ):
        if genai is None or PILImage is None:
            missing = []
            try:
                from google.genai import types as _t  # noqa
            except ImportError:
                missing.append("google-genai")
            try:
                from PIL import Image as _i  # noqa
            except ImportError:
                missing.append("Pillow")
            raise ImportError(
                f"Missing required package(s): {', '.join(missing)}. Install using: pip install {' '.join(missing)}"
            )
        self.model = model
        self.aspect_ratio = aspect_ratio
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")

        if model not in ALLOWED_MODELS:
            raise ValueError(
                f"Invalid model '{model}'. Supported: {', '.join(ALLOWED_MODELS)}"
            )
        if self.aspect_ratio not in ALLOWED_RATIOS:
            raise ValueError(
                f"Invalid aspect_ratio '{self.aspect_ratio}'. Supported: {', '.join(ALLOWED_RATIOS)}"
            )
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not set.")

    def get_tool(self) -> Tool:
        return Tool(
            name="nano_banana_create_image",
            description="Generate an image from a text prompt using Google GenAI.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Text description of the desired image",
                    },
                },
                "required": ["prompt"],
            },
            function=self._create_image,
        )

    async def _create_image(self, prompt: str) -> Dict[str, Any]:
        try:
            client = genai.Client(api_key=self.api_key)
            log_debug(f"NanoBanana generating image with prompt: {prompt}")

            cfg = types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(aspect_ratio=self.aspect_ratio),
            )

            response = client.models.generate_content(
                model=self.model,
                contents=[prompt],
                config=cfg,
            )

            if not hasattr(response, "candidates") or not response.candidates:
                return {
                    "status": "error",
                    "data": None,
                    "message": "No images generated",
                }

            images_generated = []
            for candidate in response.candidates:
                if (
                    not hasattr(candidate, "content")
                    or not candidate.content
                    or not candidate.content.parts
                ):
                    continue
                for part in candidate.content.parts:
                    if (
                        hasattr(part, "inline_data")
                        and part.inline_data
                        and part.inline_data.data
                    ):
                        image_id = str(uuid4())
                        images_generated.append(
                            {
                                "image_id": image_id,
                                "mime_type": getattr(
                                    part.inline_data, "mime_type", "image/png"
                                ),
                            }
                        )

            if images_generated:
                return {
                    "status": "success",
                    "data": images_generated,
                    "message": f"Generated {len(images_generated)} image(s)",
                }
            else:
                return {
                    "status": "error",
                    "data": None,
                    "message": "No images were generated",
                }
        except Exception as e:
            logger.error(f"NanoBanana image generation failed: {e}")
            return {"status": "error", "data": None, "message": str(e)}
