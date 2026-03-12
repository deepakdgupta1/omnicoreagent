import time
import uuid
from os import getenv
from typing import Any, Dict, Literal, Optional, TypedDict

from omnicoreagent.core.tools.local_tools_registry import Tool
from omnicoreagent.core.utils import log_info, logger

try:
    from lumaai import LumaAI  # type: ignore
except ImportError:
    LumaAI = None  # type: ignore


class KeyframeImage(TypedDict):
    type: Literal["image"]
    url: str


Keyframes = Dict[str, KeyframeImage]


class LumaBase:
    def __init__(
        self,
        api_key: Optional[str] = None,
        wait_for_completion: bool = True,
        poll_interval: int = 3,
        max_wait_time: int = 300,
    ):
        if LumaAI is None:
            raise ImportError(
                "`lumaai` not installed. Please install using `pip install lumaai`"
            )
        self.wait_for_completion = wait_for_completion
        self.poll_interval = poll_interval
        self.max_wait_time = max_wait_time
        self.api_key = api_key or getenv("LUMAAI_API_KEY")

        if not self.api_key:
            logger.error(
                "LUMAAI_API_KEY not set. Please set the LUMAAI_API_KEY environment variable."
            )

        self.client = LumaAI(auth_token=self.api_key)

    def _wait_for_generation(self, generation, video_id: str) -> Dict[str, Any]:
        if not self.wait_for_completion:
            return {
                "status": "success",
                "data": {"generation_id": generation.id},
                "message": "Generation started (async)",
            }

        seconds_waited = 0
        while seconds_waited < self.max_wait_time:
            if not generation or not generation.id:
                return {
                    "status": "error",
                    "data": None,
                    "message": "Failed to get generation ID",
                }

            generation = self.client.generations.get(generation.id)

            if generation.state == "completed" and generation.assets:
                video_url = generation.assets.video
                if video_url:
                    return {
                        "status": "success",
                        "data": {"video_url": video_url, "video_id": video_id},
                        "message": "Video generated successfully",
                    }
            elif generation.state == "failed":
                return {
                    "status": "error",
                    "data": None,
                    "message": f"Generation failed: {generation.failure_reason}",
                }

            log_info(f"Generation in progress... State: {generation.state}")
            time.sleep(self.poll_interval)
            seconds_waited += self.poll_interval

        return {
            "status": "error",
            "data": None,
            "message": f"Timed out after {self.max_wait_time} seconds",
        }


class LumaImageToVideo(LumaBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="luma_image_to_video",
            description="Generate a video from one or two images with a prompt.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Text description of the desired video",
                    },
                    "start_image_url": {
                        "type": "string",
                        "description": "URL of the starting image",
                    },
                    "end_image_url": {
                        "type": "string",
                        "description": "Optional URL of the ending image",
                    },
                    "loop": {
                        "type": "boolean",
                        "description": "Whether the video should loop",
                        "default": False,
                    },
                    "aspect_ratio": {
                        "type": "string",
                        "enum": ["1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21"],
                        "default": "16:9",
                    },
                },
                "required": ["prompt", "start_image_url"],
            },
            function=self._image_to_video,
        )

    async def _image_to_video(
        self,
        prompt: str,
        start_image_url: str,
        end_image_url: Optional[str] = None,
        loop: bool = False,
        aspect_ratio: Literal[
            "1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21"
        ] = "16:9",
    ) -> Dict[str, Any]:
        try:
            keyframes: Dict[str, Dict[str, str]] = {
                "frame0": {"type": "image", "url": start_image_url}
            }
            if end_image_url:
                keyframes["frame1"] = {"type": "image", "url": end_image_url}

            generation = self.client.generations.create(
                prompt=prompt,
                loop=loop,
                aspect_ratio=aspect_ratio,
                keyframes=keyframes,  # type: ignore
            )
            video_id = str(uuid.uuid4())
            return self._wait_for_generation(generation, video_id)
        except Exception as e:
            logger.error(f"Failed to generate video: {e}")
            return {"status": "error", "data": None, "message": str(e)}


class LumaGenerateVideo(LumaBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="luma_generate_video",
            description="Generate a video given a prompt.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Text description of the desired video",
                    },
                    "loop": {
                        "type": "boolean",
                        "description": "Whether the video should loop",
                        "default": False,
                    },
                    "aspect_ratio": {
                        "type": "string",
                        "enum": ["1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21"],
                        "default": "16:9",
                    },
                },
                "required": ["prompt"],
            },
            function=self._generate_video,
        )

    async def _generate_video(
        self,
        prompt: str,
        loop: bool = False,
        aspect_ratio: Literal[
            "1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21"
        ] = "16:9",
    ) -> Dict[str, Any]:
        try:
            generation = self.client.generations.create(
                prompt=prompt,
                loop=loop,
                aspect_ratio=aspect_ratio,
            )
            video_id = str(uuid.uuid4())
            return self._wait_for_generation(generation, video_id)
        except Exception as e:
            logger.error(f"Failed to generate video: {e}")
            return {"status": "error", "data": None, "message": str(e)}
