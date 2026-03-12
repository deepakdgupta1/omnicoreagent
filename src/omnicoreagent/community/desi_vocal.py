from os import getenv
from typing import Any, Dict, Optional

import requests

from omnicoreagent.core.tools.local_tools_registry import Tool
from omnicoreagent.core.utils import logger


class DesiVocalTools:
    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: Optional[str] = "f27d74e5-ea71-4697-be3e-f04bbd80c1a8",
    ):
        self.api_key = api_key or getenv("DESI_VOCAL_API_KEY")
        if not self.api_key:
            logger.error(
                "DESI_VOCAL_API_KEY not set. Please set the DESI_VOCAL_API_KEY environment variable."
            )
        self.voice_id = voice_id

    def get_tool(self) -> Tool:
        return Tool(
            name="desi_vocal_get_voices",
            description="Get all available voices from DesiVocal.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
            function=self._get_voices,
        )

    async def _get_voices(self) -> Dict[str, Any]:
        try:
            url = "https://prod-api2.desivocal.com/dv/api/v0/tts_api/voices"
            response = requests.get(url)
            response.raise_for_status()
            voices_data = response.json()

            results = []
            for voice_id, voice_info in voices_data.items():
                results.append(
                    {
                        "id": voice_id,
                        "name": voice_info["name"],
                        "gender": voice_info["audio_gender"],
                        "type": voice_info["voice_type"],
                        "language": ", ".join(voice_info["languages"]),
                        "preview_url": next(iter(voice_info["preview_path"].values()))
                        if voice_info["preview_path"]
                        else None,
                    }
                )
            return {
                "status": "success",
                "data": results,
                "message": f"Found {len(results)} voices",
            }
        except Exception as e:
            logger.error(f"Failed to get voices: {e}")
            return {"status": "error", "data": None, "message": str(e)}


class DesiVocalTTS(DesiVocalTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="desi_vocal_text_to_speech",
            description="Generate audio from text using DesiVocal.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Text to generate audio from",
                    },
                    "voice_id": {"type": "string", "description": "Voice ID to use"},
                },
                "required": ["prompt"],
            },
            function=self._text_to_speech,
        )

    async def _text_to_speech(
        self, prompt: str, voice_id: Optional[str] = None
    ) -> Dict[str, Any]:
        if not self.api_key:
            return {"status": "error", "data": None, "message": "API key not set"}
        try:
            url = "https://prod-api2.desivocal.com/dv/api/v0/tts_api/generate"
            payload = {"text": prompt, "voice_id": voice_id or self.voice_id}
            headers = {"X_API_KEY": self.api_key, "Content-Type": "application/json"}
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            response_json = response.json()
            audio_url = response_json["s3_path"]
            return {
                "status": "success",
                "data": {"audio_url": audio_url},
                "message": "Audio generated",
            }
        except Exception as e:
            logger.error(f"Failed to generate audio: {e}")
            return {"status": "error", "data": None, "message": str(e)}
