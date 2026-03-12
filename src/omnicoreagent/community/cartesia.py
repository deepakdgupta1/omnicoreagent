from os import getenv
from typing import Any, Dict, Optional
import base64

from omnicoreagent.core.tools.local_tools_registry import Tool

try:
    import cartesia
except ImportError:
    cartesia = None


class CartesiaTools:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: str = "sonic-2",
        default_voice_id: str = "78ab82d5-25be-4f7d-82b3-7ad64e5b85b2",
    ):
        if cartesia is None:
            raise ImportError(
                "Could not import `cartesia` python package. "
                "Please install it using `pip install cartesia`."
            )
        self.api_key = api_key or getenv("CARTESIA_API_KEY")
        self.client = None
        if self.api_key:
            try:
                self.client = cartesia.Cartesia(api_key=self.api_key)
            except Exception:
                self.client = None

        self.model_id = model_id
        self.default_voice_id = default_voice_id

    def get_tool(self) -> Tool:
        return Tool(
            name="cartesia_list_voices",
            description="List available voices from Cartesia.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
            function=self._list_voices,
        )

    async def _list_voices(self) -> Dict[str, Any]:
        if not self.client:
            return {
                "status": "error",
                "data": None,
                "message": "Client not initialized",
            }
        try:
            voices = self.client.voices.list()
            voice_objects = voices.items if voices else []

            results = []
            for v in voice_objects:
                if hasattr(v, "id"):
                    results.append(
                        {
                            "id": v.id,
                            "name": getattr(v, "name", ""),
                            "language": getattr(v, "language", ""),
                        }
                    )
            return {
                "status": "success",
                "data": results,
                "message": f"Found {len(results)} voices",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class CartesiaTTS(CartesiaTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="cartesia_text_to_speech",
            description="Convert text to speech.",
            inputSchema={
                "type": "object",
                "properties": {
                    "transcript": {"type": "string"},
                    "voice_id": {"type": "string"},
                },
                "required": ["transcript"],
            },
            function=self._text_to_speech,
        )

    async def _text_to_speech(
        self, transcript: str, voice_id: Optional[str] = None
    ) -> Dict[str, Any]:
        if not self.client:
            return {
                "status": "error",
                "data": None,
                "message": "Client not initialized",
            }

        try:
            effective_voice_id = voice_id or self.default_voice_id

            output_format = {
                "container": "mp3",
                "sample_rate": 44100,
                "bit_rate": 128000,
                "encoding": "mp3",
            }

            audio_iterator = self.client.tts.bytes(
                model_id=self.model_id,
                transcript=transcript,
                voice={"mode": "id", "id": effective_voice_id},
                output_format=output_format,
            )

            audio_data = b"".join(chunk for chunk in audio_iterator)
            b64_audio = base64.b64encode(audio_data).decode("utf-8")

            return {
                "status": "success",
                "data": {"audio_base64_length": len(b64_audio)},
                "message": "Speech generated successfully",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}
