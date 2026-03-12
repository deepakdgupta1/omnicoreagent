from os import getenv
from typing import Any, Dict, Optional, Iterator
import base64

from omnicoreagent.core.tools.local_tools_registry import Tool

try:
    from elevenlabs import ElevenLabs
except ImportError:
    ElevenLabs = None


class ElevenLabsBase:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or getenv("ELEVEN_LABS_API_KEY")
        self.client = None
        self.client = None
        if ElevenLabs is None:
            raise ImportError(
                "Could not import `elevenlabs` python package. "
                "Please install it with `pip install elevenlabs`."
            )
        elif self.api_key:
            try:
                self.client = ElevenLabs(api_key=self.api_key)
            except Exception:
                self.client = None

    def _process_audio_to_base64(self, audio_generator: Iterator[bytes]) -> str:
        audio_bytes = b"".join(audio_generator)
        return base64.b64encode(audio_bytes).decode("utf-8")


class ElevenLabsGetVoices(ElevenLabsBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="elevenlabs_get_voices",
            description="Get all available ElevenLabs voices.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
            function=self._get_voices,
        )

    async def _get_voices(self) -> Dict[str, Any]:
        if not self.client:
            return {
                "status": "error",
                "data": None,
                "message": "ELEVEN_LABS_API_KEY not set",
            }

        try:
            voices_response = self.client.voices.get_all()
            voices_list = []
            for voice in voices_response.voices:
                voices_list.append(
                    {
                        "id": voice.voice_id,
                        "name": voice.name,
                        "description": voice.description,
                    }
                )
            return {
                "status": "success",
                "data": voices_list,
                "message": f"Found {len(voices_list)} voices",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class ElevenLabsGenerateSoundEffect(ElevenLabsBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="elevenlabs_generate_sound_effect",
            description="Generate a sound effect from text.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "duration_seconds": {"type": "number"},
                },
                "required": ["prompt"],
            },
            function=self._generate,
        )

    async def _generate(
        self, prompt: str, duration_seconds: Optional[float] = None
    ) -> Dict[str, Any]:
        if not self.client:
            return {
                "status": "error",
                "data": None,
                "message": "ELEVEN_LABS_API_KEY not set",
            }

        try:
            audio_generator = self.client.text_to_sound_effects.convert(
                text=prompt, duration_seconds=duration_seconds
            )
            b64_audio = self._process_audio_to_base64(audio_generator)
            return {
                "status": "success",
                "data": {"audio_base64_length": len(b64_audio)},
                "message": "Sound effect generated",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class ElevenLabsTextToSpeech(ElevenLabsBase):
    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: str = "JBFqnCBsd6RMkjVDRZzb",
        model_id: str = "eleven_multilingual_v2",
    ):
        super().__init__(api_key)
        self.voice_id = voice_id
        self.model_id = model_id

    def get_tool(self) -> Tool:
        return Tool(
            name="elevenlabs_text_to_speech",
            description="Convert text to speech.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                },
                "required": ["prompt"],
            },
            function=self._convert,
        )

    async def _convert(self, prompt: str) -> Dict[str, Any]:
        if not self.client:
            return {
                "status": "error",
                "data": None,
                "message": "ELEVEN_LABS_API_KEY not set",
            }

        try:
            audio_generator = self.client.text_to_speech.convert(
                text=prompt,
                voice_id=self.voice_id,
                model_id=self.model_id,
            )
            b64_audio = self._process_audio_to_base64(audio_generator)
            return {
                "status": "success",
                "data": {"audio_base64_length": len(b64_audio)},
                "message": "Speech generated",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}
