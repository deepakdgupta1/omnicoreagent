from os import getenv
from typing import Any, Dict, Optional
import base64

from omnicoreagent.core.tools.local_tools_registry import Tool

try:
    from openai import OpenAI as OpenAIClient
except ImportError:
    OpenAIClient = None


class OpenAIBase:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or getenv("OPENAI_API_KEY")


class OpenAITranscribeAudio(OpenAIBase):
    def __init__(self, api_key: Optional[str] = None, model: str = "whisper-1"):
        super().__init__(api_key)
        if OpenAIClient is None:
            raise ImportError(
                "`openai` not installed. Please install using `pip install openai`"
            )
        self.model = model

    def get_tool(self) -> Tool:
        return Tool(
            name="openai_transcribe_audio",
            description="Transcribe audio file using OpenAI's Whisper API.",
            inputSchema={
                "type": "object",
                "properties": {
                    "audio_path": {
                        "type": "string",
                        "description": "Path to the audio file",
                    },
                },
                "required": ["audio_path"],
            },
            function=self._transcribe,
        )

    async def _transcribe(self, audio_path: str) -> Dict[str, Any]:
        if not self.api_key:
            return {
                "status": "error",
                "data": None,
                "message": "OPENAI_API_KEY not set",
            }

        try:
            with open(audio_path, "rb") as audio_file:
                transcript = OpenAIClient(
                    api_key=self.api_key
                ).audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    response_format="text",
                )
            return {
                "status": "success",
                "data": transcript,
                "message": "Transcription successful",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class OpenAIGenerateImage(OpenAIBase):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "dall-e-3",
        size: str = "1024x1024",
        quality: str = "standard",
        style: str = "vivid",
    ):
        super().__init__(api_key)
        self.model = model
        self.size = size
        self.quality = quality
        self.style = style

    def get_tool(self) -> Tool:
        return Tool(
            name="openai_generate_image",
            description="Generate images based on a text prompt using OpenAI DALL-E 3.",
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
            return {
                "status": "error",
                "data": None,
                "message": "OPENAI_API_KEY not set",
            }

        try:
            extra_params = {
                "size": self.size,
                "quality": self.quality,
                "style": self.style,
            }
            # Remove None values
            extra_params = {k: v for k, v in extra_params.items() if v is not None}

            client = OpenAIClient(api_key=self.api_key)
            response = client.images.generate(
                model=self.model,
                prompt=prompt,
                response_format="b64_json",
                **extra_params,
            )

            if response.data:
                data = response.data[0]
                if data.b64_json:
                    # Return base64 truncated or full? standardized response typically returns data.
                    # We can return the base64 string.
                    return {
                        "status": "success",
                        "data": {
                            "b64_json": data.b64_json[:100] + "...",
                            "revised_prompt": data.revised_prompt,
                        },
                        "message": "Image generated successfully (base64 data available)",
                    }
                if data.url:
                    return {
                        "status": "success",
                        "data": {
                            "url": data.url,
                            "revised_prompt": data.revised_prompt,
                        },
                        "message": "Image generated successfully",
                    }

            return {
                "status": "error",
                "data": None,
                "message": "No data returned from OpenAI",
            }

        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class OpenAIGenerateSpeech(OpenAIBase):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "tts-1",
        voice: str = "alloy",
        format: str = "mp3",
    ):
        super().__init__(api_key)
        if OpenAIClient is None:
            raise ImportError(
                "`openai` not installed. Please install using `pip install openai`"
            )
        self.model = model
        self.voice = voice
        self.format = format

    def get_tool(self) -> Tool:
        return Tool(
            name="openai_generate_speech",
            description="Generate speech from text using OpenAI TTS.",
            inputSchema={
                "type": "object",
                "properties": {
                    "text_input": {"type": "string"},
                },
                "required": ["text_input"],
            },
            function=self._generate,
        )

    async def _generate(self, text_input: str) -> Dict[str, Any]:
        if not self.api_key:
            return {
                "status": "error",
                "data": None,
                "message": "OPENAI_API_KEY not set",
            }
        if OpenAIClient is None:
            return {
                "status": "error",
                "data": None,
                "message": "openai not installed. Please install it using `pip install openai`.",
            }

        try:
            response = OpenAIClient(api_key=self.api_key).audio.speech.create(
                model=self.model,
                voice=self.voice,
                input=text_input,
                response_format=self.format,  # type: ignore
            )

            # We cannot easily return bytes in JSON.
            # We would typically save to file or return base64.
            # For now, let's return a success message and maybe length.
            # Or we could return base64 encoded audio.

            audio_content = response.content
            b64_audio = base64.b64encode(audio_content).decode("utf-8")

            return {
                "status": "success",
                "data": {"audio_base64_length": len(b64_audio)},
                "message": "Speech generated successfully",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}
