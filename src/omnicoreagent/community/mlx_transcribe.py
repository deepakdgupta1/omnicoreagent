"""MLX Transcribe Tools - Audio Transcription using Apple's MLX Framework"""

from pathlib import Path
from typing import Any, Dict, Optional

from omnicoreagent.core.tools.local_tools_registry import Tool
from omnicoreagent.core.utils import log_info, logger

try:
    import mlx_whisper
except ImportError:
    mlx_whisper = None


class MLXTranscribeTools:
    def __init__(
        self,
        base_dir: Optional[Path] = None,
        path_or_hf_repo: str = "mlx-community/whisper-large-v3-turbo",
        **transcription_kwargs,
    ):
        self.base_dir = (base_dir or Path.cwd()).resolve()
        self.path_or_hf_repo = path_or_hf_repo
        self.base_dir = (base_dir or Path.cwd()).resolve()
        self.path_or_hf_repo = path_or_hf_repo
        self.transcription_kwargs = transcription_kwargs

        if mlx_whisper is None:
            raise ImportError(
                "Could not import `mlx_whisper` python package. "
                "Please install it with `pip install mlx-whisper`."
            )

    def get_tool(self) -> Tool:
        return Tool(
            name="mlx_transcribe",
            description="Transcribe an audio file using Apple's MLX Whisper model.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "Name of the audio file to transcribe",
                    },
                },
                "required": ["file_name"],
            },
            function=self._transcribe,
        )

    async def _transcribe(self, file_name: str) -> Dict[str, Any]:
        try:
            file_path = self.base_dir / file_name
            if not file_path.exists():
                return {
                    "status": "error",
                    "data": None,
                    "message": f"File not found: {file_path}",
                }

            log_info(f"Transcribing: {file_path}")
            kwargs = {"path_or_hf_repo": self.path_or_hf_repo}
            kwargs.update(
                {k: v for k, v in self.transcription_kwargs.items() if v is not None}
            )

            transcription = mlx_whisper.transcribe(str(file_path), **kwargs)
            text = transcription.get("text", "")
            return {
                "status": "success",
                "data": text,
                "message": "Transcription complete",
            }
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {"status": "error", "data": None, "message": str(e)}
