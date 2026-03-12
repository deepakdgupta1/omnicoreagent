from os import getenv
from typing import Any, Dict, Optional, Iterable, Iterator
from omnicoreagent.core.tools.local_tools_registry import Tool

try:
    import replicate
    from replicate.helpers import FileOutput
except ImportError:
    replicate = None


class ReplicateBase:
    def __init__(self, api_key: Optional[str] = None):
        if replicate is None:
            raise ImportError(
                "replicate not installed. Please install it using `pip install replicate`."
            )
        self.api_key = api_key or getenv("REPLICATE_API_KEY")


class ReplicateGenerateMedia(ReplicateBase):
    def __init__(self, api_key: Optional[str] = None, model: str = "minimax/video-01"):
        super().__init__(api_key)
        self.model = model

    def get_tool(self) -> Tool:
        return Tool(
            name="replicate_generate_media",
            description="Generate media (image/video) using Replicate.",
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
                "message": "REPLICATE_API_KEY not set",
            }

        try:
            # Replicate run can be blocking
            outputs = replicate.run(ref=self.model, input={"prompt": prompt})

            # Normalize outputs
            if isinstance(outputs, FileOutput):
                outputs = [outputs]
            elif isinstance(outputs, (Iterable, Iterator)) and not isinstance(
                outputs, str
            ):
                outputs = list(outputs)

            results = []
            for output in outputs:
                if hasattr(output, "url"):
                    results.append(output.url)
                else:
                    results.append(str(output))

            return {
                "status": "success",
                "data": results,
                "message": f"Generated {len(results)} outputs",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}
