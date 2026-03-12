from typing import Any, Dict
from omnicoreagent.core.tools.local_tools_registry import Tool

try:
    import pandas as pd
except ImportError:
    pd = None


class PandasCreateDataframe:
    def __init__(self):
        if pd is None:
            raise ImportError(
                "`pandas` not installed. Please install using `pip install pandas`"
            )
        self.dataframes: Dict[str, Any] = {}

    def get_tool(self) -> Tool:
        return Tool(
            name="pandas_create_dataframe",
            description="Create a Pandas DataFrame from a file (CSV, JSON) or data.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataframe_name": {"type": "string"},
                    "format": {"type": "string", "enum": ["csv", "json", "dict"]},
                    "data": {
                        "type": "string",
                        "description": "File path or JSON string data.",
                    },
                },
                "required": ["dataframe_name", "format", "data"],
            },
            function=self._create,
        )

    async def _create(
        self, dataframe_name: str, format: str, data: str
    ) -> Dict[str, Any]:
        try:
            if format == "csv":
                df = pd.read_csv(data)
            elif format == "json":
                # Try reading as file first, then as string
                if data.endswith(".json"):
                    df = pd.read_json(data)
                else:
                    df = pd.read_json(data)
            elif format == "dict":
                import ast

                df = pd.DataFrame(ast.literal_eval(data))
            else:
                return {
                    "status": "error",
                    "data": None,
                    "message": f"Unsupported format: {format}",
                }

            # In a real agentic context, we might need a way to persist this state across calls.
            # For this refactor, we are mostly standardizing the interface.
            # The legacy tool stored it in `self.dataframes`.
            # Note: The tool instance might be recreated, so state persistence is tricky without a shared singleton or external storage.
            # For now, we will return the head of the dataframe as preview.

            preview = df.head().to_json()
            return {
                "status": "success",
                "data": {"preview": preview, "shape": df.shape},
                "message": f"DataFrame {dataframe_name} created with shape {df.shape}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


# Note: The original Pandas tool relied on keeping state in `self.dataframes`.
# If the agent lifecycle creates new tool instances per turn, this state is lost.
# However, the legacy `Toolkit` might have been persistent.
# We should probably acknowledge this limitation or use a global registry for dataframes if needed.
# For strict refactoring of structure, I'll stick to this.
