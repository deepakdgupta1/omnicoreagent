import csv
from pathlib import Path
from typing import Any, Dict
from omnicoreagent.core.tools.local_tools_registry import Tool


class CsvRead:
    def get_tool(self) -> Tool:
        return Tool(
            name="csv_read",
            description="Read a CSV file.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["file_path"],
            },
            function=self._read,
        )

    async def _read(self, file_path: str, limit: int = 10) -> Dict[str, Any]:
        try:
            path = Path(file_path)
            if not path.exists():
                return {"status": "error", "data": None, "message": "File not found"}

            data = []
            with open(path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if i >= limit:
                        break
                    data.append(row)

            return {
                "status": "success",
                "data": data,
                "message": f"Read {len(data)} rows from {file_path}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class CsvGetColumns:
    def get_tool(self) -> Tool:
        return Tool(
            name="csv_get_columns",
            description="Get columns of a CSV file.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                },
                "required": ["file_path"],
            },
            function=self._get_columns,
        )

    async def _get_columns(self, file_path: str) -> Dict[str, Any]:
        try:
            path = Path(file_path)
            if not path.exists():
                return {"status": "error", "data": None, "message": "File not found"}

            with open(path, "r", newline="") as f:
                reader = csv.DictReader(f)
                columns = reader.fieldnames

            return {
                "status": "success",
                "data": columns,
                "message": f"Columns: {columns}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}
