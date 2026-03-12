import csv
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from omnicoreagent.core.tools.local_tools_registry import Tool

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


class FileGenerationTools:
    def __init__(self, output_directory: Optional[str] = None):
        self.output_directory = Path(output_directory) if output_directory else None
        if self.output_directory:
            self.output_directory.mkdir(parents=True, exist_ok=True)

    def _save_file_to_disk(
        self, content: Union[str, bytes], filename: str
    ) -> Optional[str]:
        if not self.output_directory:
            return None
        file_path = self.output_directory / filename
        if isinstance(content, str):
            file_path.write_text(content, encoding="utf-8")
        else:
            file_path.write_bytes(content)
        return str(file_path)

    def get_tool(self) -> Tool:
        return Tool(
            name="generate_json_file",
            description="Generate a JSON file from data.",
            inputSchema={
                "type": "object",
                "properties": {
                    "data": {
                        "type": "string",
                        "description": "JSON data as string, or dict/list",
                    },
                    "filename": {"type": "string"},
                },
                "required": ["data"],
            },
            function=self._generate_json,
        )

    async def _generate_json(
        self, data: Union[str, Dict, List], filename: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            if isinstance(data, str):
                try:
                    json.loads(data)
                    json_content = data
                except json.JSONDecodeError:
                    json_content = json.dumps({"content": data}, indent=2)
            else:
                json_content = json.dumps(data, indent=2, ensure_ascii=False)

            if not filename:
                filename = f"generated_{str(uuid4())[:8]}.json"
            elif not filename.endswith(".json"):
                filename += ".json"

            file_path = self._save_file_to_disk(json_content, filename)
            msg = f"JSON file '{filename}' generated ({len(json_content)} chars)"
            if file_path:
                msg += f" at {file_path}"
            return {
                "status": "success",
                "data": {"filename": filename, "path": file_path},
                "message": msg,
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class GenerateCSVFile(FileGenerationTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="generate_csv_file",
            description="Generate a CSV file from data.",
            inputSchema={
                "type": "object",
                "properties": {
                    "data": {
                        "type": "string",
                        "description": "CSV data as JSON array string",
                    },
                    "filename": {"type": "string"},
                    "headers": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["data"],
            },
            function=self._generate_csv,
        )

    async def _generate_csv(
        self,
        data: Union[str, List],
        filename: Optional[str] = None,
        headers: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        try:
            output = io.StringIO()
            if isinstance(data, str):
                csv_content = data
            elif isinstance(data, list) and len(data) > 0:
                writer = csv.writer(output)
                if isinstance(data[0], dict):
                    fieldnames = list(data[0].keys())
                    writer.writerow(fieldnames)
                    for row in data:
                        writer.writerow(
                            [row.get(f, "") for f in fieldnames]
                            if isinstance(row, dict)
                            else [str(row)]
                        )
                elif isinstance(data[0], list):
                    if headers:
                        writer.writerow(headers)
                    writer.writerows(data)
                else:
                    if headers:
                        writer.writerow(headers)
                    for item in data:
                        writer.writerow([str(item)])
                csv_content = output.getvalue()
            else:
                csv_content = ""

            if not filename:
                filename = f"generated_{str(uuid4())[:8]}.csv"
            elif not filename.endswith(".csv"):
                filename += ".csv"

            file_path = self._save_file_to_disk(csv_content, filename)
            return {
                "status": "success",
                "data": {"filename": filename, "path": file_path},
                "message": f"CSV file '{filename}' generated",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class GeneratePDFFile(FileGenerationTools):
    def __init__(self, output_directory: Optional[str] = None):
        super().__init__(output_directory)
        if not PDF_AVAILABLE:
            raise ImportError(
                "Could not import `reportlab` python package. "
                "Please install it using `pip install reportlab`."
            )

    def get_tool(self) -> Tool:
        return Tool(
            name="generate_pdf_file",
            description="Generate a PDF file from text content.",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "filename": {"type": "string"},
                    "title": {"type": "string"},
                },
                "required": ["content"],
            },
            function=self._generate_pdf,
        )

    async def _generate_pdf(
        self, content: str, filename: Optional[str] = None, title: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=1 * inch)
            styles = getSampleStyleSheet()
            story = []
            if title:
                story.append(Paragraph(title, styles["Title"]))
                story.append(Spacer(1, 20))
            for para in content.split("\n\n"):
                if para.strip():
                    clean = para.strip().replace("<", "&lt;").replace(">", "&gt;")
                    story.append(Paragraph(clean, styles["Normal"]))
                    story.append(Spacer(1, 10))
            doc.build(story)
            pdf_content = buffer.getvalue()
            buffer.close()

            if not filename:
                filename = f"generated_{str(uuid4())[:8]}.pdf"
            elif not filename.endswith(".pdf"):
                filename += ".pdf"

            file_path = self._save_file_to_disk(pdf_content, filename)
            return {
                "status": "success",
                "data": {
                    "filename": filename,
                    "path": file_path,
                    "size": len(pdf_content),
                },
                "message": f"PDF file '{filename}' generated",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class GenerateTextFile(FileGenerationTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="generate_text_file",
            description="Generate a text file from content.",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "filename": {"type": "string"},
                },
                "required": ["content"],
            },
            function=self._generate_text,
        )

    async def _generate_text(
        self, content: str, filename: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            if not filename:
                filename = f"generated_{str(uuid4())[:8]}.txt"
            elif not filename.endswith(".txt"):
                filename += ".txt"
            file_path = self._save_file_to_disk(content, filename)
            return {
                "status": "success",
                "data": {"filename": filename, "path": file_path},
                "message": f"Text file '{filename}' generated",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}
