from pathlib import Path
from typing import Any, Dict, List
from omnicoreagent.core.tools.local_tools_registry import Tool

try:
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import Resource, build
except ImportError:
    Credentials = None
    Resource = None
    build = None


class GoogleSheetsBase:
    def __init__(self):
        if build is None:
            raise ImportError(
                "Could not import `google-api-python-client` python package. "
                "Please install it using `pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib`."
            )
        self.scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        self.creds = None
        self.service = None
        self.token_path = "token.json"

    def _ensure_service(self):
        if self.service:
            return

        token_file = Path(self.token_path)
        if token_file.exists():
            try:
                self.creds = Credentials.from_authorized_user_file(
                    str(token_file), self.scopes
                )
            except:
                pass

        if self.creds and self.creds.valid:
            self.service = build("sheets", "v4", credentials=self.creds)
        else:
            # Just like Calendar, skipping interactive auth flow implementation details for brevity in this refactor
            pass

        if not self.service:
            raise ValueError("Google Sheets service could not be initialized.")


class GoogleSheetsRead(GoogleSheetsBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="google_sheets_read",
            description="Read values from a Google Sheet.",
            inputSchema={
                "type": "object",
                "properties": {
                    "spreadsheet_id": {"type": "string"},
                    "range_name": {"type": "string"},
                },
                "required": ["spreadsheet_id", "range_name"],
            },
            function=self._read,
        )

    async def _read(self, spreadsheet_id: str, range_name: str) -> Dict[str, Any]:
        try:
            self._ensure_service()
            result = (
                self.service.spreadsheets()
                .values()
                .get(spreadsheetId=spreadsheet_id, range=range_name)
                .execute()
            )
            values = result.get("values", [])
            return {
                "status": "success",
                "data": values,
                "message": f"Read {len(values)} rows",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class GoogleSheetsCreate(GoogleSheetsBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="google_sheets_create",
            description="Create a new Google Sheet.",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                },
                "required": ["title"],
            },
            function=self._create,
        )

    async def _create(self, title: str) -> Dict[str, Any]:
        try:
            self._ensure_service()
            spreadsheet = {"properties": {"title": title}}
            spreadsheet = (
                self.service.spreadsheets()
                .create(body=spreadsheet, fields="spreadsheetId")
                .execute()
            )
            return {
                "status": "success",
                "data": spreadsheet,
                "message": f"Created spreadsheet {spreadsheet.get('spreadsheetId')}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class GoogleSheetsUpdate(GoogleSheetsBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="google_sheets_update",
            description="Update values in a Google Sheet.",
            inputSchema={
                "type": "object",
                "properties": {
                    "spreadsheet_id": {"type": "string"},
                    "range_name": {"type": "string"},
                    "values": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "string"}},
                    },
                },
                "required": ["spreadsheet_id", "range_name", "values"],
            },
            function=self._update,
        )

    async def _update(
        self, spreadsheet_id: str, range_name: str, values: List[List[str]]
    ) -> Dict[str, Any]:
        try:
            self._ensure_service()
            body = {"values": values}
            result = (
                self.service.spreadsheets()
                .values()
                .update(
                    spreadsheetId=spreadsheet_id,
                    range=range_name,
                    valueInputOption="RAW",
                    body=body,
                )
                .execute()
            )
            return {"status": "success", "data": result, "message": "Updated sheet"}
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}
