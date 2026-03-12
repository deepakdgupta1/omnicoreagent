from pathlib import Path
from typing import Any, Dict, Optional
from omnicoreagent.core.tools.local_tools_registry import Tool

try:
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import Resource, build
except ImportError:
    Credentials = None
    Resource = None
    build = None


class GoogleDriveBase:
    def __init__(self):
        if build is None:
            raise ImportError(
                "Could not import `google-api-python-client` python package. "
                "Please install it using `pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib`."
            )
        self.scopes = [
            "https://www.googleapis.com/auth/drive.readonly",
            "https://www.googleapis.com/auth/drive.file",
        ]
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
            self.service = build("drive", "v3", credentials=self.creds)
        else:
            pass

        if not self.service:
            raise ValueError("Google Drive service could not be initialized.")


class GoogleDriveListFiles(GoogleDriveBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="google_drive_list_files",
            description="List files in Google Drive.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 10},
                },
            },
            function=self._list_files,
        )

    async def _list_files(
        self, query: Optional[str] = None, limit: int = 10
    ) -> Dict[str, Any]:
        try:
            self._ensure_service()
            results = (
                self.service.files()
                .list(
                    q=query,
                    pageSize=limit,
                    fields="nextPageToken, files(id, name, mimeType)",
                )
                .execute()
            )
            files = results.get("files", [])
            return {
                "status": "success",
                "data": files,
                "message": f"Found {len(files)} files",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}
