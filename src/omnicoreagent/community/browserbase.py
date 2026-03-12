from os import getenv
from typing import Any, Dict, Optional
from omnicoreagent.core.tools.local_tools_registry import Tool

try:
    from browserbase import Browserbase
except ImportError:
    Browserbase = None


class BrowserbaseBase:
    def __init__(self, api_key: Optional[str] = None, project_id: Optional[str] = None):
        if Browserbase is None:
            raise ImportError(
                "Could not import `browserbase` python package. "
                "Please install it with `pip install browserbase`."
            )

        self.api_key = api_key or getenv("BROWSERBASE_API_KEY")
        self.project_id = project_id or getenv("BROWSERBASE_PROJECT_ID")
        self.app = None
        if self.api_key:
            try:
                self.app = Browserbase(api_key=self.api_key)
            except Exception:
                self.app = None


class BrowserbaseSessionTool(BrowserbaseBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="browserbase_create_session",
            description="Create a Browserbase session.",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_id": {"type": "string"},
                },
            },
            function=self._create_session,
        )

    async def _create_session(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        pid = project_id or self.project_id
        if not self.app or not pid:
            return {
                "status": "error",
                "data": None,
                "message": "Browserbase API key or Project ID missing",
            }

        try:
            session = self.app.sessions.create(project_id=pid)
            return {
                "status": "success",
                "data": {"id": session.id, "connect_url": session.connect_url},
                "message": "Session created",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}
