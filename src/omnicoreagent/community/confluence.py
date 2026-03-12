from os import getenv
from typing import Any, Dict, Optional
from omnicoreagent.core.tools.local_tools_registry import Tool

try:
    from atlassian import Confluence
    import requests
except ImportError:
    Confluence = None
    requests = None


class ConfluenceBase:
    def __init__(
        self,
        url: Optional[str] = None,
        username: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.url = url or getenv("CONFLUENCE_URL")
        self.username = username or getenv("CONFLUENCE_USERNAME")
        self.password = (
            api_key or getenv("CONFLUENCE_API_KEY") or getenv("CONFLUENCE_PASSWORD")
        )

        if not self.url or not self.username or not self.password:
            # We allow init without creds for Import testing, but methods will fail
            pass

        if Confluence is None:
            raise ImportError(
                "Could not import `atlassian-python-api` python package. "
                "Please install it using `pip install atlassian-python-api`."
            )
        else:
            try:
                self.confluence = Confluence(
                    url=self.url,
                    username=self.username,
                    password=self.password,
                    cloud=True,
                )
            except Exception:
                self.confluence = None


class ConfluenceGetPage(ConfluenceBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="confluence_get_page",
            description="Get a Confluence page by title.",
            inputSchema={
                "type": "object",
                "properties": {
                    "space_key": {"type": "string"},
                    "title": {"type": "string"},
                },
                "required": ["space_key", "title"],
            },
            function=self._get_page,
        )

    async def _get_page(self, space_key: str, title: str) -> Dict[str, Any]:
        try:
            if not self.confluence:
                return {
                    "status": "error",
                    "data": None,
                    "message": "Confluence client not initialized",
                }

            page = self.confluence.get_page_by_title(
                space_key, title, expand="body.storage"
            )
            if not page:
                return {"status": "error", "data": None, "message": "Page not found"}

            return {
                "status": "success",
                "data": page,
                "message": f"Retrieved page {title}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class ConfluenceCreatePage(ConfluenceBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="confluence_create_page",
            description="Create a new Confluence page.",
            inputSchema={
                "type": "object",
                "properties": {
                    "space_key": {"type": "string"},
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "parent_id": {"type": "string"},
                },
                "required": ["space_key", "title", "body"],
            },
            function=self._create_page,
        )

    async def _create_page(
        self, space_key: str, title: str, body: str, parent_id: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            if not self.confluence:
                return {
                    "status": "error",
                    "data": None,
                    "message": "Confluence client not initialized",
                }

            response = self.confluence.create_page(
                space_key, title, body, parent_id=parent_id
            )
            return {
                "status": "success",
                "data": response,
                "message": f"Created page {title} (ID: {response.get('id')})",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class ConfluenceListSpaces(ConfluenceBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="confluence_list_spaces",
            description="List all spaces.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 10},
                },
            },
            function=self._list_spaces,
        )

    async def _list_spaces(self, limit: int = 10) -> Dict[str, Any]:
        try:
            if not self.confluence:
                return {
                    "status": "error",
                    "data": None,
                    "message": "Confluence client not initialized",
                }

            results = self.confluence.get_all_spaces(start=0, limit=limit)
            return {
                "status": "success",
                "data": results.get("results", []),
                "message": f"Found {len(results.get('results', []))} spaces",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}
