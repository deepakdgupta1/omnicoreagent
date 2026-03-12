import os
from typing import Any, Dict, Optional
from omnicoreagent.core.tools.local_tools_registry import Tool

try:
    from notion_client import Client
except ImportError:
    Client = None


class NotionBase:
    def __init__(self, api_key: Optional[str] = None):
        if Client is None:
            raise ImportError(
                "Could not import `notion-client` python package. "
                "Please install it with `pip install notion-client`."
            )
        self.api_key = api_key or os.getenv("NOTION_API_KEY")

    def _get_client(self):
        if not self.api_key:
            raise ValueError("NOTION_API_KEY required.")
        return Client(auth=self.api_key)


class NotionCreatePage(NotionBase):
    def __init__(
        self, api_key: Optional[str] = None, database_id: Optional[str] = None
    ):
        super().__init__(api_key)
        self.database_id = database_id or os.getenv("NOTION_DATABASE_ID")

    def get_tool(self) -> Tool:
        return Tool(
            name="notion_create_page",
            description="Create a page in a Notion database.",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Page title."},
                    "content": {"type": "string", "description": "Page content."},
                },
                "required": ["title", "content"],
            },
            function=self._create_page,
        )

    async def _create_page(self, title: str, content: str) -> Dict[str, Any]:
        if not self.database_id:
            return {
                "status": "error",
                "data": None,
                "message": "NOTION_DATABASE_ID required.",
            }
        try:
            client = self._get_client()
            new_page = client.pages.create(
                parent={"database_id": self.database_id},
                properties={"Name": {"title": [{"text": {"content": title}}]}},
                children=[
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [
                                {"type": "text", "text": {"content": content}}
                            ]
                        },
                    }
                ],
            )
            return {
                "status": "success",
                "data": new_page,
                "message": f"Page created. URL: {new_page.get('url')}",
            }
        except Exception as e:
            return {
                "status": "error",
                "data": None,
                "message": f"Error creating Notion page: {str(e)}",
            }


class NotionSearchPage(NotionBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="notion_search_page",
            description="Search for pages in Notion.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                },
                "required": ["query"],
            },
            function=self._search_page,
        )

    async def _search_page(self, query: str) -> Dict[str, Any]:
        try:
            client = self._get_client()
            response = client.search(query=query)
            pages = []
            for result in response.get("results", []):
                if result["object"] == "page":
                    try:
                        # Try to extract title safely
                        title = "Untitled"
                        props = result.get("properties", {})
                        for key, val in props.items():
                            if val["type"] == "title" and val["title"]:
                                title = val["title"][0]["text"]["content"]
                                break
                        pages.append(f"{title} ({result['url']})")
                    except:
                        pages.append(f"Page {result['id']} ({result['url']})")

            formatted = "\n".join(pages)
            return {
                "status": "success",
                "data": response.get("results"),
                "message": f"Found pages:\n{formatted}" if pages else "No pages found.",
            }
        except Exception as e:
            return {
                "status": "error",
                "data": None,
                "message": f"Error searching Notion: {str(e)}",
            }
