from os import getenv
from typing import Any, Dict, Optional
from omnicoreagent.core.tools.local_tools_registry import Tool

try:
    import requests
except ImportError:
    requests = None


class ClickUpBase:
    def __init__(self, api_key: Optional[str] = None):
        if requests is None:
            raise ImportError(
                "Could not import `requests` python package. "
                "Please install it using `pip install requests`."
            )
        self.api_key = api_key or getenv("CLICKUP_API_KEY")
        self.base_url = "https://api.clickup.com/api/v2"
        self.headers = {"Authorization": self.api_key}
        if not self.api_key:
            raise ValueError("CLICKUP_API_KEY not set.")

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/{endpoint}"
        try:
            response = requests.request(
                method=method, url=url, headers=self.headers, params=params, json=data
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}


class ClickUpListTasks(ClickUpBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="clickup_list_tasks",
            description="List all tasks in a ClickUp space.",
            inputSchema={
                "type": "object",
                "properties": {
                    "list_id": {
                        "type": "string",
                        "description": "The ID of the list to get tasks from.",
                    },
                    "archived": {"type": "boolean", "default": False},
                },
                "required": ["list_id"],
            },
            function=self._list_tasks,
        )

    async def _list_tasks(self, list_id: str, archived: bool = False) -> Dict[str, Any]:
        try:
            endpoint = f"list/{list_id}/task"
            params = {"archived": "true" if archived else "false"}
            response = self._make_request("GET", endpoint, params=params)

            if "error" in response:
                return {"status": "error", "data": None, "message": response["error"]}

            return {
                "status": "success",
                "data": response.get("tasks", []),
                "message": f"Found {len(response.get('tasks', []))} tasks.",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class ClickUpCreateTask(ClickUpBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="clickup_create_task",
            description="Create a new task in a ClickUp list.",
            inputSchema={
                "type": "object",
                "properties": {
                    "list_id": {"type": "string"},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["list_id", "name"],
            },
            function=self._create_task,
        )

    async def _create_task(
        self, list_id: str, name: str, description: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            endpoint = f"list/{list_id}/task"
            data = {"name": name, "description": description}
            response = self._make_request("POST", endpoint, data=data)

            if "error" in response:
                return {"status": "error", "data": None, "message": response["error"]}

            return {
                "status": "success",
                "data": response,
                "message": f"Created task {response.get('id')}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class ClickUpGetTask(ClickUpBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="clickup_get_task",
            description="Get details of a specific ClickUp task.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                },
                "required": ["task_id"],
            },
            function=self._get_task,
        )

    async def _get_task(self, task_id: str) -> Dict[str, Any]:
        try:
            endpoint = f"task/{task_id}"
            response = self._make_request("GET", endpoint)

            if "error" in response:
                return {"status": "error", "data": None, "message": response["error"]}

            return {
                "status": "success",
                "data": response,
                "message": f"Retrieved task {task_id}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class ClickUpListSpaces(ClickUpBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="clickup_list_spaces",
            description="List spaces in a team.",
            inputSchema={
                "type": "object",
                "properties": {
                    "team_id": {"type": "string"},
                },
                "required": ["team_id"],
            },
            function=self._list_spaces,
        )

    async def _list_spaces(self, team_id: str) -> Dict[str, Any]:
        try:
            endpoint = f"team/{team_id}/space"
            response = self._make_request("GET", endpoint)

            if "error" in response:
                return {"status": "error", "data": None, "message": response["error"]}

            return {
                "status": "success",
                "data": response.get("spaces", []),
                "message": f"Found {len(response.get('spaces', []))} spaces",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}
