from os import getenv
from typing import Any, Dict, Optional
from omnicoreagent.core.tools.local_tools_registry import Tool

try:
    from todoist_api_python.api import TodoistAPI
except ImportError:
    TodoistAPI = None


class TodoistBase:
    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token or getenv("TODOIST_API_TOKEN")
        self.api = None
        if TodoistAPI is None:
            raise ImportError(
                "todoist-api-python not installed. Please install it using `pip install todoist-api-python`."
            )
        elif self.api_token:
            try:
                self.api = TodoistAPI(self.api_token)
            except Exception:
                pass


class TodoistCreateTask(TodoistBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="todoist_create_task",
            description="Create a new Todoist task.",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "project_id": {"type": "string"},
                    "due_string": {
                        "type": "string",
                        "description": "Natural language due date, e.g. 'tomorrow'",
                    },
                    "priority": {"type": "integer", "description": "Priority 1-4"},
                },
                "required": ["content"],
            },
            function=self._create_task,
        )

    async def _create_task(
        self,
        content: str,
        project_id: Optional[str] = None,
        due_string: Optional[str] = None,
        priority: Optional[int] = None,
    ) -> Dict[str, Any]:
        try:
            if not self.api:
                return {
                    "status": "error",
                    "data": None,
                    "message": "Todoist API token not set",
                }

            task = self.api.add_task(
                content=content,
                project_id=project_id,
                due_string=due_string,
                priority=priority,
            )
            return {
                "status": "success",
                "data": {"id": task.id, "content": task.content, "url": task.url},
                "message": f"Created task {task.content}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class TodoistGetTasks(TodoistBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="todoist_get_tasks",
            description="Get active Todoist tasks.",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_id": {"type": "string"},
                    "filter": {"type": "string"},
                },
            },
            function=self._get_tasks,
        )

    async def _get_tasks(
        self, project_id: Optional[str] = None, filter: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            if not self.api:
                return {
                    "status": "error",
                    "data": None,
                    "message": "Todoist API token not set",
                }

            tasks = self.api.get_tasks(project_id=project_id, filter=filter)
            result = [
                {
                    "id": t.id,
                    "content": t.content,
                    "due": t.due.string if t.due else None,
                }
                for t in tasks
            ]
            return {
                "status": "success",
                "data": result,
                "message": f"Found {len(result)} tasks",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class TodoistCloseTask(TodoistBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="todoist_close_task",
            description="Close (complete) a Todoist task.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                },
                "required": ["task_id"],
            },
            function=self._close_task,
        )

    async def _close_task(self, task_id: str) -> Dict[str, Any]:
        try:
            if not self.api:
                return {
                    "status": "error",
                    "data": None,
                    "message": "Todoist API token not set",
                }

            is_success = self.api.close_task(task_id=task_id)
            if is_success:
                return {
                    "status": "success",
                    "data": None,
                    "message": f"Task {task_id} closed",
                }
            return {"status": "error", "data": None, "message": "Failed to close task"}
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}
