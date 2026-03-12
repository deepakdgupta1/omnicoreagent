from os import getenv
from typing import Any, Dict, Optional, Union

from omnicoreagent.core.tools.local_tools_registry import Tool
from omnicoreagent.core.utils import log_debug, log_error

try:
    from mem0.client.main import MemoryClient
    from mem0.memory.main import Memory
except ImportError:
    MemoryClient = None
    Memory = None


class Mem0Tools:
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        user_id: Optional[str] = None,
        org_id: Optional[str] = None,
        project_id: Optional[str] = None,
        infer: bool = True,
    ):
        self.api_key = api_key or getenv("MEM0_API_KEY")
        self.user_id = user_id
        self.org_id = org_id or getenv("MEM0_ORG_ID")
        self.project_id = project_id or getenv("MEM0_PROJECT_ID")
        self.client: Union["Memory", "MemoryClient"]
        self.infer = infer

        if MemoryClient is None:
            raise ImportError(
                "`mem0ai` not installed. Please install using `pip install mem0ai`"
            )

        try:
            if self.api_key:
                log_debug("Using Mem0 Platform API key.")
                client_kwargs: Dict[str, Any] = {"api_key": self.api_key}
                if self.org_id:
                    client_kwargs["org_id"] = self.org_id
                if self.project_id:
                    client_kwargs["project_id"] = self.project_id
                self.client = MemoryClient(**client_kwargs)
            elif config is not None:
                log_debug("Using Mem0 with config.")
                self.client = Memory.from_config(config)
            else:
                log_debug("Initializing Mem0 with default settings.")
                self.client = Memory()
        except Exception as e:
            log_error(f"Failed to initialize Mem0 client: {e}")
            raise ConnectionError("Failed to initialize Mem0 client.") from e

    def get_tool(self) -> Tool:
        return Tool(
            name="mem0_add_memory",
            description="Add facts to the user's memory store.",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Facts to store in memory",
                    },
                    "user_id": {
                        "type": "string",
                        "description": "User ID (optional, uses default)",
                    },
                },
                "required": ["content"],
            },
            function=self._add_memory,
        )

    async def _add_memory(
        self, content: str, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        resolved_user_id = user_id or self.user_id
        if not resolved_user_id:
            return {"status": "error", "data": None, "message": "user_id is required"}
        try:
            messages_list = [{"role": "user", "content": content}]
            result = self.client.add(
                messages_list, user_id=resolved_user_id, infer=self.infer
            )
            return {"status": "success", "data": result, "message": "Memory added"}
        except Exception as e:
            log_error(f"Error adding memory: {e}")
            return {"status": "error", "data": None, "message": str(e)}


class Mem0SearchMemory(Mem0Tools):
    def get_tool(self) -> Tool:
        return Tool(
            name="mem0_search_memory",
            description="Search across the user's stored memories.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "user_id": {"type": "string"},
                },
                "required": ["query"],
            },
            function=self._search_memory,
        )

    async def _search_memory(
        self, query: str, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        resolved_user_id = user_id or self.user_id
        if not resolved_user_id:
            return {"status": "error", "data": None, "message": "user_id is required"}
        try:
            results = self.client.search(query=query, user_id=resolved_user_id)
            if isinstance(results, dict) and "results" in results:
                search_results = results.get("results", [])
            elif isinstance(results, list):
                search_results = results
            else:
                search_results = []
            return {
                "status": "success",
                "data": search_results,
                "message": f"Found {len(search_results)} memories",
            }
        except Exception as e:
            log_error(f"Error searching memory: {e}")
            return {"status": "error", "data": None, "message": str(e)}


class Mem0GetAllMemories(Mem0Tools):
    def get_tool(self) -> Tool:
        return Tool(
            name="mem0_get_all_memories",
            description="Return all memories for the user.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                },
            },
            function=self._get_all_memories,
        )

    async def _get_all_memories(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        resolved_user_id = user_id or self.user_id
        if not resolved_user_id:
            return {"status": "error", "data": None, "message": "user_id is required"}
        try:
            results = self.client.get_all(user_id=resolved_user_id)
            if isinstance(results, dict) and "results" in results:
                memories = results.get("results", [])
            elif isinstance(results, list):
                memories = results
            else:
                memories = []
            return {
                "status": "success",
                "data": memories,
                "message": f"Found {len(memories)} memories",
            }
        except Exception as e:
            log_error(f"Error getting memories: {e}")
            return {"status": "error", "data": None, "message": str(e)}


class Mem0DeleteAllMemories(Mem0Tools):
    def get_tool(self) -> Tool:
        return Tool(
            name="mem0_delete_all_memories",
            description="Delete all memories for the user.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                },
            },
            function=self._delete_all_memories,
        )

    async def _delete_all_memories(
        self, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        resolved_user_id = user_id or self.user_id
        if not resolved_user_id:
            return {"status": "error", "data": None, "message": "user_id is required"}
        try:
            self.client.delete_all(user_id=resolved_user_id)
            return {
                "status": "success",
                "data": None,
                "message": f"Deleted all memories for {resolved_user_id}",
            }
        except Exception as e:
            log_error(f"Error deleting memories: {e}")
            return {"status": "error", "data": None, "message": str(e)}
