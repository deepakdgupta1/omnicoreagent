import uuid
from os import getenv
from textwrap import dedent
from typing import Any, Dict, Optional

from omnicoreagent.core.tools.local_tools_registry import Tool
from omnicoreagent.core.utils import log_error

try:
    from zep_cloud import (
        BadRequestError,
        NotFoundError,
    )
    from zep_cloud import (
        Message as ZepMessage,
    )
    from zep_cloud.client import AsyncZep, Zep
except ImportError:
    BadRequestError = None
    NotFoundError = None
    ZepMessage = None
    AsyncZep = None
    Zep = None


DEFAULT_INSTRUCTIONS = dedent(
    """\
    You have access to the users memories stored in Zep. You can interact with them using the following tools:
    - `add_zep_message`: Add a message to the Zep session memory.
    - `get_zep_memory`: Get the memory for the current Zep session.
    - `search_zep_memory`: Search the Zep user graph for relevant facts.
    """
)


class ZepTools:
    def __init__(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        api_key: Optional[str] = None,
        ignore_assistant_messages: bool = False,
    ):
        if Zep is None:
            raise ImportError(
                "`zep-cloud` package not found. Please install it with `pip install zep-cloud`"
            )
        self._api_key = api_key or getenv("ZEP_API_KEY")
        if not self._api_key:
            raise ValueError("ZEP_API_KEY not set.")

        self.zep_client: Optional[Zep] = None
        self._initialized = False
        self.session_id_provided = session_id
        self.user_id_provided = user_id
        self.ignore_assistant_messages = ignore_assistant_messages
        self.session_id: Optional[str] = None
        self.user_id: Optional[str] = None
        self._initialize()

    def _initialize(self) -> bool:
        if self._initialized:
            return True
        try:
            self.zep_client = Zep(api_key=self._api_key)
            self.session_id = self.session_id_provided or f"{uuid.uuid4()}"
            self.user_id = self.user_id_provided
            if not self.user_id:
                self.user_id = f"user-{uuid.uuid4()}"
                self.zep_client.user.add(user_id=self.user_id)  # type: ignore
            else:
                try:
                    self.zep_client.user.get(self.user_id)  # type: ignore
                except NotFoundError:
                    try:
                        self.zep_client.user.add(user_id=self.user_id)  # type: ignore
                    except BadRequestError as e:
                        log_error(f"Failed to create user {self.user_id}: {e}")
                        self.zep_client = None
                        return False
            try:
                self.zep_client.thread.create(
                    thread_id=self.session_id, user_id=self.user_id
                )  # type: ignore
            except Exception:
                pass
            self._initialized = True
            return True
        except Exception as e:
            log_error(f"Failed to initialize ZepTools: {e}")
            self.zep_client = None
            return False

    def get_tool(self) -> Tool:
        return Tool(
            name="add_zep_message",
            description="Add a message to the Zep session memory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "role": {
                        "type": "string",
                        "description": "Message sender role (user, assistant, system)",
                    },
                    "content": {"type": "string"},
                },
                "required": ["role", "content"],
            },
            function=self._add_message,
        )

    async def _add_message(self, role: str, content: str) -> Dict[str, Any]:
        if not self.zep_client or not self.session_id:
            return {
                "status": "error",
                "data": None,
                "message": "Zep client/session not initialized",
            }
        try:
            zep_message = ZepMessage(role=role, content=content, role_type=role)
            ignore_roles = ["assistant"] if self.ignore_assistant_messages else None
            self.zep_client.thread.add_messages(  # type: ignore
                thread_id=self.session_id,
                messages=[zep_message],
                ignore_roles=ignore_roles,
            )
            return {
                "status": "success",
                "data": None,
                "message": f"Message from '{role}' added to session {self.session_id}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class ZepGetMemory(ZepTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="get_zep_memory",
            description="Get the memory for the current Zep session.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_type": {
                        "type": "string",
                        "enum": ["context", "messages"],
                        "default": "context",
                    },
                },
            },
            function=self._get_memory,
        )

    async def _get_memory(self, memory_type: str = "context") -> Dict[str, Any]:
        if not self.zep_client or not self.session_id:
            return {
                "status": "error",
                "data": None,
                "message": "Zep client/session not initialized",
            }
        try:
            if memory_type == "context":
                user_context = self.zep_client.thread.get_user_context(
                    thread_id=self.session_id, mode="basic"
                )  # type: ignore
                return {
                    "status": "success",
                    "data": user_context.context or "",
                    "message": "Context retrieved",
                }
            elif memory_type == "messages":
                messages_list = self.zep_client.thread.get(thread_id=self.session_id)  # type: ignore
                data = str(messages_list.messages) if messages_list.messages else ""
                return {
                    "status": "success",
                    "data": data,
                    "message": "Messages retrieved",
                }
            else:
                return {
                    "status": "error",
                    "data": None,
                    "message": f"Unsupported memory_type: {memory_type}",
                }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class ZepSearchMemory(ZepTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="search_zep_memory",
            description="Search the Zep knowledge graph for relevant facts.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "search_scope": {
                        "type": "string",
                        "enum": ["edges", "nodes"],
                        "default": "edges",
                    },
                },
                "required": ["query"],
            },
            function=self._search_memory,
        )

    async def _search_memory(
        self, query: str, search_scope: str = "edges"
    ) -> Dict[str, Any]:
        if not self.zep_client or not self.user_id:
            return {
                "status": "error",
                "data": None,
                "message": "Zep client/user not initialized",
            }
        try:
            response = self.zep_client.graph.search(
                query=query, user_id=self.user_id, scope=search_scope
            )
            if search_scope == "edges" and response.edges:
                facts = [edge.fact for edge in response.edges]
                return {
                    "status": "success",
                    "data": facts,
                    "message": f"Found {len(facts)} facts",
                }
            elif search_scope == "nodes" and response.nodes:
                nodes = [{"name": n.name, "summary": n.summary} for n in response.nodes]
                return {
                    "status": "success",
                    "data": nodes,
                    "message": f"Found {len(nodes)} nodes",
                }
            return {
                "status": "success",
                "data": [],
                "message": f"No {search_scope} found",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class ZepAsyncTools:
    def __init__(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        api_key: Optional[str] = None,
        ignore_assistant_messages: bool = False,
    ):
        self._api_key = api_key or getenv("ZEP_API_KEY")
        if not self._api_key:
            raise ValueError("ZEP_API_KEY not set.")
        self.zep_client: Optional[AsyncZep] = None
        self._initialized = False
        self.session_id_provided = session_id
        self.user_id_provided = user_id
        self.ignore_assistant_messages = ignore_assistant_messages
        self.session_id: Optional[str] = None
        self.user_id: Optional[str] = None

    async def _initialize(self) -> bool:
        if self._initialized:
            return True
        try:
            self.zep_client = AsyncZep(api_key=self._api_key)
            self.session_id = self.session_id_provided or f"{uuid.uuid4()}"
            self.user_id = self.user_id_provided
            if not self.user_id:
                self.user_id = f"user-{uuid.uuid4()}"
                await self.zep_client.user.add(user_id=self.user_id)  # type: ignore
            else:
                try:
                    await self.zep_client.user.get(self.user_id)  # type: ignore
                except NotFoundError:
                    try:
                        await self.zep_client.user.add(user_id=self.user_id)  # type: ignore
                    except BadRequestError as e:
                        log_error(f"Failed to create user {self.user_id}: {e}")
                        self.zep_client = None
                        return False
            try:
                await self.zep_client.thread.create(
                    thread_id=self.session_id, user_id=self.user_id
                )  # type: ignore
            except Exception:
                pass
            self._initialized = True
            return True
        except Exception as e:
            log_error(f"Failed to initialize ZepAsyncTools: {e}")
            self.zep_client = None
            return False

    def get_tool(self) -> Tool:
        return Tool(
            name="async_add_zep_message",
            description="Add a message to the Zep session memory (async).",
            inputSchema={
                "type": "object",
                "properties": {
                    "role": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["role", "content"],
            },
            function=self._add_message,
        )

    async def _add_message(self, role: str, content: str) -> Dict[str, Any]:
        if not self._initialized:
            await self._initialize()
        if not self.zep_client or not self.session_id:
            return {
                "status": "error",
                "data": None,
                "message": "Zep client/session not initialized",
            }
        try:
            zep_message = ZepMessage(role=role, content=content, role_type=role)
            ignore_roles = ["assistant"] if self.ignore_assistant_messages else None
            await self.zep_client.thread.add_messages(  # type: ignore
                thread_id=self.session_id,
                messages=[zep_message],
                ignore_roles=ignore_roles,
            )
            return {
                "status": "success",
                "data": None,
                "message": f"Message from '{role}' added",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class ZepAsyncGetMemory(ZepAsyncTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="async_get_zep_memory",
            description="Get the memory for the current Zep session (async).",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_type": {
                        "type": "string",
                        "enum": ["context", "messages"],
                        "default": "context",
                    },
                },
            },
            function=self._get_memory,
        )

    async def _get_memory(self, memory_type: str = "context") -> Dict[str, Any]:
        if not self._initialized:
            await self._initialize()
        if not self.zep_client or not self.session_id:
            return {
                "status": "error",
                "data": None,
                "message": "Zep client/session not initialized",
            }
        try:
            if memory_type == "context":
                ctx = await self.zep_client.thread.get_user_context(
                    thread_id=self.session_id, mode="basic"
                )  # type: ignore
                return {
                    "status": "success",
                    "data": ctx.context or "",
                    "message": "Context retrieved",
                }
            elif memory_type == "messages":
                msgs = await self.zep_client.thread.get(thread_id=self.session_id)  # type: ignore
                return {
                    "status": "success",
                    "data": str(msgs.messages) if msgs.messages else "",
                    "message": "Messages retrieved",
                }
            return {
                "status": "error",
                "data": None,
                "message": f"Unsupported memory_type: {memory_type}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class ZepAsyncSearchMemory(ZepAsyncTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="async_search_zep_memory",
            description="Search the Zep knowledge graph for relevant facts (async).",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "scope": {
                        "type": "string",
                        "enum": ["edges", "nodes"],
                        "default": "edges",
                    },
                    "limit": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
            function=self._search_memory,
        )

    async def _search_memory(
        self, query: str, scope: str = "edges", limit: int = 5
    ) -> Dict[str, Any]:
        if not self._initialized:
            await self._initialize()
        if not self.zep_client or not self.user_id:
            return {
                "status": "error",
                "data": None,
                "message": "Zep client/user not initialized",
            }
        try:
            response = await self.zep_client.graph.search(  # type: ignore
                query=query,
                user_id=self.user_id,
                scope=scope,
                limit=limit,
            )
            if scope == "edges" and response.edges:
                facts = [edge.fact for edge in response.edges]
                return {
                    "status": "success",
                    "data": facts,
                    "message": f"Found {len(facts)} facts",
                }
            elif scope == "nodes" and response.nodes:
                nodes = [{"name": n.name, "summary": n.summary} for n in response.nodes]
                return {
                    "status": "success",
                    "data": nodes,
                    "message": f"Found {len(nodes)} nodes",
                }
            return {"status": "success", "data": [], "message": f"No {scope} found"}
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}
