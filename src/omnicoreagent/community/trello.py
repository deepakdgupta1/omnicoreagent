from os import getenv
from typing import Any, Dict, Optional
from omnicoreagent.core.tools.local_tools_registry import Tool
from omnicoreagent.core.utils import logger

try:
    from trello import TrelloClient
except ImportError:
    TrelloClient = None


class TrelloBase:
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        token: Optional[str] = None,
    ):
        self.api_key = api_key or getenv("TRELLO_API_KEY")
        self.api_secret = api_secret or getenv("TRELLO_API_SECRET")
        self.token = token or getenv("TRELLO_TOKEN")
        self.client = None

        if TrelloClient is None:
            raise ImportError(
                "py-trello not installed. Please install it using `pip install py-trello`."
            )
        elif self.api_key and self.api_secret and self.token:
            try:
                self.client = TrelloClient(
                    api_key=self.api_key, api_secret=self.api_secret, token=self.token
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Trello client: {e}")


class TrelloCreateCard(TrelloBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="trello_create_card",
            description="Create a new Trello card.",
            inputSchema={
                "type": "object",
                "properties": {
                    "board_id": {"type": "string"},
                    "list_name": {"type": "string"},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["board_id", "list_name", "name"],
            },
            function=self._create_card,
        )

    async def _create_card(
        self, board_id: str, list_name: str, name: str, description: str = ""
    ) -> Dict[str, Any]:
        try:
            if not self.client:
                return {
                    "status": "error",
                    "data": None,
                    "message": "Trello client not initialized",
                }

            board = self.client.get_board(board_id)
            target_list = None
            for lst in board.list_lists():
                if lst.name.lower() == list_name.lower():
                    target_list = lst
                    break

            if not target_list:
                return {
                    "status": "error",
                    "data": None,
                    "message": f"List '{list_name}' not found",
                }

            card = target_list.add_card(name=name, desc=description)
            return {
                "status": "success",
                "data": {"id": card.id, "name": card.name, "url": card.url},
                "message": f"Created card {name}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class TrelloGetCards(TrelloBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="trello_get_cards",
            description="Get cards from a Trello list.",
            inputSchema={
                "type": "object",
                "properties": {
                    "list_id": {"type": "string"},
                },
                "required": ["list_id"],
            },
            function=self._get_cards,
        )

    async def _get_cards(self, list_id: str) -> Dict[str, Any]:
        try:
            if not self.client:
                return {
                    "status": "error",
                    "data": None,
                    "message": "Trello client not initialized",
                }

            trello_list = self.client.get_list(list_id)
            cards = trello_list.list_cards()
            result = [{"id": c.id, "name": c.name, "url": c.url} for c in cards]

            return {
                "status": "success",
                "data": result,
                "message": f"Found {len(result)} cards",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class TrelloListBoards(TrelloBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="trello_list_boards",
            description="List Trello boards.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
            function=self._list_boards,
        )

    async def _list_boards(self) -> Dict[str, Any]:
        try:
            if not self.client:
                return {
                    "status": "error",
                    "data": None,
                    "message": "Trello client not initialized",
                }

            boards = self.client.list_boards()
            result = [
                {"id": b.id, "name": b.name, "url": b.url}
                for b in boards
                if not b.closed
            ]

            return {
                "status": "success",
                "data": result,
                "message": f"Found {len(result)} open boards",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}
