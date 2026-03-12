from os import getenv
from typing import Any, Dict, Optional

from omnicoreagent.core.tools.local_tools_registry import Tool
from omnicoreagent.core.utils import logger

try:
    from webexpythonsdk import WebexAPI
    from webexpythonsdk.exceptions import ApiError
except ImportError:
    WebexAPI = None
    ApiError = None


class WebexTools:
    def __init__(self, access_token: Optional[str] = None):
        if WebexAPI is None:
            raise ImportError(
                "Could not import `webexpythonsdk` python package. "
                "Please install it with `pip install webexpythonsdk`."
            )
        access_token = access_token or getenv("WEBEX_ACCESS_TOKEN")
        if access_token is None:
            raise ValueError(
                "Webex access token is not set. Set WEBEX_ACCESS_TOKEN environment variable."
            )
        self.client = WebexAPI(access_token=access_token)

    def get_tool(self) -> Tool:
        return Tool(
            name="webex_send_message",
            description="Send a message to a Webex Room.",
            inputSchema={
                "type": "object",
                "properties": {
                    "room_id": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["room_id", "text"],
            },
            function=self._send_message,
        )

    async def _send_message(self, room_id: str, text: str) -> Dict[str, Any]:
        try:
            response = self.client.messages.create(roomId=room_id, text=text)
            return {
                "status": "success",
                "data": response.json_data,
                "message": "Message sent",
            }
        except ApiError as e:
            logger.error(f"Error sending message: {e} in room: {room_id}")
            return {"status": "error", "data": None, "message": str(e)}


class WebexListRooms(WebexTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="webex_list_rooms",
            description="List all rooms in Webex.",
            inputSchema={"type": "object", "properties": {}},
            function=self._list_rooms,
        )

    async def _list_rooms(self) -> Dict[str, Any]:
        try:
            response = self.client.rooms.list()
            rooms = [
                {"id": room.id, "title": room.title, "type": room.type}
                for room in response
            ]
            return {
                "status": "success",
                "data": rooms,
                "message": f"Found {len(rooms)} rooms",
            }
        except ApiError as e:
            logger.error(f"Error listing rooms: {e}")
            return {"status": "error", "data": None, "message": str(e)}
