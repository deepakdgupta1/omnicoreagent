import os
from typing import Any, Dict, Optional
import requests
from omnicoreagent.core.tools.local_tools_registry import Tool


class DiscordBase:
    """Base class for Discord tools to handle auth."""

    def __init__(self, bot_token: Optional[str] = None):
        self.bot_token = bot_token or os.getenv("DISCORD_BOT_TOKEN")
        if not self.bot_token:
            # Allow init without token, fail at execution
            pass
        self.base_url = "https://discord.com/api/v10"
        self.headers = {
            "Authorization": f"Bot {self.bot_token}",
            "Content-Type": "application/json",
        }

    def _make_request(
        self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if not self.bot_token:
            raise ValueError("Discord bot token is required.")
        url = f"{self.base_url}{endpoint}"
        response = requests.request(method, url, headers=self.headers, json=data)
        response.raise_for_status()
        return response.json() if response.text else {}


class DiscordSendMessage(DiscordBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="discord_send_message",
            description="Send a message to a Discord channel.",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "The ID of the channel.",
                    },
                    "message": {"type": "string", "description": "The message text."},
                },
                "required": ["channel_id", "message"],
            },
            function=self._send_message,
        )

    async def _send_message(self, channel_id: str, message: str) -> Dict[str, Any]:
        try:
            data = {"content": message}
            self._make_request("POST", f"/channels/{channel_id}/messages", data)
            return {
                "status": "success",
                "data": None,
                "message": f"Message sent successfully to channel {channel_id}",
            }
        except Exception as e:
            return {
                "status": "error",
                "data": None,
                "message": f"Error sending message: {str(e)}",
            }


class DiscordListChannels(DiscordBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="discord_list_channels",
            description="List all channels in a Discord server.",
            inputSchema={
                "type": "object",
                "properties": {
                    "guild_id": {
                        "type": "string",
                        "description": "The ID of the server (guild).",
                    },
                },
                "required": ["guild_id"],
            },
            function=self._list_channels,
        )

    async def _list_channels(self, guild_id: str) -> Dict[str, Any]:
        try:
            response = self._make_request("GET", f"/guilds/{guild_id}/channels")
            channels = [
                {"id": c["id"], "name": c["name"], "type": c["type"]} for c in response
            ]
            formatted = "\n".join([f"{c['name']} (ID: {c['id']})" for c in channels])
            return {
                "status": "success",
                "data": channels,
                "message": f"Channels:\n{formatted}",
            }
        except Exception as e:
            return {
                "status": "error",
                "data": None,
                "message": f"Error listing channels: {str(e)}",
            }


class DiscordGetMessages(DiscordBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="discord_get_messages",
            description="Get the message history of a Discord channel.",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "The ID of the channel.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max messages to fetch (default 10).",
                        "default": 10,
                    },
                },
                "required": ["channel_id"],
            },
            function=self._get_messages,
        )

    async def _get_messages(self, channel_id: str, limit: int = 10) -> Dict[str, Any]:
        try:
            response = self._make_request(
                "GET", f"/channels/{channel_id}/messages?limit={limit}"
            )
            messages = [
                {
                    "author": m["author"]["username"],
                    "content": m["content"],
                    "id": m["id"],
                }
                for m in response
            ]
            formatted = "\n".join(
                [f"[{m['author']}]: {m['content']}" for m in messages]
            )
            return {
                "status": "success",
                "data": messages,
                "message": f"Messages:\n{formatted}",
            }
        except Exception as e:
            return {
                "status": "error",
                "data": None,
                "message": f"Error getting messages: {str(e)}",
            }
