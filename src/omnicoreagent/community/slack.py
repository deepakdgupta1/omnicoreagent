import os
from typing import Any, Dict, Optional
from omnicoreagent.core.tools.local_tools_registry import Tool

try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
except ImportError:
    WebClient = None
    SlackApiError = None


class SlackBase:
    def __init__(self, token: Optional[str] = None):
        if WebClient is None:
            raise ImportError(
                "Slack tools require the `slack_sdk` package. Please install it using `pip install slack_sdk`."
            )
        self.token = token or os.getenv("SLACK_TOKEN")

    def _get_client(self):
        if not self.token:
            raise ValueError("SLACK_TOKEN is not set")
        return WebClient(token=self.token)


class SlackSendMessage(SlackBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="slack_send_message",
            description="Send a message to a Slack channel.",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel": {"type": "string", "description": "Channel ID or name."},
                    "text": {"type": "string", "description": "Message text."},
                },
                "required": ["channel", "text"],
            },
            function=self._send_message,
        )

    async def _send_message(self, channel: str, text: str) -> Dict[str, Any]:
        try:
            client = self._get_client()
            response = client.chat_postMessage(channel=channel, text=text)
            return {
                "status": "success",
                "data": response.data,
                "message": f"Message sent to {channel}",
            }
        except Exception as e:
            return {
                "status": "error",
                "data": None,
                "message": f"Error sending Slack message: {str(e)}",
            }


class SlackListChannels(SlackBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="slack_list_channels",
            description="List public channels in the workspace.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
            function=self._list_channels,
        )

    async def _list_channels(self) -> Dict[str, Any]:
        try:
            client = self._get_client()
            response = client.conversations_list(types="public_channel")
            channels = [
                {"id": c["id"], "name": c["name"]} for c in response["channels"]
            ]
            formatted = "\n".join([f"#{c['name']} ({c['id']})" for c in channels])
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


class SlackGetHistory(SlackBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="slack_get_history",
            description="Get message history from a channel.",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel": {"type": "string", "description": "Channel ID."},
                    "limit": {
                        "type": "integer",
                        "description": "Max messages.",
                        "default": 20,
                    },
                },
                "required": ["channel"],
            },
            function=self._get_history,
        )

    async def _get_history(self, channel: str, limit: int = 20) -> Dict[str, Any]:
        try:
            client = self._get_client()
            response = client.conversations_history(channel=channel, limit=limit)
            messages = []
            for msg in response.get("messages", []):
                messages.append(
                    {"user": msg.get("user", "unknown"), "text": msg.get("text", "")}
                )

            formatted = "\n".join([f"{m['user']}: {m['text']}" for m in messages])
            return {
                "status": "success",
                "data": messages,
                "message": f"History:\n{formatted}",
            }
        except Exception as e:
            return {
                "status": "error",
                "data": None,
                "message": f"Error getting history: {str(e)}",
            }
