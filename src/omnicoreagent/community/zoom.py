from base64 import b64encode
from datetime import datetime, timedelta
from os import getenv
from typing import Any, Dict, Optional

try:
    import requests
except ImportError:
    requests = None
from omnicoreagent.core.tools.local_tools_registry import Tool
from omnicoreagent.core.utils import logger


class ZoomBase:
    def __init__(
        self,
        account_id: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ):
        self.account_id = account_id or getenv("ZOOM_ACCOUNT_ID")
        self.client_id = client_id or getenv("ZOOM_CLIENT_ID")
        self.client_secret = client_secret or getenv("ZOOM_CLIENT_SECRET")
        self.__access_token = None
        self.__token_expiry = None

        if not self.account_id or not self.client_id or not self.client_secret:
            logger.error("ZOOM credentials missing.")

        if requests is None:
            raise ImportError(
                "`requests` not installed. Please install using `pip install requests`"
            )

    def _get_access_token(self) -> str:
        if (
            self.__access_token
            and self.__token_expiry
            and datetime.now() < self.__token_expiry
        ):
            return self.__access_token

        try:
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            auth_string = b64encode(
                f"{self.client_id}:{self.client_secret}".encode()
            ).decode()
            headers["Authorization"] = f"Basic {auth_string}"
            data = {"grant_type": "account_credentials", "account_id": self.account_id}

            response = requests.post(
                "https://zoom.us/oauth/token", headers=headers, data=data
            )
            response.raise_for_status()

            token_data = response.json()
            self.__access_token = token_data["access_token"]
            self.__token_expiry = datetime.now() + timedelta(
                seconds=token_data["expires_in"] - 60
            )
            return self.__access_token
        except Exception as e:
            logger.error(f"Failed to generate Zoom token: {e}")
            return ""

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        token = self._get_access_token()
        if not token:
            raise Exception("Failed to get access token")

        headers = kwargs.get("headers", {})
        headers["Authorization"] = f"Bearer {token}"
        kwargs["headers"] = headers

        url = f"https://api.zoom.us/v2{endpoint}"
        response = requests.request(method, url, **kwargs)
        response.raise_for_status()
        if response.status_code == 204:
            return {}
        return response.json()


class ZoomScheduleMeeting(ZoomBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="zoom_schedule_meeting",
            description="Schedule a new Zoom meeting.",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "start_time": {"type": "string", "description": "ISO 8601 format"},
                    "duration": {"type": "integer", "description": "Minutes"},
                    "timezone": {"type": "string", "default": "UTC"},
                },
                "required": ["topic", "start_time", "duration"],
            },
            function=self._schedule_meeting,
        )

    async def _schedule_meeting(
        self, topic: str, start_time: str, duration: int, timezone: str = "UTC"
    ) -> Dict[str, Any]:
        try:
            data = {
                "topic": topic,
                "type": 2,
                "start_time": start_time,
                "duration": duration,
                "timezone": timezone,
                "settings": {
                    "host_video": True,
                    "participant_video": True,
                    "audio": "voip",
                    "auto_recording": "none",
                },
            }
            # Zoom API calls are synchronous via requests, wrapping in async function is fine
            result = self._make_request("POST", "/users/me/meetings", json=data)
            return {
                "status": "success",
                "data": result,
                "message": f"Meeting scheduled: {result.get('id')}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class ZoomListMeetings(ZoomBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="zoom_list_meetings",
            description="List meetings for a user.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "default": "me"},
                    "type": {
                        "type": "string",
                        "default": "scheduled",
                        "enum": ["scheduled", "live", "upcoming", "previous"],
                    },
                },
            },
            function=self._list_meetings,
        )

    async def _list_meetings(
        self, user_id: str = "me", type: str = "scheduled"
    ) -> Dict[str, Any]:
        try:
            params = {"type": type, "page_size": 30}
            result = self._make_request(
                "GET", f"/users/{user_id}/meetings", params=params
            )
            meetings = result.get("meetings", [])
            return {
                "status": "success",
                "data": meetings,
                "message": f"Retrieved {len(meetings)} meetings",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class ZoomGetMeeting(ZoomBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="zoom_get_meeting",
            description="Get details of a specific Zoom meeting.",
            inputSchema={
                "type": "object",
                "properties": {
                    "meeting_id": {"type": "string"},
                },
                "required": ["meeting_id"],
            },
            function=self._get_meeting,
        )

    async def _get_meeting(self, meeting_id: str) -> Dict[str, Any]:
        try:
            result = self._make_request("GET", f"/meetings/{meeting_id}")
            return {
                "status": "success",
                "data": result,
                "message": f"Retrieved details for meeting {meeting_id}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class ZoomDeleteMeeting(ZoomBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="zoom_delete_meeting",
            description="Delete a Zoom meeting.",
            inputSchema={
                "type": "object",
                "properties": {
                    "meeting_id": {"type": "string"},
                },
                "required": ["meeting_id"],
            },
            function=self._delete_meeting,
        )

    async def _delete_meeting(self, meeting_id: str) -> Dict[str, Any]:
        try:
            self._make_request("DELETE", f"/meetings/{meeting_id}")
            return {
                "status": "success",
                "data": None,
                "message": f"Meeting {meeting_id} deleted",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class ZoomGetRecordings(ZoomBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="zoom_get_recordings",
            description="Get recordings for a meeting.",
            inputSchema={
                "type": "object",
                "properties": {
                    "meeting_id": {"type": "string"},
                },
                "required": ["meeting_id"],
            },
            function=self._get_recordings,
        )

    async def _get_recordings(self, meeting_id: str) -> Dict[str, Any]:
        try:
            result = self._make_request("GET", f"/meetings/{meeting_id}/recordings")
            files = result.get("recording_files", [])
            return {
                "status": "success",
                "data": files,
                "message": f"Retrieved {len(files)} recordings",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}
