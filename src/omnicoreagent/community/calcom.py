from datetime import datetime
from os import getenv
from typing import Any, Dict, Optional
import requests
from omnicoreagent.core.tools.local_tools_registry import Tool

try:
    import pytz
except ImportError:
    pytz = None


class CalComTools:
    def __init__(
        self,
        api_key: Optional[str] = None,
        event_type_id: Optional[int] = None,
        user_timezone: Optional[str] = None,
    ):
        if pytz is None:
            raise ImportError(
                "Could not import `pytz` python package. "
                "Please install it using `pip install pytz`."
            )
        self.api_key = api_key or getenv("CALCOM_API_KEY")
        event_type_str = getenv("CALCOM_EVENT_TYPE_ID")
        if event_type_id is not None:
            self.event_type_id = int(event_type_id)
        else:
            self.event_type_id = (
                int(event_type_str) if event_type_str is not None else 0
            )
        self.user_timezone = user_timezone or "America/New_York"

    def _convert_to_user_timezone(self, utc_time: str) -> str:
        try:
            timestamp_str = utc_time.replace("Z", "+00:00")
            utc_dt = datetime.fromisoformat(timestamp_str)
            user_tz = pytz.timezone(self.user_timezone)
            user_dt = utc_dt.astimezone(user_tz)
            return user_dt.strftime("%Y-%m-%d %H:%M %Z")
        except Exception:
            return utc_time

    def _get_headers(self, api_version: str = "2024-08-13") -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "cal-api-version": api_version,
            "Content-Type": "application/json",
        }

    # Default tool: Get available slots
    def get_tool(self) -> Tool:
        return Tool(
            name="calcom_get_available_slots",
            description="Get available time slots for booking.",
            inputSchema={
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "YYYY-MM-DD"},
                },
                "required": ["start_date", "end_date"],
            },
            function=self._get_available_slots,
        )

    async def _get_available_slots(
        self, start_date: str, end_date: str
    ) -> Dict[str, Any]:
        if not self.api_key:
            return {"status": "error", "data": None, "message": "API Key missing"}

        try:
            url = "https://api.cal.com/v2/slots/available"
            querystring = {
                "startTime": f"{start_date}T00:00:00Z",
                "endTime": f"{end_date}T23:59:59Z",
                "eventTypeId": str(self.event_type_id),
            }
            response = requests.get(
                url, headers=self._get_headers(), params=querystring
            )

            if response.status_code == 200:
                slots = response.json()["data"]["slots"]
                available_slots = []
                for date, times in slots.items():
                    for slot in times:
                        user_time = self._convert_to_user_timezone(slot["time"])
                        available_slots.append(user_time)
                return {
                    "status": "success",
                    "data": available_slots,
                    "message": "Slots retrieved",
                }

            return {
                "status": "error",
                "data": None,
                "message": f"Failed: {response.text}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class CalComCreateBooking(CalComTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="calcom_create_booking",
            description="Create a new booking.",
            inputSchema={
                "type": "object",
                "properties": {
                    "start_time": {
                        "type": "string",
                        "description": "YYYY-MM-DDTHH:MM:SSZ",
                    },
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                },
                "required": ["start_time", "name", "email"],
            },
            function=self._create_booking,
        )

    async def _create_booking(
        self, start_time: str, name: str, email: str
    ) -> Dict[str, Any]:
        if not self.api_key:
            return {"status": "error", "data": None, "message": "API Key missing"}
        try:
            url = "https://api.cal.com/v2/bookings"
            # Format time
            dt = datetime.fromisoformat(start_time).astimezone(pytz.utc)
            fmt_time = dt.isoformat(timespec="seconds")

            payload = {
                "start": fmt_time,
                "eventTypeId": self.event_type_id,
                "attendee": {
                    "name": name,
                    "email": email,
                    "timeZone": self.user_timezone,
                },
            }
            response = requests.post(url, json=payload, headers=self._get_headers())

            if response.status_code == 201:
                booking_data = response.json()["data"]
                return {
                    "status": "success",
                    "data": booking_data,
                    "message": "Booking created",
                }

            return {
                "status": "error",
                "data": None,
                "message": f"Failed: {response.text}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}
