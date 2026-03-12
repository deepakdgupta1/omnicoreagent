import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, cast
from omnicoreagent.core.tools.local_tools_registry import Tool

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import Resource, build
    from googleapiclient.errors import HttpError
except ImportError:
    Request = None
    Credentials = None
    InstalledAppFlow = None
    Resource = None
    build = None
    HttpError = None


class GoogleBase:
    DEFAULT_SCOPES = []

    def __init__(self, scopes: Optional[List[str]] = None):
        if build is None:
            raise ImportError(
                "Could not import `google-api-python-client` python package. "
                "Please install it using `pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib`."
            )
        self.scopes = scopes or self.DEFAULT_SCOPES
        self.creds = None
        self.service = None
        self.token_path = "token.json"
        self.credentials_path = "credentials.json"

    def _auth(self, service_name: str, version: str) -> None:
        if self.service:
            return

        token_file = Path(self.token_path)
        _creds_file = Path(self.credentials_path)

        if token_file.exists():
            try:
                self.creds = Credentials.from_authorized_user_file(
                    str(token_file), self.scopes
                )
            except Exception:
                pass

        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                try:
                    self.creds.refresh(Request())
                except Exception:
                    self.creds = None

            if not self.creds:
                # In a real server environment, we might rely on env vars or service accounts
                # For local dev/desktop, we use the flow.
                # Since this is running in an agent, interactive auth is hard.
                # We assume tokens exist or env vars are set for service account (omitted here for brevity matching original)
                pass

        if self.creds:
            self.service = build(service_name, version, credentials=self.creds)


class GoogleCalendarBase(GoogleBase):
    DEFAULT_SCOPES = ["https://www.googleapis.com/auth/calendar"]

    def __init__(self):
        super().__init__()
        self.calendar_id = "primary"

    def _ensure_service(self):
        self._auth("calendar", "v3")
        if not self.service:
            raise ValueError(
                "Google Calendar service could not be initialized (missing valid credentials)."
            )


class GoogleCalendarListEvents(GoogleCalendarBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="google_calendar_list_events",
            description="List upcoming calendar events.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 10},
                    "start_date": {"type": "string", "description": "ISO format date"},
                },
            },
            function=self._list_events,
        )

    async def _list_events(
        self, limit: int = 10, start_date: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            self._ensure_service()
            if not start_date:
                start_date = datetime.datetime.now(datetime.timezone.utc).isoformat()

            service = cast(Resource, self.service)
            events_result = (
                service.events()
                .list(
                    calendarId=self.calendar_id,
                    timeMin=start_date,
                    maxResults=limit,
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )

            events = events_result.get("items", [])
            return {
                "status": "success",
                "data": events,
                "message": f"Found {len(events)} events",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class GoogleCalendarCreateEvent(GoogleCalendarBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="google_calendar_create_event",
            description="Create a calendar event.",
            inputSchema={
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "start_time": {"type": "string", "description": "ISO format"},
                    "end_time": {"type": "string", "description": "ISO format"},
                    "description": {"type": "string"},
                    "location": {"type": "string"},
                    "attendees": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["summary", "start_time", "end_time"],
            },
            function=self._create_event,
        )

    async def _create_event(
        self,
        summary: str,
        start_time: str,
        end_time: str,
        description: Optional[str] = None,
        location: Optional[str] = None,
        attendees: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        try:
            self._ensure_service()
            event_body = {
                "summary": summary,
                "description": description,
                "location": location,
                "start": {"dateTime": start_time},
                "end": {"dateTime": end_time},
            }
            if attendees:
                event_body["attendees"] = [{"email": e} for e in attendees]

            service = cast(Resource, self.service)
            event = (
                service.events()
                .insert(calendarId=self.calendar_id, body=event_body)
                .execute()
            )

            return {
                "status": "success",
                "data": event,
                "message": f"Created event {event.get('htmlLink')}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}
