import re
from os import getenv
from typing import Any, Dict, Optional

from omnicoreagent.core.tools.local_tools_registry import Tool
from omnicoreagent.core.utils import log_info, logger

try:
    from twilio.base.exceptions import TwilioRestException
    from twilio.rest import Client
except ImportError:
    TwilioRestException = None
    Client = None


class TwilioTools:
    def __init__(
        self,
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        region: Optional[str] = None,
        edge: Optional[str] = None,
    ):
        self.account_sid = account_sid or getenv("TWILIO_ACCOUNT_SID")
        self.auth_token = auth_token or getenv("TWILIO_AUTH_TOKEN")
        self.api_key = api_key or getenv("TWILIO_API_KEY")
        self.api_secret = api_secret or getenv("TWILIO_API_SECRET")
        self.region = region or getenv("TWILIO_REGION")
        self.edge = edge or getenv("TWILIO_EDGE")

        if not self.account_sid:
            logger.error("TWILIO_ACCOUNT_SID not set.")

        if Client is None:
            raise ImportError(
                "Could not import `twilio` python package. "
                "Please install it with `pip install twilio`."
            )

        if self.api_key and self.api_secret:
            self.client = Client(
                self.api_key,
                self.api_secret,
                self.account_sid,
                region=self.region,
                edge=self.edge,
            )
        elif self.auth_token:
            self.client = Client(
                self.account_sid, self.auth_token, region=self.region, edge=self.edge
            )
        else:
            logger.error(
                "Neither auth_token nor api_key+api_secret provided for Twilio."
            )
            self.client = None

    @staticmethod
    def validate_phone_number(phone: str) -> bool:
        return bool(re.match(r"^\+[1-9]\d{1,14}$", phone))

    def get_tool(self) -> Tool:
        return Tool(
            name="twilio_send_sms",
            description="Send an SMS message using Twilio.",
            inputSchema={
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient phone (E.164 format)",
                    },
                    "from_": {
                        "type": "string",
                        "description": "Sender Twilio phone (E.164 format)",
                    },
                    "body": {"type": "string"},
                },
                "required": ["to", "from_", "body"],
            },
            function=self._send_sms,
        )

    async def _send_sms(self, to: str, from_: str, body: str) -> Dict[str, Any]:
        if not self.client:
            return {
                "status": "error",
                "data": None,
                "message": "Twilio client not initialized",
            }
        if not self.validate_phone_number(to):
            return {
                "status": "error",
                "data": None,
                "message": "'to' must be E.164 format",
            }
        if not self.validate_phone_number(from_):
            return {
                "status": "error",
                "data": None,
                "message": "'from_' must be E.164 format",
            }
        try:
            message = self.client.messages.create(to=to, from_=from_, body=body)
            log_info(f"SMS sent. SID: {message.sid}")
            return {
                "status": "success",
                "data": {"sid": message.sid},
                "message": f"SMS sent to {to}",
            }
        except TwilioRestException as e:
            logger.error(f"Failed to send SMS: {e}")
            return {"status": "error", "data": None, "message": str(e)}


class TwilioGetCallDetails(TwilioTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="twilio_get_call_details",
            description="Get details about a Twilio call.",
            inputSchema={
                "type": "object",
                "properties": {"call_sid": {"type": "string"}},
                "required": ["call_sid"],
            },
            function=self._get_call_details,
        )

    async def _get_call_details(self, call_sid: str) -> Dict[str, Any]:
        if not self.client:
            return {
                "status": "error",
                "data": None,
                "message": "Twilio client not initialized",
            }
        try:
            call = self.client.calls(call_sid).fetch()
            data = {
                "to": call.to,
                "from": call.from_,
                "status": call.status,
                "duration": call.duration,
                "direction": call.direction,
                "price": call.price,
                "start_time": str(call.start_time),
                "end_time": str(call.end_time),
            }
            return {
                "status": "success",
                "data": data,
                "message": "Call details retrieved",
            }
        except TwilioRestException as e:
            return {"status": "error", "data": None, "message": str(e)}


class TwilioListMessages(TwilioTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="twilio_list_messages",
            description="List recent SMS messages from Twilio.",
            inputSchema={
                "type": "object",
                "properties": {"limit": {"type": "integer", "default": 20}},
            },
            function=self._list_messages,
        )

    async def _list_messages(self, limit: int = 20) -> Dict[str, Any]:
        if not self.client:
            return {
                "status": "error",
                "data": None,
                "message": "Twilio client not initialized",
            }
        try:
            messages = []
            for msg in self.client.messages.list(limit=limit):
                messages.append(
                    {
                        "sid": msg.sid,
                        "to": msg.to,
                        "from": msg.from_,
                        "body": msg.body,
                        "status": msg.status,
                        "date_sent": str(msg.date_sent),
                    }
                )
            return {
                "status": "success",
                "data": messages,
                "message": f"Retrieved {len(messages)} messages",
            }
        except TwilioRestException as e:
            return {"status": "error", "data": None, "message": str(e)}
