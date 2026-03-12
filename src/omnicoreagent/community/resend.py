from os import getenv
from typing import Any, Dict, Optional

from omnicoreagent.core.tools.local_tools_registry import Tool
from omnicoreagent.core.utils import logger

try:
    import resend  # type: ignore
except ImportError:
    resend = None  # type: ignore


class ResendTools:
    def __init__(self, api_key: Optional[str] = None, from_email: Optional[str] = None):
        if resend is None:
            raise ImportError(
                "`resend` not installed. Please install using `pip install resend`."
            )
        self.from_email = from_email
        self.api_key = api_key or getenv("RESEND_API_KEY")
        if not self.api_key:
            logger.error("RESEND_API_KEY not set.")

    def get_tool(self) -> Tool:
        return Tool(
            name="resend_send_email",
            description="Send an email using the Resend API.",
            inputSchema={
                "type": "object",
                "properties": {
                    "to_email": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string", "description": "HTML body of the email"},
                },
                "required": ["to_email", "subject", "body"],
            },
            function=self._send_email,
        )

    async def _send_email(
        self, to_email: str, subject: str, body: str
    ) -> Dict[str, Any]:
        if not self.api_key:
            return {
                "status": "error",
                "data": None,
                "message": "RESEND_API_KEY not set",
            }
        try:
            resend.api_key = self.api_key
            params = {
                "from": self.from_email,
                "to": to_email,
                "subject": subject,
                "html": body,
            }
            result = resend.Emails.send(params)
            return {
                "status": "success",
                "data": result,
                "message": f"Email sent to {to_email}",
            }
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return {"status": "error", "data": None, "message": str(e)}
