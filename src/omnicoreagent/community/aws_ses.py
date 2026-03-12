from typing import Any, Dict, Optional
from omnicoreagent.core.tools.local_tools_registry import Tool

try:
    import boto3
except ImportError:
    boto3 = None


class AWSSETSendEmail:
    def __init__(
        self, sender_email: Optional[str] = None, region_name: str = "us-east-1"
    ):
        if boto3 is None:
            raise ImportError(
                "Could not import `boto3` python package. "
                "Please install it using `pip install boto3`."
            )
        self.sender_email = sender_email
        self.region_name = region_name

    def get_tool(self) -> Tool:
        return Tool(
            name="aws_ses_send_email",
            description="Send email via AWS SES.",
            inputSchema={
                "type": "object",
                "properties": {
                    "receiver_email": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["receiver_email", "subject", "body"],
            },
            function=self._send_email,
        )

    async def _send_email(
        self, receiver_email: str, subject: str, body: str
    ) -> Dict[str, Any]:
        try:
            client = boto3.client("ses", region_name=self.region_name)
            response = client.send_email(
                Source=self.sender_email or "noreply@example.com",
                Destination={"ToAddresses": [receiver_email]},
                Message={
                    "Subject": {"Data": subject},
                    "Body": {"Text": {"Data": body}},
                },
            )
            return {
                "status": "success",
                "data": response,
                "message": f"Email sent. ID: {response['MessageId']}",
            }
        except Exception as e:
            return {
                "status": "error",
                "data": None,
                "message": f"Error sending email: {str(e)}",
            }
