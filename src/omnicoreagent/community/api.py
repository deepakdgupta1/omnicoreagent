import json
from typing import Any, Dict, Literal, Optional
import requests
from requests.auth import HTTPBasicAuth

from omnicoreagent.core.tools.local_tools_registry import Tool
from omnicoreagent.core.utils import log_debug, logger


class CustomApiTools:
    def __init__(
        self,
        base_url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        api_key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        verify_ssl: bool = True,
        timeout: int = 30,
    ):
        self.base_url = base_url
        self.username = username
        self.password = password
        self.api_key = api_key
        self.default_headers = headers or {}
        self.verify_ssl = verify_ssl
        self.timeout = timeout

    def get_tool(self) -> Tool:
        return Tool(
            name="make_api_request",
            description="Make an HTTP request to an API.",
            inputSchema={
                "type": "object",
                "properties": {
                    "endpoint": {
                        "type": "string",
                        "description": "API endpoint (will be combined with base_url if set)",
                    },
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"],
                        "default": "GET",
                        "description": "HTTP method",
                    },
                    "params": {
                        "type": "object",
                        "description": "Query parameters",
                    },
                    "data": {
                        "type": "object",
                        "description": "Form data to send",
                    },
                    "headers": {
                        "type": "object",
                        "description": "Additional headers",
                    },
                    "json_data": {
                        "type": "object",
                        "description": "JSON data to send",
                    },
                },
                "required": ["endpoint"],
            },
            function=self._make_request,
        )

    def _get_auth(self) -> Optional[HTTPBasicAuth]:
        """Get authentication object if credentials are provided."""
        if self.username and self.password:
            return HTTPBasicAuth(self.username, self.password)
        return None

    def _get_headers(
        self, additional_headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """Combine default headers with additional headers."""
        headers = self.default_headers.copy()
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if additional_headers:
            headers.update(additional_headers)
        return headers

    async def _make_request(
        self,
        endpoint: str,
        method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"] = "GET",
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make an HTTP request to the API."""
        try:
            if self.base_url:
                url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
            else:
                url = endpoint
            log_debug(f"Making {method} request to {url}")

            response = requests.request(
                method=method,
                url=url,
                params=params,
                data=data,
                json=json_data,
                headers=self._get_headers(headers),
                auth=self._get_auth(),
                verify=self.verify_ssl,
                timeout=self.timeout,
            )

            try:
                response_data = response.json()
            except json.JSONDecodeError:
                response_data = {"text": response.text}

            result = {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "data": response_data,
            }

            if not response.ok:
                logger.error(
                    f"Request failed with status {response.status_code}: {response.text}"
                )
                return {
                    "status": "error",
                    "data": result,
                    "message": f"Request failed with status {response.status_code}",
                }

            return {
                "status": "success",
                "data": result,
                "message": "Request successful",
            }

        except requests.exceptions.RequestException as e:
            error_message = f"Request failed: {str(e)}"
            logger.error(error_message)
            return {"status": "error", "data": None, "message": error_message}
        except Exception as e:
            error_message = f"Unexpected error: {str(e)}"
            logger.error(error_message)
            return {"status": "error", "data": None, "message": error_message}
