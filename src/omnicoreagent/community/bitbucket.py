import base64
from os import getenv
from typing import Any, Dict, Optional, Union

import requests
from omnicoreagent.core.tools.local_tools_registry import Tool
from omnicoreagent.core.utils import logger


class BitbucketTools:
    def __init__(
        self,
        server_url: str = "api.bitbucket.org",
        username: Optional[str] = None,
        password: Optional[str] = None,
        token: Optional[str] = None,
        workspace: Optional[str] = None,
        repo_slug: Optional[str] = None,
        api_version: str = "2.0",
        **kwargs,
    ):
        self.username = username or getenv("BITBUCKET_USERNAME")
        self.password = password or getenv("BITBUCKET_PASSWORD")
        self.token = token or getenv("BITBUCKET_TOKEN")
        self.auth_password = self.token or self.password
        self.server_url = server_url or "api.bitbucket.org"
        self.api_version = api_version or "2.0"
        self.base_url = (
            f"https://{self.server_url}/{api_version}"
            if not self.server_url.startswith(("http://", "https://"))
            else f"{self.server_url}/{api_version}"
        )
        self.workspace = workspace
        self.repo_slug = repo_slug

        self.headers = {"Accept": "application/json"}
        if self.username and self.auth_password:
            self.headers["Authorization"] = f"Basic {self._generate_access_token()}"

    def _generate_access_token(self) -> str:
        auth_str = f"{self.username}:{self.auth_password}"
        auth_bytes = auth_str.encode("ascii")
        auth_base64 = base64.b64encode(auth_bytes).decode("ascii")
        return auth_base64

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Dict[str, Any]]:
        if not self.headers.get("Authorization"):
            raise ValueError("Username and password or token are required")

        url = f"{self.base_url}{endpoint}"
        response = requests.request(
            method, url, headers=self.headers, json=data, params=params
        )
        response.raise_for_status()
        encoding_type = response.headers.get("Content-Type", "application/json")
        if encoding_type.startswith("application/json"):
            return response.json() if response.text else {}
        elif encoding_type == "text/plain":
            return response.text

        logger.warning(f"Unsupported content type: {encoding_type}")
        return {}

    def get_tool(self) -> Tool:
        # Defaults to list_repositories if generic tool is requested,
        # but typically we'd want specific tools or a collection.
        # For now, we return list_repositories as the default "get_tool"
        return Tool(
            name="bitbucket_list_repos",
            description="Get all repositories in the workspace.",
            inputSchema={
                "type": "object",
                "properties": {
                    "count": {"type": "integer", "default": 10},
                },
            },
            function=self._list_repositories,
        )

    # Individual Tool Classes for better granularity


class BitbucketListRepos(BitbucketTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="bitbucket_list_repos",
            description="Get all repositories in the workspace.",
            inputSchema={
                "type": "object",
                "properties": {
                    "count": {"type": "integer", "default": 10},
                },
            },
            function=self._list_repositories,
        )

    async def _list_repositories(self, count: int = 10) -> Dict[str, Any]:
        try:
            if not self.workspace:
                return {
                    "status": "error",
                    "data": None,
                    "message": "Workspace is required",
                }

            count = min(count, 50)
            pagelen = min(count, 50)
            params = {"page": 1, "pagelen": pagelen}

            repo = self._make_request(
                "GET", f"/repositories/{self.workspace}", params=params
            )
            return {
                "status": "success",
                "data": repo,
                "message": f"Found repositories in {self.workspace}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class BitbucketGetRepoDetails(BitbucketTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="bitbucket_get_repo_details",
            description="Retrieves repository information.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
            function=self._get_repository_details,
        )

    async def _get_repository_details(self) -> Dict[str, Any]:
        try:
            if not self.workspace or not self.repo_slug:
                return {
                    "status": "error",
                    "data": None,
                    "message": "Workspace and Repo slug are required",
                }

            repo = self._make_request(
                "GET", f"/repositories/{self.workspace}/{self.repo_slug}"
            )
            return {
                "status": "success",
                "data": repo,
                "message": f"Details for {self.repo_slug}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class BitbucketCreateRepo(BitbucketTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="bitbucket_create_repo",
            description="Creates a new repository.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "project": {"type": "string"},
                    "is_private": {"type": "boolean"},
                    "description": {"type": "string"},
                    "language": {"type": "string"},
                },
                "required": ["name"],
            },
            function=self._create_repository,
        )

    async def _create_repository(
        self,
        name: str,
        project: Optional[str] = None,
        is_private: bool = False,
        description: Optional[str] = None,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            if not self.workspace or not self.repo_slug:
                return {
                    "status": "error",
                    "data": None,
                    "message": "Workspace and Repo slug are required",
                }

            payload = {
                "name": name,
                "scm": "git",
                "is_private": is_private,
                "description": description,
                "language": language,
            }
            if project:
                payload["project"] = {"key": project}
            repo = self._make_request(
                "POST", f"/repositories/{self.workspace}/{self.repo_slug}", data=payload
            )
            return {
                "status": "success",
                "data": repo,
                "message": f"Repository {name} created",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}
