import os
from typing import Any, Dict, Optional
from omnicoreagent.core.tools.local_tools_registry import Tool

try:
    from github import Github, Auth
except ImportError:
    Github = None
    Auth = None


class GithubBase:
    def __init__(self, access_token: Optional[str] = None):
        if Github is None:
            raise ImportError(
                "Could not import `PyGithub` python package. "
                "Please install it with `pip install PyGithub`."
            )
        self.access_token = access_token or os.getenv("GITHUB_ACCESS_TOKEN")

    def _get_client(self):
        try:
            if not self.access_token:
                raise ValueError("GitHub access token is required")
            auth = Auth.Token(self.access_token)
            return Github(auth=auth)
        except Exception as e:
            raise e


class GithubSearchRepos(GithubBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="github_search_repos",
            description="Search for repositories on GitHub.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
            function=self._search,
        )

    async def _search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        try:
            g = self._get_client()
            repos = g.search_repositories(query=query)
            results = []
            for repo in repos[:limit]:
                results.append(
                    {
                        "full_name": repo.full_name,
                        "description": repo.description,
                        "url": repo.html_url,
                        "stars": repo.stargazers_count,
                    }
                )
            return {
                "status": "success",
                "data": results,
                "message": f"Found {len(results)} repositories.",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class GithubCreateIssue(GithubBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="github_create_issue",
            description="Create an issue in a repository.",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_name": {"type": "string"},
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["repo_name", "title"],
            },
            function=self._create_issue,
        )

    async def _create_issue(
        self, repo_name: str, title: str, body: Optional[str] = ""
    ) -> Dict[str, Any]:
        try:
            g = self._get_client()
            repo = g.get_repo(repo_name)
            issue = repo.create_issue(title=title, body=body)
            return {
                "status": "success",
                "data": {"number": issue.number, "url": issue.html_url},
                "message": f"Created issue #{issue.number}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class GithubGetRepository(GithubBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="github_get_repository",
            description="Get details of a specific repository.",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_name": {"type": "string"},
                },
                "required": ["repo_name"],
            },
            function=self._get_repo,
        )

    async def _get_repo(self, repo_name: str) -> Dict[str, Any]:
        try:
            g = self._get_client()
            repo = g.get_repo(repo_name)
            data = {
                "name": repo.full_name,
                "description": repo.description,
                "url": repo.html_url,
                "stars": repo.stargazers_count,
                "language": repo.language,
                "open_issues": repo.open_issues_count,
            }
            return {
                "status": "success",
                "data": data,
                "message": f"Found repository: {repo.full_name}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}
