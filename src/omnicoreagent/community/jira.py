from os import getenv
from typing import Any, Dict, Optional
from omnicoreagent.core.tools.local_tools_registry import Tool
from omnicoreagent.core.utils import logger

try:
    from jira import JIRA
except ImportError:
    JIRA = None


class JiraBase:
    def __init__(
        self,
        server_url: Optional[str] = None,
        username: Optional[str] = None,
        token: Optional[str] = None,
    ):
        self.server_url = server_url or getenv("JIRA_SERVER_URL")
        self.username = username or getenv("JIRA_USERNAME")
        self.token = token or getenv("JIRA_TOKEN") or getenv("JIRA_PASSWORD")

        self.jira = None
        if JIRA is None:
            raise ImportError(
                "jira not installed. Please install it using `pip install jira`."
            )
        elif self.server_url and self.username and self.token:
            try:
                self.jira = JIRA(
                    server=self.server_url, basic_auth=(self.username, self.token)
                )
            except Exception as e:
                logger.error(f"Failed to initialize JIRA client: {e}")


class JiraGetIssue(JiraBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="jira_get_issue",
            description="Get details of a Jira issue.",
            inputSchema={
                "type": "object",
                "properties": {
                    "issue_key": {"type": "string"},
                },
                "required": ["issue_key"],
            },
            function=self._get_issue,
        )

    async def _get_issue(self, issue_key: str) -> Dict[str, Any]:
        try:
            if not self.jira:
                return {
                    "status": "error",
                    "data": None,
                    "message": "Jira client not initialized",
                }

            issue = self.jira.issue(issue_key)
            data = {
                "key": issue.key,
                "summary": issue.fields.summary,
                "description": issue.fields.description,
                "status": issue.fields.status.name,
                "priority": issue.fields.priority.name
                if issue.fields.priority
                else None,
                "assignee": issue.fields.assignee.displayName
                if issue.fields.assignee
                else None,
            }
            return {
                "status": "success",
                "data": data,
                "message": f"Retrieved issue {issue_key}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class JiraCreateIssue(JiraBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="jira_create_issue",
            description="Create a new Jira issue.",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_key": {"type": "string"},
                    "summary": {"type": "string"},
                    "description": {"type": "string"},
                    "issuetype": {"type": "string", "default": "Task"},
                },
                "required": ["project_key", "summary"],
            },
            function=self._create_issue,
        )

    async def _create_issue(
        self,
        project_key: str,
        summary: str,
        description: Optional[str] = None,
        issuetype: str = "Task",
    ) -> Dict[str, Any]:
        try:
            if not self.jira:
                return {
                    "status": "error",
                    "data": None,
                    "message": "Jira client not initialized",
                }

            issue_dict = {
                "project": {"key": project_key},
                "summary": summary,
                "description": description,
                "issuetype": {"name": issuetype},
            }
            new_issue = self.jira.create_issue(fields=issue_dict)
            return {
                "status": "success",
                "data": {
                    "key": new_issue.key,
                    "id": new_issue.id,
                    "url": new_issue.permalink(),
                },
                "message": f"Created issue {new_issue.key}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class JiraSearchIssues(JiraBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="jira_search_issues",
            description="Search Jira issues using JQL.",
            inputSchema={
                "type": "object",
                "properties": {
                    "jql": {"type": "string"},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["jql"],
            },
            function=self._search,
        )

    async def _search(self, jql: str, limit: int = 10) -> Dict[str, Any]:
        try:
            if not self.jira:
                return {
                    "status": "error",
                    "data": None,
                    "message": "Jira client not initialized",
                }

            issues = self.jira.search_issues(jql, maxResults=limit)
            results = []
            for issue in issues:
                results.append(
                    {
                        "key": issue.key,
                        "summary": issue.fields.summary,
                        "status": issue.fields.status.name,
                    }
                )
            return {
                "status": "success",
                "data": results,
                "message": f"Found {len(results)} issues",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}
