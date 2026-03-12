from os import getenv
from typing import Any, Dict, Optional

try:
    import requests
except ImportError:
    requests = None
from omnicoreagent.core.tools.local_tools_registry import Tool


class LinearBase:
    def __init__(self, api_key: Optional[str] = None):
        if requests is None:
            raise ImportError(
                "`requests` not installed. Please install using `pip install requests`"
            )
        self.api_key = api_key or getenv("LINEAR_API_KEY")
        self.endpoint = "https://api.linear.app/graphql"
        self.headers = {"Authorization": f"{self.api_key}"}

    def _execute_query(
        self, query: str, variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if not self.api_key:
            return {"error": "Linear API key is required"}
        try:
            response = requests.post(
                self.endpoint,
                json={"query": query, "variables": variables},
                headers=self.headers,
            )
            response.raise_for_status()
            data = response.json()
            if "errors" in data:
                return {"error": str(data["errors"])}
            return data.get("data", {})
        except Exception as e:
            return {"error": str(e)}


class LinearGetIssue(LinearBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="linear_get_issue",
            description="Get details of a Linear issue.",
            inputSchema={
                "type": "object",
                "properties": {
                    "issue_id": {"type": "string"},
                },
                "required": ["issue_id"],
            },
            function=self._get_issue,
        )

    async def _get_issue(self, issue_id: str) -> Dict[str, Any]:
        query = """
        query IssueDetails ($issueId: String!){
        issue(id: $issueId) {
          id
          title
          description
          state { name }
          assignee { name }
          }
        }
        """
        result = self._execute_query(query, {"issueId": issue_id})
        if "error" in result:
            return {"status": "error", "data": None, "message": result["error"]}

        if not result.get("issue"):
            return {"status": "error", "data": None, "message": "Issue not found"}

        return {
            "status": "success",
            "data": result["issue"],
            "message": f"Retrieved issue {result['issue'].get('title')}",
        }


class LinearCreateIssue(LinearBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="linear_create_issue",
            description="Create a new Linear issue.",
            inputSchema={
                "type": "object",
                "properties": {
                    "team_id": {"type": "string"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "priority": {"type": "integer", "description": "Priority (0-4)"},
                },
                "required": ["team_id", "title", "description"],
            },
            function=self._create_issue,
        )

    async def _create_issue(
        self, team_id: str, title: str, description: str, priority: int = 0
    ) -> Dict[str, Any]:
        query = """
        mutation IssueCreate ($title: String!, $description: String!, $teamId: String!, $priority: Int){
          issueCreate(
            input: { title: $title, description: $description, teamId: $teamId, priority: $priority}
          ) {
            success
            issue {
              id
              title
              url
            }
          }
        }
        """
        variables = {
            "title": title,
            "description": description,
            "teamId": team_id,
            "priority": priority,
        }
        result = self._execute_query(query, variables)
        if "error" in result:
            return {"status": "error", "data": None, "message": result["error"]}

        if result.get("issueCreate", {}).get("success"):
            issue = result["issueCreate"]["issue"]
            return {
                "status": "success",
                "data": issue,
                "message": f"Created issue {issue.get('title')}",
            }
        return {"status": "error", "data": None, "message": "Failed to create issue"}


class LinearGetTeams(LinearBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="linear_get_teams",
            description="Get list of Linear teams.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
            function=self._get_teams,
        )

    async def _get_teams(self) -> Dict[str, Any]:
        query = """
        query Teams {
          teams {
            nodes {
              id
              name
            }
          }
        }
        """
        result = self._execute_query(query)
        if "error" in result:
            return {"status": "error", "data": None, "message": result["error"]}

        teams = result.get("teams", {}).get("nodes", [])
        return {
            "status": "success",
            "data": teams,
            "message": f"Found {len(teams)} teams",
        }
