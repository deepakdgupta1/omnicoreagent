from os import getenv
from typing import Any, Dict, Optional
from omnicoreagent.core.tools.local_tools_registry import Tool

try:
    from apify_client import ApifyClient
except ImportError:
    ApifyClient = None


class ApifyRunActor:
    def __init__(self, api_token: Optional[str] = None):
        if ApifyClient is None:
            raise ImportError(
                "Could not import `apify-client` python package. "
                "Please install it using `pip install apify-client`."
            )
        self.api_token = api_token or getenv("APIFY_API_TOKEN")
        self.client = None
        if self.api_token:
            try:
                self.client = ApifyClient(self.api_token)
            except Exception:
                self.client = None

    def get_tool(self) -> Tool:
        return Tool(
            name="apify_run_actor",
            description="Run an Apify actor.",
            inputSchema={
                "type": "object",
                "properties": {
                    "actor_id": {
                        "type": "string",
                        "description": "e.g. apify/web-scraper",
                    },
                    "run_input": {
                        "type": "object",
                        "description": "Input JSON for the actor",
                    },
                },
                "required": ["actor_id"],
            },
            function=self._run_actor,
        )

    async def _run_actor(
        self, actor_id: str, run_input: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if not self.client:
            return {
                "status": "error",
                "data": None,
                "message": "Apify client not initialized (missing API token)",
            }

        try:
            run = self.client.actor(actor_id).call(run_input=run_input)
            if not run:
                return {
                    "status": "error",
                    "data": None,
                    "message": "Actor run failed to start",
                }

            dataset_items = (
                self.client.dataset(run["defaultDatasetId"]).list_items().items
            )
            return {
                "status": "success",
                "data": dataset_items,
                "message": f"Actor run completed with {len(dataset_items)} items",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}
