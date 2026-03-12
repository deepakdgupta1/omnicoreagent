from typing import Any, Dict

from omnicoreagent.core.tools.local_tools_registry import Tool
from omnicoreagent.core.utils import log_debug, logger

try:
    import httpx
except ImportError:
    httpx = None


class HackerNewsGetTopStories:
    def __init__(self):
        if httpx is None:
            raise ImportError(
                "`httpx` not installed. Please install using `pip install httpx`"
            )

    def get_tool(self) -> Tool:
        return Tool(
            name="hackernews_get_top_stories",
            description="Get top stories from Hacker News.",
            inputSchema={
                "type": "object",
                "properties": {
                    "num_stories": {"type": "integer", "default": 10},
                },
            },
            function=self._get_top_stories,
        )

    async def _get_top_stories(self, num_stories: int = 10) -> Dict[str, Any]:
        log_debug(f"Getting top {num_stories} stories from Hacker News")
        try:
            response = httpx.get(
                "https://hacker-news.firebaseio.com/v0/topstories.json"
            )
            story_ids = response.json()

            stories = []
            for story_id in story_ids[:num_stories]:
                story_resp = httpx.get(
                    f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
                )
                story = story_resp.json()
                if story:
                    story["username"] = story.get("by")
                    stories.append(story)
            return {
                "status": "success",
                "data": stories,
                "message": f"Retrieved {len(stories)} stories",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class HackerNewsGetUserDetails:
    def __init__(self):
        if httpx is None:
            raise ImportError(
                "`httpx` not installed. Please install using `pip install httpx`"
            )

    def get_tool(self) -> Tool:
        return Tool(
            name="hackernews_get_user_details",
            description="Get details of a Hacker News user.",
            inputSchema={
                "type": "object",
                "properties": {
                    "username": {"type": "string"},
                },
                "required": ["username"],
            },
            function=self._get_user_details,
        )

    async def _get_user_details(self, username: str) -> Dict[str, Any]:
        try:
            log_debug(f"Getting details for user: {username}")
            resp = httpx.get(
                f"https://hacker-news.firebaseio.com/v0/user/{username}.json"
            )
            user = resp.json()
            if not user:
                return {"status": "error", "data": None, "message": "User not found"}

            user_details = {
                "id": user.get("id"),
                "karma": user.get("karma"),
                "about": user.get("about"),
                "total_items_submitted": len(user.get("submitted", [])),
                "created": user.get("created"),
            }
            return {
                "status": "success",
                "data": user_details,
                "message": f"Details for {username}",
            }
        except Exception as e:
            logger.exception(e)
            return {"status": "error", "data": None, "message": str(e)}
