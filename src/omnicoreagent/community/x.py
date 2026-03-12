import os
from typing import Any, Dict, Optional
from omnicoreagent.core.tools.local_tools_registry import Tool

try:
    import tweepy
except ImportError:
    tweepy = None


class XBase:
    def __init__(self, bearer_token: Optional[str] = None, **kwargs):
        self.bearer_token = bearer_token or os.getenv("X_BEARER_TOKEN")
        self.consumer_key = kwargs.get("consumer_key") or os.getenv("X_CONSUMER_KEY")
        self.consumer_secret = kwargs.get("consumer_secret") or os.getenv(
            "X_CONSUMER_SECRET"
        )
        self.access_token = kwargs.get("access_token") or os.getenv("X_ACCESS_TOKEN")
        self.access_token_secret = kwargs.get("access_token_secret") or os.getenv(
            "X_ACCESS_TOKEN_SECRET"
        )

        if tweepy is None:
            raise ImportError(
                "Could not import `tweepy` python package. "
                "Please install it with `pip install tweepy`."
            )

    def _get_client(self):
        try:
            return tweepy.Client(
                bearer_token=self.bearer_token,
                consumer_key=self.consumer_key,
                consumer_secret=self.consumer_secret,
                access_token=self.access_token,
                access_token_secret=self.access_token_secret,
            )
        except Exception as e:
            raise ValueError(f"Error creating X client: {e}")


class XCreatePost(XBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="x_create_post",
            description="Post a tweet to X (Twitter).",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text content of the tweet.",
                    },
                },
                "required": ["text"],
            },
            function=self._create_post,
        )

    async def _create_post(self, text: str) -> Dict[str, Any]:
        try:
            client = self._get_client()
            response = client.create_tweet(text=text)
            return {
                "status": "success",
                "data": response.data,
                "message": f"Posted successfully. ID: {response.data.get('id')}",
            }
        except Exception as e:
            return {
                "status": "error",
                "data": None,
                "message": f"Error creating post: {str(e)}",
            }


class XSearchPosts(XBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="x_search_posts",
            description="Search for recent tweets on X.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "max_results": {
                        "type": "integer",
                        "description": "Max results (10-100).",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
            function=self._search_posts,
        )

    async def _search_posts(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        try:
            client = self._get_client()
            limit = max(10, min(max_results, 100))
            response = client.search_recent_tweets(
                query=query, max_results=limit, tweet_fields=["created_at", "author_id"]
            )

            if not response.data:
                return {"status": "success", "data": [], "message": "No tweets found."}

            tweets = [
                {"id": t.id, "text": t.text, "created_at": str(t.created_at)}
                for t in response.data
            ]
            formatted = "\n---\n".join(
                [f"{t['text']} ({t['created_at']})" for t in tweets]
            )

            return {"status": "success", "data": tweets, "message": formatted}
        except Exception as e:
            return {
                "status": "error",
                "data": None,
                "message": f"Error searching posts: {str(e)}",
            }
