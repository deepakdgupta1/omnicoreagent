import json
from os import getenv
from typing import Optional

from omnicoreagent.core.tools.local_tools_registry import Tool
from omnicoreagent.core.utils import logger

try:
    import praw
except ImportError:
    praw = None


class RedditBase:
    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.client_id = client_id or getenv("REDDIT_CLIENT_ID")
        self.client_secret = client_secret or getenv("REDDIT_CLIENT_SECRET")
        self.user_agent = user_agent or getenv("REDDIT_USER_AGENT", "RedditTools v1.0")
        self.username = username or getenv("REDDIT_USERNAME")
        self.password = password or getenv("REDDIT_PASSWORD")
        self.reddit = None

        if praw is None:
            raise ImportError(
                "Could not import `praw` python package. "
                "Please install it with `pip install praw`."
            )

    def _get_reddit(self):
        if self.reddit:
            return self.reddit

        if not self.client_id or not self.client_secret:
            return None

        try:
            if self.username and self.password:
                self.reddit = praw.Reddit(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    user_agent=self.user_agent,
                    username=self.username,
                    password=self.password,
                )
            else:
                self.reddit = praw.Reddit(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    user_agent=self.user_agent,
                )
            return self.reddit
        except Exception as e:
            logger.error(f"Failed to initialize Reddit: {e}")
            return None


class RedditGetUser(RedditBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="reddit_get_user",
            description="Get information about a Reddit user.",
            inputSchema={
                "type": "object",
                "properties": {
                    "username": {"type": "string"},
                },
                "required": ["username"],
            },
            function=self._get_user_info,
        )

    def _get_user_info(self, username: str) -> str:
        reddit = self._get_reddit()
        if not reddit:
            return "Reddit credentials missing"

        try:
            user = reddit.redditor(username)
            info = {
                "name": user.name,
                "comment_karma": user.comment_karma,
                "link_karma": user.link_karma,
                "created_utc": user.created_utc,
                "is_mod": user.is_mod,
            }
            return json.dumps(info, indent=2)
        except Exception as e:
            return f"Error: {e}"


class RedditGetSubreddit(RedditBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="reddit_get_subreddit",
            description="Get information about a subreddit.",
            inputSchema={
                "type": "object",
                "properties": {
                    "subreddit": {"type": "string"},
                },
                "required": ["subreddit"],
            },
            function=self._get_subreddit_info,
        )

    def _get_subreddit_info(self, subreddit: str) -> str:
        reddit = self._get_reddit()
        if not reddit:
            return "Reddit credentials missing"

        try:
            sub = reddit.subreddit(subreddit)
            info = {
                "display_name": sub.display_name,
                "title": sub.title,
                "description": sub.public_description,
                "subscribers": sub.subscribers,
                "created_utc": sub.created_utc,
            }
            return json.dumps(info, indent=2)
        except Exception as e:
            return f"Error: {e}"


class RedditGetPosts(RedditBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="reddit_get_posts",
            description="Get top posts from a subreddit.",
            inputSchema={
                "type": "object",
                "properties": {
                    "subreddit": {"type": "string"},
                    "time_filter": {
                        "type": "string",
                        "default": "week",
                        "enum": ["all", "day", "hour", "month", "week", "year"],
                    },
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["subreddit"],
            },
            function=self._get_posts,
        )

    def _get_posts(
        self, subreddit: str, time_filter: str = "week", limit: int = 10
    ) -> str:
        reddit = self._get_reddit()
        if not reddit:
            return "Reddit credentials missing"

        try:
            posts = reddit.subreddit(subreddit).top(
                time_filter=time_filter, limit=limit
            )
            data = [
                {
                    "id": p.id,
                    "title": p.title,
                    "score": p.score,
                    "url": p.url,
                    "author": str(p.author),
                    "permalink": p.permalink,
                    "created_utc": p.created_utc,
                }
                for p in posts
            ]
            return json.dumps(data, indent=2)
        except Exception as e:
            return f"Error: {e}"


class RedditCreatePost(RedditBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="reddit_create_post",
            description="Create a new post in a subreddit.",
            inputSchema={
                "type": "object",
                "properties": {
                    "subreddit": {"type": "string"},
                    "title": {"type": "string"},
                    "content": {"type": "string", "description": "Text body or URL"},
                    "is_self": {"type": "boolean", "default": True},
                },
                "required": ["subreddit", "title", "content"],
            },
            function=self._create_post,
        )

    def _create_post(
        self, subreddit: str, title: str, content: str, is_self: bool = True
    ) -> str:
        reddit = self._get_reddit()
        if not reddit:
            return "Reddit credentials missing"
        if not self.username:
            return "User auth required"

        try:
            sub = reddit.subreddit(subreddit)
            if is_self:
                submission = sub.submit(title=title, selftext=content)
            else:
                submission = sub.submit(title=title, url=content)

            return json.dumps(
                {
                    "id": submission.id,
                    "url": submission.url,
                    "permalink": submission.permalink,
                },
                indent=2,
            )
        except Exception as e:
            return f"Error: {e}"


class RedditReply(RedditBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="reddit_reply",
            description="Reply to a post or comment.",
            inputSchema={
                "type": "object",
                "properties": {
                    "thing_id": {
                        "type": "string",
                        "description": "ID of post or comment",
                    },
                    "content": {"type": "string"},
                },
                "required": ["thing_id", "content"],
            },
            function=self._reply,
        )

    def _reply(self, thing_id: str, content: str) -> str:
        reddit = self._get_reddit()
        if not reddit:
            return "Reddit credentials missing"
        if not self.username:
            return "User auth required"

        try:
            # Determine if it's a submission or comment based on prefix if available, otherwise try both
            # Simple heuristic: try to fetch as submission first
            try:
                submission = reddit.submission(id=thing_id)
                # Accessing title to force fetch
                _ = submission.title
                reply = submission.reply(body=content)
            except:
                comment = reddit.comment(id=thing_id)
                reply = comment.reply(body=content)

            return json.dumps(
                {"id": reply.id, "permalink": reply.permalink, "body": reply.body},
                indent=2,
            )

        except Exception as e:
            return f"Error: {e}"
