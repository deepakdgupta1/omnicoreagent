from typing import Any, Dict, Optional

from omnicoreagent.core.tools.local_tools_registry import Tool
from omnicoreagent.core.utils import log_debug, logger

try:
    import newspaper
except ImportError:
    newspaper = None


class NewsArticleRead:
    """Tool for reading news articles using newspaper4k."""

    def __init__(
        self,
        include_summary: bool = False,
        article_length: Optional[int] = None,
    ):
        if newspaper is None:
            raise ImportError(
                "`newspaper4k` not installed. Please install using `pip install newspaper4k`"
            )
        self.include_summary = include_summary
        self.article_length = article_length

    def get_tool(self) -> Tool:
        return Tool(
            name="news_article_read",
            description="Read and extract text from a news article URL.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the news article",
                    },
                },
                "required": ["url"],
            },
            function=self._read_article,
        )

    def _get_article_data(self, url: str) -> Optional[Dict[str, Any]]:
        try:
            article = newspaper.article(url)
            article_data: Dict[str, Any] = {}
            if article.title:
                article_data["title"] = article.title
            if article.authors:
                article_data["authors"] = article.authors
            if article.text:
                article_data["text"] = article.text
            if self.include_summary and article.summary:
                article_data["summary"] = article.summary
            try:
                if article.publish_date:
                    article_data["publish_date"] = article.publish_date.isoformat()
            except Exception:
                pass
            return article_data
        except Exception as e:
            logger.warning(f"Error reading article from {url}: {e}")
            return None

    async def _read_article(self, url: str) -> Dict[str, Any]:
        try:
            log_debug(f"Reading news: {url}")
            article_data = self._get_article_data(url)
            if not article_data:
                return {
                    "status": "error",
                    "data": None,
                    "message": f"Could not read article from {url}",
                }

            if self.article_length and "text" in article_data:
                article_data["text"] = article_data["text"][: self.article_length]

            return {
                "status": "success",
                "data": article_data,
                "message": "Article read successfully",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}
