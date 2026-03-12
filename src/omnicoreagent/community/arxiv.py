from pathlib import Path
from typing import Any, Dict, Optional
from omnicoreagent.core.tools.local_tools_registry import Tool

try:
    import arxiv
except ImportError:
    arxiv = None


class ArxivTool:
    """Arxiv Tool integration."""

    def __init__(
        self,
        download_dir: Optional[Path] = None,
    ):
        self.download_dir: Path = download_dir or Path(__file__).parent.joinpath(
            "arxiv_pdfs"
        )

    def get_tool(self) -> Tool:
        return Tool(
            name="arxiv_search",
            description="Search arXiv for a query and return the top articles.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search arXiv for.",
                    },
                    "num_articles": {
                        "type": "integer",
                        "description": "The number of articles to return.",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
            function=self._search_arxiv,
        )

    async def _search_arxiv(self, query: str, num_articles: int = 10) -> Dict[str, Any]:
        """Search arXiv and return articles."""

    async def _search_arxiv(self, query: str, num_articles: int = 10) -> Dict[str, Any]:
        """Search arXiv and return articles."""
        if arxiv is None:
            return {
                "status": "error",
                "data": None,
                "message": "`arxiv` not installed. Please install using `pip install arxiv`",
            }

        client = arxiv.Client()
        formatted_results = []
        raw_results = []

        try:
            search = arxiv.Search(
                query=query,
                max_results=num_articles,
                sort_by=arxiv.SortCriterion.Relevance,
                sort_order=arxiv.SortOrder.Descending,
            )

            # Arxiv client is synchronous, so we run it directly (or could wrap in executor if blocking)
            # For simplicity in this tool refactor, we keep it sync within async wrapper
            for result in client.results(search):
                article = {
                    "title": result.title,
                    "id": result.get_short_id(),
                    "entry_id": result.entry_id,
                    "authors": [author.name for author in result.authors],
                    "published": result.published.isoformat()
                    if result.published
                    else None,
                    "pdf_url": result.pdf_url,
                    "summary": result.summary,
                }
                raw_results.append(article)
                formatted_results.append(
                    f"Title: {article['title']}\nID: {article['id']}\nAuthors: {', '.join(article['authors'])}\nPDF: {article['pdf_url']}\nSummary: {article['summary'][:200]}...\n"
                )

            return {
                "status": "success",
                "data": raw_results,
                "message": "\n---\n".join(formatted_results)
                if formatted_results
                else "No articles found.",
            }

        except Exception as e:
            return {
                "status": "error",
                "data": None,
                "message": f"Error searching arXiv: {str(e)}",
            }
