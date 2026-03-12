from typing import Any, Dict
from omnicoreagent.core.tools.local_tools_registry import Tool

try:
    import wikipedia
except ImportError:
    wikipedia = None


class WikipediaSearchTool:
    """Wikipedia Search Tool integration."""

    """Wikipedia Search Tool integration."""

    def __init__(self):
        if wikipedia is None:
            raise ImportError(
                "Could not import `wikipedia` python package. "
                "Please install it with `pip install wikipedia`."
            )

    def get_tool(self) -> Tool:
        return Tool(
            name="wikipedia_search",
            description="Search Wikipedia for knowledge and definitions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The topic to search for.",
                    },
                    "sentences": {
                        "type": "integer",
                        "description": "Number of sentences to extract.",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
            function=self._search,
        )

    async def _search(self, query: str, sentences: int = 3) -> Dict[str, Any]:
        """Search Wikipedia."""

        try:
            # wikipedia lib is synchronous
            try:
                summary = wikipedia.summary(query, sentences=sentences)
                page = wikipedia.page(query, auto_suggest=False)

                result_data = {"title": page.title, "url": page.url, "summary": summary}

                message = f"Title: {page.title}\nURL: {page.url}\nSummary: {summary}\n"

                return {"status": "success", "data": result_data, "message": message}
            except wikipedia.DisambiguationError as e:
                return {
                    "status": "error",
                    "data": {"options": e.options},
                    "message": f"Ambiguous query. Options: {', '.join(e.options[:5])}",
                }
            except wikipedia.PageError:
                return {"status": "error", "data": None, "message": "Page not found."}

        except Exception as e:
            return {
                "status": "error",
                "data": None,
                "message": f"Error searching Wikipedia: {str(e)}",
            }
