from typing import Any, Dict, List, Optional
from xml.etree import ElementTree

try:
    import httpx
except ImportError:
    httpx = None

from omnicoreagent.core.tools.local_tools_registry import Tool
from omnicoreagent.core.utils import log_debug


class PubmedTools:
    def __init__(
        self,
        email: str = "your_email@example.com",
        max_results: int = 10,
        results_expanded: bool = False,
    ):
        if httpx is None:
            raise ImportError(
                "Could not import `httpx` python package. "
                "Please install it with `pip install httpx`."
            )
        self.email = email
        self.max_results = max_results
        self.results_expanded = results_expanded

    def get_tool(self) -> Tool:
        return Tool(
            name="pubmed_search",
            description="Search PubMed for scientific articles.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
            function=self._search_pubmed,
        )

    def _fetch_ids(self, query: str, max_results: int) -> List[str]:
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "email": self.email,
            "retmode": "json",
        }
        resp = httpx.get(url, params=params)
        data = resp.json()
        return data.get("esearchresult", {}).get("idlist", [])

    def _fetch_details(self, ids: List[str]) -> ElementTree.Element:
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {"db": "pubmed", "id": ",".join(ids), "retmode": "xml"}
        resp = httpx.get(url, params=params)
        return ElementTree.fromstring(resp.text)

    def _parse_article(self, article: ElementTree.Element) -> Dict[str, Any]:
        medline = article.find("MedlineCitation")
        if medline is None:
            return {}

        pmid_elem = medline.find("PMID")
        art_elem = medline.find("Article")
        if art_elem is None:
            return {}

        title_elem = art_elem.find("ArticleTitle")
        abstract_elem = art_elem.find("Abstract/AbstractText")

        result: Dict[str, Any] = {
            "pmid": pmid_elem.text if pmid_elem is not None else None,
            "title": title_elem.text if title_elem is not None else None,
        }

        if abstract_elem is not None and abstract_elem.text:
            result["abstract"] = abstract_elem.text

        # Authors
        authors = []
        author_list = art_elem.find("AuthorList")
        if author_list is not None:
            for author in author_list.findall("Author"):
                last = author.find("LastName")
                fore = author.find("ForeName")
                if last is not None and fore is not None:
                    authors.append(f"{fore.text} {last.text}")
        if authors:
            result["authors"] = authors

        return result

    async def _search_pubmed(
        self, query: str, max_results: Optional[int] = None
    ) -> Dict[str, Any]:
        try:
            n = max_results or self.max_results
            log_debug(f"Searching PubMed: {query}")
            ids = self._fetch_ids(query, n)
            if not ids:
                return {"status": "success", "data": [], "message": "No results found"}

            root = self._fetch_details(ids)
            articles = []
            for article in root.findall("PubmedArticle"):
                parsed = self._parse_article(article)
                if parsed:
                    articles.append(parsed)

            return {
                "status": "success",
                "data": articles,
                "message": f"Found {len(articles)} articles",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}
