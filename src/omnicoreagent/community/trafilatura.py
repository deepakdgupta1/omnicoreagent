from typing import Any, Dict, List, Optional

from omnicoreagent.core.tools.local_tools_registry import Tool

try:
    from trafilatura import extract, extract_metadata, fetch_url, html2txt
    from trafilatura.meta import reset_caches

    try:
        from trafilatura.spider import focused_crawler

        SPIDER_AVAILABLE = True
    except ImportError:
        focused_crawler = None
        SPIDER_AVAILABLE = False
except ImportError:
    extract = None
    extract_metadata = None
    fetch_url = None
    html2txt = None
    reset_caches = None
    focused_crawler = None
    SPIDER_AVAILABLE = False


class TrafilaturaTools:
    def __init__(
        self,
        output_format: str = "txt",
        include_comments: bool = True,
        include_tables: bool = True,
        include_images: bool = False,
        include_formatting: bool = False,
        include_links: bool = False,
        with_metadata: bool = False,
        favor_precision: bool = False,
        favor_recall: bool = False,
        target_language: Optional[str] = None,
        deduplicate: bool = False,
        max_tree_size: Optional[int] = None,
        max_crawl_urls: int = 10,
        max_known_urls: int = 100000,
    ):
        if extract is None:
            raise ImportError(
                "`trafilatura` not installed. Please install using `pip install trafilatura`"
            )
        self.output_format = output_format
        self.include_comments = include_comments
        self.include_tables = include_tables
        self.include_images = include_images
        self.include_formatting = include_formatting
        self.include_links = include_links
        self.with_metadata = with_metadata
        self.favor_precision = favor_precision
        self.favor_recall = favor_recall
        self.target_language = target_language
        self.deduplicate = deduplicate
        self.max_tree_size = max_tree_size
        self.max_crawl_urls = max_crawl_urls
        self.max_known_urls = max_known_urls

    def _get_extraction_params(self, **overrides) -> Dict[str, Any]:
        params = {
            "output_format": overrides.get("output_format") or self.output_format,
            "include_comments": self.include_comments,
            "include_tables": self.include_tables,
            "include_images": self.include_images,
            "include_formatting": self.include_formatting,
            "include_links": self.include_links,
            "with_metadata": self.with_metadata,
            "favor_precision": self.favor_precision,
            "favor_recall": self.favor_recall,
            "target_language": self.target_language,
            "deduplicate": self.deduplicate,
            "max_tree_size": self.max_tree_size,
        }
        return params

    def get_tool(self) -> Tool:
        return Tool(
            name="trafilatura_extract_text",
            description="Extract main text content from a web page URL using Trafilatura.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "output_format": {
                        "type": "string",
                        "description": "txt, json, xml, markdown, csv, html",
                    },
                },
                "required": ["url"],
            },
            function=self._extract_text,
        )

    async def _extract_text(
        self, url: str, output_format: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            html_content = fetch_url(url)
            if not html_content:
                return {
                    "status": "error",
                    "data": None,
                    "message": f"Could not fetch {url}",
                }
            params = self._get_extraction_params(output_format=output_format)
            result = extract(html_content, url=url, **params)
            reset_caches()
            if result is None:
                return {
                    "status": "error",
                    "data": None,
                    "message": f"Could not extract from {url}",
                }
            return {"status": "success", "data": result, "message": "Text extracted"}
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class TrafilaturaExtractMetadata(TrafilaturaTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="trafilatura_extract_metadata",
            description="Extract metadata from a web page URL.",
            inputSchema={
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
            function=self._extract_metadata,
        )

    async def _extract_metadata(self, url: str) -> Dict[str, Any]:
        try:
            html_content = fetch_url(url)
            if not html_content:
                return {
                    "status": "error",
                    "data": None,
                    "message": f"Could not fetch {url}",
                }
            metadata_doc = extract_metadata(
                html_content, default_url=url, extensive=True
            )
            reset_caches()
            if metadata_doc is None:
                return {
                    "status": "error",
                    "data": None,
                    "message": f"No metadata from {url}",
                }
            return {
                "status": "success",
                "data": metadata_doc.as_dict(),
                "message": "Metadata extracted",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class TrafilaturaHtmlToText(TrafilaturaTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="trafilatura_html_to_text",
            description="Convert HTML content to plain text.",
            inputSchema={
                "type": "object",
                "properties": {
                    "html_content": {"type": "string"},
                    "clean": {"type": "boolean", "default": True},
                },
                "required": ["html_content"],
            },
            function=self._html_to_text,
        )

    async def _html_to_text(
        self, html_content: str, clean: bool = True
    ) -> Dict[str, Any]:
        try:
            result = html2txt(html_content, clean=clean)
            reset_caches()
            if not result:
                return {
                    "status": "error",
                    "data": None,
                    "message": "Could not extract text from HTML",
                }
            return {
                "status": "success",
                "data": result,
                "message": "HTML converted to text",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class TrafilaturaBatchExtract(TrafilaturaTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="trafilatura_batch_extract",
            description="Extract content from multiple URLs in batch.",
            inputSchema={
                "type": "object",
                "properties": {
                    "urls": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["urls"],
            },
            function=self._extract_batch,
        )

    async def _extract_batch(self, urls: List[str]) -> Dict[str, Any]:
        try:
            results = {}
            failed = []
            params = self._get_extraction_params()
            for url in urls:
                try:
                    html = fetch_url(url)
                    if html:
                        content = extract(html, url=url, **params)
                        if content:
                            results[url] = content
                        else:
                            failed.append(url)
                    else:
                        failed.append(url)
                except Exception:
                    failed.append(url)
            reset_caches()
            data = {
                "successful": len(results),
                "failed": len(failed),
                "results": results,
                "failed_urls": failed,
            }
            return {
                "status": "success",
                "data": data,
                "message": f"Extracted {len(results)}/{len(urls)} URLs",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class TrafilaturaCrawl(TrafilaturaTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="trafilatura_crawl_website",
            description="Crawl a website and discover links.",
            inputSchema={
                "type": "object",
                "properties": {
                    "homepage_url": {"type": "string"},
                    "extract_content": {"type": "boolean", "default": False},
                },
                "required": ["homepage_url"],
            },
            function=self._crawl,
        )

    async def _crawl(
        self, homepage_url: str, extract_content: bool = False
    ) -> Dict[str, Any]:
        if not SPIDER_AVAILABLE:
            return {
                "status": "error",
                "data": None,
                "message": "Trafilatura spider module not available",
            }
        try:
            to_visit, known_links = focused_crawler(
                homepage=homepage_url,
                max_seen_urls=self.max_crawl_urls,
                max_known_urls=self.max_known_urls,
                lang=self.target_language,
            )
            data: Dict[str, Any] = {
                "to_visit": list(to_visit or []),
                "known_links": list(known_links or []),
            }
            if extract_content and known_links:
                extracted = {}
                params = self._get_extraction_params()
                for url in list(known_links)[:10]:
                    try:
                        html = fetch_url(url)
                        if html:
                            content = extract(html, url=url, **params)
                            if content:
                                extracted[url] = content
                    except Exception:
                        pass
                data["extracted_content"] = extracted
            reset_caches()
            return {
                "status": "success",
                "data": data,
                "message": f"Crawled {homepage_url}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}
