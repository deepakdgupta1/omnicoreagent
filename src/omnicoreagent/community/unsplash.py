import json
from os import getenv
from typing import Any, Dict, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from omnicoreagent.core.tools.local_tools_registry import Tool
from omnicoreagent.core.utils import logger


class UnsplashBase:
    base_url = "https://api.unsplash.com"

    def __init__(self, access_key: Optional[str] = None):
        self.access_key = access_key or getenv("UNSPLASH_ACCESS_KEY")
        if not self.access_key:
            logger.warning(
                "No Unsplash API key provided. Set UNSPLASH_ACCESS_KEY environment variable."
            )

    def _make_request(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        if params:
            url = f"{url}?{urlencode(params)}"

        headers = {
            "Authorization": f"Client-ID {self.access_key}",
            "Accept-Version": "v1",
        }

        request = Request(url, headers=headers)
        with urlopen(request) as response:
            return json.loads(response.read().decode())

    def _format_photo(self, photo: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": photo.get("id"),
            "description": photo.get("description") or photo.get("alt_description"),
            "width": photo.get("width"),
            "height": photo.get("height"),
            "urls": photo.get("urls", {}),
            "author": photo.get("user", {}).get("name"),
            "links": photo.get("links", {}),
        }


class UnsplashSearchPhotos(UnsplashBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="unsplash_search_photos",
            description="Search for photos on Unsplash by keyword.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "per_page": {"type": "integer", "default": 10},
                    "page": {"type": "integer", "default": 1},
                    "orientation": {
                        "type": "string",
                        "enum": ["landscape", "portrait", "squarish"],
                    },
                },
                "required": ["query"],
            },
            function=self._search_photos,
        )

    def _search_photos(
        self,
        query: str,
        per_page: int = 10,
        page: int = 1,
        orientation: Optional[str] = None,
    ) -> str:
        if not self.access_key:
            return "No API Key"
        try:
            params = {"query": query, "per_page": per_page, "page": page}
            if orientation:
                params["orientation"] = orientation

            response = self._make_request("/search/photos", params)
            photos = [self._format_photo(p) for p in response.get("results", [])]
            return json.dumps(photos, indent=2)
        except Exception as e:
            return f"Error: {e}"


class UnsplashGetPhoto(UnsplashBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="unsplash_get_photo",
            description="Get detailed information about a specific photo.",
            inputSchema={
                "type": "object",
                "properties": {
                    "photo_id": {"type": "string"},
                },
                "required": ["photo_id"],
            },
            function=self._get_photo,
        )

    def _get_photo(self, photo_id: str) -> str:
        if not self.access_key:
            return "No API Key"
        try:
            photo = self._make_request(f"/photos/{photo_id}")
            return json.dumps(self._format_photo(photo), indent=2)
        except Exception as e:
            return f"Error: {e}"


class UnsplashGetRandomPhoto(UnsplashBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="unsplash_get_random_photo",
            description="Get random photo(s) from Unsplash.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "count": {"type": "integer", "default": 1},
                },
            },
            function=self._get_random_photo,
        )

    def _get_random_photo(self, query: Optional[str] = None, count: int = 1) -> str:
        if not self.access_key:
            return "No API Key"
        try:
            params = {"count": count}
            if query:
                params["query"] = query
            response = self._make_request("/photos/random", params)

            photos = response if isinstance(response, list) else [response]
            return json.dumps([self._format_photo(p) for p in photos], indent=2)
        except Exception as e:
            return f"Error: {e}"


class UnsplashDownloadPhoto(UnsplashBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="unsplash_download_photo",
            description="Trigger a download event for a photo (required by API).",
            inputSchema={
                "type": "object",
                "properties": {
                    "photo_id": {"type": "string"},
                },
                "required": ["photo_id"],
            },
            function=self._download_photo,
        )

    def _download_photo(self, photo_id: str) -> str:
        if not self.access_key:
            return "No API Key"
        try:
            response = self._make_request(f"/photos/{photo_id}/download")
            return json.dumps({"url": response.get("url")})
        except Exception as e:
            return f"Error: {e}"
