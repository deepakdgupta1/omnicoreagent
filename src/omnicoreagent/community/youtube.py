import json
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen

from omnicoreagent.core.tools.local_tools_registry import Tool

try:
    from youtube_transcript_api import YouTubeTranscriptApi
except ImportError:
    YouTubeTranscriptApi = None


class YouTubeTools:
    def __init__(
        self,
        languages: Optional[List[str]] = None,
        proxies: Optional[Dict[str, Any]] = None,
    ):
        if YouTubeTranscriptApi is None:
            raise ImportError(
                "`youtube_transcript_api` not installed. Please install using `pip install youtube_transcript_api`"
            )
        self.languages = languages
        self.proxies = proxies

    @staticmethod
    def get_video_id(url: str) -> Optional[str]:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if hostname == "youtu.be":
            return parsed.path[1:]
        if hostname in ("www.youtube.com", "youtube.com"):
            if parsed.path == "/watch":
                return parse_qs(parsed.query).get("v", [None])[0]
            if parsed.path.startswith(("/embed/", "/v/")):
                return parsed.path.split("/")[2]
        return None

    def get_tool(self) -> Tool:
        return Tool(
            name="youtube_get_captions",
            description="Get captions/transcript from a YouTube video.",
            inputSchema={
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
            function=self._get_captions,
        )

    async def _get_captions(self, url: str) -> Dict[str, Any]:
        if not url:
            return {"status": "error", "data": None, "message": "No URL provided"}
        try:
            video_id = self.get_video_id(url)
            if not video_id:
                return {
                    "status": "error",
                    "data": None,
                    "message": "Could not extract video ID",
                }
            kwargs: Dict[str, Any] = {}
            if self.languages:
                kwargs["languages"] = self.languages
            if self.proxies:
                kwargs["proxies"] = self.proxies
            captions = YouTubeTranscriptApi().fetch(video_id, **kwargs)
            text = " ".join(line.text for line in captions) if captions else ""
            if not text:
                return {"status": "success", "data": "", "message": "No captions found"}
            return {"status": "success", "data": text, "message": "Captions retrieved"}
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class YouTubeGetVideoData(YouTubeTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="youtube_get_video_data",
            description="Get metadata for a YouTube video (title, author, thumbnail, etc).",
            inputSchema={
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
            function=self._get_video_data,
        )

    async def _get_video_data(self, url: str) -> Dict[str, Any]:
        if not url:
            return {"status": "error", "data": None, "message": "No URL provided"}
        try:
            video_id = self.get_video_id(url)
            if not video_id:
                return {
                    "status": "error",
                    "data": None,
                    "message": "Could not extract video ID",
                }
            oembed_url = f"https://www.youtube.com/oembed?format=json&url=https://www.youtube.com/watch?v={video_id}"
            with urlopen(oembed_url) as response:
                video_data = json.loads(response.read().decode())
            clean = {
                "title": video_data.get("title"),
                "author_name": video_data.get("author_name"),
                "author_url": video_data.get("author_url"),
                "thumbnail_url": video_data.get("thumbnail_url"),
            }
            return {
                "status": "success",
                "data": clean,
                "message": "Video data retrieved",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class YouTubeGetTimestamps(YouTubeTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="youtube_get_timestamps",
            description="Generate timestamps from YouTube video captions.",
            inputSchema={
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
            function=self._get_timestamps,
        )

    async def _get_timestamps(self, url: str) -> Dict[str, Any]:
        if not url:
            return {"status": "error", "data": None, "message": "No URL provided"}
        try:
            video_id = self.get_video_id(url)
            if not video_id:
                return {
                    "status": "error",
                    "data": None,
                    "message": "Could not extract video ID",
                }
            kwargs: Dict[str, Any] = {}
            if self.languages:
                kwargs["languages"] = self.languages
            if self.proxies:
                kwargs["proxies"] = self.proxies
            captions = YouTubeTranscriptApi().fetch(video_id, **kwargs)
            timestamps = []
            for line in captions:
                start = int(line.start)
                minutes, seconds = divmod(start, 60)
                timestamps.append(f"{minutes}:{seconds:02d} - {line.text}")
            text = "\n".join(timestamps)
            return {
                "status": "success",
                "data": text,
                "message": "Timestamps generated",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}
