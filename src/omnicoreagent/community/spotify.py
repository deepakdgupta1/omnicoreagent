import json
from typing import Any, Dict, List, Optional

try:
    import httpx
except ImportError:
    httpx = None
from omnicoreagent.core.tools.local_tools_registry import Tool
from omnicoreagent.core.utils import log_debug


class SpotifyBase:
    def __init__(
        self,
        access_token: str,
        default_market: Optional[str] = "US",
        timeout: int = 30,
    ):
        if httpx is None:
            raise ImportError(
                "`httpx` not installed. Please install using `pip install httpx`"
            )
        self.access_token = access_token
        self.default_market = default_market
        self.timeout = timeout
        self.base_url = "https://api.spotify.com/v1"

    def _make_request(
        self,
        endpoint: str,
        method: str = "GET",
        body: Optional[dict] = None,
        params: Optional[dict] = None,
    ) -> dict:
        url = f"{self.base_url}/{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=self.timeout) as client:
            response = client.request(
                method=method,
                url=url,
                headers=headers,
                json=body,
                params=params,
            )

            if response.status_code == 204:
                return {"success": True}

            try:
                return response.json()
            except json.JSONDecodeError:
                return {"error": f"Failed to parse response: {response.text}"}


class SpotifySearch(SpotifyBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="spotify_search",
            description="Search for tracks, artists, albums, or playlists on Spotify.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": ["track", "artist", "album", "playlist"],
                        "default": "track",
                    },
                    "max_results": {"type": "integer", "default": 10},
                    "market": {"type": "string", "default": "US"},
                },
                "required": ["query"],
            },
            function=self._search,
        )

    def _search(
        self,
        query: str,
        type: str = "track",
        max_results: int = 10,
        market: Optional[str] = None,
    ) -> str:
        log_debug(f"Searching Spotify for {type}: {query}")
        params = {
            "q": query,
            "type": type,
            "limit": min(max_results, 50),
            "market": market or self.default_market,
        }
        result = self._make_request("search", params=params)

        if "error" in result:
            return json.dumps(result, indent=2)

        data = []
        if type == "track":
            tracks = result.get("tracks", {}).get("items", [])
            data = [
                {
                    "id": t["id"],
                    "name": t["name"],
                    "artists": [a["name"] for a in t["artists"]],
                    "album": t["album"]["name"],
                    "uri": t["uri"],
                    "preview_url": t.get("preview_url"),
                    "popularity": t.get("popularity"),
                }
                for t in tracks
            ]
        elif type == "artist":
            artists = result.get("artists", {}).get("items", [])
            data = [
                {
                    "id": a["id"],
                    "name": a["name"],
                    "genres": a.get("genres", []),
                    "popularity": a.get("popularity"),
                    "uri": a["uri"],
                }
                for a in artists
            ]
        elif type == "album":
            albums = result.get("albums", {}).get("items", [])
            data = [
                {
                    "id": a["id"],
                    "name": a["name"],
                    "artists": [art["name"] for art in a["artists"]],
                    "release_date": a.get("release_date"),
                    "total_tracks": a.get("total_tracks"),
                    "uri": a["uri"],
                }
                for a in albums
            ]
        elif type == "playlist":
            playlists = result.get("playlists", {}).get("items", [])
            data = [
                {
                    "id": p["id"],
                    "name": p["name"],
                    "owner": p["owner"]["display_name"],
                    "track_count": p["tracks"]["total"],
                    "uri": p["uri"],
                }
                for p in playlists
                if p
            ]

        return json.dumps(data, indent=2)


class SpotifyPlay(SpotifyBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="spotify_play",
            description="Start/Resume playback or get currently playing track.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["play", "current"],
                        "default": "play",
                    },
                    "track_uri": {"type": "string"},
                    "context_uri": {"type": "string"},
                    "device_id": {"type": "string"},
                },
            },
            function=self._play_control,
        )

    def _play_control(
        self,
        action: str = "play",
        track_uri: Optional[str] = None,
        context_uri: Optional[str] = None,
        device_id: Optional[str] = None,
    ) -> str:
        if action == "current":
            log_debug("Fetching currently playing track")
            result = self._make_request("me/player/currently-playing")
            if not result or result.get("error"):
                return json.dumps(result or {"message": "Nothing playing"})

            item = result.get("item", {})
            return json.dumps(
                {
                    "is_playing": result.get("is_playing"),
                    "track": item.get("name"),
                    "artist": item.get("artists", [{}])[0].get("name"),
                    "uri": item.get("uri"),
                },
                indent=2,
            )

        # Play action
        log_debug(f"Starting playback: track={track_uri}, context={context_uri}")
        params = {}
        if device_id:
            params["device_id"] = device_id

        body: Dict[str, Any] = {}
        if track_uri:
            body["uris"] = [track_uri]
        if context_uri:
            body["context_uri"] = context_uri

        result = self._make_request(
            "me/player/play",
            method="PUT",
            body=body if body else None,
            params=params if params else None,
        )
        return json.dumps(result, indent=2)


class SpotifyPlaylist(SpotifyBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="spotify_playlist",
            description="Manage playlists (create, add tracks, list user playlists).",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "add", "list_user"],
                        "default": "list_user",
                    },
                    "name": {"type": "string", "description": "For create"},
                    "description": {"type": "string", "description": "For create"},
                    "public": {"type": "boolean", "default": False},
                    "playlist_id": {"type": "string", "description": "For add"},
                    "track_uris": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "For create/add",
                    },
                    "limit": {"type": "integer", "default": 20},
                },
            },
            function=self._manage_playlist,
        )

    def _manage_playlist(
        self,
        action: str = "list_user",
        name: Optional[str] = None,
        description: Optional[str] = None,
        public: bool = False,
        playlist_id: Optional[str] = None,
        track_uris: Optional[List[str]] = None,
        limit: int = 20,
    ) -> str:
        if action == "list_user":
            result = self._make_request("me/playlists", params={"limit": limit})
            items = result.get("items", [])
            data = [
                {"id": p["id"], "name": p["name"], "uri": p["uri"]} for p in items if p
            ]
            return json.dumps(data, indent=2)

        elif action == "create":
            if not name:
                return "Name required for create"
            user = self._make_request("me")
            user_id = user.get("id")
            if not user_id:
                return "Failed to get user ID"

            body = {"name": name, "description": description or "", "public": public}
            playlist = self._make_request(
                f"users/{user_id}/playlists", method="POST", body=body
            )

            if "error" in playlist:
                return json.dumps(playlist)

            if track_uris:
                self._make_request(
                    f"playlists/{playlist['id']}/tracks",
                    method="POST",
                    body={"uris": track_uris[:100]},
                )

            return json.dumps(
                {"status": "created", "id": playlist["id"], "uri": playlist["uri"]}
            )

        elif action == "add":
            if not playlist_id or not track_uris:
                return "playlist_id and track_uris required"
            result = self._make_request(
                f"playlists/{playlist_id}/tracks",
                method="POST",
                body={"uris": track_uris[:100]},
            )
            return json.dumps(result)

        return "Invalid action"


class SpotifyUser(SpotifyBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="spotify_user",
            description="Get user profile and top tracks/artists.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["profile", "top_tracks", "top_artists"],
                        "default": "profile",
                    },
                    "time_range": {"type": "string", "default": "medium_term"},
                    "limit": {"type": "integer", "default": 20},
                },
            },
            function=self._user_info,
        )

    def _user_info(
        self, action: str = "profile", time_range: str = "medium_term", limit: int = 20
    ) -> str:
        if action == "profile":
            return json.dumps(self._make_request("me"), indent=2)

        endpoint = f"me/top/{'tracks' if action == 'top_tracks' else 'artists'}"
        result = self._make_request(
            endpoint, params={"time_range": time_range, "limit": limit}
        )

        items = result.get("items", [])
        data = [{"id": i["id"], "name": i["name"], "uri": i["uri"]} for i in items]
        return json.dumps(data, indent=2)


class SpotifyRecommendations(SpotifyBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="spotify_recommendations",
            description="Get track recommendations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "seed_tracks": {"type": "array", "items": {"type": "string"}},
                    "seed_artists": {"type": "array", "items": {"type": "string"}},
                    "seed_genres": {"type": "array", "items": {"type": "string"}},
                    "limit": {"type": "integer", "default": 20},
                },
            },
            function=self._recommendations,
        )

    def _recommendations(
        self,
        seed_tracks: Optional[List[str]] = None,
        seed_artists: Optional[List[str]] = None,
        seed_genres: Optional[List[str]] = None,
        limit: int = 20,
    ) -> str:
        params: Dict[str, Any] = {"limit": limit}
        if seed_tracks:
            params["seed_tracks"] = ",".join(seed_tracks[:5])
        if seed_artists:
            params["seed_artists"] = ",".join(seed_artists[:5])
        if seed_genres:
            params["seed_genres"] = ",".join(seed_genres[:5])

        result = self._make_request("recommendations", params=params)
        tracks = result.get("tracks", [])
        data = [
            {"name": t["name"], "artist": t["artists"][0]["name"], "uri": t["uri"]}
            for t in tracks
        ]
        return json.dumps(data, indent=2)
