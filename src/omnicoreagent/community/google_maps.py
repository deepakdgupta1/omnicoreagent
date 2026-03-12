"""Google Maps Tools - Search, directions, geocoding, and more."""

from os import getenv
from typing import Any, Dict, List, Optional

from omnicoreagent.core.tools.local_tools_registry import Tool

try:
    import googlemaps
    from google.maps import places_v1
except ImportError:
    googlemaps = None
    places_v1 = None


class GoogleMapTools:
    def __init__(self, key: Optional[str] = None):
        self.api_key = key or getenv("GOOGLE_MAPS_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_MAPS_API_KEY is not set.")
        if googlemaps is None:
            raise ImportError(
                "googlemaps not installed. pip install googlemaps google-maps-places"
            )
        self.client = googlemaps.Client(key=self.api_key)
        self.places_client = places_v1.PlacesClient() if places_v1 else None

    def get_tool(self) -> Tool:
        return Tool(
            name="google_maps_search_places",
            description="Search for places using Google Maps Places API.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g., 'restaurants in NYC')",
                    },
                },
                "required": ["query"],
            },
            function=self._search_places,
        )

    async def _search_places(self, query: str) -> Dict[str, Any]:
        if not self.places_client:
            return {
                "status": "error",
                "data": None,
                "message": "google-maps-places not installed",
            }
        try:
            request = places_v1.SearchTextRequest(text_query=query)
            response = self.places_client.search_text(
                request=request, metadata=[("x-goog-fieldmask", "*")]
            )
            places = []
            for place in response.places:
                place_info = {
                    "name": place.display_name.text,
                    "address": place.formatted_address,
                    "rating": place.rating,
                    "place_id": place.id,
                    "phone": place.international_phone_number,
                    "website": place.website_uri,
                }
                places.append(place_info)
            return {
                "status": "success",
                "data": places,
                "message": f"Found {len(places)} places",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class GoogleMapsDirections(GoogleMapTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="google_maps_get_directions",
            description="Get directions between two locations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "mode": {
                        "type": "string",
                        "enum": ["driving", "walking", "bicycling", "transit"],
                        "default": "driving",
                    },
                },
                "required": ["origin", "destination"],
            },
            function=self._get_directions,
        )

    async def _get_directions(
        self, origin: str, destination: str, mode: str = "driving"
    ) -> Dict[str, Any]:
        try:
            result = self.client.directions(origin, destination, mode=mode)
            return {
                "status": "success",
                "data": result,
                "message": "Directions retrieved",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class GoogleMapsGeocode(GoogleMapTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="google_maps_geocode",
            description="Convert an address into geographic coordinates.",
            inputSchema={
                "type": "object",
                "properties": {
                    "address": {"type": "string"},
                },
                "required": ["address"],
            },
            function=self._geocode,
        )

    async def _geocode(self, address: str) -> Dict[str, Any]:
        try:
            result = self.client.geocode(address)
            return {
                "status": "success",
                "data": result,
                "message": "Geocoded successfully",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class GoogleMapsReverseGeocode(GoogleMapTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="google_maps_reverse_geocode",
            description="Convert coordinates into an address.",
            inputSchema={
                "type": "object",
                "properties": {
                    "lat": {"type": "number"},
                    "lng": {"type": "number"},
                },
                "required": ["lat", "lng"],
            },
            function=self._reverse_geocode,
        )

    async def _reverse_geocode(self, lat: float, lng: float) -> Dict[str, Any]:
        try:
            result = self.client.reverse_geocode((lat, lng))
            return {"status": "success", "data": result, "message": "Reverse geocoded"}
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class GoogleMapsDistanceMatrix(GoogleMapTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="google_maps_distance_matrix",
            description="Calculate distance and time between origins and destinations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "origins": {"type": "array", "items": {"type": "string"}},
                    "destinations": {"type": "array", "items": {"type": "string"}},
                    "mode": {"type": "string", "default": "driving"},
                },
                "required": ["origins", "destinations"],
            },
            function=self._distance_matrix,
        )

    async def _distance_matrix(
        self, origins: List[str], destinations: List[str], mode: str = "driving"
    ) -> Dict[str, Any]:
        try:
            result = self.client.distance_matrix(origins, destinations, mode=mode)
            return {
                "status": "success",
                "data": result,
                "message": "Distance matrix calculated",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}
