from os import getenv
from typing import Any, Dict, Optional

try:
    import requests
except ImportError:
    requests = None

from omnicoreagent.core.tools.local_tools_registry import Tool
from omnicoreagent.core.utils import logger


class OpenWeatherTools:
    def __init__(
        self,
        api_key: Optional[str] = None,
        units: str = "metric",
    ):
        if requests is None:
            raise ImportError(
                "`requests` not installed. Please install using `pip install requests`"
            )
        self.api_key = api_key or getenv("OPENWEATHER_API_KEY")
        if not self.api_key:
            logger.error("OPENWEATHER_API_KEY not set.")
        self.units = units
        self.base_url = "https://api.openweathermap.org"

    def _make_request(self, url: str, params: Dict) -> Optional[Dict]:
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"API request failed: {e}")
            return None

    def _geocode(self, location: str) -> Optional[Dict]:
        url = f"{self.base_url}/geo/1.0/direct"
        params = {"q": location, "limit": 1, "appid": self.api_key}
        data = self._make_request(url, params)
        if data and len(data) > 0:
            return {
                "lat": data[0]["lat"],
                "lon": data[0]["lon"],
                "name": data[0].get("name", location),
            }
        return None

    def get_tool(self) -> Tool:
        return Tool(
            name="openweather_current",
            description="Get current weather for a location.",
            inputSchema={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name (e.g., 'London', 'New York')",
                    },
                },
                "required": ["location"],
            },
            function=self._get_current_weather,
        )

    async def _get_current_weather(self, location: str) -> Dict[str, Any]:
        try:
            geo = self._geocode(location)
            if not geo:
                return {
                    "status": "error",
                    "data": None,
                    "message": f"Could not geocode '{location}'",
                }

            url = f"{self.base_url}/data/2.5/weather"
            params = {
                "lat": geo["lat"],
                "lon": geo["lon"],
                "appid": self.api_key,
                "units": self.units,
            }
            data = self._make_request(url, params)
            if not data:
                return {
                    "status": "error",
                    "data": None,
                    "message": "Failed to get weather data",
                }
            return {
                "status": "success",
                "data": data,
                "message": f"Weather for {location}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class OpenWeatherForecast(OpenWeatherTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="openweather_forecast",
            description="Get weather forecast for a location.",
            inputSchema={
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "days": {
                        "type": "integer",
                        "default": 5,
                        "description": "Number of days (max 5)",
                    },
                },
                "required": ["location"],
            },
            function=self._get_forecast,
        )

    async def _get_forecast(self, location: str, days: int = 5) -> Dict[str, Any]:
        try:
            geo = self._geocode(location)
            if not geo:
                return {
                    "status": "error",
                    "data": None,
                    "message": f"Could not geocode '{location}'",
                }

            url = f"{self.base_url}/data/2.5/forecast"
            params = {
                "lat": geo["lat"],
                "lon": geo["lon"],
                "appid": self.api_key,
                "units": self.units,
                "cnt": days * 8,
            }
            data = self._make_request(url, params)
            if not data:
                return {
                    "status": "error",
                    "data": None,
                    "message": "Failed to get forecast",
                }
            return {
                "status": "success",
                "data": data,
                "message": f"Forecast for {location}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class OpenWeatherAirPollution(OpenWeatherTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="openweather_air_pollution",
            description="Get air pollution data for a location.",
            inputSchema={
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                },
                "required": ["location"],
            },
            function=self._get_air_pollution,
        )

    async def _get_air_pollution(self, location: str) -> Dict[str, Any]:
        try:
            geo = self._geocode(location)
            if not geo:
                return {
                    "status": "error",
                    "data": None,
                    "message": f"Could not geocode '{location}'",
                }

            url = f"{self.base_url}/data/2.5/air_pollution"
            params = {"lat": geo["lat"], "lon": geo["lon"], "appid": self.api_key}
            data = self._make_request(url, params)
            if not data:
                return {
                    "status": "error",
                    "data": None,
                    "message": "Failed to get air pollution data",
                }
            return {
                "status": "success",
                "data": data,
                "message": f"Air pollution for {location}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}
