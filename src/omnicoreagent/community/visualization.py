import json
import os
from typing import Any, Dict, List, Optional, Union

from omnicoreagent.core.tools.local_tools_registry import Tool


class VisualizationTools:
    def __init__(self, output_dir: str = "charts"):
        try:
            import matplotlib

            matplotlib.use("Agg")
        except ImportError:
            raise ImportError(
                "matplotlib is not installed. Please install it using: `pip install matplotlib`"
            )

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.output_dir = output_dir

    def _normalize_data(self, data: Union[Dict, List, str]) -> Dict[str, float]:
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                return {"Data": 1.0}

        if isinstance(data, dict):
            return {
                str(k): float(v) if isinstance(v, (int, float)) else 0
                for k, v in data.items()
            }
        elif isinstance(data, list) and data:
            if isinstance(data[0], dict):
                result = {}
                for item in data:
                    if isinstance(item, dict):
                        keys = list(item.keys())
                        if len(keys) >= 2:
                            result[str(item[keys[0]])] = (
                                float(item[keys[1]])
                                if isinstance(item[keys[1]], (int, float))
                                else 0
                            )
                return result
            return {
                f"Item {i + 1}": float(v) if isinstance(v, (int, float)) else 0
                for i, v in enumerate(data)
            }
        return {"Data": 1.0}

    def get_tool(self) -> Tool:
        return Tool(
            name="create_bar_chart",
            description="Create a bar chart from data.",
            inputSchema={
                "type": "object",
                "properties": {
                    "data": {"description": "Dict or list of data points"},
                    "title": {"type": "string", "default": "Bar Chart"},
                    "x_label": {"type": "string", "default": "Categories"},
                    "y_label": {"type": "string", "default": "Values"},
                    "filename": {"type": "string"},
                },
                "required": ["data"],
            },
            function=self._create_bar_chart,
        )

    async def _create_bar_chart(
        self,
        data: Any,
        title: str = "Bar Chart",
        x_label: str = "Categories",
        y_label: str = "Values",
        filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            import matplotlib.pyplot as plt

            norm = self._normalize_data(data)
            plt.figure(figsize=(10, 6))
            plt.bar(list(norm.keys()), list(norm.values()))
            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            fname = filename or f"bar_chart_{len(os.listdir(self.output_dir)) + 1}.png"
            path = os.path.join(self.output_dir, fname)
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()
            return {
                "status": "success",
                "data": {"file_path": path},
                "message": f"Bar chart saved to {path}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class VisualizationLineChart(VisualizationTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="create_line_chart",
            description="Create a line chart from data.",
            inputSchema={
                "type": "object",
                "properties": {
                    "data": {"description": "Dict or list of data points"},
                    "title": {"type": "string", "default": "Line Chart"},
                    "x_label": {"type": "string", "default": "X-axis"},
                    "y_label": {"type": "string", "default": "Y-axis"},
                },
                "required": ["data"],
            },
            function=self._create_line_chart,
        )

    async def _create_line_chart(
        self,
        data: Any,
        title: str = "Line Chart",
        x_label: str = "X-axis",
        y_label: str = "Y-axis",
        filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            import matplotlib.pyplot as plt

            norm = self._normalize_data(data)
            plt.figure(figsize=(10, 6))
            plt.plot(list(norm.keys()), list(norm.values()), marker="o", linewidth=2)
            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.xticks(rotation=45, ha="right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            fname = filename or f"line_chart_{len(os.listdir(self.output_dir)) + 1}.png"
            path = os.path.join(self.output_dir, fname)
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()
            return {
                "status": "success",
                "data": {"file_path": path},
                "message": f"Line chart saved to {path}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class VisualizationPieChart(VisualizationTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="create_pie_chart",
            description="Create a pie chart from data.",
            inputSchema={
                "type": "object",
                "properties": {
                    "data": {"description": "Dict or list of data points"},
                    "title": {"type": "string", "default": "Pie Chart"},
                },
                "required": ["data"],
            },
            function=self._create_pie_chart,
        )

    async def _create_pie_chart(
        self, data: Any, title: str = "Pie Chart", filename: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            import matplotlib.pyplot as plt

            norm = self._normalize_data(data)
            plt.figure(figsize=(10, 8))
            plt.pie(
                list(norm.values()),
                labels=list(norm.keys()),
                autopct="%1.1f%%",
                startangle=90,
            )
            plt.title(title)
            plt.axis("equal")
            fname = filename or f"pie_chart_{len(os.listdir(self.output_dir)) + 1}.png"
            path = os.path.join(self.output_dir, fname)
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()
            return {
                "status": "success",
                "data": {"file_path": path},
                "message": f"Pie chart saved to {path}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class VisualizationScatterPlot(VisualizationTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="create_scatter_plot",
            description="Create a scatter plot.",
            inputSchema={
                "type": "object",
                "properties": {
                    "x_data": {"type": "array", "items": {"type": "number"}},
                    "y_data": {"type": "array", "items": {"type": "number"}},
                    "title": {"type": "string", "default": "Scatter Plot"},
                },
                "required": ["x_data", "y_data"],
            },
            function=self._create_scatter,
        )

    async def _create_scatter(
        self,
        x_data: List[float],
        y_data: List[float],
        title: str = "Scatter Plot",
        x_label: str = "X-axis",
        y_label: str = "Y-axis",
        filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            import matplotlib.pyplot as plt

            if len(x_data) != len(y_data):
                return {
                    "status": "error",
                    "data": None,
                    "message": "x_data and y_data must have same length",
                }
            plt.figure(figsize=(10, 6))
            plt.scatter(x_data, y_data, alpha=0.7, s=50)
            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            fname = filename or f"scatter_{len(os.listdir(self.output_dir)) + 1}.png"
            path = os.path.join(self.output_dir, fname)
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()
            return {
                "status": "success",
                "data": {"file_path": path},
                "message": f"Scatter plot saved to {path}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class VisualizationHistogram(VisualizationTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="create_histogram",
            description="Create a histogram from numeric data.",
            inputSchema={
                "type": "object",
                "properties": {
                    "data": {"type": "array", "items": {"type": "number"}},
                    "bins": {"type": "integer", "default": 10},
                    "title": {"type": "string", "default": "Histogram"},
                },
                "required": ["data"],
            },
            function=self._create_histogram,
        )

    async def _create_histogram(
        self,
        data: List[float],
        bins: int = 10,
        title: str = "Histogram",
        x_label: str = "Values",
        y_label: str = "Frequency",
        filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            import matplotlib.pyplot as plt

            numeric = [float(v) for v in data if isinstance(v, (int, float))]
            if not numeric:
                return {
                    "status": "error",
                    "data": None,
                    "message": "No valid numeric data",
                }
            plt.figure(figsize=(10, 6))
            plt.hist(numeric, bins=bins, alpha=0.7, edgecolor="black")
            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            fname = filename or f"histogram_{len(os.listdir(self.output_dir)) + 1}.png"
            path = os.path.join(self.output_dir, fname)
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()
            return {
                "status": "success",
                "data": {"file_path": path},
                "message": f"Histogram saved to {path}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}
