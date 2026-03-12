from typing import Any, Dict, Optional
from omnicoreagent.core.tools.local_tools_registry import Tool

try:
    import docker
except ImportError:
    docker = None


class DockerBase:
    def __init__(self):
        if docker is None:
            raise ImportError(
                "Could not import `docker` python package. "
                "Please install it using `pip install docker`."
            )

    def _get_client(self):
        try:
            return docker.from_env()
        except Exception as e:
            raise ValueError(f"Docker client error: {e}")


class DockerListContainers(DockerBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="docker_list_containers",
            description="List Docker containers.",
            inputSchema={
                "type": "object",
                "properties": {
                    "all": {"type": "boolean", "default": False},
                },
            },
            function=self._list,
        )

    async def _list(self, all: bool = False) -> Dict[str, Any]:
        try:
            client = self._get_client()
            containers = client.containers.list(all=all)
            data = [
                {
                    "id": c.id[:12],
                    "name": c.name,
                    "status": c.status,
                    "image": str(c.image),
                }
                for c in containers
            ]
            return {
                "status": "success",
                "data": data,
                "message": f"Found {len(data)} containers.",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class DockerRunContainer(DockerBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="docker_run_container",
            description="Run a Docker container.",
            inputSchema={
                "type": "object",
                "properties": {
                    "image": {"type": "string"},
                    "command": {"type": "string"},
                },
                "required": ["image"],
            },
            function=self._run,
        )

    async def _run(self, image: str, command: Optional[str] = None) -> Dict[str, Any]:
        try:
            client = self._get_client()
            container = client.containers.run(image, command, detach=True)
            return {
                "status": "success",
                "data": {"id": container.id},
                "message": f"Container started: {container.id[:12]}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


# Add more classes for other Docker functions as needed (ListImages, etc.)
# For brevity in this batch, implemented core functionality.
