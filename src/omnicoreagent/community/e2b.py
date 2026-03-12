from os import getenv
from pathlib import Path
from typing import Any, Dict, Optional

from omnicoreagent.core.tools.local_tools_registry import Tool
from omnicoreagent.core.utils import prepare_python_code
from omnicoreagent.core.utils import logger

try:
    from e2b_code_interpreter import Sandbox
except ImportError:
    Sandbox = None


class E2BTools:
    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 300,
        sandbox_options: Optional[Dict[str, Any]] = None,
    ):
        if Sandbox is None:
            raise ImportError(
                "Could not import `e2b-code-interpreter` python package. "
                "Please install it using `pip install e2b-code-interpreter`."
            )
        self.api_key = api_key or getenv("E2B_API_KEY")
        if not self.api_key:
            raise ValueError(
                "E2B_API_KEY not set. Please set the E2B_API_KEY environment variable."
            )

        self.sandbox_options = sandbox_options or {}
        try:
            self.sandbox = Sandbox.create(
                api_key=self.api_key, timeout=timeout, **self.sandbox_options
            )
        except Exception as e:
            logger.error(f"Warning: Could not create sandbox: {e}")
            raise e

        self.last_execution = None
        self.downloaded_files: Dict[int, str] = {}

    def get_tool(self) -> Tool:
        return Tool(
            name="e2b_run_python",
            description="Run Python code in an isolated E2B sandbox.",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                },
                "required": ["code"],
            },
            function=self._run_python_code,
        )

    async def _run_python_code(self, code: str) -> Dict[str, Any]:
        try:
            executable_code = prepare_python_code(code)
            execution = self.sandbox.run_code(executable_code)
            self.last_execution = execution

            if execution.error:
                return {
                    "status": "error",
                    "data": {
                        "name": execution.error.name,
                        "value": execution.error.value,
                        "traceback": execution.error.traceback,
                    },
                    "message": f"Error: {execution.error.name}: {execution.error.value}",
                }

            results = []
            if hasattr(execution, "logs") and execution.logs:
                results.append(f"Logs: {execution.logs}")
            for i, result in enumerate(execution.results):
                if hasattr(result, "text") and result.text:
                    results.append(f"Result {i + 1}: {result.text}")
                elif hasattr(result, "png") and result.png:
                    results.append(f"Result {i + 1}: Generated PNG image")
                elif hasattr(result, "chart") and result.chart:
                    results.append(f"Result {i + 1}: Generated chart")
                else:
                    results.append(f"Result {i + 1}: Output available")

            return {
                "status": "success",
                "data": results if results else "Code executed with no output",
                "message": "Code executed successfully",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class E2BUploadFile(E2BTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="e2b_upload_file",
            description="Upload a file to the E2B sandbox.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "sandbox_path": {"type": "string"},
                },
                "required": ["file_path"],
            },
            function=self._upload_file,
        )

    async def _upload_file(
        self, file_path: str, sandbox_path: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            if not sandbox_path:
                sandbox_path = Path(file_path).name
            with open(file_path, "rb") as f:
                file_in_sandbox = self.sandbox.files.write(sandbox_path, f)
            return {
                "status": "success",
                "data": file_in_sandbox.path,
                "message": "File uploaded",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class E2BDownloadFile(E2BTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="e2b_download_file",
            description="Download a file from the E2B sandbox.",
            inputSchema={
                "type": "object",
                "properties": {
                    "sandbox_path": {"type": "string"},
                    "local_path": {"type": "string"},
                },
                "required": ["sandbox_path"],
            },
            function=self._download_file,
        )

    async def _download_file(
        self, sandbox_path: str, local_path: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            if not local_path:
                local_path = Path(sandbox_path).name
            content = self.sandbox.files.read(sandbox_path)
            with open(local_path, "wb") as f:
                f.write(content)
            return {
                "status": "success",
                "data": local_path,
                "message": "File downloaded",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class E2BRunCommand(E2BTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="e2b_run_command",
            description="Run a shell command in the E2B sandbox.",
            inputSchema={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                },
                "required": ["command"],
            },
            function=self._run_command,
        )

    async def _run_command(self, command: str) -> Dict[str, Any]:
        try:
            result = self.sandbox.commands.run(command)
            output = []
            if hasattr(result, "stdout") and result.stdout:
                output.append(f"STDOUT:\n{result.stdout}")
            if hasattr(result, "stderr") and result.stderr:
                output.append(f"STDERR:\n{result.stderr}")
            return {
                "status": "success",
                "data": "\n".join(output)
                if output
                else "Command executed with no output",
                "message": "Command executed",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class E2BListFiles(E2BTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="e2b_list_files",
            description="List files in the E2B sandbox.",
            inputSchema={
                "type": "object",
                "properties": {
                    "directory_path": {"type": "string", "default": "/"},
                },
            },
            function=self._list_files,
        )

    async def _list_files(self, directory_path: str = "/") -> Dict[str, Any]:
        try:
            files = self.sandbox.files.list(directory_path)
            if not files:
                return {
                    "status": "success",
                    "data": [],
                    "message": f"No files in {directory_path}",
                }
            result = []
            for f in files:
                file_type = "Directory" if f.type == "directory" else "File"
                size = f"{f.size} bytes" if f.size is not None else "Unknown"
                result.append({"name": f.name, "type": file_type, "size": size})
            return {
                "status": "success",
                "data": result,
                "message": f"Contents of {directory_path}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class E2BReadFile(E2BTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="e2b_read_file",
            description="Read a file's content from the E2B sandbox.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                },
                "required": ["file_path"],
            },
            function=self._read_file,
        )

    async def _read_file(self, file_path: str) -> Dict[str, Any]:
        try:
            content = self.sandbox.files.read(file_path)
            if isinstance(content, bytes):
                try:
                    content = content.decode("utf-8")
                except UnicodeDecodeError:
                    return {
                        "status": "success",
                        "data": f"Binary data ({len(content)} bytes)",
                        "message": "File read",
                    }
            return {
                "status": "success",
                "data": content,
                "message": "File read successfully",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class E2BWriteFile(E2BTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="e2b_write_file",
            description="Write content to a file in the E2B sandbox.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["file_path", "content"],
            },
            function=self._write_file,
        )

    async def _write_file(self, file_path: str, content: str) -> Dict[str, Any]:
        try:
            bytes_content = content.encode("utf-8")
            file_info = self.sandbox.files.write(file_path, bytes_content)
            return {
                "status": "success",
                "data": file_info.path,
                "message": "File written",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class E2BGetSandboxStatus(E2BTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="e2b_get_sandbox_status",
            description="Get the current status of the E2B sandbox.",
            inputSchema={"type": "object", "properties": {}},
            function=self._get_status,
        )

    async def _get_status(self) -> Dict[str, Any]:
        try:
            sandbox_id = getattr(self.sandbox, "id", "Unknown")
            return {
                "status": "success",
                "data": {"sandbox_id": sandbox_id},
                "message": "Sandbox status",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class E2BShutdownSandbox(E2BTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="e2b_shutdown_sandbox",
            description="Shutdown the E2B sandbox.",
            inputSchema={"type": "object", "properties": {}},
            function=self._shutdown,
        )

    async def _shutdown(self) -> Dict[str, Any]:
        try:
            self.sandbox.kill()
            return {"status": "success", "data": None, "message": "Sandbox shut down"}
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}
