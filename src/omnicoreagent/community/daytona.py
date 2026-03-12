from os import getenv
from pathlib import Path
from typing import Any, Dict, Optional

from omnicoreagent.core.tools.local_tools_registry import Tool
from omnicoreagent.core.utils import prepare_python_code
from omnicoreagent.core.utils import log_debug, log_info, log_warning

try:
    from daytona import (
        CodeLanguage,
        CreateSandboxFromSnapshotParams,
        Daytona,
        DaytonaConfig,
        Sandbox,
    )
except ImportError:
    CodeLanguage = None
    CreateSandboxFromSnapshotParams = None
    Daytona = None
    DaytonaConfig = None
    Sandbox = None


class DaytonaTools:
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        sandbox_id: Optional[str] = None,
        sandbox_language: Optional[CodeLanguage] = None,
        sandbox_target: Optional[str] = None,
        sandbox_os: Optional[str] = None,
        auto_stop_interval: Optional[int] = 60,
        sandbox_os_user: Optional[str] = None,
        sandbox_env_vars: Optional[Dict[str, str]] = None,
        sandbox_labels: Optional[Dict[str, str]] = None,
        sandbox_public: Optional[bool] = None,
        organization_id: Optional[str] = None,
        timeout: int = 300,
        auto_create_sandbox: bool = True,
        verify_ssl: Optional[bool] = False,
        persistent: bool = True,
    ):
        if Daytona is None:
            raise ImportError(
                "Could not import `daytona-sdk` python package. "
                "Please install it using `pip install daytona-sdk`."
            )
        self.api_key = api_key or getenv("DAYTONA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DAYTONA_API_KEY not set. Please set the DAYTONA_API_KEY environment variable."
            )

        self.api_url = api_url or getenv("DAYTONA_API_URL")
        self.sandbox_id = sandbox_id
        self.sandbox_target = sandbox_target
        self.organization_id = organization_id
        self.sandbox_language = sandbox_language or CodeLanguage.PYTHON
        self.sandbox_os = sandbox_os
        self.auto_stop_interval = auto_stop_interval
        self.sandbox_os_user = sandbox_os_user
        self.sandbox_env_vars = sandbox_env_vars
        self.sandbox_labels = sandbox_labels or {}
        self.sandbox_public = sandbox_public
        self.timeout = timeout
        self.auto_create_sandbox = auto_create_sandbox
        self.persistent = persistent
        self.verify_ssl = verify_ssl

        if not self.verify_ssl:
            self._disable_ssl_verification()

        self.config = DaytonaConfig(
            api_key=self.api_key,
            api_url=self.api_url,
            target=self.sandbox_target,
            organization_id=self.organization_id,
        )
        self.daytona = Daytona(self.config)
        self._sandbox = None

    def _disable_ssl_verification(self) -> None:
        try:
            from daytona_api_client import Configuration

            original_init = Configuration.__init__

            def patched_init(self, *args, **kwargs):
                original_init(self, *args, **kwargs)
                self.verify_ssl = False

            setattr(Configuration, "__init__", patched_init)
            import urllib3

            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            log_debug("SSL certificate verification is disabled")
        except ImportError:
            log_warning(
                "Could not import daytona_api_client.Configuration for SSL patching"
            )

    def _get_or_create_sandbox(self) -> "Sandbox":
        if self._sandbox is not None:
            return self._sandbox

        sandbox = None
        if self.sandbox_id:
            try:
                sandbox = self.daytona.get(self.sandbox_id)
                log_debug(f"Using explicit sandbox: {self.sandbox_id}")
            except Exception as e:
                log_debug(f"Failed to get sandbox {self.sandbox_id}: {e}")

        if sandbox is None:
            labels = self.sandbox_labels.copy()
            labels.setdefault("created_by", "omnicoreagent_daytona_toolkit")
            labels.setdefault("language", str(self.sandbox_language))
            if self.persistent:
                labels.setdefault("persistent", "true")

            params = CreateSandboxFromSnapshotParams(
                language=self.sandbox_language,
                os_user=self.sandbox_os_user,
                env_vars=self.sandbox_env_vars,
                auto_stop_interval=self.auto_stop_interval,
                labels=labels,
                public=self.sandbox_public,
            )
            sandbox = self.daytona.create(params, timeout=self.timeout)
            log_info(f"Created new Daytona sandbox: {sandbox.id}")

        if sandbox.state != "started":
            log_info(f"Starting sandbox {sandbox.id}")
            self.daytona.start(sandbox, timeout=self.timeout)

        self._sandbox = sandbox
        return sandbox

    def get_tool(self) -> Tool:
        return Tool(
            name="daytona_run_code",
            description="Execute code in a persistent Daytona sandbox.",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Code to execute"},
                },
                "required": ["code"],
            },
            function=self._run_code,
        )

    async def _run_code(self, code: str) -> Dict[str, Any]:
        try:
            sandbox = self._get_or_create_sandbox()
            if self.sandbox_language == CodeLanguage.PYTHON:
                code = prepare_python_code(code)
            response = sandbox.process.code_run(code)
            return {
                "status": "success",
                "data": response.result,
                "message": "Code executed",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class DaytonaRunShellCommand(DaytonaTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="daytona_run_shell_command",
            description="Execute a shell command in the Daytona sandbox.",
            inputSchema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute",
                    },
                    "cwd": {"type": "string", "description": "Working directory"},
                },
                "required": ["command"],
            },
            function=self._run_shell,
        )

    async def _run_shell(
        self, command: str, cwd: str = "/home/daytona"
    ) -> Dict[str, Any]:
        try:
            sandbox = self._get_or_create_sandbox()
            response = sandbox.process.exec(command, cwd=cwd)
            return {
                "status": "success",
                "data": response.result,
                "message": "Command executed",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class DaytonaCreateFile(DaytonaTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="daytona_create_file",
            description="Create or update a file in the Daytona sandbox.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["file_path", "content"],
            },
            function=self._create_file,
        )

    async def _create_file(self, file_path: str, content: str) -> Dict[str, Any]:
        try:
            sandbox = self._get_or_create_sandbox()
            path = Path(file_path)
            path_str = str(path)

            parent_dir = str(path.parent)
            if parent_dir and parent_dir != "/":
                sandbox.process.exec(f"mkdir -p {parent_dir}")

            escaped_content = content.replace("'", "'\"'\"'")
            command = f"cat > '{path_str}' << 'EOF'\n{escaped_content}\nEOF"
            result = sandbox.process.exec(command)

            if result.exit_code != 0:
                return {
                    "status": "error",
                    "data": None,
                    "message": f"Failed: {result.result}",
                }

            return {
                "status": "success",
                "data": path_str,
                "message": f"File created: {path_str}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class DaytonaReadFile(DaytonaTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="daytona_read_file",
            description="Read a file from the Daytona sandbox.",
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
            sandbox = self._get_or_create_sandbox()
            result = sandbox.process.exec(f"cat '{file_path}'")
            if result.exit_code != 0:
                return {
                    "status": "error",
                    "data": None,
                    "message": f"Error: {result.result}",
                }
            return {
                "status": "success",
                "data": result.result,
                "message": "File read successfully",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class DaytonaListFiles(DaytonaTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="daytona_list_files",
            description="List files in a directory in the Daytona sandbox.",
            inputSchema={
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "default": "/home/daytona"},
                },
            },
            function=self._list_files,
        )

    async def _list_files(self, directory: str = "/home/daytona") -> Dict[str, Any]:
        try:
            sandbox = self._get_or_create_sandbox()
            result = sandbox.process.exec(f"ls -la '{directory}'")
            if result.exit_code != 0:
                return {
                    "status": "error",
                    "data": None,
                    "message": f"Error: {result.result}",
                }
            return {
                "status": "success",
                "data": result.result,
                "message": f"Contents of {directory}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class DaytonaDeleteFile(DaytonaTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="daytona_delete_file",
            description="Delete a file or directory from the Daytona sandbox.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                },
                "required": ["file_path"],
            },
            function=self._delete_file,
        )

    async def _delete_file(self, file_path: str) -> Dict[str, Any]:
        try:
            sandbox = self._get_or_create_sandbox()
            result = sandbox.process.exec(f"rm -rf '{file_path}'")
            if result.exit_code != 0:
                return {
                    "status": "error",
                    "data": None,
                    "message": f"Failed: {result.result}",
                }
            return {
                "status": "success",
                "data": file_path,
                "message": f"Deleted: {file_path}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}
