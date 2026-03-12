import functools
import runpy
from pathlib import Path
from typing import Optional, Tuple

from omnicoreagent.core.tools.local_tools_registry import Tool
from omnicoreagent.core.utils import log_debug, log_error, log_info, logger


@functools.lru_cache(maxsize=None)
def warn() -> None:
    logger.warning(
        "PythonTools can run arbitrary code, please provide human supervision."
    )


class PythonBase:
    def __init__(
        self,
        base_dir: Optional[Path] = None,
        safe_globals: Optional[dict] = None,
        safe_locals: Optional[dict] = None,
        restrict_to_base_dir: bool = True,
    ):
        self.base_dir: Path = (base_dir or Path.cwd()).resolve()
        self.restrict_to_base_dir = restrict_to_base_dir
        self.safe_globals: dict = safe_globals or globals()
        self.safe_locals: dict = safe_locals or locals()

    def _check_path(self, file_name: str) -> Tuple[bool, Path]:
        try:
            file_path = (self.base_dir / file_name).resolve()
            if self.restrict_to_base_dir and not str(file_path).startswith(
                str(self.base_dir)
            ):
                return False, file_path
            return True, file_path
        except Exception:
            return False, Path(file_name)


class PythonSaveAndRun(PythonBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="python_save_and_run",
            description="Save Python code to a file and run it.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "The name of the file to save code to.",
                    },
                    "code": {
                        "type": "string",
                        "description": "The code to save and run.",
                    },
                    "variable_to_return": {
                        "type": "string",
                        "description": "The variable to return after execution.",
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": "Overwrite the file if it already exists.",
                        "default": True,
                    },
                },
                "required": ["file_name", "code"],
            },
            function=self._save_to_file_and_run,
        )

    def _save_to_file_and_run(
        self,
        file_name: str,
        code: str,
        variable_to_return: Optional[str] = None,
        overwrite: bool = True,
    ) -> str:
        try:
            warn()
            safe, file_path = self._check_path(file_name)
            if not safe:
                return (
                    f"Error: Path '{file_name}' is outside the allowed base directory"
                )

            log_debug(f"Saving code to {file_path}")
            if not file_path.parent.exists():
                file_path.parent.mkdir(parents=True, exist_ok=True)
            if file_path.exists() and not overwrite:
                return f"File {file_name} already exists"

            file_path.write_text(code, encoding="utf-8")
            log_info(f"Saved: {file_path}")
            log_info(f"Running {file_path}")

            globals_after_run = runpy.run_path(
                str(file_path), init_globals=self.safe_globals, run_name="__main__"
            )

            if variable_to_return:
                variable_value = globals_after_run.get(variable_to_return)
                return (
                    str(variable_value)
                    if variable_value is not None
                    else f"Variable {variable_to_return} not found"
                )

            return f"successfully ran {str(file_path)}"
        except Exception as e:
            logger.error(f"Error saving and running code: {e}")
            return f"Error saving and running code: {e}"


class PythonRunFile(PythonBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="python_run_file",
            description="Run code in a Python file.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "The name of the file to run.",
                    },
                    "variable_to_return": {
                        "type": "string",
                        "description": "The variable to return after execution.",
                    },
                },
                "required": ["file_name"],
            },
            function=self._run_python_file,
        )

    def _run_python_file(
        self, file_name: str, variable_to_return: Optional[str] = None
    ) -> str:
        try:
            warn()
            safe, file_path = self._check_path(file_name)
            if not safe:
                return (
                    f"Error: Path '{file_name}' is outside the allowed base directory"
                )

            log_info(f"Running {file_path}")
            globals_after_run = runpy.run_path(
                str(file_path), init_globals=self.safe_globals, run_name="__main__"
            )

            if variable_to_return:
                variable_value = globals_after_run.get(variable_to_return)
                return (
                    str(variable_value)
                    if variable_value is not None
                    else f"Variable {variable_to_return} not found"
                )

            return f"successfully ran {str(file_path)}"
        except Exception as e:
            logger.error(f"Error running file: {e}")
            return f"Error running file: {e}"


class PythonReadFile(PythonBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="python_read_file",
            description="Read content from a file.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "The name of the file to read.",
                    },
                },
                "required": ["file_name"],
            },
            function=self._read_file,
        )

    def _read_file(self, file_name: str) -> str:
        try:
            log_info(f"Reading file: {file_name}")
            safe, file_path = self._check_path(file_name)
            if not safe:
                log_error(f"Attempted to read file outside base directory: {file_name}")
                return "Error reading file: path outside allowed directory"
            return str(file_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return f"Error reading file: {e}"


class PythonListFiles(PythonBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="python_list_files",
            description="List files in the base directory.",
            inputSchema={"type": "object", "properties": {}},
            function=self._list_files,
        )

    def _list_files(self) -> str:
        try:
            log_info(f"Reading files in : {self.base_dir}")
            files = [str(file_path.name) for file_path in self.base_dir.iterdir()]
            return ", ".join(files)
        except Exception as e:
            logger.error(f"Error reading files: {e}")
            return f"Error reading files: {e}"


class PythonRunCode(PythonBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="python_run_code",
            description="Run Python code in the current environment.",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "The code to run."},
                    "variable_to_return": {
                        "type": "string",
                        "description": "The variable to return.",
                    },
                },
                "required": ["code"],
            },
            function=self._run_python_code,
        )

    def _run_python_code(
        self, code: str, variable_to_return: Optional[str] = None
    ) -> str:
        try:
            warn()
            log_debug(f"Running code:\n\n{code}\n\n")
            exec(code, self.safe_globals, self.safe_locals)

            if variable_to_return:
                variable_value = self.safe_locals.get(variable_to_return)
                return (
                    str(variable_value)
                    if variable_value is not None
                    else f"Variable {variable_to_return} not found"
                )

            return "successfully ran python code"
        except Exception as e:
            logger.error(f"Error running python code: {e}")
            return f"Error running python code: {e}"


class PythonPipInstall:
    def get_tool(self) -> Tool:
        return Tool(
            name="python_pip_install",
            description="Install a package using pip.",
            inputSchema={
                "type": "object",
                "properties": {
                    "package_name": {
                        "type": "string",
                        "description": "The name of the package to install.",
                    },
                },
                "required": ["package_name"],
            },
            function=self._pip_install_package,
        )

    def _pip_install_package(self, package_name: str) -> str:
        try:
            warn()
            log_debug(f"Installing package {package_name}")
            import subprocess
            import sys

            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package_name]
            )
            return f"successfully installed package {package_name}"
        except Exception as e:
            logger.error(f"Error installing package {package_name}: {e}")
            return f"Error installing package {package_name}: {e}"
