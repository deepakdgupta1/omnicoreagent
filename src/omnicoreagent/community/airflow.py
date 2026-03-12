from pathlib import Path
from typing import Any, Dict, Optional, Union

from omnicoreagent.core.tools.local_tools_registry import Tool
from omnicoreagent.core.utils import log_debug, log_info, logger


class AirflowTools:
    def __init__(
        self,
        dags_dir: Optional[Union[Path, str]] = None,
    ):
        """
        quick start to work with airflow : https://airflow.apache.org/docs/apache-airflow/stable/start.html
        """
        _dags_dir: Optional[Path] = None
        if dags_dir is not None:
            if isinstance(dags_dir, str):
                _dags_dir = Path.cwd().joinpath(dags_dir)
            else:
                _dags_dir = dags_dir
        self.dags_dir: Path = _dags_dir or Path.cwd()

    def get_tool(self) -> Tool:
        return Tool(
            name="save_dag_file",
            description="Saves python code for an Airflow DAG to a file called `dag_file` and returns the file path if successful.",
            inputSchema={
                "type": "object",
                "properties": {
                    "contents": {
                        "type": "string",
                        "description": "The contents of the DAG.",
                    },
                    "dag_file": {
                        "type": "string",
                        "description": "The name of the file to save to.",
                    },
                },
                "required": ["contents", "dag_file"],
            },
            function=self._save_dag_file,
        )

    async def _save_dag_file(self, contents: str, dag_file: str) -> Dict[str, Any]:
        """Saves python code for an Airflow DAG to a file called `dag_file` and returns the file path if successful."""
        try:
            file_path = self.dags_dir.joinpath(dag_file)
            log_debug(f"Saving contents to {file_path}")
            if not file_path.parent.exists():
                file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(contents)
            log_info(f"Saved: {file_path}")

            return {
                "status": "success",
                "data": str(file_path),
                "message": f"DAG saved to {file_path}",
            }
        except Exception as e:
            logger.error(f"Error saving to file: {e}")
            return {
                "status": "error",
                "data": None,
                "message": f"Error saving to file: {e}",
            }


class AirflowReadDAG:
    def __init__(
        self,
        dags_dir: Optional[Union[Path, str]] = None,
    ):
        _dags_dir: Optional[Path] = None
        if dags_dir is not None:
            if isinstance(dags_dir, str):
                _dags_dir = Path.cwd().joinpath(dags_dir)
            else:
                _dags_dir = dags_dir
        self.dags_dir: Path = _dags_dir or Path.cwd()

    def get_tool(self) -> Tool:
        return Tool(
            name="read_dag_file",
            description="Reads an Airflow DAG file `dag_file` and returns the contents if successful.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dag_file": {
                        "type": "string",
                        "description": "The name of the file to read.",
                    },
                },
                "required": ["dag_file"],
            },
            function=self._read_dag_file,
        )

    async def _read_dag_file(self, dag_file: str) -> Dict[str, Any]:
        """Reads an Airflow DAG file `dag_file` and returns the contents if successful."""
        try:
            log_info(f"Reading file: {dag_file}")
            file_path = self.dags_dir.joinpath(dag_file)
            contents = file_path.read_text()

            return {
                "status": "success",
                "data": contents,
                "message": "DAG file read successfully",
            }
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return {
                "status": "error",
                "data": None,
                "message": f"Error reading file: {e}",
            }
