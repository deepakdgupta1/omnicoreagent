import subprocess
from typing import Any, Dict
from omnicoreagent.core.tools.local_tools_registry import Tool


class LocalPython:
    """
    Executes Python code locally.
    WARNING: This executes code directly on the host machine. Use with caution.
    """

    def __init__(self):
        pass

    def _execute(self, code: str) -> Dict[str, Any]:
        """
        Execute Python code and return stdout/stderr.
        """
        try:
            # Simple execution using python -c
            # Note: This has limitations (no persistent state between calls)
            result = subprocess.run(
                ["python3", "-c", code],
                capture_output=True,
                text=True,
                timeout=30,  # 30 seconds timeout
            )

            output = ""
            if result.stdout:
                output += f"STDOUT:\n{result.stdout}\n"
            if result.stderr:
                output += f"STDERR:\n{result.stderr}\n"

            return {
                "status": "success",
                "data": output
                if output
                else "Code executed successfully with no output.",
                "message": "Execution complete",
            }

        except subprocess.TimeoutExpired:
            return {"status": "error", "data": None, "message": "Execution timed out."}
        except Exception as e:
            return {
                "status": "error",
                "data": None,
                "message": f"Execution error: {str(e)}",
            }

    def get_tool(self) -> Tool:
        return Tool(
            name="python_interpreter",
            description="Execute Python code locally. Use this for calculations, data processing, or running quick scripts. Code runs in a stateless environment.",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute.",
                    },
                },
                "required": ["code"],
            },
            function=self._execute,
        )


class LocalBash:
    """
    Executes Bash commands locally.
    WARNING: This executes commands directly on the host machine. Use with caution.
    """

    def __init__(self):
        pass

    def _execute(self, command: str) -> Dict[str, Any]:
        """
        Execute Bash command and return stdout/stderr.
        """
        try:
            # Execute command
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=30
            )

            output = ""
            if result.stdout:
                output += f"STDOUT:\n{result.stdout}\n"
            if result.stderr:
                output += f"STDERR:\n{result.stderr}\n"

            return {
                "status": "success",
                "data": output
                if output
                else "Command executed successfully with no output.",
                "message": "Execution complete",
            }

        except subprocess.TimeoutExpired:
            return {"status": "error", "data": None, "message": "Command timed out."}
        except Exception as e:
            return {
                "status": "error",
                "data": None,
                "message": f"Execution error: {str(e)}",
            }

    def get_tool(self) -> Tool:
        return Tool(
            name="bash_command",
            description="Execute Bash commands locally. Use this for file operations, system checks, or running shell scripts.",
            inputSchema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The Bash command to execute.",
                    },
                },
                "required": ["command"],
            },
            function=self._execute,
        )
