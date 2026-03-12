import math
from typing import Any, Dict, Optional
from omnicoreagent.core.tools.local_tools_registry import Tool


class CalculatorTool:
    def get_tool(self) -> Tool:
        return Tool(
            name="calculator",
            description="Perform mathematical operations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": [
                            "add",
                            "subtract",
                            "multiply",
                            "divide",
                            "power",
                            "sqrt",
                        ],
                    },
                    "a": {"type": "number"},
                    "b": {
                        "type": "number",
                        "description": "Second operand (not needed for sqrt).",
                    },
                },
                "required": ["operation", "a"],
            },
            function=self._calculate,
        )

    async def _calculate(
        self, operation: str, a: float, b: Optional[float] = None
    ) -> Dict[str, Any]:
        try:
            result = None
            if operation == "add":
                result = a + (b or 0)
            elif operation == "subtract":
                result = a - (b or 0)
            elif operation == "multiply":
                result = a * (b or 1)
            elif operation == "divide":
                if b == 0:
                    return {
                        "status": "error",
                        "data": None,
                        "message": "Division by zero",
                    }
                result = a / (b or 1)
            elif operation == "power":
                result = math.pow(a, b or 1)
            elif operation == "sqrt":
                result = math.sqrt(a)
            else:
                return {
                    "status": "error",
                    "data": None,
                    "message": f"Unknown operation: {operation}",
                }

            return {"status": "success", "data": result, "message": f"Result: {result}"}
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}
