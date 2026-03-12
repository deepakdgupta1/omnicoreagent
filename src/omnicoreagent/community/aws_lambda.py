from typing import Any, Dict
from omnicoreagent.core.tools.local_tools_registry import Tool

try:
    import boto3
except ImportError:
    boto3 = None


class AWSLambdaBase:
    def __init__(self, region_name: str = "us-east-1"):
        if boto3 is None:
            raise ImportError(
                "Could not import `boto3` python package. "
                "Please install it using `pip install boto3`."
            )
        self.region_name = region_name

    def _get_client(self):
        try:
            return boto3.client("lambda", region_name=self.region_name)
        except Exception as e:
            raise ValueError(f"Failed to create lambda client: {e}")


class AWSLambdaListFunctions(AWSLambdaBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="aws_lambda_list_functions",
            description="List AWS Lambda functions.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
            function=self._list_functions,
        )

    async def _list_functions(self) -> Dict[str, Any]:
        try:
            client = self._get_client()
            response = client.list_functions()
            functions = [f["FunctionName"] for f in response.get("Functions", [])]
            return {
                "status": "success",
                "data": functions,
                "message": f"Functions: {', '.join(functions)}",
            }
        except Exception as e:
            return {
                "status": "error",
                "data": None,
                "message": f"Error listing functions: {str(e)}",
            }


class AWSLambdaInvoke(AWSLambdaBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="aws_lambda_invoke",
            description="Invoke an AWS Lambda function.",
            inputSchema={
                "type": "object",
                "properties": {
                    "function_name": {"type": "string"},
                    "payload": {"type": "string", "default": "{}"},
                },
                "required": ["function_name"],
            },
            function=self._invoke,
        )

    async def _invoke(self, function_name: str, payload: str = "{}") -> Dict[str, Any]:
        try:
            client = self._get_client()
            response = client.invoke(FunctionName=function_name, Payload=payload)
            payload_resp = response["Payload"].read().decode("utf-8")
            return {
                "status": "success",
                "data": {
                    "status_code": response["StatusCode"],
                    "payload": payload_resp,
                },
                "message": f"Invoked {function_name}",
            }
        except Exception as e:
            return {
                "status": "error",
                "data": None,
                "message": f"Error invoking function: {str(e)}",
            }
