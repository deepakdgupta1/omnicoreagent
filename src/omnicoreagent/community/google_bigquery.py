from os import getenv
from typing import Any, Dict, Optional

from omnicoreagent.core.tools.local_tools_registry import Tool
from omnicoreagent.core.utils import logger

try:
    from google.cloud import bigquery
except ImportError:
    bigquery = None


class GoogleBigQueryBase:
    def __init__(
        self,
        dataset: str,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[Any] = None,
    ):
        if bigquery is None:
            raise ImportError(
                "Could not import `google-cloud-bigquery` python package. "
                "Please install it using `pip install google-cloud-bigquery`."
            )
        self.project = project or getenv("GOOGLE_CLOUD_PROJECT")
        self.location = location or getenv("GOOGLE_CLOUD_LOCATION")
        self.dataset = dataset
        self.credentials = credentials

        if not self.project:
            logger.warning("GOOGLE_CLOUD_PROJECT not set")

        self.client = None

    def _get_client(self):
        if not self.client:
            self.client = bigquery.Client(
                project=self.project, credentials=self.credentials
            )
        return self.client

    def _clean_sql(self, sql: str) -> str:
        return sql.replace("\\n", " ").replace("\n", " ")


class GoogleBigQueryListTables(GoogleBigQueryBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="bigquery_list_tables",
            description="List tables in the BigQuery dataset.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
            function=self._list_tables,
        )

    async def _list_tables(self) -> Dict[str, Any]:
        try:
            client = self._get_client()
            tables = client.list_tables(self.dataset)
            table_ids = [table.table_id for table in tables]
            return {
                "status": "success",
                "data": table_ids,
                "message": f"Tables: {', '.join(table_ids)}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class GoogleBigQueryRunQuery(GoogleBigQueryBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="bigquery_run_query",
            description="Run a SQL query on BigQuery.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
            function=self._run_query,
        )

    async def _run_query(self, query: str) -> Dict[str, Any]:
        try:
            client = self._get_client()
            cleaned_query = self._clean_sql(query)
            job_config = bigquery.QueryJobConfig(
                default_dataset=f"{self.project}.{self.dataset}"
            )

            query_job = client.query(cleaned_query, job_config)
            results = query_job.result()

            data = [dict(row) for row in results]

            return {
                "status": "success",
                "data": data,
                "message": f"Returned {len(data)} rows",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}
