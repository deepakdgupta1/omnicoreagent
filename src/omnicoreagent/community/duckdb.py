from typing import Any, Dict, List, Optional

from omnicoreagent.core.tools.local_tools_registry import Tool
from omnicoreagent.core.utils import log_info, logger

try:
    import duckdb
except ImportError:
    duckdb = None


class DuckDbBase:
    def __init__(
        self,
        db_path: Optional[str] = None,
        connection: Optional[Any] = None,
        read_only: bool = False,
        config: Optional[dict] = None,
        init_commands: Optional[List[str]] = None,
    ):
        if duckdb is None:
            raise ImportError(
                "Could not import `duckdb` python package. "
                "Please install it using `pip install duckdb`."
            )
        self.db_path = db_path
        self.read_only = read_only
        self.config = config
        self._connection = connection
        self.init_commands = init_commands

    @property
    def connection(self) -> Any:
        if self._connection is None:
            connection_kwargs: Dict[str, Any] = {}
            if self.db_path is not None:
                connection_kwargs["database"] = self.db_path
            if self.read_only:
                connection_kwargs["read_only"] = self.read_only
            if self.config is not None:
                connection_kwargs["config"] = self.config
            self._connection = duckdb.connect(**connection_kwargs)
            try:
                if self.init_commands is not None:
                    for command in self.init_commands:
                        self._connection.sql(command)
            except Exception as e:
                logger.exception(e)
                logger.warning("Failed to run duckdb init commands")

        return self._connection


class DuckDbShowTables(DuckDbBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="duckdb_show_tables",
            description="Show tables in the DuckDB database.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
            function=self._show_tables,
        )

    async def _show_tables(self) -> Dict[str, Any]:
        try:
            stmt = "SHOW TABLES;"
            result = self.connection.sql(stmt).fetchall()
            tables = [r[0] for r in result]
            return {
                "status": "success",
                "data": tables,
                "message": f"Tables: {', '.join(tables)}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class DuckDbRunQuery(DuckDbBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="duckdb_run_query",
            description="Run a SQL query on DuckDB.",
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
            # Simple sanitization
            formatted_sql = query.replace("`", "").split(";")[0]
            log_info(f"Running: {formatted_sql}")

            query_result = self.connection.sql(formatted_sql)

            if query_result is None:
                return {
                    "status": "success",
                    "data": None,
                    "message": "Query executed with no output",
                }

            try:
                # Try to get column names and rows
                columns = query_result.columns
                rows = query_result.fetchall()
                data = [dict(zip(columns, row)) for row in rows]
                return {
                    "status": "success",
                    "data": data,
                    "message": f"Returned {len(data)} rows",
                }
            except AttributeError:
                # Fallback for non-select queries that might return simple status
                return {
                    "status": "success",
                    "data": str(query_result),
                    "message": "Query executed",
                }

        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class DuckDbDescribeTable(DuckDbBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="duckdb_describe_table",
            description="Describe a DuckDB table.",
            inputSchema={
                "type": "object",
                "properties": {
                    "table": {"type": "string"},
                },
                "required": ["table"],
            },
            function=self._describe,
        )

    async def _describe(self, table: str) -> Dict[str, Any]:
        try:
            result = self.connection.sql(f"DESCRIBE {table};").fetchall()
            return {
                "status": "success",
                "data": str(result),
                "message": f"Description for {table}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}
