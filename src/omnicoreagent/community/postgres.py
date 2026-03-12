from typing import Any, Dict, Optional
from omnicoreagent.core.tools.local_tools_registry import Tool

try:
    import psycopg
    from psycopg import sql
    from psycopg.rows import dict_row
except ImportError:
    psycopg = None
    sql = None
    dict_row = None


class PostgresBase:
    def __init__(
        self,
        db_name: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        table_schema: str = "public",
        connection: Optional[Any] = None,
    ):
        if psycopg is None:
            raise ImportError(
                "psycopg not installed. Please install it using `pip install psycopg[binary]`."
            )
        self.db_name = db_name
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.table_schema = table_schema
        self._connection = connection

    def _get_connection(self):
        if self._connection and not self._connection.closed:
            return self._connection
        if not all([self.db_name, self.user, self.host]):
            # If params missing, rely on caller to have provided connection or env vars
            # but standard refactor implies usage.
            pass

        conn_kwargs = {
            "dbname": self.db_name,
            "user": self.user,
            "password": self.password,
            "host": self.host,
            "port": self.port,
            "row_factory": dict_row,
            "options": f"-c search_path={self.table_schema}",
        }
        # Filter None
        conn_kwargs = {k: v for k, v in conn_kwargs.items() if v is not None}

        self._connection = psycopg.connect(**conn_kwargs)
        return self._connection


class PostgresShowTables(PostgresBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="postgres_show_tables",
            description="List tables in the Postgres database.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
            function=self._show_tables,
        )

    async def _show_tables(self) -> Dict[str, Any]:
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema = %s;",
                    (self.table_schema,),
                )
                tables = [row["table_name"] for row in cur.fetchall()]
                return {
                    "status": "success",
                    "data": tables,
                    "message": f"Tables: {', '.join(tables)}",
                }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class PostgresRunQuery(PostgresBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="postgres_run_query",
            description="Run a SQL query on Postgres.",
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
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute(query)
                if cur.description:
                    result = cur.fetchall()
                    data = [dict(row) for row in result]
                    return {
                        "status": "success",
                        "data": data,
                        "message": f"Returned {len(data)} rows",
                    }
                else:
                    return {
                        "status": "success",
                        "data": None,
                        "message": "Query executed",
                    }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}
