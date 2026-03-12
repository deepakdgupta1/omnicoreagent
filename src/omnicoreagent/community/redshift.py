from os import getenv
from typing import Any, Dict, Optional

from omnicoreagent.core.tools.local_tools_registry import Tool

try:
    import redshift_connector
except ImportError:
    redshift_connector = None


class RedshiftBase:
    def __init__(
        self,
        host: Optional[str] = None,
        port: int = 5439,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        iam: bool = False,
        cluster_identifier: Optional[str] = None,
        region: Optional[str] = None,
        db_user: Optional[str] = None,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        session_token: Optional[str] = None,
        profile: Optional[str] = None,
        ssl: bool = True,
        table_schema: str = "public",
    ):
        self.host = host or getenv("REDSHIFT_HOST")
        self.port = port
        self.database = database or getenv("REDSHIFT_DATABASE")
        self.user = user
        self.password = password
        self.iam = iam
        self.cluster_identifier = cluster_identifier or getenv(
            "REDSHIFT_CLUSTER_IDENTIFIER"
        )
        self.region = region or getenv("AWS_REGION")
        self.db_user = db_user or getenv("REDSHIFT_DB_USER")
        self.access_key_id = access_key_id or getenv("AWS_ACCESS_KEY_ID")
        self.secret_access_key = secret_access_key or getenv("AWS_SECRET_ACCESS_KEY")
        self.session_token = session_token or getenv("AWS_SESSION_TOKEN")
        self.profile = profile or getenv("AWS_PROFILE")
        self.ssl = ssl
        if redshift_connector is None:
            raise ImportError(
                "redshift_connector not installed. Please install it using `pip install redshift_connector`."
            )
        self.table_schema = table_schema
        self._connection = None

    def _get_connection(self):
        if self._connection:
            return self._connection

        kwargs = {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "ssl": self.ssl,
        }

        if self.iam:
            kwargs["iam"] = True
            if self.cluster_identifier:
                kwargs["cluster_identifier"] = self.cluster_identifier
            if self.db_user:
                kwargs["db_user"] = self.db_user
            if self.region:
                kwargs["region"] = self.region
            if self.profile:
                kwargs["profile"] = self.profile
            else:
                if self.access_key_id:
                    kwargs["access_key_id"] = self.access_key_id
                if self.secret_access_key:
                    kwargs["secret_access_key"] = self.secret_access_key
                if self.session_token:
                    kwargs["session_token"] = self.session_token
        else:
            if self.user:
                kwargs["user"] = self.user
            if self.password:
                kwargs["password"] = self.password

        # Filter None
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        self._connection = redshift_connector.connect(**kwargs)
        return self._connection


class RedshiftShowTables(RedshiftBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="redshift_show_tables",
            description="List tables in Redshift.",
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
                tables = [row[0] for row in cur.fetchall()]
                return {
                    "status": "success",
                    "data": tables,
                    "message": f"Tables: {', '.join(tables)}",
                }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class RedshiftRunQuery(RedshiftBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="redshift_run_query",
            description="Run a SQL query on Redshift.",
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
                    columns = [desc[0] for desc in cur.description]
                    rows = cur.fetchall()
                    data = [dict(zip(columns, row)) for row in rows]
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
