from typing import Any, Dict, Optional

from omnicoreagent.core.tools.local_tools_registry import Tool

try:
    from sqlalchemy import Engine, create_engine, text, inspect
    from sqlalchemy.orm import Session, sessionmaker
except ImportError:
    Engine = None
    create_engine = None
    text = None
    inspect = None
    Session = None
    sessionmaker = None


class SQLBase:
    def __init__(
        self,
        db_url: Optional[str] = None,
        db_engine: Optional[Any] = None,
    ):
        self.db_engine = db_engine
        if not self.db_engine and db_url:
            if create_engine is None:
                raise ImportError(
                    "sqlalchemy not installed. Please install it using `pip install sqlalchemy`."
                )
            else:
                self.db_engine = create_engine(db_url)

        if not self.db_engine:
            # Allow lazy init or failure depending on usage
            pass


class SQLListTables(SQLBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="sql_list_tables",
            description="List tables in the database.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
            function=self._list_tables,
        )

    async def _list_tables(self) -> Dict[str, Any]:
        try:
            inspector = inspect(self.db_engine)
            tables = inspector.get_table_names()
            return {
                "status": "success",
                "data": tables,
                "message": f"Tables: {', '.join(tables)}",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class SQLRunQuery(SQLBase):
    def get_tool(self) -> Tool:
        return Tool(
            name="sql_run_query",
            description="Run a SQL query using SQLAlchemy.",
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
            with self.db_engine.connect() as conn:
                result = conn.execute(text(query))
                if result.returns_rows:
                    rows = result.fetchall()
                    keys = result.keys()
                    data = [dict(zip(keys, row)) for row in rows]
                    return {
                        "status": "success",
                        "data": data,
                        "message": f"Returned {len(data)} rows",
                    }
                else:
                    return {
                        "status": "success",
                        "data": None,
                        "message": "Query executed (no rows returned)",
                    }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}
