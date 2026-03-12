import os
from typing import Any, Dict, Optional

from omnicoreagent.core.tools.local_tools_registry import Tool
from omnicoreagent.core.utils import log_debug, logger

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None


class Neo4jTools:
    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ):
        uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = user or os.getenv("NEO4J_USERNAME")
        password = password or os.getenv("NEO4J_PASSWORD")

        if GraphDatabase is None:
            raise ImportError(
                "`neo4j` not installed. Please install using `pip install neo4j`"
            )

        if user is None or password is None:
            raise ValueError("Username or password for Neo4j not provided")

        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            log_debug("Connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

        self.database = database or "neo4j"

    def get_tool(self) -> Tool:
        return Tool(
            name="neo4j_run_cypher",
            description="Execute a Cypher query against a Neo4j database.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Cypher query to execute",
                    },
                },
                "required": ["query"],
            },
            function=self._run_cypher_query,
        )

    async def _run_cypher_query(self, query: str) -> Dict[str, Any]:
        try:
            log_debug(f"Running Cypher query: {query}")
            with self.driver.session(database=self.database) as session:
                result = session.run(query)
                data = result.data()
            return {
                "status": "success",
                "data": data,
                "message": f"Query returned {len(data)} results",
            }
        except Exception as e:
            logger.error(f"Error running Cypher query: {e}")
            return {"status": "error", "data": None, "message": str(e)}


class Neo4jListLabels(Neo4jTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="neo4j_list_labels",
            description="List all node labels in the Neo4j database.",
            inputSchema={"type": "object", "properties": {}},
            function=self._list_labels,
        )

    async def _list_labels(self) -> Dict[str, Any]:
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("CALL db.labels()")
                labels = [record["label"] for record in result]
            return {
                "status": "success",
                "data": labels,
                "message": f"Found {len(labels)} labels",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class Neo4jListRelationships(Neo4jTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="neo4j_list_relationships",
            description="List all relationship types in the Neo4j database.",
            inputSchema={"type": "object", "properties": {}},
            function=self._list_relationships,
        )

    async def _list_relationships(self) -> Dict[str, Any]:
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("CALL db.relationshipTypes()")
                types = [record["relationshipType"] for record in result]
            return {
                "status": "success",
                "data": types,
                "message": f"Found {len(types)} relationship types",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}


class Neo4jGetSchema(Neo4jTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="neo4j_get_schema",
            description="Retrieve the database schema visualization.",
            inputSchema={"type": "object", "properties": {}},
            function=self._get_schema,
        )

    async def _get_schema(self) -> Dict[str, Any]:
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("CALL db.schema.visualization()")
                schema_data = result.data()
            return {
                "status": "success",
                "data": schema_data,
                "message": "Schema retrieved",
            }
        except Exception as e:
            return {"status": "error", "data": None, "message": str(e)}
