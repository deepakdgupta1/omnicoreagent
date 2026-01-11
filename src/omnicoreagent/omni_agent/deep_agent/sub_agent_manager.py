"""
SubAgentManager: Tools for autonomous sub-agent spawning.

Provides tools that DeepAgent can call to spawn:
- Immediate sub-agents (via OmniCoreAgent)
- Parallel sub-agents (via ParallelAgent)
- Background agents (via BackgroundAgentManager)
"""

from typing import Dict, Any, List, Optional
from omnicoreagent.omni_agent.agent import OmniCoreAgent
from omnicoreagent.omni_agent.workflow.parallel_agent import ParallelAgent
from omnicoreagent.omni_agent.background_agent.background_agent_manager import (
    BackgroundAgentManager,
)
from omnicoreagent.core.tools.local_tools_registry import ToolRegistry
from omnicoreagent.core.utils import logger
import asyncio
import uuid


class SubAgentManager:
    """
    Manages autonomous sub-agent spawning for DeepAgent.
    
    The DeepAgent calls these tools when it decides it needs help:
    - spawn_sub_agent: Quick synchronous helper
    - spawn_parallel_agents: Multiple concurrent helpers
    - spawn_background_agent: Long-running async task
    - get_background_result: Check async task status
    
    All spawned sub-agents are full OmniCoreAgents with access to
    the same memory, MCP tools, and capabilities.
    """
    
    def __init__(
        self,
        base_model_config: Dict[str, Any],
        memory_router=None,
        event_router=None,
        agent_config: Dict[str, Any] = None,
        mcp_tools: List[Dict] = None,
        project_memory_base: str = "/memories/projects/default",
    ):
        """
        Initialize SubAgentManager.
        
        Args:
            base_model_config: Model configuration for spawned agents
            memory_router: Shared memory router
            event_router: Shared event router
            agent_config: Base agent config (memory_tool_backend, etc.)
            mcp_tools: MCP tools available to spawned agents
            project_memory_base: Base memory path for this project
        """
        self.base_model_config = base_model_config
        self.memory_router = memory_router
        self.event_router = event_router
        self.agent_config = agent_config or {"memory_tool_backend": "local"}
        self.mcp_tools = mcp_tools
        self.project_memory_base = project_memory_base
        
        # Track spawned agents
        self._sync_agents: Dict[str, OmniCoreAgent] = {}
        self._background_manager: Optional[BackgroundAgentManager] = None
        self._background_tasks: Dict[str, Dict] = {}
    
    def _get_background_manager(self) -> BackgroundAgentManager:
        """Lazy init BackgroundAgentManager."""
        if not self._background_manager:
            self._background_manager = BackgroundAgentManager(
                memory_router=self.memory_router,
                event_router=self.event_router,
            )
        return self._background_manager
    
    async def spawn(
        self,
        name: str,
        instruction: str,
        task: str,
        save_result_to: Optional[str] = None,
    ) -> str:
        """
        Spawn a sub-agent, run it, and return result.
        
        Args:
            name: Agent name
            instruction: What this agent does (system instruction)
            task: Task to execute
            save_result_to: Optional memory path to save result
            
        Returns:
            The agent's response as a string
        """
        # Add project context to instruction
        full_instruction = f"""{instruction}

PROJECT MEMORY: {self.project_memory_base}/
You have access to memory tools for persistent storage.
Save important findings to the project memory path.
"""

        agent = OmniCoreAgent(
            name=name,
            system_instruction=full_instruction,
            model_config=self.base_model_config,
            memory_router=self.memory_router,
            event_router=self.event_router,
            agent_config=self.agent_config,
            mcp_tools=self.mcp_tools,
        )
        
        self._sync_agents[name] = agent
        
        try:
            if self.mcp_tools:
                await agent.connect_mcp_servers()
            
            result = await agent.run(task)
            response = result.get("response", str(result))
            
            # Optionally save to memory
            if save_result_to:
                save_prompt = f"""
Use memory_create_update to save this result:

Path: {save_result_to}
Mode: create
Content:
# Result from {name}

{response}
"""
                await agent.run(save_prompt)
            
            logger.info(f"SubAgentManager: Agent '{name}' completed task")
            return response
            
        except Exception as e:
            logger.error(f"SubAgentManager: Agent '{name}' failed: {e}")
            return f"Error: Agent '{name}' failed with: {str(e)}"
            
        finally:
            await agent.cleanup()
            if name in self._sync_agents:
                del self._sync_agents[name]
    
    async def spawn_parallel(
        self,
        agents_config: List[Dict[str, str]],
    ) -> Dict[str, str]:
        """
        Spawn multiple sub-agents in parallel.
        
        Args:
            agents_config: List of dicts with keys:
                - name: Agent name
                - instruction: System instruction
                - task: Task to perform
                - save_result_to: (optional) Memory path to save result
                
        Returns:
            Dict mapping agent name to response
        """
        if not agents_config:
            return {}
        
        # Create all agents
        agents = []
        for config in agents_config:
            full_instruction = f"""{config.get('instruction', '')}

PROJECT MEMORY: {self.project_memory_base}/
You have access to memory tools for persistent storage.
"""
            
            agent = OmniCoreAgent(
                name=config["name"],
                system_instruction=full_instruction,
                model_config=self.base_model_config,
                memory_router=self.memory_router,
                event_router=self.event_router,
                agent_config=self.agent_config,
                mcp_tools=self.mcp_tools,
            )
            agents.append(agent)
        
        # Use ParallelAgent for concurrent execution
        parallel = ParallelAgent(sub_agents=agents)
        
        try:
            await parallel.initialize()
            
            # Build task map
            tasks = {config["name"]: config["task"] for config in agents_config}
            results = await parallel.run(agent_tasks=tasks)
            
            # Extract responses
            response_map = {}
            for name, res in results.items():
                if isinstance(res, dict):
                    response_map[name] = res.get("response", str(res))
                else:
                    response_map[name] = str(res)
            
            logger.info(f"SubAgentManager: Parallel agents completed: {list(response_map.keys())}")
            return response_map
            
        except Exception as e:
            logger.error(f"SubAgentManager: Parallel execution failed: {e}")
            return {"error": str(e)}
            
        finally:
            await parallel.shutdown()
    
    async def spawn_background(
        self,
        name: str,
        instruction: str,
        task: str,
        timeout: int = 600,
    ) -> str:
        """
        Spawn a background agent for long-running tasks.
        
        Args:
            name: Agent name
            instruction: System instruction
            task: Task to execute
            timeout: Task timeout in seconds (default 10 minutes)
            
        Returns:
            Task ID to check status later
        """
        manager = self._get_background_manager()
        task_id = f"bg_{name}_{uuid.uuid4().hex[:8]}"
        
        full_instruction = f"""{instruction}

PROJECT MEMORY: {self.project_memory_base}/
You have access to memory tools for persistent storage.
IMPORTANT: Save your final results to memory before completing.
"""

        try:
            await manager.create_agent({
                "agent_id": task_id,
                "name": name,
                "system_instruction": full_instruction,
                "model_config": self.base_model_config,
                "agent_config": self.agent_config,
                "mcp_tools": self.mcp_tools or [],
                "task_config": {
                    "query": task,
                    "timeout": timeout,
                    "max_retries": 2,
                    "interval": timeout + 60,  # One-shot task
                },
            })
            
            self._background_tasks[task_id] = {
                "agent_id": task_id,
                "name": name,
                "status": "running",
            }
            
            logger.info(f"SubAgentManager: Background agent '{name}' started as {task_id}")
            return f"Background task started with ID: {task_id}. Use get_background_result('{task_id}') to check status."
            
        except Exception as e:
            logger.error(f"SubAgentManager: Failed to start background agent: {e}")
            return f"Error starting background agent: {str(e)}"
    
    async def get_background_result(self, task_id: str) -> str:
        """
        Check status of a background task.
        
        Args:
            task_id: The task ID returned by spawn_background
            
        Returns:
            Status message or result
        """
        if task_id not in self._background_tasks:
            return f"Task '{task_id}' not found. Available tasks: {list(self._background_tasks.keys())}"
        
        manager = self._get_background_manager()
        status = await manager.get_agent_status(task_id)
        
        if not status:
            return f"Task '{task_id}' status unavailable"
        
        if status.get("is_running"):
            run_count = status.get("run_count", 0)
            return f"Task '{task_id}' is still running (run count: {run_count}). Check again later or check {self.project_memory_base}/ for partial results."
        
        # Task completed - direct to memory for results
        return f"Task '{task_id}' completed. Results should be saved to {self.project_memory_base}/. Use memory_view to check."
    
    async def cleanup(self):
        """Clean up all spawned agents and resources."""
        # Clean up sync agents
        for name, agent in list(self._sync_agents.items()):
            try:
                await agent.cleanup()
                logger.info(f"SubAgentManager: Cleaned up agent '{name}'")
            except Exception as e:
                logger.warning(f"SubAgentManager: Error cleaning up '{name}': {e}")
        self._sync_agents.clear()
        
        # Clean up background manager
        if self._background_manager:
            try:
                await self._background_manager.shutdown()
                logger.info("SubAgentManager: Background manager shutdown complete")
            except Exception as e:
                logger.warning(f"SubAgentManager: Error shutting down background manager: {e}")
        
        self._background_tasks.clear()


def build_sub_agent_tools(manager: SubAgentManager, registry: ToolRegistry) -> ToolRegistry:
    """
    Register sub-agent spawning tools in a ToolRegistry.
    
    These tools allow the DeepAgent to autonomously spawn sub-agents
    when it determines doing so would be beneficial.
    """
    
    @registry.register_tool(
        name="spawn_sub_agent",
        description="""
Spawn a specialized sub-agent to help with a specific task.

Use when:
- Need specialized expertise for a subtask
- Want to delegate focused work with fresh context
- Task benefits from a dedicated agent

The agent runs immediately and returns its result.
Sub-agents have full OmniCoreAgent capabilities including memory and MCP tools.
""",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Descriptive name for this helper agent (e.g., 'DataAnalyzer', 'ResearchAssistant')",
                },
                "instruction": {
                    "type": "string",
                    "description": "System instruction defining what this agent specializes in and how it should behave",
                },
                "task": {
                    "type": "string",
                    "description": "The specific task for this agent to perform",
                },
                "save_result_to": {
                    "type": "string",
                    "description": "Optional: memory path to save the result (e.g., '/memories/projects/myproject/research/findings.md')",
                },
            },
            "required": ["name", "instruction", "task"],
            "additionalProperties": False,
        },
    )
    async def spawn_sub_agent(
        name: str,
        instruction: str,
        task: str,
        save_result_to: str = None,
    ) -> str:
        return await manager.spawn(name, instruction, task, save_result_to)
    
    @registry.register_tool(
        name="spawn_parallel_agents",
        description="""
Spawn multiple sub-agents to work in parallel.

Use when:
- Task can be split into independent parts
- Research needs multiple perspectives
- Want to parallelize for speed
- Multiple independent subtasks exist

All agents run concurrently and results are collected.
Each agent is a full OmniCoreAgent with all capabilities.
""",
        inputSchema={
            "type": "object",
            "properties": {
                "agents": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Agent name"},
                            "instruction": {"type": "string", "description": "System instruction"},
                            "task": {"type": "string", "description": "Task to perform"},
                        },
                        "required": ["name", "instruction", "task"],
                    },
                    "description": "List of agents to spawn, each with name, instruction, and task",
                },
            },
            "required": ["agents"],
            "additionalProperties": False,
        },
    )
    async def spawn_parallel_agents(agents: list) -> str:
        results = await manager.spawn_parallel(agents)
        # Format results nicely
        output_lines = ["PARALLEL AGENT RESULTS:"]
        for name, result in results.items():
            output_lines.append(f"\n=== {name} ===")
            output_lines.append(result)
        return "\n".join(output_lines)
    
    @registry.register_tool(
        name="spawn_background_agent",
        description="""
Spawn a background agent for long-running tasks.

Use when:
- Task may take several minutes
- Don't want to block waiting for completion
- Can check results later
- Task is independent from current flow

Returns a task_id. Use get_background_result to check status later.
The agent will save results to project memory when complete.
""",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Agent name",
                },
                "instruction": {
                    "type": "string",
                    "description": "System instruction for the background agent",
                },
                "task": {
                    "type": "string",
                    "description": "Task to perform in background",
                },
                "timeout": {
                    "type": "integer",
                    "default": 600,
                    "description": "Timeout in seconds (default 600 = 10 minutes)",
                },
            },
            "required": ["name", "instruction", "task"],
            "additionalProperties": False,
        },
    )
    async def spawn_background_agent(
        name: str,
        instruction: str,
        task: str,
        timeout: int = 600,
    ) -> str:
        return await manager.spawn_background(name, instruction, task, timeout)
    
    @registry.register_tool(
        name="get_background_result",
        description="""
Check status or result of a background task started earlier.

Use after spawning a background agent to check if it has completed
and where to find its results.
""",
        inputSchema={
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "The task_id returned by spawn_background_agent",
                },
            },
            "required": ["task_id"],
            "additionalProperties": False,
        },
    )
    async def get_background_result(task_id: str) -> str:
        return await manager.get_background_result(task_id)
    
    return registry
