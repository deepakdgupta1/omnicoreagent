"""
DeepAgent: Advanced OmniCoreAgent with Autonomous Sub-Agent Spawning and RPI Workflow.

DeepAgent extends OmniCoreAgent with:
- Autonomous sub-agent spawning (sync, parallel, background)
- RPI (Research, Plan, Implement) workflow methods
- Persistent memory via memory_tool_backend (always enabled)
- Project-based file organization

The agent DECIDES ON ITS OWN when to spawn sub-agents.
User can also provide pre-configured sub_agents if desired.
"""

from typing import Optional, List, Dict, Any
from omnicoreagent.omni_agent.agent import OmniCoreAgent
from omnicoreagent.core.memory_store.memory_router import MemoryRouter
from omnicoreagent.core.events.event_router import EventRouter
from omnicoreagent.core.tools.local_tools_registry import ToolRegistry
from .sub_agent_manager import SubAgentManager, build_sub_agent_tools
from .prompts import build_enhanced_instruction
from omnicoreagent.core.utils import logger
from datetime import datetime
import uuid


class DeepAgent:
    """
    Advanced OmniCoreAgent with autonomous sub-agent spawning and RPI workflow.
    
    As easy to use as OmniCoreAgent, but with powerful autonomous capabilities:
    
    1. **Autonomous Sub-Agent Spawning**: The agent can spawn helper agents
       when it determines doing so would be beneficial. Supports:
       - Synchronous helpers (spawn_sub_agent)
       - Parallel execution (spawn_parallel_agents)
       - Background/async tasks (spawn_background_agent)
    
    2. **RPI Workflow**: Built-in methods for Research, Plan, Implement, Iterate
       phases to handle complex multi-step tasks systematically.
    
    3. **Persistent Memory**: memory_tool_backend is ALWAYS enabled. The agent
       actively uses file-based persistence for research, plans, and progress.
    
    4. **Project Organization**: All files organized under /memories/projects/{project}/
    
    Example:
        ```python
        agent = DeepAgent(
            name="Analyst",
            system_instruction="You are a market research analyst.",
            model_config={"provider": "openai", "model": "gpt-4o"},
        )
        
        await agent.initialize()
        
        # Use RPI workflow
        await agent.research("Analyze competitor landscape")
        await agent.plan("Develop market entry strategy")
        
        # Or direct use (agent may spawn sub-agents autonomously)
        await agent.run("What's the current market size?")
        
        await agent.cleanup()
        ```
    """
    
    def __init__(
        self,
        name: str,
        system_instruction: str,
        model_config: Dict[str, Any],
        project_name: Optional[str] = None,
        sub_agents: Optional[List[OmniCoreAgent]] = None,
        memory_router: Optional[MemoryRouter] = None,
        event_router: Optional[EventRouter] = None,
        mcp_tools: Optional[List[Dict]] = None,
        local_tools: Optional[ToolRegistry] = None,
        agent_config: Optional[Dict[str, Any]] = None,
        debug: bool = False,
    ):
        """
        Initialize DeepAgent.
        
        Args:
            name: Agent name
            system_instruction: What this agent does (app builder defines this!)
            model_config: LLM configuration (provider, model, etc.)
            project_name: Project identifier for file organization (auto-generated if not provided)
            sub_agents: Optional pre-configured sub-agents (agent can also spawn its own)
            memory_router: Memory router (defaults to in_memory)
            event_router: Event router (defaults to in_memory)
            mcp_tools: MCP tools available to agent and spawned sub-agents
            local_tools: Additional local tools
            agent_config: Additional agent config (memory_tool_backend ALWAYS added)
            debug: Enable debug logging
        """
        self.name = name
        self.system_instruction = system_instruction
        self.model_config = model_config
        self.project_name = project_name or f"project_{datetime.now().strftime('%Y%m%d_%H%M')}"
        self.user_sub_agents = sub_agents or []
        self.memory_router = memory_router or MemoryRouter("in_memory")
        self.event_router = event_router or EventRouter("in_memory")
        self.mcp_tools = mcp_tools
        self.user_local_tools = local_tools
        self.debug = debug
        
        # ALWAYS enable memory_tool_backend for persistent storage
        self.agent_config = agent_config.copy() if agent_config else {}
        self.agent_config["memory_tool_backend"] = "local"
        
        # Memory paths for this project
        self.memory_base = f"/memories/projects/{self.project_name}"
        
        # Internal state
        self._sub_agent_manager: Optional[SubAgentManager] = None
        self._orchestrator: Optional[OmniCoreAgent] = None
        self._initialized = False
    
    async def initialize(self):
        """
        Initialize the DeepAgent.
        
        This must be called before using the agent. Sets up:
        - SubAgentManager with spawning tools
        - Orchestrator OmniCoreAgent with enhanced instructions
        - MCP server connections if configured
        """
        if self._initialized:
            return
        
        logger.info(f"DeepAgent '{self.name}': Initializing for project '{self.project_name}'")
        
        # Create SubAgentManager with all spawning tools
        self._sub_agent_manager = SubAgentManager(
            base_model_config=self.model_config,
            memory_router=self.memory_router,
            event_router=self.event_router,
            agent_config=self.agent_config,
            mcp_tools=self.mcp_tools,
            project_memory_base=self.memory_base,
        )
        
        # Build tools registry with sub-agent spawning tools
        tools = self.user_local_tools or ToolRegistry()
        build_sub_agent_tools(self._sub_agent_manager, tools)
        
        # Build enhanced instruction with all extensions
        enhanced_instruction = build_enhanced_instruction(
            base_instruction=self.system_instruction,
            project_name=self.project_name,
            memory_base=self.memory_base,
        )
        
        # Create the orchestrator OmniCoreAgent
        self._orchestrator = OmniCoreAgent(
            name=self.name,
            system_instruction=enhanced_instruction,
            model_config=self.model_config,
            memory_router=self.memory_router,
            event_router=self.event_router,
            agent_config=self.agent_config,
            mcp_tools=self.mcp_tools,
            local_tools=tools,
            sub_agents={"sub_agents": self.user_sub_agents} if self.user_sub_agents else None,
            debug=self.debug,
        )
        
        # Connect MCP servers if configured
        if self.mcp_tools:
            await self._orchestrator.connect_mcp_servers()
        
        self._initialized = True
        logger.info(f"DeepAgent '{self.name}': Initialization complete")
    
    # =========================================================================
    # RPI Workflow Methods
    # =========================================================================
    
    async def research(
        self,
        query: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute the Research phase.
        
        Gathers information and documents findings. The agent may autonomously
        spawn parallel sub-agents for comprehensive research.
        
        Args:
            query: What to research
            session_id: Optional session ID for continuity
            
        Returns:
            Agent response with research findings
        """
        if not self._initialized:
            await self.initialize()
        
        session_id = session_id or str(uuid.uuid4())
        timestamp = datetime.now().strftime('%Y-%m-%d-%H%M')
        
        prompt = f"""
RESEARCH TASK: {query}

Instructions:
1. First, check {self.memory_base}/ for any existing relevant research or context
2. Conduct thorough research on the topic
3. Consider if spawning parallel sub-agents would help gather information faster
4. Save your findings to {self.memory_base}/research/{timestamp}-research.md
5. Update {self.memory_base}/progress.md with current status

Be comprehensive. Document findings objectively without bias.
Organize findings in clear sections with headers.
"""
        return await self._orchestrator.run(prompt, session_id=session_id)
    
    async def plan(
        self,
        goal: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute the Plan phase.
        
        Creates a detailed implementation plan based on research.
        
        Args:
            goal: What to plan for
            session_id: Optional session ID for continuity
            
        Returns:
            Agent response with plan
        """
        if not self._initialized:
            await self.initialize()
        
        session_id = session_id or str(uuid.uuid4())
        timestamp = datetime.now().strftime('%Y-%m-%d-%H%M')
        
        prompt = f"""
PLANNING TASK: {goal}

Instructions:
1. Read all research from {self.memory_base}/research/
2. Create a detailed, phased implementation plan with:
   - Clear phases with [ ] checkboxes for each step
   - Success criteria for each phase
   - Verification steps
3. If requirements are ambiguous, list clarifying questions
4. Save to {self.memory_base}/plans/{timestamp}-plan.md
5. Update {self.memory_base}/progress.md with planning status
"""
        return await self._orchestrator.run(prompt, session_id=session_id)
    
    async def implement(
        self,
        plan_path: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute the Implement phase.
        
        Executes the plan step by step with verification.
        
        Args:
            plan_path: Path to the plan in memory (e.g., '/memories/projects/x/plans/plan.md')
            session_id: Optional session ID for continuity
            
        Returns:
            Agent response with implementation status
        """
        if not self._initialized:
            await self.initialize()
        
        session_id = session_id or str(uuid.uuid4())
        
        prompt = f"""
IMPLEMENTATION TASK

Plan: {plan_path}

Instructions:
1. Read the plan from memory using memory_view
2. Execute each phase step by step
3. After completing a step, mark it [x] using memory_str_replace
4. Update {self.memory_base}/progress.md after each milestone
5. If a step fails, document the error and pause for review
6. Consider spawning sub-agents for parallel implementation tasks
7. Run verification after completing each phase
"""
        return await self._orchestrator.run(prompt, session_id=session_id)
    
    async def iterate(
        self,
        plan_path: str,
        feedback: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute the Iterate phase.
        
        Updates the plan based on feedback without full re-research.
        
        Args:
            plan_path: Path to the plan in memory
            feedback: Feedback to incorporate
            session_id: Optional session ID for continuity
            
        Returns:
            Agent response with updated plan
        """
        if not self._initialized:
            await self.initialize()
        
        session_id = session_id or str(uuid.uuid4())
        
        prompt = f"""
ITERATION TASK

Plan: {plan_path}
Feedback: {feedback}

Instructions:
1. Read the current plan from memory
2. Research only what's needed to address the feedback
3. Make targeted updates to the plan (don't rewrite unchanged sections)
4. Document what changed and why
5. Update {self.memory_base}/progress.md with iteration status
"""
        return await self._orchestrator.run(prompt, session_id=session_id)
    
    # =========================================================================
    # Direct Run
    # =========================================================================
    
    async def run(
        self,
        query: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the agent directly, like OmniCoreAgent.
        
        The agent has all enhanced capabilities and may autonomously
        spawn sub-agents if it determines doing so would be beneficial.
        
        Args:
            query: User query or task
            session_id: Optional session ID for continuity
            
        Returns:
            Agent response
        """
        if not self._initialized:
            await self.initialize()
        
        session_id = session_id or str(uuid.uuid4())
        return await self._orchestrator.run(query, session_id=session_id)
    
    # =========================================================================
    # Lifecycle Management
    # =========================================================================
    
    async def cleanup(self):
        """
        Clean up all resources.
        
        Shuts down:
        - SubAgentManager and all spawned agents
        - Orchestrator and MCP connections
        """
        logger.info(f"DeepAgent '{self.name}': Cleaning up...")
        
        if self._sub_agent_manager:
            await self._sub_agent_manager.cleanup()
        
        if self._orchestrator:
            await self._orchestrator.cleanup()
        
        self._initialized = False
        logger.info(f"DeepAgent '{self.name}': Cleanup complete")
    
    async def connect_mcp_servers(self):
        """Connect MCP servers (called automatically in initialize)."""
        if self._orchestrator:
            await self._orchestrator.connect_mcp_servers()
    
    async def cleanup_mcp_servers(self):
        """Disconnect MCP servers."""
        if self._orchestrator:
            await self._orchestrator.cleanup_mcp_servers()
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def get_metrics(self):
        """Get metrics from the orchestrator."""
        return self._orchestrator.get_metrics if self._orchestrator else None
    
    @property
    def is_initialized(self) -> bool:
        """Check if the agent is initialized."""
        return self._initialized
