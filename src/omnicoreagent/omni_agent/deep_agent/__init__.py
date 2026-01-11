"""
DeepAgent: Advanced OmniCoreAgent with Autonomous Sub-Agent Spawning and RPI Workflow.

This module provides DeepAgent, which extends OmniCoreAgent with:
- Autonomous sub-agent spawning (sync, parallel, background)
- RPI (Research, Plan, Implement) workflow methods
- Persistent memory via memory_tool_backend (always enabled)
- Project-based file organization

Example:
    ```python
    from omnicoreagent import DeepAgent
    
    agent = DeepAgent(
        name="Analyst",
        system_instruction="You are a market research analyst.",
        model_config={"provider": "openai", "model": "gpt-4o"},
    )
    
    await agent.initialize()
    await agent.research("Analyze competitor landscape")
    await agent.plan("Develop market strategy")
    await agent.cleanup()
    ```
"""

from .deep_agent import DeepAgent
from .sub_agent_manager import SubAgentManager, build_sub_agent_tools
from .prompts import (
    SUB_AGENT_EXTENSION,
    MEMORY_PERSISTENCE_EXTENSION,
    RPI_WORKFLOW_EXTENSION,
    build_enhanced_instruction,
)

__all__ = [
    "DeepAgent",
    "SubAgentManager",
    "build_sub_agent_tools",
    "SUB_AGENT_EXTENSION",
    "MEMORY_PERSISTENCE_EXTENSION",
    "RPI_WORKFLOW_EXTENSION",
    "build_enhanced_instruction",
]
