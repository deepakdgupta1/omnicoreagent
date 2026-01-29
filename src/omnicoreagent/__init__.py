"""
OmniCoreAgent AI Framework

A comprehensive AI agent framework with MCP client capabilities.
"""

from .core.agents import ReactAgent
from .core.memory_store import MemoryRouter
from .core.llm import LLMConnection
from .core.events import EventRouter
from .core.memory_store import DatabaseMessageStore
from .core.tools import ToolRegistry, Tool
from .core.utils import logger

from .omni_agent.agent import OmniCoreAgent, OmniAgent
from .omni_agent.background_agent import (
    BackgroundOmniCoreAgent,
    BackgroundAgentManager,
    TaskRegistry,
    APSchedulerBackend,
    BackgroundTaskScheduler,
)
from .omni_agent.omni_serve import OmniServe, OmniServeConfig

# OmniServe resilience and observability utilities
from .omni_agent.omni_serve.resilience import (
    RetryConfig,
    RetryStrategy,
    CircuitBreaker,
    CircuitBreakerConfig,
    with_retry,
    retry_async,
)
from .omni_agent.omni_serve.observability import get_metrics

from .mcp_clients_connection import MCPClient, Configuration

from .omni_agent.workflow.parallel_agent import ParallelAgent
from .omni_agent.workflow.sequential_agent import SequentialAgent
from .omni_agent.workflow.router_agent import RouterAgent

from .omni_agent.deep_agent import DeepAgent

__all__ = [
    # Core
    "ReactAgent",
    "MemoryRouter",
    "LLMConnection",
    "EventRouter",
    "DatabaseMessageStore",
    "ToolRegistry",
    "Tool",
    "logger",
    # Agents
    "OmniCoreAgent",
    "OmniAgent",
    "BackgroundOmniCoreAgent",
    "BackgroundAgentManager",
    "TaskRegistry",
    "APSchedulerBackend",
    "BackgroundTaskScheduler",
    "ParallelAgent",
    "SequentialAgent",
    "RouterAgent",
    "DeepAgent",
    # MCP
    "MCPClient",
    "Configuration",
    # OmniServe
    "OmniServe",
    "OmniServeConfig",
    # Resilience utilities
    "RetryConfig",
    "RetryStrategy",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "with_retry",
    "retry_async",
    "get_metrics",
]
