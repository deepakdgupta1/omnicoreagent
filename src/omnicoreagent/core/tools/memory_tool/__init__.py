"""
Memory Tool backends for OmniCoreAgent.

Provides pluggable storage backends for the memory tool:
- LocalMemoryBackend: Filesystem-based storage
- S3MemoryBackend: AWS S3 storage
- R2MemoryBackend: Cloudflare R2 storage

Usage:
    # In agent config
    agent_config = {
        "memory_tool_backend": "local",  # or "s3" or "r2"
    }
"""

from omnicoreagent.core.tools.memory_tool.base import AbstractMemoryBackend
from omnicoreagent.core.tools.memory_tool.local_storage import LocalMemoryBackend
from omnicoreagent.core.tools.memory_tool.s3_storage import S3MemoryBackend
from omnicoreagent.core.tools.memory_tool.r2_storage import R2MemoryBackend
from omnicoreagent.core.tools.memory_tool.factory import create_memory_backend
from omnicoreagent.core.tools.memory_tool.memory_tool import MemoryTool, build_tool_registry_memory_tool

__all__ = [
    # Base class
    "AbstractMemoryBackend",
    # Backends
    "LocalMemoryBackend",
    "S3MemoryBackend",
    "R2MemoryBackend",
    # Factory
    "create_memory_backend",
    # Memory tool
    "MemoryTool",
    "build_tool_registry_memory_tool",
]
