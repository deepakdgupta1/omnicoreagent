
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from omnicoreagent.omni_agent.deep_agent import DeepAgent
from omnicoreagent import OmniCoreAgent

@pytest.fixture
def model_config():
    return {"provider": "openai", "model": "gpt-4"}

@pytest.fixture
def deep_agent(model_config):
    return DeepAgent(
        name="TestDeepAgent",
        system_instruction="Test Instruction",
        model_config=model_config,
        project_name="test_project"
    )

@pytest.mark.asyncio
async def test_deep_agent_initialization(deep_agent):
    """Test that Deep Agent initializes correctly with sub-components."""
    await deep_agent.initialize()
    
    assert deep_agent._initialized
    assert deep_agent._sub_agent_manager is not None
    assert deep_agent._orchestrator is not None
    # Verify memory_tool_backend is enabled
    assert deep_agent.agent_config["memory_tool_backend"] == "local"
    
    await deep_agent.cleanup()

@pytest.mark.asyncio
async def test_deep_agent_rpi_methods(deep_agent):
    """Test RPI workflow methods delegate to orchestrator."""
    await deep_agent.initialize()
    
    # Mock orchestrator.run
    mock_response = {"response": "Mocked Response", "metric": {}}
    deep_agent._orchestrator.run = AsyncMock(return_value=mock_response)
    
    # Test Research
    res = await deep_agent.research("Test Query")
    assert res == mock_response
    args, kwargs = deep_agent._orchestrator.run.call_args
    assert "RESEARCH TASK" in args[0]
    assert "/memories/projects/test_project/research/" in args[0]
    
    # Test Plan
    res = await deep_agent.plan("Test Goal")
    assert res == mock_response
    args, kwargs = deep_agent._orchestrator.run.call_args
    assert "PLANNING TASK" in args[0]
    
    # Test Implement
    res = await deep_agent.implement("plan.md")
    assert res == mock_response
    args, kwargs = deep_agent._orchestrator.run.call_args
    assert "IMPLEMENTATION TASK" in args[0]
    
    await deep_agent.cleanup()

@pytest.mark.asyncio
async def test_autonomous_tools_registration(deep_agent):
    """Verify autonomous spawning tools are registered."""
    await deep_agent.initialize() # This builds the tool registry
    
    registry = deep_agent._orchestrator.local_tools
    tools = registry.list_tools()
    
    tool_names = [t.name for t in tools]
    assert "spawn_sub_agent" in tool_names
    assert "spawn_parallel_agents" in tool_names
    assert "spawn_background_agent" in tool_names
    
    await deep_agent.cleanup()

@pytest.mark.asyncio
async def test_prompt_injection():
    """Verify prompt extensions are injected."""
    agent = DeepAgent(
        name="Test",
        system_instruction="Base Instruction",
        model_config={"provider": "openai", "model": "test"},
        project_name="proj"
    )
    await agent.initialize()
    
    full_instruction = agent._orchestrator.system_instruction
    
    # Check for our extensions
    assert "autonomous_sub_agents" in full_instruction
    assert "memory_persistence" in full_instruction
    assert "rpi_workflow" in full_instruction
    assert "PROJECT: proj" in full_instruction
    
    await agent.cleanup()
