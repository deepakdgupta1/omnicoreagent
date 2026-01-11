"""
Deep Agent Cookbook Example

This example demonstrates how to use the Deep Agent for complex tasks requiring
autonomous orchestration, parallel research, and persistent memory.

Key Features Demonstrated:
1. Autonomous Sub-Agent Spawning (agent decides when to parallelize)
2. RPI Workflow (Research -> Plan -> Implement)
3. Background Task Execution (for long-running operations)
4. Persistent Memory Integration
"""

import asyncio
import os
import sys

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from omnicoreagent import DeepAgent
from omnicoreagent.core.utils import logger
from dotenv import load_dotenv

# Load environment variables (API keys)
load_dotenv()


async def main():
    # 1. Initialize the Deep Agent
    # Notice how simple the setup is - just like a regular OmniCoreAgent
    agent = DeepAgent(
        name="MarketAnalyst",
        system_instruction="""
        You are a senior market research analyst. 
        Your goal is to provide comprehensive market analysis and strategy.
        """,
        model_config={
            "provider": "openai", 
            "model": "gpt-4o",
            "api_key": os.getenv("OPENAI_API_KEY")
        },
        project_name="ev_market_analysis_2025",
        debug=True
    )
    
    try:
        print("\n" + "="*50)
        print("🤖 Initializing Deep Agent...")
        print("="*50)
        await agent.initialize()
        
        # 2. Research Phase
        # The agent will autonomously decide if it needs to spawn sub-agents
        # based on the complexity of the query.
        print("\n" + "="*50)
        print("🔍 PHASE 1: RESEARCH (Autonomous Parallelization)")
        print("="*50)
        
        research_query = """
        Conduct a comprehensive analysis of the Electric Vehicle (EV) market for 2025.
        Focus on these 3 specific areas:
        1. US Market Trends and Policy Changes
        2. EU Market Growth and Regulations
        3. Emerging Battery Technologies
        
        Please research these in parallel for efficiency.
        """
        
        research_result = await agent.research(research_query)
        print(f"\n✅ Research Complete!")
        print(f"Response Summary: {str(research_result.get('response'))[:200]}...")
        
        # 3. Plan Phase
        # Creates a detailed implementation plan based on the research
        print("\n" + "="*50)
        print("📝 PHASE 2: PLANNING")
        print("="*50)
        
        plan_goal = "Create a strategic roadmap for a new EV startup entering the US market."
        plan_result = await agent.plan(plan_goal)
        print(f"\n✅ Plan Created!")
        print(f"Plan Summary: {str(plan_result.get('response'))[:200]}...")
        
        # 4. Background Task Example
        # Demonstrating how to manually trigger a long-running background task
        # (The agent can also do this autonomously)
        print("\n" + "="*50)
        print("⚙️ DEMO: AUTONOMOUS SUB-AGENT SPAWNING")
        print("="*50)
        
        # We'll ask the agent to do something that triggers its internal tool usage
        complex_task = """
        I need to process a large dataset of customer feedback reviews. 
        This is a long-running task that should happen in the background.
        Please spawn a background agent to handle this simulation.
        Task: 'Process 10,000 simulations of customer sentiment analysis'
        """
        
        response = await agent.run(complex_task)
        print(f"\n✅ Agent Response: {response.get('response')}")
        
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\n" + "="*50)
        print("🧹 Cleanup")
        print("="*50)
        await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
