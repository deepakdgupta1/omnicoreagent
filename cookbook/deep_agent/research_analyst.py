"""
Research Analyst with DeepAgent.

Shows how DeepAgent uses RPI workflow for complex research.

Run: python cookbook/deep_agent/research_analyst.py
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from omnicoreagent import DeepAgent  # noqa: E402


async def main():
    agent = DeepAgent(
        name="MarketResearcher",
        system_instruction="""
You are a senior market research analyst.
You conduct thorough research, create structured analyses, and provide strategic recommendations.
""",
        model_config={
            "provider": os.getenv("LLM_PROVIDER", "openai"),
            "model": os.getenv("LLM_MODEL", "gpt-4o"),
        },
        project_name="market_research",
    )

    await agent.initialize()
    print("✓ Research Agent initialized")

    result = await agent.run("""
    Analyze the top 3 cloud providers (AWS, Azure, GCP) for AI workloads.

    Research their AI/ML offerings, then create a comparison and recommendation.
    """)

    print(f"\n📋 Result:\n{result['response'][:1000]}...")

    await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
