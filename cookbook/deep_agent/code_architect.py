"""
Code Architect with DeepAgent.

Shows DeepAgent for system design using RPI workflow.

Run: python cookbook/deep_agent/code_architect.py
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from omnicoreagent import DeepAgent


async def main():
    agent = DeepAgent(
        name="CodeArchitect",
        system_instruction="""
You are a senior software architect.
You research best practices, create detailed designs, and provide implementation guidance.
""",
        model_config={
            "provider": os.getenv("LLM_PROVIDER", "openai"),
            "model": os.getenv("LLM_MODEL", "gpt-4o"),
        },
        project_name="auth_system",
    )

    await agent.initialize()
    print("✓ Architect initialized")

    result = await agent.run("""
    Design an authentication system for a SaaS application.

    Requirements:
    - Email/password and social login (Google, GitHub)
    - JWT with refresh tokens
    - Role-based access control

    Research best practices, then create a detailed plan.
    """)

    print(f"\n📋 Result:\n{result['response'][:1200]}...")

    await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
