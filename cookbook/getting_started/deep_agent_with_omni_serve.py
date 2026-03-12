"""
OmniServe with DeepAgent Cookbook Example

This example demonstrates how to turn a DeepAgent into a
production-ready FastAPI server using OmniServe.

Run this script:
    cd /path/to/omnicoreagent
    uv run python cookbook/getting_started/deep_agent_with_omni_serve.py

Then test the endpoints:
    # Health check
    curl http://localhost:8000/health

    # Readiness check (shows if DeepAgent is initialized)
    curl http://localhost:8000/ready

    # Run a query
    curl -X POST http://localhost:8000/run/sync \
        -H "Content-Type: application/json" \
        -d '{"query": "Research and summarize the key benefits of AI agents."}'

Visit http://localhost:8000/docs for the interactive Swagger UI.
"""

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from omnicoreagent import DeepAgent, OmniServe, OmniServeConfig


def main():
    """Run the OmniServe server with a DeepAgent."""

    # Create the DeepAgent
    agent = DeepAgent(
        name="ResearchAgent",
        system_instruction="""You are a research assistant with multi-agent
        orchestration capabilities. You can spawn subagents to help with
        complex research tasks. Break down problems into subtasks and
        coordinate their execution.""",
        model_config={
            "provider": "gemini",
            "model": "gemini-2.0-flash",
        },
        debug=False,
    )

    # Configure the server
    config = OmniServeConfig(
        host="0.0.0.0",
        port=8000,
        cors_enabled=True,
        enable_docs=True,
        request_logging=True,
    )

    # Create and start the server
    server = OmniServe(
        agent=agent,
        config=config,
        title="Research Agent API",
        description="A multi-agent research API powered by DeepAgent and OmniServe",
    )

    print("\n" + "=" * 60)
    print("🧠 OmniServe with DeepAgent")
    print("=" * 60)
    print(f"Server: http://localhost:{config.port}")
    print(f"Swagger UI: http://localhost:{config.port}/docs")
    print("=" * 60)
    print("\nNote: DeepAgent will be initialized on first request.")
    print("Check /ready endpoint for initialization status.\n")

    # Start the server (blocking)
    server.start()


if __name__ == "__main__":
    main()
