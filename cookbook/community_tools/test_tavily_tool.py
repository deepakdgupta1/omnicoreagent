from omnicoreagent.community import TavilyTools
from omnicoreagent.omni_agent import OmniAgent


def main():
    print("Initialize TavilyTools...")
    # TavilyTools is a Toolkit
    tavily_toolkit = TavilyTools(api_key="tvly-test")

    print(f"Toolkit initialized: {tavily_toolkit.name}")

    # NEW: Get compatible Tool objects
    print("Calling get_tools()...")
    tools = tavily_toolkit.get_tools()

    print(f"Got {len(tools)} tools:")
    for t in tools:
        print(f" - {t.name}: {t.description[:50]}...")
        print(f"   Async: {t.is_async}")
        print(f"   Schema: {list(t.inputSchema['properties'].keys())}")

    # Create a dummy agent to test registration
    print("\nInitializing Agent with these tools...")
    _agent = OmniAgent(tools=tools)
    print("Agent initialized successfully with ported tools!")


if __name__ == "__main__":
    main()
