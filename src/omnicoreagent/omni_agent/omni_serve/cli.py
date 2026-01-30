"""
OmniServe CLI - Command-line interface for quick server deployment.

Usage:
    omniserve run --agent path/to/agent.py --port 8000
    omniserve quickstart --provider gemini --model gemini-2.0-flash
    omniserve config --show
"""

import os
import sys
import importlib.util
from pathlib import Path
from typing import Optional

import click

from omnicoreagent.core.utils import logger


def _load_agent_from_file(path: str):
    """
    Load an agent from a Python file.

    The file should define an `agent` variable or an `create_agent()` function.
    """
    file_path = Path(path).resolve()
    if not file_path.exists():
        raise click.ClickException(f"Agent file not found: {path}")

    # Load the module
    spec = importlib.util.spec_from_file_location("agent_module", file_path)
    if spec is None or spec.loader is None:
        raise click.ClickException(f"Failed to load module from: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["agent_module"] = module

    # Add the file's directory to path for relative imports
    sys.path.insert(0, str(file_path.parent))

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise click.ClickException(f"Error loading agent file: {e}")

    # Look for agent variable or create_agent function
    if hasattr(module, "agent"):
        return module.agent
    elif hasattr(module, "create_agent"):
        return module.create_agent()
    else:
        raise click.ClickException(
            f"Agent file must define an 'agent' variable or 'create_agent()' function"
        )


@click.group()
@click.version_option(version="0.0.1", prog_name="omniserve")
def cli():
    """OmniServe - Production-ready API server for AI agents.

    Deploy OmniCoreAgent or DeepAgent as a REST/SSE API with a single command.
    """
    pass


@cli.command()
@click.option(
    "--agent", "-a",
    type=click.Path(exists=True),
    help="Path to Python file containing the agent",
)
@click.option("--host", "-h", default=None, help="Host to bind to (default: 0.0.0.0)")
@click.option("--port", "-p", default=None, type=int, help="Port to bind to (default: 8000)")
@click.option("--workers", "-w", default=1, type=int, help="Number of workers")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
@click.option("--no-docs", is_flag=True, help="Disable Swagger UI")
@click.option("--cors-origins", default="*", help="Comma-separated CORS origins")
@click.option("--auth-token", default=None, help="Enable auth with this token")
@click.option("--rate-limit", default=None, type=int, help="Rate limit (requests per minute)")
def run(
    agent: Optional[str],
    host: Optional[str],
    port: Optional[int],
    workers: int,
    reload: bool,
    no_docs: bool,
    cors_origins: str,
    auth_token: Optional[str],
    rate_limit: Optional[int],
):
    """Run an agent as an API server.

    Example:
        omniserve run --agent my_agent.py --port 8000
    """
    from omnicoreagent import OmniServe, OmniServeConfig

    if agent is None:
        raise click.ClickException(
            "Please specify an agent file with --agent or use 'omniserve quickstart'"
        )

    # Load the agent
    click.echo(f"📦 Loading agent from: {agent}")
    loaded_agent = _load_agent_from_file(agent)
    click.echo(f"✅ Loaded agent: {loaded_agent.name}")

    # Build config
    config = OmniServeConfig(
        host=host or "0.0.0.0",
        port=port or 8000,
        workers=workers,
        enable_docs=not no_docs,
        cors_origins=[o.strip() for o in cors_origins.split(",")],
        auth_enabled=auth_token is not None,
        auth_token=auth_token,
        rate_limit_enabled=rate_limit is not None,
        rate_limit_requests=rate_limit or 100,
        rate_limit_window=60,
    )

    # Start server
    click.echo("")
    click.echo("=" * 50)
    click.echo("🚀 OmniServe v0.0.1")
    click.echo("=" * 50)
    click.echo(f"Agent: {loaded_agent.name}")
    click.echo(f"Server: http://{config.host}:{config.port}")
    if config.enable_docs:
        click.echo(f"Docs: http://{config.host}:{config.port}/docs")
    click.echo(f"Metrics: http://{config.host}:{config.port}/prometheus")
    click.echo("")
    click.echo("Features Enabled:")
    click.echo(f"  • Auth: {'✓ (Bearer token)' if config.auth_enabled else '✗ (use --auth-token to enable)'}")
    click.echo(f"  • Rate Limit: {'✓ ' + str(config.rate_limit_requests) + '/min' if config.rate_limit_enabled else '✗ (use --rate-limit N to enable)'}")
    click.echo(f"  • CORS: {config.cors_origins}")
    click.echo("")
    click.echo("💡 Available options: --auth-token, --rate-limit, --cors-origins, --no-docs, --reload")
    click.echo("   Run 'omniserve run --help' for all options")
    click.echo("=" * 50)
    click.echo("")

    server = OmniServe(loaded_agent, config=config)
    server.start(reload=reload)


@cli.command()
@click.option("--provider", "-p", default="gemini", help="LLM provider (openai, gemini, anthropic)")
@click.option("--model", "-m", default="gemini-2.0-flash", help="Model name")
@click.option("--name", "-n", default="QuickAgent", help="Agent name")
@click.option("--instruction", "-i", default="You are a helpful AI assistant.", help="System instruction")
@click.option("--port", default=8000, type=int, help="Port to bind to")
@click.option("--host", default="0.0.0.0", help="Host to bind to")
def quickstart(
    provider: str,
    model: str,
    name: str,
    instruction: str,
    port: int,
    host: str,
):
    """Start a quick agent server without writing any code.

    Example:
        omniserve quickstart --provider openai --model gpt-4o --port 8000
    """
    from omnicoreagent import OmniCoreAgent, OmniServe, OmniServeConfig

    click.echo(f"🚀 Creating {provider}/{model} agent...")

    # Create agent
    agent = OmniCoreAgent(
        name=name,
        system_instruction=instruction,
        model_config={
            "provider": provider,
            "model": model,
        },
        debug=False,
    )

    config = OmniServeConfig(
        host=host,
        port=port,
        enable_docs=True,
        cors_origins=["*"],
    )

    click.echo("")
    click.echo("=" * 50)
    click.echo("🚀 OmniServe v0.0.1 - Quickstart")
    click.echo("=" * 50)
    click.echo(f"Agent: {name}")
    click.echo(f"Model: {provider}/{model}")
    click.echo(f"Server: http://{host}:{port}")
    click.echo(f"Docs: http://{host}:{port}/docs")
    click.echo(f"Metrics: http://{host}:{port}/prometheus")
    click.echo("")
    click.echo("Features (default):")
    click.echo("  • Auth: ✗ (use 'omniserve run' with --auth-token to enable)")
    click.echo("  • Rate Limit: ✗ (use 'omniserve run' with --rate-limit to enable)")
    click.echo("  • CORS: * (all origins)")
    click.echo("")
    click.echo("💡 For more control, use 'omniserve run --agent my_agent.py'")
    click.echo("   Options: --auth-token, --rate-limit, --cors-origins, --reload")
    click.echo("=" * 50)
    click.echo("")
    click.echo("Test with:")
    click.echo(f'  curl -X POST http://{host}:{port}/run/sync \\')
    click.echo('    -H "Content-Type: application/json" \\')
    click.echo('    -d \'{"query": "Hello!"}\'')
    click.echo("")

    server = OmniServe(agent, config=config)
    server.start()


@cli.command("config")
@click.option("--show", is_flag=True, help="Show current configuration from environment")
@click.option("--env-example", is_flag=True, help="Print example .env file")
def config_cmd(show: bool, env_example: bool):
    """View or generate configuration.

    Example:
        omniserve config --show
        omniserve config --env-example > .env
    """
    from omnicoreagent.omni_agent.omni_serve import OmniServeConfig

    if env_example:
        click.echo("""# OmniServe Configuration
# Copy this to .env and modify as needed

# Server
OMNISERVE_HOST=0.0.0.0
OMNISERVE_PORT=8000
OMNISERVE_WORKERS=1

# API
OMNISERVE_API_PREFIX=
OMNISERVE_ENABLE_DOCS=true
OMNISERVE_ENABLE_REDOC=true

# CORS
OMNISERVE_CORS_ENABLED=true
OMNISERVE_CORS_ORIGINS=*
OMNISERVE_CORS_CREDENTIALS=true

# Authentication
OMNISERVE_AUTH_ENABLED=false
OMNISERVE_AUTH_TOKEN=

# Logging
OMNISERVE_REQUEST_LOGGING=true
OMNISERVE_LOG_LEVEL=INFO

# Rate Limiting
OMNISERVE_RATE_LIMIT_ENABLED=false
OMNISERVE_RATE_LIMIT_REQUESTS=100
OMNISERVE_RATE_LIMIT_WINDOW=60

# Timeout
OMNISERVE_REQUEST_TIMEOUT=300
""")
        return

    if show:
        config = OmniServeConfig.from_env()
        click.echo("Current OmniServe Configuration:")
        click.echo("-" * 40)
        for key, value in config.model_dump().items():
            # Mask auth token
            if key == "auth_token" and value:
                value = value[:4] + "****" + value[-4:] if len(value) > 8 else "****"
            click.echo(f"  {key}: {value}")
        return

    click.echo("Use --show to view current config or --env-example for template")


@cli.command("generate-deployment")
@click.option("--file", "-f", "file_path", type=click.Path(exists=True), help="Path to your python file (e.g. app.py or agent.py)")
@click.option("--output-dir", "-o", default=".", help="Output directory for generated files")
def generate_deployment(file_path: str, output_dir: str):
    """Generate Docker deployment configuration.
    
    Interactively generates a clean docker-compose.yml and .env file 
    configured specifically for your application.
    """
    import rich
    import json
    from rich.console import Console
    from rich.prompt import Prompt, Confirm
    
    console = Console()
    console.print("[bold blue]🚀 OmniServe Deployment Generator[/bold blue]")
    console.print("This wizard will generate a production-ready Docker configuration.\n")
    
    # 1. App/Agent Configuration
    if not file_path:
        file_path = Prompt.ask("Path to your python file (e.g., app.py or agent.py)")
        
    if not os.path.exists(file_path):
        console.print(f"[bold red]Error:[/bold red] File '{file_path}' not found.")
        return

    # Inspect file content to determine mode (App vs Script)
    is_full_app = False
    try:
        with open(file_path, "r") as f:
            content = f.read()
            if "OmniServe(" in content and ".start()" in content:
                is_full_app = True
                console.print("[green]✓ Detected self-contained OmniServe app.[/green]")
            else:
                console.print("[green]✓ Detected agent definition file.[/green]")
    except:
        pass

    # 2. Inspect Agent for Memory Usage (Universal)
    uses_memory = False
    memory_backend = "local"
    
    try:
        console.print("[dim]Inspecting configuration...[/dim]")
        loaded_agent = _load_agent_from_file(file_path)
        
        # Check explicit config first
        if loaded_agent.agent_config and isinstance(loaded_agent.agent_config, dict):
            if loaded_agent.agent_config.get("memory_tool_backend"):
                uses_memory = True
                memory_backend = loaded_agent.agent_config.get("memory_tool_backend")
                
        # Check tools if not found in config
        if not uses_memory:
            # Check local tools
            if loaded_agent.local_tools:
                tools_list = loaded_agent.local_tools.get_available_tools()
                for tool in tools_list:
                    tool_name = tool.get("name") if isinstance(tool, dict) else tool.name
                    if tool_name.startswith("memory_"):
                        uses_memory = True
                        break
                        
        if uses_memory:
            console.print(f"[green]✓ Memory usage detected (backend: {memory_backend})[/green]")
        else:
            console.print("[dim]No memory tools detected. Storage configuration skipped.[/dim]")
            
    except Exception as e:
        console.print(f"[yellow]Warning: Could not inspect agent ({e}). Defaulting to interactive mode.[/yellow]")
        # If inspection fails (e.g. complex app structure), we ask the user as fallback
        if Confirm.ask("Could not verify memory usage. Does your app use memory tools (S3/R2)?", default=False):
            uses_memory = True
            memory_backend = None

    # 3. Storage Configuration
    env_vars = {}
    storage_type = None

    if uses_memory:
        console.print("\n[bold yellow]Storage Configuration[/bold yellow]")
        
        # If we detected a specific backend, use it as default or skip asking if obvious
        if memory_backend and memory_backend in ["s3", "r2", "local"]:
             storage_type = memory_backend
             # console.print(f"Using configured backend: [bold cyan]{storage_type}[/bold cyan]")
        else:
            storage_type = Prompt.ask(
                "Select storage backend", 
                choices=["local", "s3", "r2"], 
                default="local"
            )
        
        if storage_type == "s3":
            console.print("[dim]Added AWS S3 placeholders to .env[/dim]")
            env_vars["AWS_S3_BUCKET"] = ""
            env_vars["AWS_REGION"] = "us-east-1"
            env_vars["AWS_ACCESS_KEY_ID"] = ""
            env_vars["AWS_SECRET_ACCESS_KEY"] = ""
            
        elif storage_type == "r2":
            console.print("[dim]Added Cloudflare R2 placeholders to .env[/dim]")
            env_vars["R2_BUCKET_NAME"] = ""
            env_vars["R2_ACCOUNT_ID"] = ""
            env_vars["R2_ACCESS_KEY_ID"] = ""
            env_vars["R2_SECRET_ACCESS_KEY"] = ""
    else:
        storage_type = "none"

    # 4. Automatic LLM Configuration (Placeholder)
    # We automatically add a placeholder for LLM_API_KEY to ensure the user knows to set it.
    env_vars["LLM_API_KEY"] = "" 

    # Generate Files
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)
    
    # .env
    env_content = "# OmniServe Environment Variables\n"
    if not is_full_app:
        env_content += f"OMNISERVE_HOST=0.0.0.0\nOMNISERVE_PORT=8000\n"

    for k, v in env_vars.items():
        if v:
            env_content += f"{k}={v}\n"
        else:
            env_content += f"{k}=  # <--- UPDATE THIS\n"
        
    env_file = out_path / ".env"
    with open(env_file, "w") as f:
        f.write(env_content)
        
    # docker-compose.yml
    # We maintain the folder structure to ensure imports work correctly.
    # So we mount context (.) to /app
    
    # Calculate relative path from current working directory (where docker-compose is)
    # to the target file.
    try:
        rel_path = os.path.relpath(Path(file_path).resolve(), Path.cwd())
    except ValueError:
        # Fallback if on different drives (Windows) or other issues
        rel_path = os.path.basename(file_path)

    # Determine command based on mode
    # Since we mount . to /app, the file is at /app/{rel_path}
    if is_full_app:
        # Run python directly
        cmd = ["python", f"/app/{rel_path}"]
        agent_msg = f"python /app/{rel_path}"
    else:
        # Run via omniserve CLI wrapper
        cmd = ["omniserve", "run", "--agent", f"/app/{rel_path}"]
        agent_msg = f"omniserve run --agent /app/{rel_path}"
    
    # Create necessary directories for persistence (so they are owned by user, not root)
    # We create them in the current directory (where docker-compose is run)
    
    # We assume output_dir is where we want to run docker-compose
    # But usually docker-compose is run at the root. 
    # Let's assume the user runs generate-deployment at the root.
    
    artifacts_dir = Path(".omnicoreagent_artifacts")
    skills_dir = Path(".agents/skills")
    
    # Ensure output dirs exist so we can mount them
    artifacts_dir.mkdir(exist_ok=True)
    
    compose_content = f"""version: '3.8'

services:
  omniserver:
    build: 
      context: .
      dockerfile: Dockerfile
    container_name: omniserver
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - .:/app
      - ./.omnicoreagent_artifacts:/app/.omnicoreagent_artifacts"""

    # Only mount skills if they exist
    if skills_dir.exists():
        compose_content += f"\n      - ./{skills_dir}:/app/.agents/skills"

    if storage_type == "local":
        compose_content += "\n      - ./memories:/app/memories"
    
    compose_content += f"""
    command: {json.dumps(cmd)}
    restart: unless-stopped
"""

    compose_file = out_path / "docker-compose.yml"
    with open(compose_file, "w") as f:
        f.write(compose_content)

    # Dockerfile (End-User / Production version)
    dockerfile_content = f"""FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir omnicoreagent
# RUN pip install uv && uv init && uv add omnicoreagent

# COPY requirements.txt .
# RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["omniserve", "quickstart"]
"""
    
    dockerfile_path = out_path / "Dockerfile"
    with open(dockerfile_path, "w") as f:
        f.write(dockerfile_content)

    console.print(f"\n[bold green]✓ Generated configuration in {output_dir}:[/bold green]")
    console.print(f"  - {env_file}")
    console.print(f"  - {compose_file}")
    console.print(f"  - {dockerfile_path}")
    console.print("\n[bold]Next Steps:[/bold]")
    console.print("1. [bold yellow]Update .env with your LLM_API_KEY keys![/bold yellow]")
    
    if storage_type == "s3":
        console.print("   [bold yellow]Update .env with your AWS S3 configuration![/bold yellow]")
    elif storage_type == "r2":
        console.print("   [bold yellow]Update .env with your Cloudflare R2 configuration![/bold yellow]")
        
    console.print("2. [dim]Run command to start server:[/dim]")
    console.print(f"   docker-compose up -d --build")


def main():
    """Entry point for the CLI."""
    cli()



if __name__ == "__main__":
    main()
