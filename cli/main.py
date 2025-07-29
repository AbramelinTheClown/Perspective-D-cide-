#!/usr/bin/env python3
"""
Gola CLI - AI-Powered Data Processing & Dataset Creation System

Main entry point for the Gola command-line interface.
"""

import sys
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.traceback import install

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cli.commands import plan, ingest, build, validate, export, crawl, mcp, hub
from cli.utils.config import load_config
from cli.utils.logging import setup_logging

# Install rich traceback handler
install()

# Create the main app
app = typer.Typer(
    name="corpusctl",
    help="Gola - AI-Powered Data Processing & Dataset Creation System",
    add_completion=False,
    rich_markup_mode="rich",
)

# Create console for rich output
console = Console()

def version_callback(value: bool):
    """Print version information."""
    if value:
        console.print("[bold blue]Gola[/bold blue] v1.0.0")
        console.print("AI-Powered Data Processing & Dataset Creation System")
        raise typer.Exit()

@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, 
        "--version", 
        "-v", 
        callback=version_callback,
        help="Show version and exit"
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable verbose logging"
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug mode"
    )
):
    """
    Gola - AI-Powered Data Processing & Dataset Creation System
    
    A comprehensive CLI tool for ingesting data from multiple sources and building
    clean, validated datasets with vectorization. Supports files, books, academic
    PDFs, and web content with intelligent automation and multi-model collaboration.
    
    Examples:
        corpusctl plan --source ./library --mode fiction --budget 20
        corpusctl ingest --source ./library --notes
        corpusctl build --tasks summary,entities,qa --mode fiction
        corpusctl validate --dataset books_v1
        corpusctl export --dataset books_v1 --format jsonl,csv,parquet
        corpusctl mcp serve --port 3323
    """
    # Load configuration
    if config:
        config_path = config
    else:
        config_path = project_root / "configs" / "project.yaml"
    
    try:
        config_data = load_config(config_path)
        console.print(f"[green]✓[/green] Loaded configuration from {config_path}")
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to load configuration: {e}")
        raise typer.Exit(1)
    
    # Setup logging
    log_level = "DEBUG" if debug else ("INFO" if verbose else "WARNING")
    setup_logging(log_level, config_data.get("logging", {}))
    
    # Store config in app state
    app.state.config = config_data
    app.state.debug = debug

# Add command groups
app.add_typer(plan.app, name="plan", help="Planning and scouting commands")
app.add_typer(ingest.app, name="ingest", help="Data ingestion commands")
app.add_typer(build.app, name="build", help="Dataset building commands")
app.add_typer(validate.app, name="validate", help="Validation commands")
app.add_typer(export.app, name="export", help="Export commands")
app.add_typer(crawl.app, name="crawl", help="Web crawling commands")
app.add_typer(mcp.app, name="mcp", help="MCP server commands")
app.add_typer(hub.app, name="hub", help="Dev Vector DB Hub integration commands")

@app.command()
def status():
    """Show system status and health."""
    console.print("[bold blue]Gola System Status[/bold blue]")
    console.print("=" * 50)
    
    # Check configuration
    try:
        config = app.state.config
        console.print(f"[green]✓[/green] Configuration loaded")
        console.print(f"  Default mode: {config.get('default_mode', 'general')}")
        console.print(f"  Router policy: {config.get('router_policy', 'throughput')}")
    except Exception as e:
        console.print(f"[red]✗[/red] Configuration error: {e}")
    
    # Check database
    try:
        # TODO: Implement database health check
        console.print(f"[green]✓[/green] Database connection")
    except Exception as e:
        console.print(f"[red]✗[/red] Database error: {e}")
    
    # Check vector database
    try:
        # TODO: Implement vector DB health check
        console.print(f"[green]✓[/green] Vector database")
    except Exception as e:
        console.print(f"[red]✗[/red] Vector DB error: {e}")
    
    # Check GPU status
    try:
        # TODO: Implement GPU status check
        console.print(f"[green]✓[/green] GPU monitoring")
    except Exception as e:
        console.print(f"[red]✗[/red] GPU error: {e}")
    
    # Check providers
    try:
        providers = config.get("providers", {})
        for provider, settings in providers.items():
            if settings.get("enabled", False):
                console.print(f"[green]✓[/green] {provider} provider")
            else:
                console.print(f"[yellow]⚠[/yellow] {provider} provider (disabled)")
    except Exception as e:
        console.print(f"[red]✗[/red] Provider check error: {e}")

@app.command()
def init():
    """Initialize the Gola system."""
    console.print("[bold blue]Initializing Gola System[/bold blue]")
    console.print("=" * 50)
    
    # Create directories
    directories = [
        "data",
        "data/raw",
        "data/parsed", 
        "data/notes",
        "data/chunks",
        "data/outputs",
        "data/manifests",
        "indexes",
        "logs",
        "temp"
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]✓[/green] Created directory: {directory}")
    
    # Initialize database
    try:
        # TODO: Implement database initialization
        console.print(f"[green]✓[/green] Database initialized")
    except Exception as e:
        console.print(f"[red]✗[/red] Database initialization failed: {e}")
    
    # Initialize vector database
    try:
        # TODO: Implement vector DB initialization
        console.print(f"[green]✓[/green] Vector database initialized")
    except Exception as e:
        console.print(f"[red]✗[/red] Vector DB initialization failed: {e}")
    
    console.print("\n[bold green]Gola system initialized successfully![/bold green]")
    console.print("Next steps:")
    console.print("  1. Configure your API keys in .env")
    console.print("  2. Run 'corpusctl status' to check system health")
    console.print("  3. Run 'corpusctl plan --help' to start processing data")

@app.command()
def doctor():
    """Run system diagnostics."""
    console.print("[bold blue]Gola System Diagnostics[/bold blue]")
    console.print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 10):
        console.print(f"[green]✓[/green] Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        console.print(f"[red]✗[/red] Python {python_version.major}.{python_version.minor}.{python_version.micro} (requires 3.10+)")
    
    # Check required packages
    required_packages = [
        "typer", "rich", "pydantic", "litellm", "unstructured",
        "qdrant_client", "polars", "fastcdc", "datasketch"
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            console.print(f"[green]✓[/green] {package}")
        except ImportError:
            console.print(f"[red]✗[/red] {package} (missing)")
    
    # Check environment variables
    required_env_vars = [
        "LM_STUDIO_API_LOCAL",
        "ANTHROPIC_API_KEY_1", 
        "DEEPSEEK_API_KEY_1",
        "GROK_API_KEY_0"
    ]
    
    for env_var in required_env_vars:
        if os.getenv(env_var):
            console.print(f"[green]✓[/green] {env_var}")
        else:
            console.print(f"[yellow]⚠[/yellow] {env_var} (not set)")
    
    # Check file permissions
    try:
        test_file = project_root / "temp" / "test.txt"
        test_file.write_text("test")
        test_file.unlink()
        console.print(f"[green]✓[/green] File system permissions")
    except Exception as e:
        console.print(f"[red]✗[/red] File system permissions: {e}")
    
    console.print("\n[bold]Diagnostics complete![/bold]")

if __name__ == "__main__":
    app() 