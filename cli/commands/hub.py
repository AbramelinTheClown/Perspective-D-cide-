"""
Hub CLI Commands
Commands for interacting with the Dev Vector DB Hub.
"""

import json
import time
from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from hub.client_rest import HubRestClient, HubItem
from hub.sync_adapters import HubSyncAdapter

console = Console()

app = typer.Typer(
    name="hub",
    help="Dev Vector DB Hub integration commands",
    rich_markup_mode="rich",
)

def get_hub_client(config: dict) -> HubRestClient:
    """Get hub client from configuration."""
    hub_config = config.get("hub", {})
    return HubRestClient(hub_config)

def get_sync_adapter(config: dict) -> HubSyncAdapter:
    """Get sync adapter from configuration."""
    hub_config = config.get("hub", {})
    hub_client = HubRestClient(hub_config)
    return HubSyncAdapter(hub_client, hub_config)

@app.command()
def register(
    project_name: str = typer.Argument(..., help="Name of the project to register"),
    description: str = typer.Option("", "--description", "-d", help="Project description"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Config file path")
):
    """Register a new project with the Dev Vector DB Hub."""
    # Load config
    if config:
        import yaml
        with open(config) as f:
            config_data = yaml.safe_load(f)
    else:
        # Use default config
        config_data = {"hub": {}}
    
    hub_client = get_hub_client(config_data)
    
    try:
        result = hub_client.register_project(project_name, description)
        
        console.print(Panel(
            f"[bold green]Project Registered Successfully![/bold green]\n\n"
            f"Project Name: {project_name}\n"
            f"Project ID: {result.get('project_id')}\n"
            f"Description: {description or 'No description provided'}",
            title="Hub Registration"
        ))
        
        # Save project ID to config
        if "hub" not in config_data:
            config_data["hub"] = {}
        config_data["hub"]["project_id"] = result.get("project_id")
        
        # Save updated config
        if config:
            with open(config, 'w') as f:
                yaml.dump(config_data, f)
            console.print(f"[green]✓[/green] Updated config file: {config}")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to register project: {e}")
        raise typer.Exit(1)

@app.command()
def push(
    dataset_slug: str = typer.Argument(..., help="Dataset slug to push to hub"),
    project_name: str = typer.Option(None, "--project", "-p", help="Project name"),
    content_types: Optional[List[str]] = typer.Option(
        None, "--types", "-t", 
        help="Content types to push (pattern, concept, documentation, idea)"
    ),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Config file path")
):
    """Push dataset insights to the Dev Vector DB Hub."""
    # Load config
    if config:
        import yaml
        with open(config) as f:
            config_data = yaml.safe_load(f)
    else:
        config_data = {"hub": {}}
    
    # Load dataset data
    dataset_path = Path(f"data/outputs/{dataset_slug}/manifest.json")
    if not dataset_path.exists():
        console.print(f"[red]✗[/red] Dataset manifest not found: {dataset_path}")
        raise typer.Exit(1)
    
    with open(dataset_path) as f:
        dataset_data = json.load(f)
    
    # Initialize hub components
    hub_client = get_hub_client(config_data)
    sync_adapter = get_sync_adapter(config_data)
    
    # Set project if specified
    if project_name:
        try:
            result = hub_client.register_project(project_name)
            config_data["hub"]["project_id"] = result.get("project_id")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not register project: {e}[/yellow]")
    
    try:
        # Sync dataset to hub
        result = sync_adapter.sync_dataset_to_hub(dataset_slug, dataset_data)
        
        # Display results
        table = Table(title=f"Hub Sync Results for {dataset_slug}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Insights", str(result["total_insights"]))
        table.add_row("Filtered Insights", str(result["filtered_insights"]))
        table.add_row("Stored Insights", str(result["stored_insights"]))
        
        console.print(table)
        
        # Show insights by type
        if result["insights_by_type"]:
            type_table = Table(title="Insights by Type")
            type_table.add_column("Type", style="cyan")
            type_table.add_column("Count", style="green")
            
            for insight_type, count in result["insights_by_type"].items():
                type_table.add_row(insight_type, str(count))
            
            console.print(type_table)
        
        console.print(f"[green]✓[/green] Successfully synced dataset {dataset_slug} to hub")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to push dataset: {e}")
        raise typer.Exit(1)

@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    content_types: Optional[List[str]] = typer.Option(
        None, "--types", "-t",
        help="Content types to search (pattern, concept, documentation, idea, code)"
    ),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum number of results"),
    threshold: float = typer.Option(0.7, "--threshold", help="Similarity threshold"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Config file path")
):
    """Search the Dev Vector DB Hub for relevant content."""
    # Load config
    if config:
        import yaml
        with open(config) as f:
            config_data = yaml.safe_load(f)
    else:
        config_data = {"hub": {}}
    
    hub_client = get_hub_client(config_data)
    
    try:
        results = hub_client.search(query, content_types, limit, threshold)
        
        if not results:
            console.print(f"[yellow]No results found for query: {query}[/yellow]")
            return
        
        # Display results
        table = Table(title=f"Search Results for: {query}")
        table.add_column("Content Type", style="cyan")
        table.add_column("Content", style="green")
        table.add_column("Similarity", style="yellow")
        table.add_column("Project", style="blue")
        
        for result in results:
            content = result.get("content", "")[:100] + "..." if len(result.get("content", "")) > 100 else result.get("content", "")
            similarity = f"{result.get('similarity', 0):.3f}"
            project = result.get("metadata", {}).get("project_id", "unknown")
            
            table.add_row(
                result.get("content_type", "unknown"),
                content,
                similarity,
                project
            )
        
        console.print(table)
        console.print(f"[blue]Found {len(results)} results[/blue]")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Search failed: {e}")
        raise typer.Exit(1)

@app.command()
def stats(
    project_id: Optional[str] = typer.Option(None, "--project", "-p", help="Project ID for specific stats"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Config file path")
):
    """Get statistics from the Dev Vector DB Hub."""
    # Load config
    if config:
        import yaml
        with open(config) as f:
            config_data = yaml.safe_load(f)
    else:
        config_data = {"hub": {}}
    
    hub_client = get_hub_client(config_data)
    
    try:
        stats_data = hub_client.get_stats(project_id)
        
        # Display stats
        if project_id:
            title = f"Project Stats for {project_id}"
        else:
            title = "Global Hub Stats"
        
        table = Table(title=title)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in stats_data.items():
            if isinstance(value, dict):
                # Handle nested stats
                for sub_key, sub_value in value.items():
                    table.add_row(f"{key}.{sub_key}", str(sub_value))
            else:
                table.add_row(key, str(value))
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to get stats: {e}")
        raise typer.Exit(1)

@app.command()
def context(
    project_id: Optional[str] = typer.Option(None, "--project", "-p", help="Project ID"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Config file path")
):
    """Get project context from the Dev Vector DB Hub."""
    # Load config
    if config:
        import yaml
        with open(config) as f:
            config_data = yaml.safe_load(f)
    else:
        config_data = {"hub": {}}
    
    hub_client = get_hub_client(config_data)
    
    try:
        context_data = hub_client.get_project_context(project_id)
        
        # Display context
        console.print(Panel(
            f"[bold blue]Project Context[/bold blue]\n\n"
            f"Project ID: {project_id or 'Default'}\n"
            f"Context Data: {json.dumps(context_data, indent=2)}",
            title="Hub Context"
        ))
        
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to get context: {e}")
        raise typer.Exit(1)

@app.command()
def store(
    content: str = typer.Argument(..., help="Content to store"),
    content_type: str = typer.Option(..., "--type", "-t", help="Content type (pattern, concept, documentation, idea, code)"),
    metadata: Optional[str] = typer.Option(None, "--metadata", "-m", help="JSON metadata"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Config file path")
):
    """Store a single item in the Dev Vector DB Hub."""
    # Load config
    if config:
        import yaml
        with open(config) as f:
            config_data = yaml.safe_load(f)
    else:
        config_data = {"hub": {}}
    
    hub_client = get_hub_client(config_data)
    
    # Parse metadata
    item_metadata = {}
    if metadata:
        try:
            item_metadata = json.loads(metadata)
        except json.JSONDecodeError:
            console.print(f"[red]✗[/red] Invalid JSON metadata: {metadata}")
            raise typer.Exit(1)
    
    try:
        # Create hub item
        hub_item = HubItem(
            content=content,
            content_type=content_type,
            metadata=item_metadata
        )
        
        # Store in hub
        result = hub_client.store_item(hub_item)
        
        console.print(Panel(
            f"[bold green]Item Stored Successfully![/bold green]\n\n"
            f"Content Type: {content_type}\n"
            f"Item ID: {result.get('id')}\n"
            f"Content: {content[:100]}{'...' if len(content) > 100 else ''}",
            title="Hub Storage"
        ))
        
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to store item: {e}")
        raise typer.Exit(1)

@app.command()
def coordinates(
    query: str = typer.Argument(..., help="Query for development coordinates"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Config file path")
):
    """Get development coordinates for a query."""
    # Load config
    if config:
        import yaml
        with open(config) as f:
            config_data = yaml.safe_load(f)
    else:
        config_data = {"hub": {}}
    
    hub_client = get_hub_client(config_data)
    
    try:
        coordinates = hub_client.get_development_coordinates(query)
        
        console.print(Panel(
            f"[bold blue]Development Coordinates[/bold blue]\n\n"
            f"Query: {query}\n"
            f"Coordinates: {json.dumps(coordinates, indent=2)}",
            title="Development Coordinates"
        ))
        
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to get coordinates: {e}")
        raise typer.Exit(1) 