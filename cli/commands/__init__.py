"""
Gola CLI Commands Package

This package contains all the command modules for the Gola CLI.
"""

from . import plan
from . import ingest
from . import build
from . import export
from . import hub

# Placeholder modules for future implementation
try:
    from . import validate
except ImportError:
    # Create a placeholder validate module
    import typer
    validate = typer.Typer(name="validate", help="Validation commands (not implemented)")

try:
    from . import crawl
except ImportError:
    # Create a placeholder crawl module
    import typer
    crawl = typer.Typer(name="crawl", help="Web crawling commands (not implemented)")

try:
    from . import mcp
except ImportError:
    # Create a placeholder mcp module
    import typer
    mcp = typer.Typer(name="mcp", help="MCP server commands (not implemented)")

__all__ = [
    "plan",
    "ingest", 
    "build",
    "validate",
    "export",
    "crawl",
    "mcp",
    "hub"
] 