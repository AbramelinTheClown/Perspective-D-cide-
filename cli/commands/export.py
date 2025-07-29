"""
Export command for Gola CLI - handles data export and vectorization.
"""

import typer
from pathlib import Path
from typing import Optional, List
import json
import yaml
import csv
from datetime import datetime
import pandas as pd

from cli.utils.config import load_config
from cli.utils.logging import get_logger
from schemas.base import RunMetadata

app = typer.Typer(name="export", help="Export datasets and create vector stores")

logger = get_logger(__name__)


@app.command()
def dataset(
    dataset_path: Path = typer.Argument(..., help="Dataset file to export"),
    output_dir: Path = typer.Option(Path("data/exports"), "--output", "-o", help="Output directory"),
    formats: List[str] = typer.Option(["jsonl", "csv", "parquet"], "--formats", "-f", help="Export formats"),
    vectorize: bool = typer.Option(True, "--vectorize", help="Create vector embeddings"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Export a dataset in multiple formats."""
    
    try:
        # Load configuration
        config_data = load_config(config) if config else {}
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate run metadata
        run_metadata = RunMetadata(
            run_id=f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            source_path=str(dataset_path),
            created_at=datetime.now(),
            status="exporting"
        )
        
        logger.info(f"Starting dataset export: {dataset_path} -> {output_dir}")
        logger.info(f"Formats: {', '.join(formats)}")
        
        # Load dataset
        dataset = load_dataset(dataset_path)
        
        if not dataset:
            logger.error(f"Failed to load dataset: {dataset_path}")
            raise typer.Exit(1)
        
        # Export in requested formats
        export_files = {}
        
        for format_type in formats:
            try:
                export_file = export_dataset(dataset, format_type, output_dir, run_metadata.run_id)
                export_files[format_type] = str(export_file)
                
                if verbose:
                    logger.info(f"Exported {format_type}: {export_file}")
                    
            except Exception as e:
                logger.error(f"Failed to export {format_type}: {e}")
        
        # Vectorize if requested
        vector_store_path = None
        if vectorize:
            try:
                vector_store_path = create_vector_store(dataset, output_dir, run_metadata.run_id, config_data, verbose)
                export_files["vector_store"] = str(vector_store_path)
                
                if verbose:
                    logger.info(f"Vector store created: {vector_store_path}")
                    
            except Exception as e:
                logger.error(f"Failed to create vector store: {e}")
        
        # Update run metadata
        run_metadata.status = "completed"
        run_metadata.completed_at = datetime.now()
        run_metadata.files_processed = 1
        run_metadata.chunks_created = len(dataset.get("data", []))
        
        # Save export metadata
        export_metadata = {
            "run_metadata": run_metadata.dict(),
            "export_files": export_files,
            "export_stats": {
                "total_chunks": len(dataset.get("data", [])),
                "formats_exported": list(export_files.keys()),
                "vectorized": vector_store_path is not None
            }
        }
        
        metadata_file = output_dir / f"export_metadata_{run_metadata.run_id}.json"
        with open(metadata_file, 'w') as f:
            json.dump(export_metadata, f, indent=2, default=str)
        
        logger.info(f"Export completed: {metadata_file}")
        
    except Exception as e:
        logger.error(f"Failed to export dataset: {e}")
        raise typer.Exit(1)


@app.command()
def status(
    output_dir: Path = typer.Option(Path("data/exports"), "--output", "-o", help="Output directory"),
    format: str = typer.Option("table", "--format", "-f", help="Output format (table, json, yaml)")
):
    """Show export status and statistics."""
    
    try:
        if not output_dir.exists():
            logger.warning(f"Output directory does not exist: {output_dir}")
            return
        
        # Find metadata files
        metadata_files = list(output_dir.glob("export_metadata_*.json"))
        
        if not metadata_files:
            logger.info("No export runs found")
            return
        
        # Load and analyze metadata
        runs = []
        for metadata_file in sorted(metadata_files, key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                with open(metadata_file, 'r') as f:
                    export_data = json.load(f)
                    run_data = export_data.get("run_metadata", {})
                    export_stats = export_data.get("export_stats", {})
                    
                    runs.append({
                        "file": metadata_file.name,
                        "run_id": run_data.get("run_id"),
                        "source": run_data.get("source_path"),
                        "status": run_data.get("status"),
                        "created": run_data.get("created_at"),
                        "completed": run_data.get("completed_at"),
                        "total_chunks": export_stats.get("total_chunks", 0),
                        "formats_exported": export_stats.get("formats_exported", []),
                        "vectorized": export_stats.get("vectorized", False)
                    })
            except Exception as e:
                logger.warning(f"Failed to load metadata {metadata_file}: {e}")
        
        if format == "json":
            typer.echo(json.dumps(runs, indent=2, default=str))
        elif format == "yaml":
            typer.echo(yaml.dump(runs, default_flow_style=False))
        else:
            display_export_status(runs)
            
    except Exception as e:
        logger.error(f"Failed to show export status: {e}")
        raise typer.Exit(1)


@app.command()
def vectorize(
    dataset_path: Path = typer.Argument(..., help="Dataset file to vectorize"),
    output_dir: Path = typer.Option(Path("data/vector_stores"), "--output", "-o", help="Output directory"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Create vector embeddings for a dataset."""
    
    try:
        # Load configuration
        config_data = load_config(config) if config else {}
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting vectorization: {dataset_path} -> {output_dir}")
        
        # Load dataset
        dataset = load_dataset(dataset_path)
        
        if not dataset:
            logger.error(f"Failed to load dataset: {dataset_path}")
            raise typer.Exit(1)
        
        # Create vector store
        vector_store_path = create_vector_store(dataset, output_dir, f"vector_{datetime.now().strftime('%Y%m%d_%H%M%S')}", config_data, verbose)
        
        logger.info(f"Vectorization completed: {vector_store_path}")
        
    except Exception as e:
        logger.error(f"Failed to vectorize dataset: {e}")
        raise typer.Exit(1)


def load_dataset(dataset_path: Path) -> dict:
    """Load dataset from file."""
    
    if dataset_path.suffix.lower() == ".jsonl":
        # Load JSONL format
        data = []
        with open(dataset_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        
        return {
            "metadata": {
                "source": str(dataset_path),
                "total_chunks": len(data),
                "created_at": datetime.now().isoformat()
            },
            "data": data
        }
    else:
        # Load JSON format
        with open(dataset_path, 'r') as f:
            return json.load(f)


def export_dataset(dataset: dict, format_type: str, output_dir: Path, run_id: str) -> Path:
    """Export dataset in the specified format."""
    
    data = dataset.get("data", [])
    
    if format_type == "jsonl":
        # Export as JSONL
        export_file = output_dir / f"dataset_{run_id}.jsonl"
        with open(export_file, 'w') as f:
            for item in data:
                f.write(json.dumps(item, default=str) + '\n')
        
    elif format_type == "csv":
        # Export as CSV
        export_file = output_dir / f"dataset_{run_id}.csv"
        
        # Flatten the data for CSV export
        flattened_data = []
        for item in data:
            flat_item = {
                "chunk_id": item.get("chunk_id", ""),
                "file_path": item.get("file_path", ""),
                "text": item.get("text", "")
            }
            
            # Add task results
            tasks = item.get("tasks", {})
            for task_name, task_result in tasks.items():
                if isinstance(task_result, dict):
                    for key, value in task_result.items():
                        flat_item[f"{task_name}_{key}"] = str(value)
                else:
                    flat_item[task_name] = str(task_result)
            
            flattened_data.append(flat_item)
        
        if flattened_data:
            df = pd.DataFrame(flattened_data)
            df.to_csv(export_file, index=False)
        
    elif format_type == "parquet":
        # Export as Parquet
        export_file = output_dir / f"dataset_{run_id}.parquet"
        
        # Flatten the data for Parquet export
        flattened_data = []
        for item in data:
            flat_item = {
                "chunk_id": item.get("chunk_id", ""),
                "file_path": item.get("file_path", ""),
                "text": item.get("text", "")
            }
            
            # Add task results
            tasks = item.get("tasks", {})
            for task_name, task_result in tasks.items():
                if isinstance(task_result, dict):
                    for key, value in task_result.items():
                        flat_item[f"{task_name}_{key}"] = str(value)
                else:
                    flat_item[task_name] = str(task_result)
            
            flattened_data.append(flat_item)
        
        if flattened_data:
            df = pd.DataFrame(flattened_data)
            df.to_parquet(export_file, index=False)
        
    elif format_type == "json":
        # Export as JSON
        export_file = output_dir / f"dataset_{run_id}.json"
        with open(export_file, 'w') as f:
            json.dump(dataset, f, indent=2, default=str)
        
    else:
        raise ValueError(f"Unsupported format: {format_type}")
    
    return export_file


def create_vector_store(dataset: dict, output_dir: Path, run_id: str, config: dict, verbose: bool) -> Path:
    """Create vector embeddings for the dataset."""
    
    try:
        # Import vector store components
        from pipeline.export.vector_store import VectorStore
        
        # Initialize vector store
        vector_store = VectorStore(config.get("vector_store", {}))
        
        # Create embeddings for text chunks
        data = dataset.get("data", [])
        texts = []
        metadata = []
        
        for item in data:
            text = item.get("text", "")
            if text.strip():
                texts.append(text)
                metadata.append({
                    "chunk_id": item.get("chunk_id", ""),
                    "file_path": item.get("file_path", ""),
                    "tasks": list(item.get("tasks", {}).keys())
                })
        
        if verbose:
            logger.info(f"Creating embeddings for {len(texts)} text chunks")
        
        # Create embeddings
        vector_store.create_embeddings(texts, metadata)
        
        # Save vector store
        vector_store_path = output_dir / f"vector_store_{run_id}"
        vector_store.save(vector_store_path)
        
        return vector_store_path
        
    except ImportError:
        logger.warning("Vector store components not available. Skipping vectorization.")
        return None
    except Exception as e:
        logger.error(f"Failed to create vector store: {e}")
        return None


def display_export_status(runs: List[dict]):
    """Display export status in a table format."""
    
    if not runs:
        return
    
    typer.echo("\n" + "="*120)
    typer.echo(f"{'Run ID':<15} {'Source':<25} {'Status':<10} {'Chunks':<8} {'Formats':<20} {'Vectorized':<12} {'Created':<20}")
    typer.echo("="*120)
    
    for run in runs:
        source = run["source"][:23] + "..." if len(run["source"]) > 25 else run["source"]
        created = run["created"][:19] if run["created"] else "N/A"
        formats = ", ".join(run["formats_exported"])[:18] + "..." if len(", ".join(run["formats_exported"])) > 20 else ", ".join(run["formats_exported"])
        vectorized = "Yes" if run["vectorized"] else "No"
        
        typer.echo(f"{run['run_id']:<15} {source:<25} {run['status']:<10} {run['total_chunks']:<8} "
                  f"{formats:<20} {vectorized:<12} {created:<20}")
    
    typer.echo("="*120)


if __name__ == "__main__":
    app() 