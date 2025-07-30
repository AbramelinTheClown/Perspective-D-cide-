"""
Build command for Gola CLI - handles AI processing and dataset building.
"""

import typer
from pathlib import Path
from typing import Optional, List
import json
import yaml
from datetime import datetime
import asyncio

from cli.utils.config import load_config
from cli.utils.logging import get_logger
from schemas.base import RunMetadata, QualityMetrics
from pipeline.builders.base import BaseBuilder
from pipeline.builders.summary_builder import SummaryBuilder
from pipeline.builders.entities_builder import EntitiesBuilder
from pipeline.router.llm_router import LLMRouter
from pipeline.validate.validator import OutputValidator as Validator
from pipeline.monitoring.gpu import GPUMonitor

app = typer.Typer(name="build", help="Build datasets using AI processing")

logger = get_logger(__name__)


@app.command()
def dataset(
    input_dir: Path = typer.Argument(..., help="Input directory with ingested chunks"),
    output_dir: Path = typer.Option(Path("data/datasets"), "--output", "-o", help="Output directory"),
    tasks: List[str] = typer.Option(["summary", "entities"], "--tasks", "-t", help="Tasks to run"),
    mode: str = typer.Option("general", "--mode", "-m", help="Processing mode"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file"),
    parallel: int = typer.Option(4, "--parallel", "-p", help="Number of parallel workers"),
    validate: bool = typer.Option(True, "--validate", help="Validate outputs"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Build a dataset from ingested chunks using AI processing."""
    
    try:
        # Load configuration
        config_data = load_config(config) if config else {}
        
        # Initialize components
        gpu_monitor = GPUMonitor()
        llm_router = LLMRouter(config_data.get("providers", {}))
        validator = Validator() if validate else None
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate run metadata
        run_metadata = RunMetadata(
            run_id=f"build_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            source_path=str(input_dir),
            mode=mode,
            created_at=datetime.now(),
            status="building"
        )
        
        logger.info(f"Starting dataset build: {input_dir} -> {output_dir}")
        logger.info(f"Tasks: {', '.join(tasks)}")
        
        # Load chunks from input directory
        chunks = load_chunks_from_directory(input_dir)
        
        if not chunks:
            logger.error(f"No chunks found in {input_dir}")
            raise typer.Exit(1)
        
        logger.info(f"Loaded {len(chunks)} chunks for processing")
        
        # Initialize builders
        builders = initialize_builders(tasks, llm_router, config_data)
        
        # Process chunks
        results = process_chunks(chunks, builders, parallel, verbose)
        
        # Validate results if requested
        if validate and validator:
            validation_results = validate_results(results, validator, verbose)
        else:
            validation_results = None
        
        # Save dataset
        dataset_path = save_dataset(results, validation_results, run_metadata, output_dir)
        
        # Update run metadata
        run_metadata.status = "completed"
        run_metadata.completed_at = datetime.now()
        run_metadata.files_processed = len(chunks)
        run_metadata.chunks_created = len(results)
        
        # Save run metadata
        metadata_file = output_dir / f"build_metadata_{run_metadata.run_id}.json"
        with open(metadata_file, 'w') as f:
            json.dump(run_metadata.dict(), f, indent=2, default=str)
        
        logger.info(f"Dataset build completed: {dataset_path}")
        
    except Exception as e:
        logger.error(f"Failed to build dataset: {e}")
        raise typer.Exit(1)


@app.command()
def status(
    output_dir: Path = typer.Option(Path("data/datasets"), "--output", "-o", help="Output directory"),
    format: str = typer.Option("table", "--format", "-f", help="Output format (table, json, yaml)")
):
    """Show dataset build status and statistics."""
    
    try:
        if not output_dir.exists():
            logger.warning(f"Output directory does not exist: {output_dir}")
            return
        
        # Find metadata files
        metadata_files = list(output_dir.glob("build_metadata_*.json"))
        
        if not metadata_files:
            logger.info("No build runs found")
            return
        
        # Load and analyze metadata
        runs = []
        for metadata_file in sorted(metadata_files, key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                with open(metadata_file, 'r') as f:
                    run_data = json.load(f)
                    runs.append({
                        "file": metadata_file.name,
                        "run_id": run_data.get("run_id"),
                        "source": run_data.get("source_path"),
                        "mode": run_data.get("mode"),
                        "status": run_data.get("status"),
                        "created": run_data.get("created_at"),
                        "completed": run_data.get("completed_at"),
                        "files_processed": run_data.get("files_processed", 0),
                        "chunks_created": run_data.get("chunks_created", 0),
                        "quality_score": run_data.get("quality_score", 0)
                    })
            except Exception as e:
                logger.warning(f"Failed to load metadata {metadata_file}: {e}")
        
        if format == "json":
            typer.echo(json.dumps(runs, indent=2, default=str))
        elif format == "yaml":
            typer.echo(yaml.dump(runs, default_flow_style=False))
        else:
            display_build_status(runs)
            
    except Exception as e:
        logger.error(f"Failed to show build status: {e}")
        raise typer.Exit(1)


@app.command()
def validate(
    dataset_path: Path = typer.Argument(..., help="Dataset file to validate"),
    output_dir: Path = typer.Option(Path("data/datasets"), "--output", "-o", help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Validate a built dataset."""
    
    try:
        if not dataset_path.exists():
            logger.error(f"Dataset file not found: {dataset_path}")
            raise typer.Exit(1)
        
        # Initialize validator
        validator = Validator()
        
        # Load dataset
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        # Validate dataset
        validation_results = validator.validate_dataset(dataset, verbose)
        
        # Display validation results
        display_validation_results(validation_results)
        
        # Save validation report
        validation_file = output_dir / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        validation_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(validation_file, 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        
        logger.info(f"Validation report saved: {validation_file}")
        
    except Exception as e:
        logger.error(f"Failed to validate dataset: {e}")
        raise typer.Exit(1)


def load_chunks_from_directory(input_dir: Path) -> List[dict]:
    """Load chunks from the input directory."""
    
    chunks = []
    
    # Look for chunk files
    chunk_files = list(input_dir.rglob("*_chunks.jsonl"))
    
    for chunk_file in chunk_files:
        try:
            with open(chunk_file, 'r') as f:
                for line in f:
                    if line.strip():
                        chunk_data = json.loads(line)
                        chunks.append(chunk_data)
        except Exception as e:
            logger.warning(f"Failed to load chunks from {chunk_file}: {e}")
    
    return chunks


def initialize_builders(tasks: List[str], llm_router: LLMRouter, config: dict) -> dict:
    """Initialize AI builders for the specified tasks."""
    
    builders = {}
    
    for task in tasks:
        if task == "summary":
            builders[task] = SummaryBuilder(llm_router, config)
        elif task == "entities":
            builders[task] = EntitiesBuilder(llm_router, config)
        else:
            logger.warning(f"Unknown task: {task}")
    
    return builders


def process_chunks(chunks: List[dict], builders: dict, parallel: int, verbose: bool) -> List[dict]:
    """Process chunks using the specified builders."""
    
    results = []
    
    # Process chunks in batches
    batch_size = max(1, len(chunks) // parallel)
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        
        if verbose:
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
        
        # Process each chunk with each builder
        for chunk in batch:
            chunk_result = {
                "chunk_id": chunk.get("chunk_id"),
                "file_path": chunk.get("file_path"),
                "text": chunk.get("text"),
                "tasks": {}
            }
            
            for task_name, builder in builders.items():
                try:
                    task_result = builder.process_chunk(chunk)
                    chunk_result["tasks"][task_name] = task_result
                except Exception as e:
                    logger.error(f"Failed to process chunk {chunk.get('chunk_id')} with {task_name}: {e}")
                    chunk_result["tasks"][task_name] = {"error": str(e)}
            
            results.append(chunk_result)
    
    return results


def validate_results(results: List[dict], validator: Validator, verbose: bool) -> dict:
    """Validate processing results."""
    
    validation_results = {
        "total_chunks": len(results),
        "validation_errors": [],
        "quality_metrics": {},
        "task_validation": {}
    }
    
    # Validate each task type
    task_types = set()
    for result in results:
        task_types.update(result.get("tasks", {}).keys())
    
    for task_type in task_types:
        task_results = [r.get("tasks", {}).get(task_type) for r in results if r.get("tasks", {}).get(task_type)]
        
        try:
            task_validation = validator.validate_task_results(task_type, task_results)
            validation_results["task_validation"][task_type] = task_validation
        except Exception as e:
            logger.error(f"Failed to validate {task_type} results: {e}")
            validation_results["validation_errors"].append(f"{task_type}: {e}")
    
    # Calculate overall quality metrics
    try:
        quality_metrics = validator.calculate_quality_metrics(results)
        validation_results["quality_metrics"] = quality_metrics
    except Exception as e:
        logger.error(f"Failed to calculate quality metrics: {e}")
        validation_results["validation_errors"].append(f"Quality metrics: {e}")
    
    if verbose:
        logger.info(f"Validation completed: {len(validation_results['validation_errors'])} errors")
    
    return validation_results


def save_dataset(results: List[dict], validation_results: dict, run_metadata: RunMetadata, output_dir: Path) -> Path:
    """Save the built dataset."""
    
    # Create dataset structure
    dataset = {
        "metadata": {
            "run_id": run_metadata.run_id,
            "created_at": run_metadata.created_at.isoformat(),
            "mode": run_metadata.mode,
            "total_chunks": len(results),
            "tasks": list(set().union(*[set(r.get("tasks", {}).keys()) for r in results]))
        },
        "validation": validation_results,
        "data": results
    }
    
    # Save as JSONL for easy processing
    dataset_file = output_dir / f"dataset_{run_metadata.run_id}.jsonl"
    
    with open(dataset_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result, default=str) + '\n')
    
    # Save full dataset as JSON
    full_dataset_file = output_dir / f"dataset_{run_metadata.run_id}_full.json"
    with open(full_dataset_file, 'w') as f:
        json.dump(dataset, f, indent=2, default=str)
    
    # Save summary
    summary = {
        "run_id": run_metadata.run_id,
        "created_at": run_metadata.created_at.isoformat(),
        "total_chunks": len(results),
        "tasks": dataset["metadata"]["tasks"],
        "quality_score": validation_results.get("quality_metrics", {}).get("overall_score", 0) if validation_results else 0,
        "files": {
            "dataset_jsonl": str(dataset_file),
            "dataset_full": str(full_dataset_file)
        }
    }
    
    summary_file = output_dir / f"dataset_{run_metadata.run_id}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    return dataset_file


def display_build_status(runs: List[dict]):
    """Display build status in a table format."""
    
    if not runs:
        return
    
    typer.echo("\n" + "="*120)
    typer.echo(f"{'Run ID':<15} {'Source':<25} {'Mode':<10} {'Status':<10} {'Files':<8} {'Chunks':<8} {'Quality':<8} {'Created':<20}")
    typer.echo("="*120)
    
    for run in runs:
        source = run["source"][:23] + "..." if len(run["source"]) > 25 else run["source"]
        created = run["created"][:19] if run["created"] else "N/A"
        quality = f"{run['quality_score']:.2f}" if run['quality_score'] else "N/A"
        
        typer.echo(f"{run['run_id']:<15} {source:<25} {run['mode']:<10} {run['status']:<10} "
                  f"{run['files_processed']:<8} {run['chunks_created']:<8} {quality:<8} {created:<20}")
    
    typer.echo("="*120)


def display_validation_results(validation_results: dict):
    """Display validation results."""
    
    typer.echo("\n" + "="*60)
    typer.echo("📋 DATASET VALIDATION RESULTS")
    typer.echo("="*60)
    
    typer.echo(f"Total Chunks: {validation_results.get('total_chunks', 0)}")
    
    # Display quality metrics
    quality_metrics = validation_results.get("quality_metrics", {})
    if quality_metrics:
        typer.echo(f"\n📊 QUALITY METRICS:")
        typer.echo(f"  Overall Score: {quality_metrics.get('overall_score', 0):.2f}")
        typer.echo(f"  Completeness: {quality_metrics.get('completeness', 0):.2f}")
        typer.echo(f"  Consistency: {quality_metrics.get('consistency', 0):.2f}")
        typer.echo(f"  Accuracy: {quality_metrics.get('accuracy', 0):.2f}")
    
    # Display task validation
    task_validation = validation_results.get("task_validation", {})
    if task_validation:
        typer.echo(f"\n🔍 TASK VALIDATION:")
        for task, validation in task_validation.items():
            status = "✅" if validation.get("passed", False) else "❌"
            typer.echo(f"  {status} {task}: {validation.get('message', 'No message')}")
    
    # Display errors
    errors = validation_results.get("validation_errors", [])
    if errors:
        typer.echo(f"\n❌ VALIDATION ERRORS:")
        for error in errors:
            typer.echo(f"  • {error}")
    
    typer.echo("="*60)


if __name__ == "__main__":
    app() 
