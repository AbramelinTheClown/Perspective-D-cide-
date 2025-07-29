"""
Ingest command for Gola CLI - handles file ingestion and processing.
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
from schemas.base import FileMetadata, ChunkMetadata, RunMetadata
from pipeline.ingest.watcher import FileWatcher
from pipeline.ingest.parser import DocumentParser
from pipeline.normalize.chunker import ContentChunker
from pipeline.normalize.dedup import Deduplicator
from pipeline.monitoring.gpu import GPUMonitor

app = typer.Typer(name="ingest", help="Ingest and process files")

logger = get_logger(__name__)


@app.command()
def files(
    source: Path = typer.Argument(..., help="Source directory or file to ingest"),
    output_dir: Path = typer.Option(Path("data/ingested"), "--output", "-o", help="Output directory"),
    mode: str = typer.Option("general", "--mode", "-m", help="Processing mode"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file"),
    watch: bool = typer.Option(False, "--watch", "-w", help="Watch for new files"),
    recursive: bool = typer.Option(True, "--recursive", "-r", help="Process subdirectories recursively"),
    file_types: Optional[List[str]] = typer.Option(None, "--types", "-t", help="File types to process"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Ingest files from the specified source."""
    
    try:
        # Load configuration
        config_data = load_config(config) if config else {}
        
        # Initialize components
        gpu_monitor = GPUMonitor()
        parser = DocumentParser()
        chunker = ContentChunker()
        deduplicator = Deduplicator()
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate run metadata
        run_metadata = RunMetadata(
            run_id=f"ingest_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            source_path=str(source),
            mode=mode,
            created_at=datetime.now(),
            status="ingesting"
        )
        
        logger.info(f"Starting ingestion: {source} -> {output_dir}")
        
        if watch:
            # Start file watcher
            asyncio.run(watch_files(source, output_dir, run_metadata, parser, chunker, deduplicator, config_data, verbose))
        else:
            # Process existing files
            process_files(source, output_dir, run_metadata, parser, chunker, deduplicator, config_data, recursive, file_types, verbose)
        
        # Update run status
        run_metadata.status = "completed"
        run_metadata.completed_at = datetime.now()
        
        # Save run metadata
        metadata_file = output_dir / f"ingest_metadata_{run_metadata.run_id}.json"
        with open(metadata_file, 'w') as f:
            json.dump(run_metadata.dict(), f, indent=2, default=str)
        
        logger.info(f"Ingestion completed: {metadata_file}")
        
    except Exception as e:
        logger.error(f"Failed to ingest files: {e}")
        raise typer.Exit(1)


@app.command()
def status(
    output_dir: Path = typer.Option(Path("data/ingested"), "--output", "-o", help="Output directory"),
    format: str = typer.Option("table", "--format", "-f", help="Output format (table, json, yaml)")
):
    """Show ingestion status and statistics."""
    
    try:
        if not output_dir.exists():
            logger.warning(f"Output directory does not exist: {output_dir}")
            return
        
        # Find metadata files
        metadata_files = list(output_dir.glob("ingest_metadata_*.json"))
        
        if not metadata_files:
            logger.info("No ingestion runs found")
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
                        "duplicates_found": run_data.get("duplicates_found", 0)
                    })
            except Exception as e:
                logger.warning(f"Failed to load metadata {metadata_file}: {e}")
        
        if format == "json":
            typer.echo(json.dumps(runs, indent=2, default=str))
        elif format == "yaml":
            typer.echo(yaml.dump(runs, default_flow_style=False))
        else:
            display_ingestion_status(runs)
            
    except Exception as e:
        logger.error(f"Failed to show ingestion status: {e}")
        raise typer.Exit(1)


@app.command()
def validate(
    output_dir: Path = typer.Option(Path("data/ingested"), "--output", "-o", help="Output directory"),
    run_id: Optional[str] = typer.Option(None, "--run-id", "-r", help="Specific run ID to validate"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Validate ingested data integrity and quality."""
    
    try:
        if not output_dir.exists():
            logger.error(f"Output directory does not exist: {output_dir}")
            raise typer.Exit(1)
        
        # Find metadata files
        metadata_files = list(output_dir.glob("ingest_metadata_*.json"))
        
        if run_id:
            metadata_files = [f for f in metadata_files if run_id in f.name]
        
        if not metadata_files:
            logger.error("No ingestion runs found")
            raise typer.Exit(1)
        
        # Validate each run
        validation_results = []
        for metadata_file in metadata_files:
            try:
                result = validate_ingestion_run(metadata_file, output_dir, verbose)
                validation_results.append(result)
            except Exception as e:
                logger.error(f"Failed to validate {metadata_file}: {e}")
        
        # Display validation summary
        display_validation_summary(validation_results)
        
    except Exception as e:
        logger.error(f"Failed to validate ingested data: {e}")
        raise typer.Exit(1)


def process_files(source: Path, output_dir: Path, run_metadata: RunMetadata, parser: DocumentParser, 
                 chunker: ContentChunker, deduplicator: Deduplicator, config: dict, 
                 recursive: bool, file_types: List[str], verbose: bool):
    """Process existing files in the source directory."""
    
    # Get files to process
    files_to_process = get_files_to_process(source, recursive, file_types)
    
    if not files_to_process:
        logger.warning(f"No files found to process in {source}")
        return
    
    logger.info(f"Processing {len(files_to_process)} files")
    
    # Process each file
    processed_files = 0
    total_chunks = 0
    total_duplicates = 0
    
    for file_path in files_to_process:
        try:
            if verbose:
                logger.info(f"Processing: {file_path}")
            
            # Extract file metadata
            file_metadata = extract_file_metadata(file_path)
            
            # Parse document
            parsed_content = parser.parse_document(file_path)
            
            if not parsed_content:
                logger.warning(f"Failed to parse: {file_path}")
                continue
            
            # Chunk content
            chunks = chunker.chunk_content(parsed_content, file_metadata)
            
            # Deduplicate chunks
            unique_chunks, duplicates = deduplicator.deduplicate_chunks(chunks)
            
            # Save processed data
            save_processed_file(file_path, file_metadata, unique_chunks, output_dir, run_metadata.run_id)
            
            processed_files += 1
            total_chunks += len(unique_chunks)
            total_duplicates += len(duplicates)
            
            if verbose:
                logger.info(f"  Chunks: {len(unique_chunks)}, Duplicates: {len(duplicates)}")
                
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
    
    # Update run metadata
    run_metadata.files_processed = processed_files
    run_metadata.chunks_created = total_chunks
    run_metadata.duplicates_found = total_duplicates
    
    logger.info(f"Processing completed: {processed_files} files, {total_chunks} chunks, {total_duplicates} duplicates")


async def watch_files(source: Path, output_dir: Path, run_metadata: RunMetadata, parser: DocumentParser,
                     chunker: ContentChunker, deduplicator: Deduplicator, config: dict, verbose: bool):
    """Watch for new files and process them automatically."""
    
    logger.info(f"Starting file watcher for: {source}")
    
    # Initialize file watcher
    watcher = FileWatcher(source, output_dir)
    
    # Set up event handlers
    @watcher.on_file_created
    async def handle_file_created(file_path: Path):
        try:
            if verbose:
                logger.info(f"New file detected: {file_path}")
            
            # Extract file metadata
            file_metadata = extract_file_metadata(file_path)
            
            # Parse document
            parsed_content = parser.parse_document(file_path)
            
            if not parsed_content:
                logger.warning(f"Failed to parse: {file_path}")
                return
            
            # Chunk content
            chunks = chunker.chunk_content(parsed_content, file_metadata)
            
            # Deduplicate chunks
            unique_chunks, duplicates = deduplicator.deduplicate_chunks(chunks)
            
            # Save processed data
            save_processed_file(file_path, file_metadata, unique_chunks, output_dir, run_metadata.run_id)
            
            # Update run metadata
            run_metadata.files_processed += 1
            run_metadata.chunks_created += len(unique_chunks)
            run_metadata.duplicates_found += len(duplicates)
            
            if verbose:
                logger.info(f"  Processed: {len(unique_chunks)} chunks, {len(duplicates)} duplicates")
                
        except Exception as e:
            logger.error(f"Failed to process new file {file_path}: {e}")
    
    # Start watching
    await watcher.start()


def get_files_to_process(source: Path, recursive: bool, file_types: List[str]) -> List[Path]:
    """Get list of files to process."""
    
    if source.is_file():
        return [source]
    
    # Define supported file types
    supported_types = {
        ".pdf", ".docx", ".doc", ".txt", ".md", ".html", ".htm", 
        ".json", ".csv", ".xml", ".yaml", ".yml"
    }
    
    if file_types:
        supported_types = set(file_types)
    
    files = []
    
    if recursive:
        # Recursive search
        for file_path in source.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_types:
                files.append(file_path)
    else:
        # Non-recursive search
        for file_path in source.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_types:
                files.append(file_path)
    
    return sorted(files)


def extract_file_metadata(file_path: Path) -> FileMetadata:
    """Extract metadata from a file."""
    
    stat = file_path.stat()
    
    return FileMetadata(
        file_path=str(file_path),
        file_name=file_path.name,
        file_size=stat.st_size,
        file_type=file_path.suffix.lower(),
        created_at=datetime.fromtimestamp(stat.st_ctime),
        modified_at=datetime.fromtimestamp(stat.st_mtime),
        file_hash=calculate_file_hash(file_path)
    )


def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA256 hash of file."""
    
    import hashlib
    
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    
    return hash_sha256.hexdigest()


def save_processed_file(file_path: Path, file_metadata: FileMetadata, chunks: List[ChunkMetadata], 
                       output_dir: Path, run_id: str):
    """Save processed file data."""
    
    # Create run-specific directory
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save file metadata
    metadata_file = run_dir / f"{file_path.stem}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(file_metadata.dict(), f, indent=2, default=str)
    
    # Save chunks
    chunks_file = run_dir / f"{file_path.stem}_chunks.jsonl"
    with open(chunks_file, 'w') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.dict(), default=str) + '\n')
    
    # Save summary
    summary = {
        "file_path": str(file_path),
        "file_metadata": file_metadata.dict(),
        "chunk_count": len(chunks),
        "total_chars": sum(len(chunk.text) for chunk in chunks),
        "processed_at": datetime.now().isoformat(),
        "run_id": run_id
    }
    
    summary_file = run_dir / f"{file_path.stem}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)


def validate_ingestion_run(metadata_file: Path, output_dir: Path, verbose: bool) -> dict:
    """Validate a specific ingestion run."""
    
    with open(metadata_file, 'r') as f:
        run_data = json.load(f)
    
    run_id = run_data.get("run_id")
    run_dir = output_dir / run_id
    
    validation_result = {
        "run_id": run_id,
        "metadata_file": str(metadata_file),
        "run_dir_exists": run_dir.exists(),
        "files_processed": run_data.get("files_processed", 0),
        "chunks_created": run_data.get("chunks_created", 0),
        "duplicates_found": run_data.get("duplicates_found", 0),
        "validation_errors": [],
        "data_integrity": "unknown"
    }
    
    if not run_dir.exists():
        validation_result["validation_errors"].append("Run directory does not exist")
        validation_result["data_integrity"] = "failed"
        return validation_result
    
    # Check for expected files
    expected_files = list(run_dir.glob("*_metadata.json"))
    expected_chunks = list(run_dir.glob("*_chunks.jsonl"))
    expected_summaries = list(run_dir.glob("*_summary.json"))
    
    validation_result["metadata_files"] = len(expected_files)
    validation_result["chunk_files"] = len(expected_chunks)
    validation_result["summary_files"] = len(expected_summaries)
    
    # Validate file counts match
    if len(expected_files) != run_data.get("files_processed", 0):
        validation_result["validation_errors"].append(
            f"File count mismatch: expected {run_data.get('files_processed', 0)}, found {len(expected_files)}"
        )
    
    # Validate chunk files
    total_chunks = 0
    for chunk_file in expected_chunks:
        try:
            with open(chunk_file, 'r') as f:
                chunk_count = sum(1 for line in f if line.strip())
                total_chunks += chunk_count
        except Exception as e:
            validation_result["validation_errors"].append(f"Failed to read chunk file {chunk_file}: {e}")
    
    validation_result["actual_chunks"] = total_chunks
    
    if total_chunks != run_data.get("chunks_created", 0):
        validation_result["validation_errors"].append(
            f"Chunk count mismatch: expected {run_data.get('chunks_created', 0)}, found {total_chunks}"
        )
    
    # Determine data integrity
    if validation_result["validation_errors"]:
        validation_result["data_integrity"] = "failed"
    else:
        validation_result["data_integrity"] = "passed"
    
    if verbose:
        logger.info(f"Validation for {run_id}: {validation_result['data_integrity']}")
        if validation_result["validation_errors"]:
            for error in validation_result["validation_errors"]:
                logger.warning(f"  {error}")
    
    return validation_result


def display_ingestion_status(runs: List[dict]):
    """Display ingestion status in a table format."""
    
    if not runs:
        return
    
    typer.echo("\n" + "="*120)
    typer.echo(f"{'Run ID':<15} {'Source':<25} {'Mode':<10} {'Status':<10} {'Files':<8} {'Chunks':<8} {'Duplicates':<10} {'Created':<20}")
    typer.echo("="*120)
    
    for run in runs:
        source = run["source"][:23] + "..." if len(run["source"]) > 25 else run["source"]
        created = run["created"][:19] if run["created"] else "N/A"
        
        typer.echo(f"{run['run_id']:<15} {source:<25} {run['mode']:<10} {run['status']:<10} "
                  f"{run['files_processed']:<8} {run['chunks_created']:<8} {run['duplicates_found']:<10} {created:<20}")
    
    typer.echo("="*120)


def display_validation_summary(validation_results: List[dict]):
    """Display validation summary."""
    
    if not validation_results:
        return
    
    typer.echo("\n" + "="*80)
    typer.echo("📋 INGESTION VALIDATION SUMMARY")
    typer.echo("="*80)
    
    total_runs = len(validation_results)
    passed_runs = sum(1 for r in validation_results if r["data_integrity"] == "passed")
    failed_runs = total_runs - passed_runs
    
    typer.echo(f"Total Runs: {total_runs}")
    typer.echo(f"Passed: {passed_runs}")
    typer.echo(f"Failed: {failed_runs}")
    typer.echo(f"Success Rate: {(passed_runs/total_runs)*100:.1f}%")
    
    if failed_runs > 0:
        typer.echo(f"\n❌ FAILED RUNS:")
        for result in validation_results:
            if result["data_integrity"] == "failed":
                typer.echo(f"  {result['run_id']}:")
                for error in result["validation_errors"]:
                    typer.echo(f"    - {error}")
    
    typer.echo("="*80)


if __name__ == "__main__":
    app() 