"""
Plan command for Gola CLI - handles planning and scouting operations.
"""

import typer
from pathlib import Path
from typing import Optional, List
import json
import yaml
from datetime import datetime

from cli.utils.config import load_config
from cli.utils.logging import get_logger
from schemas.base import PlanSpec, RunMetadata
from pipeline.monitoring.gpu import GPUMonitor
from pipeline.router.llm_router import LLMRouter

app = typer.Typer(name="plan", help="Plan and scout data processing operations")

logger = get_logger(__name__)


@app.command()
def create(
    source: Path = typer.Argument(..., help="Source directory or file to process"),
    mode: str = typer.Option("general", "--mode", "-m", help="Processing mode (general, fiction, technical, legal)"),
    budget: float = typer.Option(25.0, "--budget", "-b", help="Daily budget in USD"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file for plan"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Create a processing plan for the given source."""
    
    try:
        # Load configuration
        config_data = load_config(config) if config else {}
        
        # Initialize components
        gpu_monitor = GPUMonitor()
        llm_router = LLMRouter(config_data.get("providers", {}))
        
        # Scout the source
        logger.info(f"Scouting source: {source}")
        plan_spec = scout_source(source, mode, budget, config_data, gpu_monitor, llm_router)
        
        # Generate plan metadata
        run_metadata = RunMetadata(
            run_id=f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            source_path=str(source),
            mode=mode,
            budget=budget,
            created_at=datetime.now(),
            status="planned"
        )
        
        # Create plan output
        plan_output = {
            "plan_spec": plan_spec.dict(),
            "run_metadata": run_metadata.dict(),
            "gpu_status": gpu_monitor.get_status().dict(),
            "estimated_costs": estimate_costs(plan_spec, llm_router),
            "risk_assessment": assess_risks(plan_spec, source),
            "recommendations": generate_recommendations(plan_spec, gpu_monitor)
        }
        
        # Save plan
        if output:
            output_path = output
        else:
            output_path = Path(f"plans/plan_{run_metadata.run_id}.json")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(plan_output, f, indent=2, default=str)
        
        # Display plan summary
        display_plan_summary(plan_output, verbose)
        
        logger.info(f"Plan created: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to create plan: {e}")
        raise typer.Exit(1)


@app.command()
def list(
    plans_dir: Path = typer.Option(Path("plans"), "--dir", "-d", help="Plans directory"),
    format: str = typer.Option("table", "--format", "-f", help="Output format (table, json, yaml)")
):
    """List existing processing plans."""
    
    try:
        if not plans_dir.exists():
            logger.warning(f"Plans directory does not exist: {plans_dir}")
            return
        
        plan_files = list(plans_dir.glob("plan_*.json"))
        
        if not plan_files:
            logger.info("No plans found")
            return
        
        plans = []
        for plan_file in sorted(plan_files, key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                with open(plan_file, 'r') as f:
                    plan_data = json.load(f)
                    plans.append({
                        "file": plan_file.name,
                        "run_id": plan_data.get("run_metadata", {}).get("run_id"),
                        "source": plan_data.get("run_metadata", {}).get("source_path"),
                        "mode": plan_data.get("run_metadata", {}).get("mode"),
                        "status": plan_data.get("run_metadata", {}).get("status"),
                        "created": plan_data.get("run_metadata", {}).get("created_at"),
                        "estimated_cost": plan_data.get("estimated_costs", {}).get("total", 0)
                    })
            except Exception as e:
                logger.warning(f"Failed to load plan {plan_file}: {e}")
        
        if format == "json":
            typer.echo(json.dumps(plans, indent=2, default=str))
        elif format == "yaml":
            typer.echo(yaml.dump(plans, default_flow_style=False))
        else:
            display_plans_table(plans)
            
    except Exception as e:
        logger.error(f"Failed to list plans: {e}")
        raise typer.Exit(1)


@app.command()
def show(
    plan_file: Path = typer.Argument(..., help="Plan file to display"),
    format: str = typer.Option("summary", "--format", "-f", help="Display format (summary, full, yaml)")
):
    """Show details of a specific plan."""
    
    try:
        if not plan_file.exists():
            logger.error(f"Plan file not found: {plan_file}")
            raise typer.Exit(1)
        
        with open(plan_file, 'r') as f:
            plan_data = json.load(f)
        
        if format == "yaml":
            typer.echo(yaml.dump(plan_data, default_flow_style=False))
        elif format == "full":
            typer.echo(json.dumps(plan_data, indent=2, default=str))
        else:
            display_plan_details(plan_data)
            
    except Exception as e:
        logger.error(f"Failed to show plan: {e}")
        raise typer.Exit(1)


def scout_source(source: Path, mode: str, budget: float, config: dict, gpu_monitor, llm_router) -> PlanSpec:
    """Scout the source and create a plan specification."""
    
    # Analyze source structure
    source_info = analyze_source_structure(source)
    
    # Estimate processing requirements
    processing_requirements = estimate_processing_requirements(source_info, mode)
    
    # Check GPU availability
    gpu_status = gpu_monitor.get_status()
    
    # Create plan specification
    plan_spec = PlanSpec(
        source_path=str(source),
        mode=mode,
        budget=budget,
        estimated_files=source_info["file_count"],
        estimated_size_mb=source_info["total_size_mb"],
        processing_tasks=processing_requirements["tasks"],
        estimated_tokens=processing_requirements["estimated_tokens"],
        gpu_available=gpu_status.gpu_count > 0,
        gpu_memory_gb=gpu_status.total_memory_gb if gpu_status.gpu_count > 0 else 0,
        local_processing_capacity=gpu_status.gpu_count > 0,
        cloud_fallback_required=processing_requirements["estimated_tokens"] > 1000000,  # 1M tokens
        risk_level=assess_risk_level(source_info, processing_requirements),
        recommendations=generate_plan_recommendations(source_info, processing_requirements, gpu_status)
    )
    
    return plan_spec


def analyze_source_structure(source: Path) -> dict:
    """Analyze the structure of the source directory or file."""
    
    source_info = {
        "file_count": 0,
        "total_size_mb": 0,
        "file_types": {},
        "subdirectories": [],
        "largest_files": [],
        "estimated_processing_time": 0
    }
    
    if source.is_file():
        # Single file
        source_info["file_count"] = 1
        source_info["total_size_mb"] = source.stat().st_size / (1024 * 1024)
        source_info["file_types"][source.suffix] = 1
        source_info["largest_files"].append({
            "path": str(source),
            "size_mb": source_info["total_size_mb"]
        })
    else:
        # Directory
        for file_path in source.rglob("*"):
            if file_path.is_file():
                source_info["file_count"] += 1
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                source_info["total_size_mb"] += file_size_mb
                
                # Track file types
                ext = file_path.suffix.lower()
                source_info["file_types"][ext] = source_info["file_types"].get(ext, 0) + 1
                
                # Track largest files
                source_info["largest_files"].append({
                    "path": str(file_path),
                    "size_mb": file_size_mb
                })
        
        # Sort largest files and keep top 10
        source_info["largest_files"].sort(key=lambda x: x["size_mb"], reverse=True)
        source_info["largest_files"] = source_info["largest_files"][:10]
        
        # Get subdirectories
        source_info["subdirectories"] = [str(d) for d in source.iterdir() if d.is_dir()]
    
    # Estimate processing time (rough estimate: 1 minute per MB)
    source_info["estimated_processing_time"] = source_info["total_size_mb"] * 60
    
    return source_info


def estimate_processing_requirements(source_info: dict, mode: str) -> dict:
    """Estimate processing requirements based on source and mode."""
    
    # Base tasks for all modes
    base_tasks = ["summary", "entities", "dedup"]
    
    # Mode-specific tasks
    mode_tasks = {
        "general": ["summary", "entities", "topics", "qa_pairs"],
        "fiction": ["characters", "timeline", "themes", "dialogue_acts"],
        "technical": ["equations", "figures", "tables", "citations"],
        "legal": ["sections", "clauses", "obligations", "rights"]
    }
    
    tasks = mode_tasks.get(mode, base_tasks)
    
    # Estimate tokens based on file size and content
    # Rough estimate: 1 token per 4 characters, average file has 50% text content
    estimated_tokens = int(source_info["total_size_mb"] * 1024 * 1024 * 0.5 / 4)
    
    return {
        "tasks": tasks,
        "estimated_tokens": estimated_tokens,
        "estimated_api_calls": len(tasks) * source_info["file_count"],
        "processing_complexity": "high" if estimated_tokens > 1000000 else "medium" if estimated_tokens > 100000 else "low"
    }


def estimate_costs(plan_spec: PlanSpec, llm_router: LLMRouter) -> dict:
    """Estimate processing costs."""
    
    # Rough cost estimates per 1K tokens
    cost_per_1k_tokens = {
        "local": 0.0,  # LM Studio is free
        "anthropic": 0.015,  # Claude 3.5 Sonnet
        "openai": 0.002,  # GPT-4
        "grok": 0.01,  # Grok-4
        "deepseek": 0.002  # DeepSeek Chat
    }
    
    estimated_tokens = plan_spec.estimated_tokens
    total_cost = 0
    
    # Estimate local vs cloud split
    if plan_spec.local_processing_capacity:
        local_tokens = min(estimated_tokens, 500000)  # Assume 500K tokens local capacity
        cloud_tokens = estimated_tokens - local_tokens
        total_cost = (cloud_tokens / 1000) * cost_per_1k_tokens["anthropic"]  # Use Claude as baseline
    else:
        total_cost = (estimated_tokens / 1000) * cost_per_1k_tokens["anthropic"]
    
    return {
        "total": round(total_cost, 2),
        "local_cost": 0.0,
        "cloud_cost": round(total_cost, 2),
        "cost_per_1k_tokens": cost_per_1k_tokens,
        "budget_utilization": round((total_cost / plan_spec.budget) * 100, 1)
    }


def assess_risks(plan_spec: PlanSpec, source: Path) -> dict:
    """Assess risks for the processing plan."""
    
    risks = {
        "duplication_risk": "low",
        "pii_risk": "medium",
        "processing_risk": "low",
        "cost_overrun_risk": "low",
        "quality_risk": "low"
    }
    
    # Assess duplication risk
    if plan_spec.estimated_files > 1000:
        risks["duplication_risk"] = "high"
    elif plan_spec.estimated_files > 100:
        risks["duplication_risk"] = "medium"
    
    # Assess PII risk based on file types
    pii_extensions = {".pdf", ".docx", ".txt", ".csv", ".json"}
    if any(ext in pii_extensions for ext in [".pdf", ".docx"]):
        risks["pii_risk"] = "high"
    
    # Assess processing risk
    if plan_spec.estimated_tokens > 5000000:  # 5M tokens
        risks["processing_risk"] = "high"
    elif plan_spec.estimated_tokens > 1000000:  # 1M tokens
        risks["processing_risk"] = "medium"
    
    # Assess cost overrun risk
    cost_estimate = (plan_spec.estimated_tokens / 1000) * 0.015  # Claude cost
    if cost_estimate > plan_spec.budget * 0.8:
        risks["cost_overrun_risk"] = "high"
    elif cost_estimate > plan_spec.budget * 0.5:
        risks["cost_overrun_risk"] = "medium"
    
    return risks


def generate_recommendations(plan_spec: PlanSpec, gpu_monitor) -> List[str]:
    """Generate recommendations for the processing plan."""
    
    recommendations = []
    
    # GPU recommendations
    if not plan_spec.local_processing_capacity:
        recommendations.append("Consider setting up LM Studio for local processing to reduce costs")
    
    if gpu_monitor.get_status().gpu_count > 0:
        recommendations.append("GPU available - will use local processing for initial tasks")
    
    # Cost recommendations
    if plan_spec.estimated_tokens > 1000000:
        recommendations.append("Large dataset detected - consider using batch processing for cost efficiency")
    
    # Quality recommendations
    if plan_spec.estimated_files > 100:
        recommendations.append("Large file count - recommend enabling cross-validation for quality assurance")
    
    # Risk mitigation
    if plan_spec.risk_level == "high":
        recommendations.append("High-risk plan - recommend starting with a small subset for validation")
    
    return recommendations


def assess_risk_level(source_info: dict, processing_requirements: dict) -> str:
    """Assess overall risk level."""
    
    risk_score = 0
    
    # File count risk
    if source_info["file_count"] > 1000:
        risk_score += 3
    elif source_info["file_count"] > 100:
        risk_score += 2
    elif source_info["file_count"] > 10:
        risk_score += 1
    
    # Size risk
    if source_info["total_size_mb"] > 1000:
        risk_score += 3
    elif source_info["total_size_mb"] > 100:
        risk_score += 2
    elif source_info["total_size_mb"] > 10:
        risk_score += 1
    
    # Processing complexity risk
    if processing_requirements["processing_complexity"] == "high":
        risk_score += 3
    elif processing_requirements["processing_complexity"] == "medium":
        risk_score += 2
    
    if risk_score >= 6:
        return "high"
    elif risk_score >= 3:
        return "medium"
    else:
        return "low"


def generate_plan_recommendations(source_info: dict, processing_requirements: dict, gpu_status) -> List[str]:
    """Generate specific recommendations for the plan."""
    
    recommendations = []
    
    if source_info["file_count"] > 100:
        recommendations.append("Consider processing in batches to manage memory usage")
    
    if processing_requirements["estimated_tokens"] > 1000000:
        recommendations.append("Large token count - recommend using batch processing APIs")
    
    if gpu_status.gpu_count == 0:
        recommendations.append("No GPU detected - will rely on cloud processing")
    
    return recommendations


def display_plan_summary(plan_output: dict, verbose: bool):
    """Display a summary of the created plan."""
    
    plan_spec = plan_output["plan_spec"]
    run_metadata = plan_output["run_metadata"]
    estimated_costs = plan_output["estimated_costs"]
    risk_assessment = plan_output["risk_assessment"]
    
    typer.echo("\n" + "="*60)
    typer.echo("📋 PROCESSING PLAN SUMMARY")
    typer.echo("="*60)
    
    typer.echo(f"Run ID: {run_metadata['run_id']}")
    typer.echo(f"Source: {run_metadata['source_path']}")
    typer.echo(f"Mode: {run_metadata['mode']}")
    typer.echo(f"Status: {run_metadata['status']}")
    
    typer.echo(f"\n📊 ESTIMATES:")
    typer.echo(f"  Files: {plan_spec['estimated_files']:,}")
    typer.echo(f"  Size: {plan_spec['estimated_size_mb']:.1f} MB")
    typer.echo(f"  Tokens: {plan_spec['estimated_tokens']:,}")
    typer.echo(f"  Tasks: {', '.join(plan_spec['processing_tasks'])}")
    
    typer.echo(f"\n💰 COSTS:")
    typer.echo(f"  Estimated: ${estimated_costs['total']:.2f}")
    typer.echo(f"  Budget: ${plan_spec['budget']:.2f}")
    typer.echo(f"  Utilization: {estimated_costs['budget_utilization']}%")
    
    typer.echo(f"\n⚠️  RISKS:")
    for risk, level in risk_assessment.items():
        emoji = "🔴" if level == "high" else "🟡" if level == "medium" else "🟢"
        typer.echo(f"  {emoji} {risk.replace('_', ' ').title()}: {level}")
    
    if verbose and plan_output.get("recommendations"):
        typer.echo(f"\n💡 RECOMMENDATIONS:")
        for rec in plan_output["recommendations"]:
            typer.echo(f"  • {rec}")
    
    typer.echo("="*60)


def display_plans_table(plans: List[dict]):
    """Display plans in a table format."""
    
    if not plans:
        return
    
    # Create table header
    headers = ["File", "Run ID", "Source", "Mode", "Status", "Cost"]
    typer.echo("\n" + "="*100)
    typer.echo(f"{'File':<20} {'Run ID':<15} {'Source':<25} {'Mode':<10} {'Status':<10} {'Cost':<10}")
    typer.echo("="*100)
    
    for plan in plans:
        source = plan["source"][:23] + "..." if len(plan["source"]) > 25 else plan["source"]
        cost = f"${plan['estimated_cost']:.2f}" if plan['estimated_cost'] else "N/A"
        
        typer.echo(f"{plan['file']:<20} {plan['run_id']:<15} {source:<25} {plan['mode']:<10} {plan['status']:<10} {cost:<10}")
    
    typer.echo("="*100)


def display_plan_details(plan_data: dict):
    """Display detailed plan information."""
    
    plan_spec = plan_data.get("plan_spec", {})
    run_metadata = plan_data.get("run_metadata", {})
    estimated_costs = plan_data.get("estimated_costs", {})
    risk_assessment = plan_data.get("risk_assessment", {})
    
    typer.echo("\n" + "="*60)
    typer.echo("📋 PLAN DETAILS")
    typer.echo("="*60)
    
    typer.echo(f"Run ID: {run_metadata.get('run_id', 'N/A')}")
    typer.echo(f"Source: {run_metadata.get('source_path', 'N/A')}")
    typer.echo(f"Mode: {run_metadata.get('mode', 'N/A')}")
    typer.echo(f"Budget: ${run_metadata.get('budget', 0):.2f}")
    typer.echo(f"Created: {run_metadata.get('created_at', 'N/A')}")
    typer.echo(f"Status: {run_metadata.get('status', 'N/A')}")
    
    typer.echo(f"\n📊 SPECIFICATIONS:")
    typer.echo(f"  Estimated Files: {plan_spec.get('estimated_files', 0):,}")
    typer.echo(f"  Estimated Size: {plan_spec.get('estimated_size_mb', 0):.1f} MB")
    typer.echo(f"  Estimated Tokens: {plan_spec.get('estimated_tokens', 0):,}")
    typer.echo(f"  Processing Tasks: {', '.join(plan_spec.get('processing_tasks', []))}")
    typer.echo(f"  GPU Available: {plan_spec.get('gpu_available', False)}")
    typer.echo(f"  GPU Memory: {plan_spec.get('gpu_memory_gb', 0):.1f} GB")
    typer.echo(f"  Risk Level: {plan_spec.get('risk_level', 'unknown')}")
    
    typer.echo(f"\n💰 COST BREAKDOWN:")
    typer.echo(f"  Total Estimated: ${estimated_costs.get('total', 0):.2f}")
    typer.echo(f"  Local Cost: ${estimated_costs.get('local_cost', 0):.2f}")
    typer.echo(f"  Cloud Cost: ${estimated_costs.get('cloud_cost', 0):.2f}")
    typer.echo(f"  Budget Utilization: {estimated_costs.get('budget_utilization', 0):.1f}%")
    
    typer.echo(f"\n⚠️  RISK ASSESSMENT:")
    for risk, level in risk_assessment.items():
        emoji = "🔴" if level == "high" else "🟡" if level == "medium" else "🟢"
        typer.echo(f"  {emoji} {risk.replace('_', ' ').title()}: {level}")
    
    if plan_data.get("recommendations"):
        typer.echo(f"\n💡 RECOMMENDATIONS:")
        for rec in plan_data["recommendations"]:
            typer.echo(f"  • {rec}")
    
    typer.echo("="*60)


if __name__ == "__main__":
    app() 