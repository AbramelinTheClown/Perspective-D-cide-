"""
LLM router for intelligent model selection and provider management.
"""

import time
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict

from schemas.base import GPUStatus, RouterDecision, ProviderConfig
from pipeline.monitoring.gpu import get_gpu_monitor
from cli.utils.logging import gola_logger

class TaskType(Enum):
    """Task types for routing decisions."""
    SUMMARY = "summary"
    ENTITIES = "entities"
    TRIPLES = "triples"
    QA_PAIRS = "qa_pairs"
    TOPICS = "topics"
    CLASSIFICATION = "classification"
    TRANSLATION = "translation"
    CODE_GENERATION = "code_generation"

class RouterPolicy(Enum):
    """Router policies."""
    THROUGHPUT = "throughput"  # Maximize throughput
    COST = "cost"  # Minimize cost
    QUALITY = "quality"  # Maximize quality
    LATENCY = "latency"  # Minimize latency
    PRIVACY = "privacy"  # Prefer local models

@dataclass
class RouterConfig:
    """Router configuration."""
    default_policy: RouterPolicy = RouterPolicy.THROUGHPUT
    gpu_memory_threshold_mb: int = 1000
    gpu_utilization_threshold: float = 85.0
    cost_budget_usd: float = 25.0
    daily_cost_limit_usd: float = 50.0
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    enable_fallback: bool = True
    local_first: bool = True
    pii_sensitive_tasks: List[str] = None

class ProviderManager:
    """Manages LLM providers and their status."""
    
    def __init__(self, providers: Dict[str, ProviderConfig]):
        """
        Initialize provider manager.
        
        Args:
            providers: Provider configurations
        """
        self.providers = providers
        self.provider_status: Dict[str, Dict[str, Any]] = {}
        self.daily_costs: Dict[str, float] = defaultdict(float)
        self.request_counts: Dict[str, int] = defaultdict(int)
        self.last_reset = datetime.utcnow()
        
        # Initialize provider status
        for provider_name in providers:
            self.provider_status[provider_name] = {
                "enabled": providers[provider_name].enabled,
                "last_request": None,
                "error_count": 0,
                "consecutive_errors": 0,
                "circuit_breaker": False,
                "circuit_breaker_until": None
            }
    
    def get_available_providers(self, task_type: TaskType, 
                               pii_level: int = 0) -> List[str]:
        """
        Get available providers for a task.
        
        Args:
            task_type: Type of task
            pii_level: PII sensitivity level
            
        Returns:
            List of available provider names
        """
        available = []
        
        for provider_name, provider in self.providers.items():
            if not self._is_provider_available(provider_name, pii_level):
                continue
            
            # Check if provider supports the task
            if self._supports_task(provider, task_type):
                available.append(provider_name)
        
        return available
    
    def _is_provider_available(self, provider_name: str, pii_level: int) -> bool:
        """
        Check if provider is available.
        
        Args:
            provider_name: Provider name
            pii_level: PII sensitivity level
            
        Returns:
            True if provider is available
        """
        if provider_name not in self.provider_status:
            return False
        
        status = self.provider_status[provider_name]
        
        # Check if disabled
        if not status["enabled"]:
            return False
        
        # Check circuit breaker
        if status["circuit_breaker"]:
            if status["circuit_breaker_until"] and datetime.utcnow() < status["circuit_breaker_until"]:
                return False
            else:
                # Reset circuit breaker
                status["circuit_breaker"] = False
                status["consecutive_errors"] = 0
        
        # Check PII restrictions
        if pii_level > 1 and provider_name != "lmstudio":
            return False
        
        return True
    
    def _supports_task(self, provider: ProviderConfig, task_type: TaskType) -> bool:
        """
        Check if provider supports the task type.
        
        Args:
            provider: Provider configuration
            task_type: Task type
            
        Returns:
            True if provider supports the task
        """
        # All providers support basic tasks
        basic_tasks = [TaskType.SUMMARY, TaskType.ENTITIES, TaskType.TRIPLES, TaskType.QA_PAIRS]
        
        if task_type in basic_tasks:
            return True
        
        # Check for specialized tasks
        if task_type == TaskType.CODE_GENERATION:
            # Prefer providers with code models
            code_models = ["gpt-4", "claude-3", "grok-4"]
            return any(model in provider.models for model in code_models)
        
        return True
    
    def record_request(self, provider_name: str, cost_usd: float = 0.0, 
                      success: bool = True) -> None:
        """
        Record a request to a provider.
        
        Args:
            provider_name: Provider name
            cost_usd: Cost in USD
            success: Whether request was successful
        """
        if provider_name not in self.provider_status:
            return
        
        status = self.provider_status[provider_name]
        status["last_request"] = datetime.utcnow()
        
        if success:
            status["consecutive_errors"] = 0
            self.request_counts[provider_name] += 1
            self.daily_costs[provider_name] += cost_usd
        else:
            status["error_count"] += 1
            status["consecutive_errors"] += 1
            
            # Trigger circuit breaker if too many consecutive errors
            if status["consecutive_errors"] >= 5:
                status["circuit_breaker"] = True
                status["circuit_breaker_until"] = datetime.utcnow() + timedelta(minutes=15)
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """
        Get provider statistics.
        
        Returns:
            Provider statistics
        """
        return {
            "providers": self.provider_status,
            "daily_costs": dict(self.daily_costs),
            "request_counts": dict(self.request_counts)
        }
    
    def reset_daily_stats(self) -> None:
        """Reset daily statistics."""
        self.daily_costs.clear()
        self.request_counts.clear()
        self.last_reset = datetime.utcnow()

class LLMRouter:
    """LLM router for intelligent model selection."""
    
    def __init__(self, providers: Dict[str, ProviderConfig], 
                 config: Optional[RouterConfig] = None):
        """
        Initialize LLM router.
        
        Args:
            providers: Provider configurations
            config: Router configuration
        """
        self.config = config or RouterConfig()
        self.provider_manager = ProviderManager(providers)
        self.gpu_monitor = get_gpu_monitor()
        self.routing_history: List[RouterDecision] = []
        
        # Initialize PII sensitive tasks
        if self.config.pii_sensitive_tasks is None:
            self.config.pii_sensitive_tasks = [
                "entities", "triples", "classification"
            ]
    
    def route_request(self, task_type: str, chunk_hash: str, 
                     text_length: int, pii_level: int = 0,
                     policy: Optional[RouterPolicy] = None) -> RouterDecision:
        """
        Route a request to the best available provider.
        
        Args:
            task_type: Type of task
            chunk_hash: Chunk hash
            text_length: Length of text
            pii_level: PII sensitivity level
            policy: Routing policy (overrides default)
            
        Returns:
            Router decision
        """
        # Convert task type to enum
        try:
            task_enum = TaskType(task_type)
        except ValueError:
            task_enum = TaskType.SUMMARY
        
        # Use default policy if not specified
        if policy is None:
            policy = self.config.default_policy
        
        # Get available providers
        available_providers = self.provider_manager.get_available_providers(
            task_enum, pii_level
        )
        
        if not available_providers:
            raise RuntimeError("No available providers for request")
        
        # Get GPU status
        gpu_status = None
        if self.gpu_monitor:
            gpu_summary = self.gpu_monitor.get_gpu_summary()
            if gpu_summary["available_gpus"] > 0:
                best_gpu = self.gpu_monitor.get_best_gpu()
                if best_gpu is not None:
                    gpu_status = self.gpu_monitor.get_gpu_status(best_gpu)
        
        # Select provider based on policy
        selected_provider, selected_model, reasoning = self._select_provider(
            available_providers, task_enum, policy, gpu_status, text_length, pii_level
        )
        
        # Estimate cost
        estimated_cost = self._estimate_cost(selected_provider, selected_model, text_length)
        
        # Create router decision
        decision = RouterDecision(
            task_type=task_type,
            chunk_hash=chunk_hash,
            selected_provider=selected_provider,
            selected_model=selected_model,
            reasoning=reasoning,
            gpu_status=gpu_status,
            pii_level=pii_level,
            estimated_cost=estimated_cost
        )
        
        # Record decision
        self.routing_history.append(decision)
        
        gola_logger.info(f"Routed {task_type} to {selected_provider}/{selected_model}: {reasoning}")
        
        return decision
    
    def _select_provider(self, available_providers: List[str], 
                        task_type: TaskType, policy: RouterPolicy,
                        gpu_status: Optional[GPUStatus], text_length: int,
                        pii_level: int) -> Tuple[str, str, str]:
        """
        Select provider based on policy.
        
        Args:
            available_providers: List of available providers
            task_type: Task type
            policy: Routing policy
            gpu_status: GPU status
            text_length: Text length
            pii_level: PII level
            
        Returns:
            Tuple of (provider, model, reasoning)
        """
        # PII-sensitive tasks go to local first
        if pii_level > 1 and "lmstudio" in available_providers:
            return "lmstudio", "local/llama-3.2-1b-instruct", "PII-sensitive task, using local model"
        
        # Policy-based selection
        if policy == RouterPolicy.PRIVACY:
            return self._select_private_provider(available_providers, task_type)
        elif policy == RouterPolicy.COST:
            return self._select_cost_optimized_provider(available_providers, task_type, text_length)
        elif policy == RouterPolicy.QUALITY:
            return self._select_quality_optimized_provider(available_providers, task_type)
        elif policy == RouterPolicy.LATENCY:
            return self._select_latency_optimized_provider(available_providers, task_type, gpu_status)
        else:  # THROUGHPUT
            return self._select_throughput_optimized_provider(available_providers, task_type, gpu_status)
    
    def _select_private_provider(self, available_providers: List[str], 
                                task_type: TaskType) -> Tuple[str, str, str]:
        """Select provider prioritizing privacy."""
        if "lmstudio" in available_providers:
            return "lmstudio", "local/llama-3.2-1b-instruct", "Privacy policy: using local model"
        
        # Fallback to other providers if local not available
        provider = available_providers[0]
        model = self.provider_manager.providers[provider].models[0]
        return provider, model, f"Privacy policy: fallback to {provider}"
    
    def _select_cost_optimized_provider(self, available_providers: List[str],
                                       task_type: TaskType, text_length: int) -> Tuple[str, str, str]:
        """Select provider optimizing for cost."""
        best_provider = None
        best_cost = float('inf')
        
        for provider_name in available_providers:
            provider = self.provider_manager.providers[provider_name]
            if provider.cost_per_1k_tokens:
                # Estimate cost for this request
                estimated_tokens = text_length // 4  # Rough estimate
                cost = (estimated_tokens / 1000) * provider.cost_per_1k_tokens
                
                if cost < best_cost:
                    best_cost = cost
                    best_provider = provider_name
        
        if best_provider:
            model = self.provider_manager.providers[best_provider].models[0]
            return best_provider, model, f"Cost optimization: selected {best_provider} (${best_cost:.4f})"
        
        # Fallback
        provider = available_providers[0]
        model = self.provider_manager.providers[provider].models[0]
        return provider, model, "Cost optimization: fallback selection"
    
    def _select_quality_optimized_provider(self, available_providers: List[str],
                                          task_type: TaskType) -> Tuple[str, str, str]:
        """Select provider optimizing for quality."""
        # Prefer high-quality models
        quality_providers = ["anthropic", "openai", "xai"]
        
        for provider in quality_providers:
            if provider in available_providers:
                provider_config = self.provider_manager.providers[provider]
                # Select the best model available
                model = provider_config.models[-1] if provider_config.models else "default"
                return provider, model, f"Quality optimization: selected {provider}"
        
        # Fallback
        provider = available_providers[0]
        model = self.provider_manager.providers[provider].models[0]
        return provider, model, "Quality optimization: fallback selection"
    
    def _select_latency_optimized_provider(self, available_providers: List[str],
                                          task_type: TaskType, 
                                          gpu_status: Optional[GPUStatus]) -> Tuple[str, str, str]:
        """Select provider optimizing for latency."""
        # Prefer local if GPU is available
        if "lmstudio" in available_providers and gpu_status and not gpu_status.is_overloaded:
            return "lmstudio", "local/llama-3.2-1b-instruct", "Latency optimization: using local GPU"
        
        # Prefer fast cloud providers
        fast_providers = ["openai", "anthropic"]
        for provider in fast_providers:
            if provider in available_providers:
                model = self.provider_manager.providers[provider].models[0]
                return provider, model, f"Latency optimization: selected {provider}"
        
        # Fallback
        provider = available_providers[0]
        model = self.provider_manager.providers[provider].models[0]
        return provider, model, "Latency optimization: fallback selection"
    
    def _select_throughput_optimized_provider(self, available_providers: List[str],
                                             task_type: TaskType,
                                             gpu_status: Optional[GPUStatus]) -> Tuple[str, str, str]:
        """Select provider optimizing for throughput."""
        # Use local if GPU is available and not overloaded
        if "lmstudio" in available_providers and gpu_status and not gpu_status.is_overloaded:
            return "lmstudio", "local/llama-3.2-1b-instruct", "Throughput optimization: using local GPU"
        
        # Distribute across available providers
        provider = available_providers[0]
        model = self.provider_manager.providers[provider].models[0]
        return provider, model, f"Throughput optimization: selected {provider}"
    
    def _estimate_cost(self, provider_name: str, model_name: str, 
                      text_length: int) -> float:
        """
        Estimate cost for a request.
        
        Args:
            provider_name: Provider name
            model_name: Model name
            text_length: Text length
            
        Returns:
            Estimated cost in USD
        """
        if provider_name not in self.provider_manager.providers:
            return 0.0
        
        provider = self.provider_manager.providers[provider_name]
        if not provider.cost_per_1k_tokens:
            return 0.0
        
        # Estimate tokens (rough approximation)
        estimated_tokens = text_length // 4
        
        # Estimate cost
        cost = (estimated_tokens / 1000) * provider.cost_per_1k_tokens
        
        return cost
    
    def record_result(self, decision: RouterDecision, success: bool, 
                     actual_cost: float = 0.0, duration_seconds: float = 0.0) -> None:
        """
        Record the result of a routing decision.
        
        Args:
            decision: Router decision
            success: Whether the request was successful
            actual_cost: Actual cost incurred
            duration_seconds: Request duration
        """
        self.provider_manager.record_request(
            decision.selected_provider, actual_cost, success
        )
        
        # Update decision with results
        decision.actual_cost = actual_cost
        decision.duration_seconds = duration_seconds
        decision.success = success
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """
        Get routing statistics.
        
        Returns:
            Routing statistics
        """
        return {
            "total_decisions": len(self.routing_history),
            "provider_stats": self.provider_manager.get_provider_stats(),
            "recent_decisions": [
                {
                    "task_type": d.task_type,
                    "provider": d.selected_provider,
                    "model": d.selected_model,
                    "success": getattr(d, 'success', None),
                    "cost": getattr(d, 'actual_cost', None),
                    "duration": getattr(d, 'duration_seconds', None)
                }
                for d in self.routing_history[-10:]  # Last 10 decisions
            ]
        }

# Global router instance
llm_router = None

def get_llm_router() -> Optional[LLMRouter]:
    """Get the global LLM router instance."""
    return llm_router

def init_llm_router(providers: Dict[str, ProviderConfig], 
                   config: Optional[RouterConfig] = None) -> LLMRouter:
    """
    Initialize the global LLM router.
    
    Args:
        providers: Provider configurations
        config: Router configuration
        
    Returns:
        LLM router instance
    """
    global llm_router
    llm_router = LLMRouter(providers, config)
    return llm_router 