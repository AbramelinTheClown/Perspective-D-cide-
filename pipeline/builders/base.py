"""
Base builder interface for AI processing tasks.
"""

import time
import hashlib
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass

from schemas.base import ChunkMetadata, RunMetadata, QualityMetrics, EvidenceSpan, BaseOutput
from pipeline.router.llm_router import get_llm_router, RouterPolicy
from cli.utils.logging import gola_logger

@dataclass
class BuilderConfig:
    """Configuration for AI builders."""
    task_type: str
    prompt_version: str = "v1.0"
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout_seconds: float = 30.0
    router_policy: RouterPolicy = RouterPolicy.THROUGHPUT
    enable_validation: bool = True
    enable_caching: bool = True
    cache_ttl_hours: int = 24

class BuilderResult:
    """Result from AI builder processing."""
    
    def __init__(self, success: bool, output: Optional[BaseOutput] = None, 
                 error: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        self.success = success
        self.output = output
        self.error = error
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow()

class BaseBuilder(ABC):
    """Base class for all AI builders."""
    
    def __init__(self, config: BuilderConfig):
        """
        Initialize base builder.
        
        Args:
            config: Builder configuration
        """
        self.config = config
        self.router = get_llm_router()
        self.cache: Dict[str, BuilderResult] = {}
        
        if not self.router:
            raise RuntimeError("LLM router not initialized")
    
    def process_chunk(self, chunk: ChunkMetadata, 
                     context: Optional[Dict[str, Any]] = None) -> BuilderResult:
        """
        Process a chunk and generate structured output.
        
        Args:
            chunk: Chunk metadata to process
            context: Additional context for processing
            
        Returns:
            Builder result with output or error
        """
        # Check cache first
        cache_key = self._generate_cache_key(chunk, context)
        if self.config.enable_caching and cache_key in self.cache:
            cached_result = self.cache[cache_key]
            if self._is_cache_valid(cached_result):
                gola_logger.debug(f"Using cached result for chunk {chunk.chunk_hash}")
                return cached_result
        
        # Process the chunk
        start_time = time.time()
        result = self._process_chunk_internal(chunk, context)
        duration = time.time() - start_time
        
        # Cache the result
        if self.config.enable_caching and result.success:
            self.cache[cache_key] = result
        
        # Log the result
        self._log_result(chunk, result, duration)
        
        return result
    
    def _process_chunk_internal(self, chunk: ChunkMetadata, 
                               context: Optional[Dict[str, Any]]) -> BuilderResult:
        """
        Internal chunk processing with retries.
        
        Args:
            chunk: Chunk metadata
            context: Processing context
            
        Returns:
            Builder result
        """
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                # Route the request
                router_decision = self.router.route_request(
                    task_type=self.config.task_type,
                    chunk_hash=chunk.chunk_hash,
                    text_length=len(chunk.text_norm),
                    pii_level=chunk.metadata.get('pii_level', 0),
                    policy=self.config.router_policy
                )
                
                # Generate the prompt
                prompt = self._generate_prompt(chunk, context, router_decision)
                
                # Call the LLM
                llm_response = self._call_llm(prompt, router_decision)
                
                # Parse the response
                output = self._parse_response(llm_response, chunk)
                
                # Validate the output
                if self.config.enable_validation:
                    validation_result = self._validate_output(output, chunk)
                    if not validation_result.success:
                        raise ValueError(f"Validation failed: {validation_result.error}")
                
                # Record successful result
                self.router.record_result(
                    router_decision, 
                    success=True,
                    duration_seconds=time.time() - time.time()
                )
                
                return BuilderResult(success=True, output=output)
                
            except Exception as e:
                last_error = str(e)
                gola_logger.warning(f"Attempt {attempt + 1} failed for chunk {chunk.chunk_hash}: {e}")
                
                # Record failed result
                if 'router_decision' in locals():
                    self.router.record_result(
                        router_decision,
                        success=False,
                        duration_seconds=time.time() - time.time()
                    )
                
                # Wait before retry
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (2 ** attempt))  # Exponential backoff
        
        # All attempts failed
        return BuilderResult(
            success=False,
            error=f"All {self.config.max_retries} attempts failed. Last error: {last_error}"
        )
    
    @abstractmethod
    def _generate_prompt(self, chunk: ChunkMetadata, 
                        context: Optional[Dict[str, Any]], 
                        router_decision) -> str:
        """
        Generate prompt for the LLM.
        
        Args:
            chunk: Chunk metadata
            context: Processing context
            router_decision: Router decision
            
        Returns:
            Generated prompt
        """
        pass
    
    @abstractmethod
    def _parse_response(self, response: str, chunk: ChunkMetadata) -> BaseOutput:
        """
        Parse LLM response into structured output.
        
        Args:
            response: LLM response
            chunk: Source chunk
            
        Returns:
            Structured output
        """
        pass
    
    def _validate_output(self, output: BaseOutput, chunk: ChunkMetadata) -> BuilderResult:
        """
        Validate the output.
        
        Args:
            output: Generated output
            chunk: Source chunk
            
        Returns:
            Validation result
        """
        try:
            # Basic validation
            if not output.quality_metrics:
                return BuilderResult(success=False, error="Missing quality metrics")
            
            if output.quality_metrics.confidence < 0.5:
                return BuilderResult(success=False, error="Low confidence output")
            
            # Custom validation
            custom_validation = self._custom_validation(output, chunk)
            if not custom_validation.success:
                return custom_validation
            
            return BuilderResult(success=True)
            
        except Exception as e:
            return BuilderResult(success=False, error=f"Validation error: {e}")
    
    def _custom_validation(self, output: BaseOutput, chunk: ChunkMetadata) -> BuilderResult:
        """
        Custom validation specific to the builder type.
        
        Args:
            output: Generated output
            chunk: Source chunk
            
        Returns:
            Validation result
        """
        # Default implementation - always pass
        return BuilderResult(success=True)
    
    def _call_llm(self, prompt: str, router_decision) -> str:
        """
        Call the LLM with the given prompt.
        
        Args:
            prompt: Prompt to send
            router_decision: Router decision
            
        Returns:
            LLM response
        """
        # This is a placeholder - in a real implementation, this would call the actual LLM API
        # For now, we'll return a mock response
        gola_logger.debug(f"Calling LLM: {router_decision.selected_provider}/{router_decision.selected_model}")
        
        # Mock response - replace with actual LLM call
        return self._mock_llm_response(prompt, router_decision)
    
    def _mock_llm_response(self, prompt: str, router_decision) -> str:
        """
        Mock LLM response for testing.
        
        Args:
            prompt: Input prompt
            router_decision: Router decision
            
        Returns:
            Mock response
        """
        # This should be replaced with actual LLM API calls
        return f"Mock response for {self.config.task_type}: {prompt[:100]}..."
    
    def _generate_cache_key(self, chunk: ChunkMetadata, 
                           context: Optional[Dict[str, Any]]) -> str:
        """
        Generate cache key for the chunk.
        
        Args:
            chunk: Chunk metadata
            context: Processing context
            
        Returns:
            Cache key
        """
        # Create a hash of the chunk content and context
        content = f"{chunk.chunk_hash}_{self.config.task_type}_{self.config.prompt_version}"
        if context:
            content += f"_{hash(str(context))}"
        
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _is_cache_valid(self, result: BuilderResult) -> bool:
        """
        Check if cached result is still valid.
        
        Args:
            result: Cached result
            
        Returns:
            True if cache is valid
        """
        if not result.timestamp:
            return False
        
        age_hours = (datetime.utcnow() - result.timestamp).total_seconds() / 3600
        return age_hours < self.config.cache_ttl_hours
    
    def _log_result(self, chunk: ChunkMetadata, result: BuilderResult, 
                   duration: float) -> None:
        """
        Log processing result.
        
        Args:
            chunk: Processed chunk
            result: Processing result
            duration: Processing duration
        """
        if result.success:
            gola_logger.info(f"Successfully processed chunk {chunk.chunk_hash} "
                           f"({self.config.task_type}) in {duration:.2f}s")
        else:
            gola_logger.error(f"Failed to process chunk {chunk.chunk_hash} "
                            f"({self.config.task_type}): {result.error}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get builder statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "task_type": self.config.task_type,
            "prompt_version": self.config.prompt_version,
            "cache_size": len(self.cache),
            "cache_hits": 0,  # TODO: Implement cache hit tracking
            "total_processed": 0,  # TODO: Implement processing counter
        }
    
    def clear_cache(self) -> None:
        """Clear the builder cache."""
        self.cache.clear()
        gola_logger.info(f"Cleared cache for {self.config.task_type} builder") 