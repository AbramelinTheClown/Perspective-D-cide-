"""
Summary builder for generating document summaries.
"""

import json
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from pydantic import BaseModel, Field

from pipeline.builders.base import BaseBuilder, BuilderConfig, BuilderResult
from schemas.base import BaseOutput, QualityMetrics, EvidenceSpan
from cli.utils.logging import gola_logger

class SummaryPoint(BaseModel):
    """A summary point with evidence."""
    text: str = Field(..., description="Summary point text")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    evidence_spans: List[Dict[str, int]] = Field(default_factory=list, description="Evidence spans")
    type: str = Field("key_point", description="Point type (key_point, fact, insight)")

class SummaryOutput(BaseOutput):
    """Summary output with extractive and abstractive summaries."""
    extractive_summary: List[SummaryPoint] = Field(default_factory=list, description="Extractive summary points")
    abstractive_summary: str = Field("", description="Abstractive summary text")
    key_quotes: List[Dict[str, Any]] = Field(default_factory=list, description="Key quotes with spans")
    topics: List[str] = Field(default_factory=list, description="Main topics")
    reading_level: str = Field("general", description="Reading level (basic, general, technical)")
    word_count: int = Field(0, description="Word count of summary")
    
    class Config:
        json_encoders = {
            SummaryPoint: lambda v: v.dict()
        }

class SummaryBuilder(BaseBuilder):
    """Builder for generating document summaries."""
    
    def __init__(self, config: Optional[BuilderConfig] = None):
        """
        Initialize summary builder.
        
        Args:
            config: Builder configuration
        """
        if config is None:
            config = BuilderConfig(
                task_type="summary",
                prompt_version="v1.0",
                router_policy=self.router_policy.THROUGHPUT
            )
        super().__init__(config)
    
    def _generate_prompt(self, chunk, context, router_decision) -> str:
        """
        Generate prompt for summary generation.
        
        Args:
            chunk: Chunk metadata
            context: Processing context
            router_decision: Router decision
            
        Returns:
            Generated prompt
        """
        # Build context information
        context_info = ""
        if context:
            if 'document_type' in context:
                context_info += f"\nDocument type: {context['document_type']}"
            if 'target_audience' in context:
                context_info += f"\nTarget audience: {context['target_audience']}"
            if 'summary_length' in context:
                context_info += f"\nDesired summary length: {context['summary_length']}"
        
        prompt = f"""You are an expert document summarizer. Analyze the following text and create a comprehensive summary.

{context_info}

TEXT TO SUMMARIZE:
{chunk.text_norm}

INSTRUCTIONS:
1. Create an extractive summary with 3-5 key points, each with evidence from the text
2. Create an abstractive summary that captures the main ideas in 2-3 sentences
3. Identify 2-4 key quotes that best represent the content
4. List the main topics discussed
5. Assess the reading level (basic, general, technical)
6. Provide confidence scores for each summary point

OUTPUT FORMAT (JSON):
{{
    "extractive_summary": [
        {{
            "text": "Key point text",
            "confidence": 0.95,
            "evidence_spans": [{{"start": 10, "end": 50}}],
            "type": "key_point"
        }}
    ],
    "abstractive_summary": "Overall summary in 2-3 sentences",
    "key_quotes": [
        {{
            "text": "Quote text",
            "confidence": 0.9,
            "span": {{"start": 100, "end": 150}},
            "importance": "high"
        }}
    ],
    "topics": ["topic1", "topic2"],
    "reading_level": "general",
    "quality_metrics": {{
        "confidence": 0.85,
        "coverage": 0.9,
        "evidence_span_coverage": 0.8
    }}
}}

Ensure all evidence spans reference actual text positions in the input."""
        
        return prompt
    
    def _parse_response(self, response: str, chunk) -> SummaryOutput:
        """
        Parse LLM response into summary output.
        
        Args:
            response: LLM response
            chunk: Source chunk
            
        Returns:
            Summary output
        """
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in response")
            
            json_str = json_match.group()
            data = json.loads(json_str)
            
            # Create summary output
            output = SummaryOutput(
                run_id=f"summary_{chunk.chunk_hash}",
                quality_metrics=QualityMetrics(
                    confidence=data.get("quality_metrics", {}).get("confidence", 0.8),
                    coverage=data.get("quality_metrics", {}).get("coverage", 0.8),
                    evidence_span_coverage=data.get("quality_metrics", {}).get("evidence_span_coverage", 0.8)
                ),
                evidence_spans=self._extract_evidence_spans(data, chunk),
                metadata={
                    "reading_level": data.get("reading_level", "general"),
                    "topics": data.get("topics", [])
                }
            )
            
            # Parse extractive summary
            for point_data in data.get("extractive_summary", []):
                point = SummaryPoint(
                    text=point_data.get("text", ""),
                    confidence=point_data.get("confidence", 0.8),
                    evidence_spans=point_data.get("evidence_spans", []),
                    type=point_data.get("type", "key_point")
                )
                output.extractive_summary.append(point)
            
            # Parse abstractive summary
            output.abstractive_summary = data.get("abstractive_summary", "")
            
            # Parse key quotes
            for quote_data in data.get("key_quotes", []):
                quote = {
                    "text": quote_data.get("text", ""),
                    "confidence": quote_data.get("confidence", 0.8),
                    "span": quote_data.get("span", {}),
                    "importance": quote_data.get("importance", "medium")
                }
                output.key_quotes.append(quote)
            
            # Parse topics
            output.topics = data.get("topics", [])
            output.reading_level = data.get("reading_level", "general")
            
            # Calculate word count
            output.word_count = len(output.abstractive_summary.split())
            
            return output
            
        except Exception as e:
            gola_logger.error(f"Error parsing summary response: {e}")
            # Return a basic summary as fallback
            return self._create_fallback_summary(chunk)
    
    def _extract_evidence_spans(self, data: Dict[str, Any], chunk) -> List[EvidenceSpan]:
        """
        Extract evidence spans from parsed data.
        
        Args:
            data: Parsed response data
            chunk: Source chunk
            
        Returns:
            List of evidence spans
        """
        spans = []
        
        # Extract spans from extractive summary
        for point in data.get("extractive_summary", []):
            for span_data in point.get("evidence_spans", []):
                span = EvidenceSpan(
                    start=span_data.get("start", 0),
                    end=span_data.get("end", 0),
                    text=chunk.text_norm[span_data.get("start", 0):span_data.get("end", 0)]
                )
                spans.append(span)
        
        # Extract spans from key quotes
        for quote in data.get("key_quotes", []):
            span_data = quote.get("span", {})
            if span_data:
                span = EvidenceSpan(
                    start=span_data.get("start", 0),
                    end=span_data.get("end", 0),
                    text=chunk.text_norm[span_data.get("start", 0):span_data.get("end", 0)]
                )
                spans.append(span)
        
        return spans
    
    def _create_fallback_summary(self, chunk) -> SummaryOutput:
        """
        Create a fallback summary when parsing fails.
        
        Args:
            chunk: Source chunk
            
        Returns:
            Fallback summary output
        """
        # Create a simple extractive summary
        sentences = chunk.text_norm.split('.')
        key_points = []
        
        for i, sentence in enumerate(sentences[:3]):  # Take first 3 sentences
            if len(sentence.strip()) > 10:
                point = SummaryPoint(
                    text=sentence.strip(),
                    confidence=0.6,
                    evidence_spans=[{"start": chunk.text_norm.find(sentence), 
                                   "end": chunk.text_norm.find(sentence) + len(sentence)}],
                    type="key_point"
                )
                key_points.append(point)
        
        # Create simple abstractive summary
        abstractive = f"Document contains {len(sentences)} sentences covering various topics."
        
        return SummaryOutput(
            run_id=f"summary_fallback_{chunk.chunk_hash}",
            quality_metrics=QualityMetrics(
                confidence=0.5,
                coverage=0.3,
                evidence_span_coverage=0.4
            ),
            extractive_summary=key_points,
            abstractive_summary=abstractive,
            topics=["general"],
            reading_level="general",
            word_count=len(abstractive.split())
        )
    
    def _custom_validation(self, output: SummaryOutput, chunk) -> BuilderResult:
        """
        Custom validation for summary output.
        
        Args:
            output: Summary output
            chunk: Source chunk
            
        Returns:
            Validation result
        """
        try:
            # Check if we have at least one summary point
            if not output.extractive_summary and not output.abstractive_summary:
                return BuilderResult(success=False, error="No summary content generated")
            
            # Check if abstractive summary is reasonable length
            if output.abstractive_summary:
                word_count = len(output.abstractive_summary.split())
                if word_count < 5:
                    return BuilderResult(success=False, error="Abstractive summary too short")
                if word_count > 100:
                    return BuilderResult(success=False, error="Abstractive summary too long")
            
            # Check if extractive points have evidence
            for point in output.extractive_summary:
                if not point.evidence_spans:
                    return BuilderResult(success=False, error="Extractive point missing evidence")
            
            return BuilderResult(success=True)
            
        except Exception as e:
            return BuilderResult(success=False, error=f"Summary validation error: {e}")

def create_summary_builder(config: Optional[BuilderConfig] = None) -> SummaryBuilder:
    """
    Create a summary builder instance.
    
    Args:
        config: Builder configuration
        
    Returns:
        Summary builder instance
    """
    return SummaryBuilder(config) 