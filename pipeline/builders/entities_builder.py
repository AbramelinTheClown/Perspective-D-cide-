"""
Entities builder for extracting named entities from text.
"""

import json
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from pydantic import BaseModel, Field

from pipeline.builders.base import BaseBuilder, BuilderConfig, BuilderResult
from schemas.base import BaseOutput, QualityMetrics, EvidenceSpan
from cli.utils.logging import gola_logger

class Entity(BaseModel):
    """A named entity with metadata."""
    text: str = Field(..., description="Entity text")
    type: str = Field(..., description="Entity type (PERSON, ORGANIZATION, LOCATION, etc.)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    evidence_spans: List[Dict[str, int]] = Field(default_factory=list, description="Evidence spans")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    normalized_form: Optional[str] = Field(None, description="Normalized entity form")

class EntitiesOutput(BaseOutput):
    """Entities extraction output."""
    entities: List[Entity] = Field(default_factory=list, description="Extracted entities")
    entity_types: List[str] = Field(default_factory=list, description="Entity types found")
    entity_count: int = Field(0, description="Total entity count")
    most_common_type: Optional[str] = Field(None, description="Most common entity type")
    
    class Config:
        json_encoders = {
            Entity: lambda v: v.dict()
        }

class EntitiesBuilder(BaseBuilder):
    """Builder for extracting named entities."""
    
    def __init__(self, config: Optional[BuilderConfig] = None):
        """
        Initialize entities builder.
        
        Args:
            config: Builder configuration
        """
        if config is None:
            config = BuilderConfig(
                task_type="entities",
                prompt_version="v1.0",
                router_policy=self.router_policy.THROUGHPUT
            )
        super().__init__(config)
    
    def _generate_prompt(self, chunk, context, router_decision) -> str:
        """
        Generate prompt for entity extraction.
        
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
            if 'entity_types' in context:
                context_info += f"\nFocus on entity types: {', '.join(context['entity_types'])}"
            if 'domain' in context:
                context_info += f"\nDomain: {context['domain']}"
        
        prompt = f"""You are an expert named entity recognition system. Extract all named entities from the following text.

{context_info}

TEXT TO ANALYZE:
{chunk.text_norm}

INSTRUCTIONS:
1. Identify all named entities (people, organizations, locations, dates, etc.)
2. For each entity, provide:
   - The exact text as it appears
   - The entity type (PERSON, ORGANIZATION, LOCATION, DATE, MONEY, PERCENT, etc.)
   - Confidence score (0.0-1.0)
   - Evidence spans (character positions in the text)
   - Normalized form if applicable
3. Include any additional metadata (e.g., title for persons, country for locations)
4. Ensure all evidence spans are accurate and reference actual text

OUTPUT FORMAT (JSON):
{{
    "entities": [
        {{
            "text": "Entity text as it appears",
            "type": "PERSON",
            "confidence": 0.95,
            "evidence_spans": [{{"start": 10, "end": 25}}],
            "normalized_form": "Normalized entity name",
            "metadata": {{
                "title": "Dr.",
                "nationality": "American"
            }}
        }}
    ],
    "quality_metrics": {{
        "confidence": 0.85,
        "coverage": 0.9,
        "evidence_span_coverage": 0.8
    }}
}}

Use standard entity types: PERSON, ORGANIZATION, LOCATION, DATE, TIME, MONEY, PERCENT, QUANTITY, ORDINAL, CARDINAL, FACILITY, PRODUCT, EVENT, WORK_OF_ART, LAW, LANGUAGE, NATIONALITY, RELIGION, IDEOLOGY, OTHER."""
        
        return prompt
    
    def _parse_response(self, response: str, chunk) -> EntitiesOutput:
        """
        Parse LLM response into entities output.
        
        Args:
            response: LLM response
            chunk: Source chunk
            
        Returns:
            Entities output
        """
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in response")
            
            json_str = json_match.group()
            data = json.loads(json_str)
            
            # Create entities output
            output = EntitiesOutput(
                run_id=f"entities_{chunk.chunk_hash}",
                quality_metrics=QualityMetrics(
                    confidence=data.get("quality_metrics", {}).get("confidence", 0.8),
                    coverage=data.get("quality_metrics", {}).get("coverage", 0.8),
                    evidence_span_coverage=data.get("quality_metrics", {}).get("evidence_span_coverage", 0.8)
                ),
                evidence_spans=self._extract_evidence_spans(data, chunk)
            )
            
            # Parse entities
            for entity_data in data.get("entities", []):
                entity = Entity(
                    text=entity_data.get("text", ""),
                    type=entity_data.get("type", "OTHER"),
                    confidence=entity_data.get("confidence", 0.8),
                    evidence_spans=entity_data.get("evidence_spans", []),
                    normalized_form=entity_data.get("normalized_form"),
                    metadata=entity_data.get("metadata", {})
                )
                output.entities.append(entity)
            
            # Calculate statistics
            output.entity_count = len(output.entities)
            output.entity_types = list(set(entity.type for entity in output.entities))
            
            # Find most common entity type
            if output.entities:
                type_counts = {}
                for entity in output.entities:
                    type_counts[entity.type] = type_counts.get(entity.type, 0) + 1
                output.most_common_type = max(type_counts, key=type_counts.get)
            
            return output
            
        except Exception as e:
            gola_logger.error(f"Error parsing entities response: {e}")
            # Return a basic entities output as fallback
            return self._create_fallback_entities(chunk)
    
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
        
        for entity in data.get("entities", []):
            for span_data in entity.get("evidence_spans", []):
                span = EvidenceSpan(
                    start=span_data.get("start", 0),
                    end=span_data.get("end", 0),
                    text=chunk.text_norm[span_data.get("start", 0):span_data.get("end", 0)]
                )
                spans.append(span)
        
        return spans
    
    def _create_fallback_entities(self, chunk) -> EntitiesOutput:
        """
        Create fallback entities when parsing fails.
        
        Args:
            chunk: Source chunk
            
        Returns:
            Fallback entities output
        """
        # Simple fallback: look for capitalized words as potential entities
        words = chunk.text_norm.split()
        entities = []
        
        for i, word in enumerate(words):
            if len(word) > 2 and word[0].isupper() and word[1:].islower():
                # Simple heuristic for potential proper nouns
                entity = Entity(
                    text=word,
                    type="OTHER",
                    confidence=0.3,
                    evidence_spans=[{"start": chunk.text_norm.find(word), 
                                   "end": chunk.text_norm.find(word) + len(word)}],
                    normalized_form=word
                )
                entities.append(entity)
        
        return EntitiesOutput(
            run_id=f"entities_fallback_{chunk.chunk_hash}",
            quality_metrics=QualityMetrics(
                confidence=0.3,
                coverage=0.2,
                evidence_span_coverage=0.3
            ),
            entities=entities,
            entity_count=len(entities),
            entity_types=["OTHER"] if entities else []
        )
    
    def _custom_validation(self, output: EntitiesOutput, chunk) -> BuilderResult:
        """
        Custom validation for entities output.
        
        Args:
            output: Entities output
            chunk: Source chunk
            
        Returns:
            Validation result
        """
        try:
            # Check if we have entities
            if not output.entities:
                return BuilderResult(success=False, error="No entities extracted")
            
            # Check entity quality
            for entity in output.entities:
                if not entity.text.strip():
                    return BuilderResult(success=False, error="Empty entity text")
                
                if entity.confidence < 0.1:
                    return BuilderResult(success=False, error="Entity confidence too low")
                
                if not entity.evidence_spans:
                    return BuilderResult(success=False, error="Entity missing evidence spans")
                
                # Check if entity text appears in the chunk
                if entity.text not in chunk.text_norm:
                    return BuilderResult(success=False, error="Entity text not found in chunk")
            
            return BuilderResult(success=True)
            
        except Exception as e:
            return BuilderResult(success=False, error=f"Entities validation error: {e}")
    
    def get_entity_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about extracted entities.
        
        Args:
            Statistics dictionary
        """
        if not hasattr(self, '_last_output') or not self._last_output:
            return {}
        
        output = self._last_output
        
        # Count entities by type
        type_counts = {}
        for entity in output.entities:
            type_counts[entity.type] = type_counts.get(entity.type, 0) + 1
        
        # Calculate average confidence
        avg_confidence = sum(entity.confidence for entity in output.entities) / len(output.entities) if output.entities else 0
        
        return {
            "total_entities": output.entity_count,
            "entity_types": output.entity_types,
            "type_counts": type_counts,
            "most_common_type": output.most_common_type,
            "average_confidence": avg_confidence,
            "high_confidence_entities": len([e for e in output.entities if e.confidence > 0.8])
        }

def create_entities_builder(config: Optional[BuilderConfig] = None) -> EntitiesBuilder:
    """
    Create an entities builder instance.
    
    Args:
        config: Builder configuration
        
    Returns:
        Entities builder instance
    """
    return EntitiesBuilder(config) 