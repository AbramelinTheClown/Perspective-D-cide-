"""
Sync Adapters for Dev Vector DB Hub Integration
Maps Gola pipeline data to hub format and handles re-embedding.
"""

import time
import hashlib
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path

from .client_rest import HubItem, HubRestClient

@dataclass
class DatasetInsight:
    """Represents an insight extracted from a dataset."""
    content: str
    content_type: str
    metadata: Dict[str, Any]
    confidence: float = 1.0
    source_dataset: str = ""
    source_task: str = ""

class HubSyncAdapter:
    """Adapter for syncing Gola pipeline data to the Dev Vector DB Hub."""
    
    def __init__(self, hub_client: HubRestClient, config: Dict[str, Any]):
        """Initialize the sync adapter."""
        self.hub_client = hub_client
        self.config = config
        self.sync_policies = config.get("sync_policies", {})
        
    def extract_patterns_from_runs(self, runs: List[Dict[str, Any]], dataset_slug: str) -> List[DatasetInsight]:
        """Extract patterns from successful runs."""
        insights = []
        
        # Pattern: Successful model combinations
        model_combinations = {}
        task_model_success = {}
        
        for run in runs:
            if run.get("status") == "ok":
                provider = run.get("provider", "unknown")
                model_id = run.get("model_id", "unknown")
                task_type = run.get("task_type", "unknown")
                
                # Track model combinations
                combo_key = f"{provider}/{model_id}"
                model_combinations[combo_key] = model_combinations.get(combo_key, 0) + 1
                
                # Track task-specific success
                task_key = f"{task_type}/{combo_key}"
                task_model_success[task_key] = task_model_success.get(task_key, 0) + 1
        
        # Create pattern insights
        for combo, count in model_combinations.items():
            if count >= 3:  # Only patterns with multiple successes
                insights.append(DatasetInsight(
                    content=f"Model combination {combo} consistently succeeds for dataset {dataset_slug}",
                    content_type="pattern",
                    metadata={
                        "model_combination": combo,
                        "success_count": count,
                        "total_runs": len(runs),
                        "success_rate": count / len(runs),
                        "dataset": dataset_slug
                    },
                    source_dataset=dataset_slug,
                    source_task="model_selection"
                ))
        
        # Task-specific patterns
        for task_combo, count in task_model_success.items():
            if count >= 2:
                task_type, model_combo = task_combo.split("/", 1)
                insights.append(DatasetInsight(
                    content=f"Task '{task_type}' works well with {model_combo} for {dataset_slug}",
                    content_type="pattern",
                    metadata={
                        "task_type": task_type,
                        "model_combination": model_combo,
                        "success_count": count,
                        "dataset": dataset_slug
                    },
                    source_dataset=dataset_slug,
                    source_task=task_type
                ))
        
        return insights
    
    def extract_concepts_from_entities(self, entities: List[Dict[str, Any]], dataset_slug: str) -> List[DatasetInsight]:
        """Extract concepts from entity analysis."""
        insights = []
        
        if not entities:
            return insights
        
        # Concept: Entity type distribution
        entity_types = {}
        entity_texts = {}
        
        for entity in entities:
            entity_type = entity.get("entity_type", "unknown")
            entity_text = entity.get("entity_text", "")
            
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
            if entity_text:
                entity_texts[entity_text] = entity_texts.get(entity_text, 0) + 1
        
        # Significant entity types
        for entity_type, count in entity_types.items():
            if count >= 5:  # Only significant patterns
                insights.append(DatasetInsight(
                    content=f"Entity type '{entity_type}' is prominent in {dataset_slug}",
                    content_type="concept",
                    metadata={
                        "entity_type": entity_type,
                        "frequency": count,
                        "total_entities": len(entities),
                        "percentage": count / len(entities),
                        "dataset": dataset_slug
                    },
                    source_dataset=dataset_slug,
                    source_task="entity_extraction"
                ))
        
        # Common entity texts
        for entity_text, count in entity_texts.items():
            if count >= 3:
                insights.append(DatasetInsight(
                    content=f"Entity '{entity_text}' appears frequently in {dataset_slug}",
                    content_type="concept",
                    metadata={
                        "entity_text": entity_text,
                        "frequency": count,
                        "dataset": dataset_slug
                    },
                    source_dataset=dataset_slug,
                    source_task="entity_extraction"
                ))
        
        return insights
    
    def extract_concepts_from_triples(self, triples: List[Dict[str, Any]], dataset_slug: str) -> List[DatasetInsight]:
        """Extract concepts from triple analysis."""
        insights = []
        
        if not triples:
            return insights
        
        # Concept: Common predicates
        predicates = {}
        subjects = {}
        objects = {}
        
        for triple in triples:
            pred = triple.get("predicate", "")
            subj = triple.get("subject", "")
            obj = triple.get("object", "")
            
            if pred:
                predicates[pred] = predicates.get(pred, 0) + 1
            if subj:
                subjects[subj] = subjects.get(subj, 0) + 1
            if obj:
                objects[obj] = objects.get(obj, 0) + 1
        
        # Significant predicates
        for predicate, count in predicates.items():
            if count >= 3:
                insights.append(DatasetInsight(
                    content=f"Predicate '{predicate}' is common in {dataset_slug}",
                    content_type="concept",
                    metadata={
                        "predicate": predicate,
                        "frequency": count,
                        "total_triples": len(triples),
                        "dataset": dataset_slug
                    },
                    source_dataset=dataset_slug,
                    source_task="triple_extraction"
                ))
        
        return insights
    
    def extract_ideas_from_qa_pairs(self, qa_pairs: List[Dict[str, Any]], dataset_slug: str) -> List[DatasetInsight]:
        """Extract ideas from QA pair analysis."""
        insights = []
        
        if not qa_pairs:
            return insights
        
        # Idea: Question patterns
        question_types = {}
        answer_types = {}
        
        for qa in qa_pairs:
            question = qa.get("question", "")
            answer_type = qa.get("answer_type", "unknown")
            
            # Simple question type detection
            if "what" in question.lower():
                question_types["what"] = question_types.get("what", 0) + 1
            elif "how" in question.lower():
                question_types["how"] = question_types.get("how", 0) + 1
            elif "why" in question.lower():
                question_types["why"] = question_types.get("why", 0) + 1
            elif "when" in question.lower():
                question_types["when"] = question_types.get("when", 0) + 1
            elif "where" in question.lower():
                question_types["where"] = question_types.get("where", 0) + 1
            
            answer_types[answer_type] = answer_types.get(answer_type, 0) + 1
        
        # Question type insights
        for q_type, count in question_types.items():
            if count >= 2:
                insights.append(DatasetInsight(
                    content=f"'{q_type.capitalize()}' questions are common in {dataset_slug}",
                    content_type="idea",
                    metadata={
                        "question_type": q_type,
                        "frequency": count,
                        "total_qa_pairs": len(qa_pairs),
                        "dataset": dataset_slug
                    },
                    source_dataset=dataset_slug,
                    source_task="qa_generation"
                ))
        
        # Answer type insights
        for a_type, count in answer_types.items():
            if count >= 3:
                insights.append(DatasetInsight(
                    content=f"Answer type '{a_type}' is prevalent in {dataset_slug}",
                    content_type="idea",
                    metadata={
                        "answer_type": a_type,
                        "frequency": count,
                        "total_qa_pairs": len(qa_pairs),
                        "dataset": dataset_slug
                    },
                    source_dataset=dataset_slug,
                    source_task="qa_generation"
                ))
        
        return insights
    
    def create_documentation_insights(self, dataset_data: Dict[str, Any], dataset_slug: str) -> List[DatasetInsight]:
        """Create documentation insights from dataset metadata."""
        insights = []
        
        # Documentation: Dataset manifest
        manifest_content = f"Dataset {dataset_slug} processed with {dataset_data.get('total_runs', 0)} runs"
        if "quality_metrics" in dataset_data:
            metrics = dataset_data["quality_metrics"]
            manifest_content += f", quality score: {metrics.get('overall_score', 'N/A')}"
        
        insights.append(DatasetInsight(
            content=manifest_content,
            content_type="documentation",
            metadata={
                "dataset": dataset_slug,
                "total_runs": dataset_data.get("total_runs", 0),
                "quality_metrics": dataset_data.get("quality_metrics", {}),
                "created_at": time.time()
            },
            source_dataset=dataset_slug,
            source_task="documentation"
        ))
        
        return insights
    
    def sync_dataset_to_hub(self, dataset_slug: str, dataset_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sync a complete dataset to the hub."""
        insights = []
        
        # Extract patterns from runs
        if "runs" in dataset_data:
            run_insights = self.extract_patterns_from_runs(dataset_data["runs"], dataset_slug)
            insights.extend(run_insights)
        
        # Extract concepts from outputs
        if "outputs" in dataset_data:
            outputs = dataset_data["outputs"]
            
            # Entity insights
            if "entities" in outputs:
                entity_insights = self.extract_concepts_from_entities(outputs["entities"], dataset_slug)
                insights.extend(entity_insights)
            
            # Triple insights
            if "triples" in outputs:
                triple_insights = self.extract_concepts_from_triples(outputs["triples"], dataset_slug)
                insights.extend(triple_insights)
            
            # QA insights
            if "qa_pairs" in outputs:
                qa_insights = self.extract_ideas_from_qa_pairs(outputs["qa_pairs"], dataset_slug)
                insights.extend(qa_insights)
        
        # Documentation insights
        doc_insights = self.create_documentation_insights(dataset_data, dataset_slug)
        insights.extend(doc_insights)
        
        # Filter insights based on sync policies
        filtered_insights = self._filter_insights(insights)
        
        # Convert to HubItems and store
        hub_items = []
        for insight in filtered_insights:
            hub_item = HubItem(
                content=insight.content,
                content_type=insight.content_type,
                metadata={
                    **insight.metadata,
                    "confidence": insight.confidence,
                    "source_dataset": insight.source_dataset,
                    "source_task": insight.source_task,
                    "extracted_at": time.time()
                }
            )
            hub_items.append(hub_item)
        
        # Store in hub
        results = []
        for hub_item in hub_items:
            try:
                result = self.hub_client.store_item(hub_item)
                results.append(result)
            except Exception as e:
                print(f"Warning: Failed to store insight: {e}")
        
        return {
            "dataset_slug": dataset_slug,
            "total_insights": len(insights),
            "filtered_insights": len(filtered_insights),
            "stored_insights": len(results),
            "insights_by_type": self._group_insights_by_type(filtered_insights)
        }
    
    def _filter_insights(self, insights: List[DatasetInsight]) -> List[DatasetInsight]:
        """Filter insights based on sync policies."""
        filtered = []
        
        for insight in insights:
            # Check if this content type should be pushed
            push_key = f"push_{insight.content_type}s"
            if not self.sync_policies.get(push_key, True):
                continue
            
            # Check quality thresholds
            if insight.confidence < self.sync_policies.get("min_confidence", 0.7):
                continue
            
            # Check frequency thresholds
            frequency = insight.metadata.get("frequency", 0)
            if frequency < 2:  # Minimum frequency threshold
                continue
            
            filtered.append(insight)
        
        return filtered
    
    def _group_insights_by_type(self, insights: List[DatasetInsight]) -> Dict[str, int]:
        """Group insights by content type."""
        grouped = {}
        for insight in insights:
            grouped[insight.content_type] = grouped.get(insight.content_type, 0) + 1
        return grouped
    
    def get_relevant_patterns(self, query: str, content_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get relevant patterns from the hub for a query."""
        return self.hub_client.search(query, content_types or ["pattern", "concept"])
    
    def get_project_context(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """Get project context from the hub."""
        return self.hub_client.get_project_context(project_id) 