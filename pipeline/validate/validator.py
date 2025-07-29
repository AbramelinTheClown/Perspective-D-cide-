"""
Validation system for quality assurance and output validation.
"""

import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel, Field, ValidationError

from schemas.base import BaseOutput, QualityMetrics, EvidenceSpan
from cli.utils.logging import gola_logger

@dataclass
class ValidationRule:
    """A validation rule for outputs."""
    name: str
    description: str
    severity: str  # "error", "warning", "info"
    enabled: bool = True

@dataclass
class ValidationResult:
    """Result of a validation check."""
    rule_name: str
    passed: bool
    message: str
    severity: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

class ValidationReport:
    """Complete validation report."""
    
    def __init__(self, output_id: str):
        self.output_id = output_id
        self.results: List[ValidationResult] = []
        self.timestamp = datetime.utcnow()
        self.overall_passed = True
        self.error_count = 0
        self.warning_count = 0
        self.info_count = 0
    
    def add_result(self, result: ValidationResult) -> None:
        """Add a validation result."""
        self.results.append(result)
        
        if result.severity == "error" and not result.passed:
            self.overall_passed = False
            self.error_count += 1
        elif result.severity == "warning" and not result.passed:
            self.warning_count += 1
        elif result.severity == "info":
            self.info_count += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        return {
            "output_id": self.output_id,
            "timestamp": self.timestamp.isoformat(),
            "overall_passed": self.overall_passed,
            "total_checks": len(self.results),
            "errors": self.error_count,
            "warnings": self.warning_count,
            "info": self.info_count,
            "pass_rate": (len([r for r in self.results if r.passed]) / len(self.results)) if self.results else 0.0
        }
    
    def get_failed_checks(self) -> List[ValidationResult]:
        """Get all failed validation checks."""
        return [r for r in self.results if not r.passed]
    
    def get_errors(self) -> List[ValidationResult]:
        """Get all error-level failures."""
        return [r for r in self.results if r.severity == "error" and not r.passed]

class BaseValidator:
    """Base validator class."""
    
    def __init__(self, rules: Optional[List[ValidationRule]] = None):
        """
        Initialize validator.
        
        Args:
            rules: List of validation rules
        """
        self.rules = rules or self._get_default_rules()
    
    def validate_output(self, output: BaseOutput, chunk_text: str) -> ValidationReport:
        """
        Validate an output against all rules.
        
        Args:
            output: Output to validate
            chunk_text: Source chunk text
            
        Returns:
            Validation report
        """
        report = ValidationReport(output.run_id)
        
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            try:
                result = self._apply_rule(rule, output, chunk_text)
                report.add_result(result)
            except Exception as e:
                gola_logger.error(f"Error applying rule {rule.name}: {e}")
                result = ValidationResult(
                    rule_name=rule.name,
                    passed=False,
                    message=f"Rule application failed: {e}",
                    severity="error"
                )
                report.add_result(result)
        
        return report
    
    def _apply_rule(self, rule: ValidationRule, output: BaseOutput, 
                   chunk_text: str) -> ValidationResult:
        """
        Apply a specific validation rule.
        
        Args:
            rule: Validation rule
            output: Output to validate
            chunk_text: Source chunk text
            
        Returns:
            Validation result
        """
        # This should be overridden by subclasses
        return ValidationResult(
            rule_name=rule.name,
            passed=True,
            message="Rule not implemented",
            severity=rule.severity
        )
    
    def _get_default_rules(self) -> List[ValidationRule]:
        """Get default validation rules."""
        return [
            ValidationRule(
                name="schema_validation",
                description="Validate output schema",
                severity="error"
            ),
            ValidationRule(
                name="quality_metrics",
                description="Check quality metrics",
                severity="warning"
            ),
            ValidationRule(
                name="evidence_spans",
                description="Validate evidence spans",
                severity="error"
            ),
            ValidationRule(
                name="content_coverage",
                description="Check content coverage",
                severity="warning"
            )
        ]

class OutputValidator(BaseValidator):
    """Validator for general outputs."""
    
    def _apply_rule(self, rule: ValidationRule, output: BaseOutput, 
                   chunk_text: str) -> ValidationResult:
        """
        Apply validation rule to output.
        
        Args:
            rule: Validation rule
            output: Output to validate
            chunk_text: Source chunk text
            
        Returns:
            Validation result
        """
        if rule.name == "schema_validation":
            return self._validate_schema(output)
        elif rule.name == "quality_metrics":
            return self._validate_quality_metrics(output)
        elif rule.name == "evidence_spans":
            return self._validate_evidence_spans(output, chunk_text)
        elif rule.name == "content_coverage":
            return self._validate_content_coverage(output, chunk_text)
        else:
            return ValidationResult(
                rule_name=rule.name,
                passed=True,
                message="Rule not implemented",
                severity=rule.severity
            )
    
    def _validate_schema(self, output: BaseOutput) -> ValidationResult:
        """Validate output schema."""
        try:
            # Check required fields
            if not hasattr(output, 'run_id') or not output.run_id:
                return ValidationResult(
                    rule_name="schema_validation",
                    passed=False,
                    message="Missing run_id",
                    severity="error"
                )
            
            if not hasattr(output, 'quality_metrics') or not output.quality_metrics:
                return ValidationResult(
                    rule_name="schema_validation",
                    passed=False,
                    message="Missing quality_metrics",
                    severity="error"
                )
            
            return ValidationResult(
                rule_name="schema_validation",
                passed=True,
                message="Schema validation passed",
                severity="error"
            )
            
        except Exception as e:
            return ValidationResult(
                rule_name="schema_validation",
                passed=False,
                message=f"Schema validation error: {e}",
                severity="error"
            )
    
    def _validate_quality_metrics(self, output: BaseOutput) -> ValidationResult:
        """Validate quality metrics."""
        try:
            metrics = output.quality_metrics
            
            # Check confidence
            if metrics.confidence < 0.3:
                return ValidationResult(
                    rule_name="quality_metrics",
                    passed=False,
                    message=f"Low confidence: {metrics.confidence}",
                    severity="warning"
                )
            
            # Check coverage
            if metrics.coverage < 0.5:
                return ValidationResult(
                    rule_name="quality_metrics",
                    passed=False,
                    message=f"Low coverage: {metrics.coverage}",
                    severity="warning"
                )
            
            # Check evidence span coverage
            if metrics.evidence_span_coverage < 0.3:
                return ValidationResult(
                    rule_name="quality_metrics",
                    passed=False,
                    message=f"Low evidence coverage: {metrics.evidence_span_coverage}",
                    severity="warning"
                )
            
            return ValidationResult(
                rule_name="quality_metrics",
                passed=True,
                message="Quality metrics acceptable",
                severity="warning"
            )
            
        except Exception as e:
            return ValidationResult(
                rule_name="quality_metrics",
                passed=False,
                message=f"Quality metrics validation error: {e}",
                severity="warning"
            )
    
    def _validate_evidence_spans(self, output: BaseOutput, chunk_text: str) -> ValidationResult:
        """Validate evidence spans."""
        try:
            if not output.evidence_spans:
                return ValidationResult(
                    rule_name="evidence_spans",
                    passed=False,
                    message="No evidence spans provided",
                    severity="error"
                )
            
            # Check each evidence span
            for i, span in enumerate(output.evidence_spans):
                # Check span bounds
                if span.start < 0 or span.end > len(chunk_text):
                    return ValidationResult(
                        rule_name="evidence_spans",
                        passed=False,
                        message=f"Evidence span {i} out of bounds: {span.start}-{span.end}",
                        severity="error"
                    )
                
                if span.start >= span.end:
                    return ValidationResult(
                        rule_name="evidence_spans",
                        passed=False,
                        message=f"Evidence span {i} invalid: start >= end",
                        severity="error"
                    )
                
                # Check if span text matches
                span_text = chunk_text[span.start:span.end]
                if span.text and span.text != span_text:
                    return ValidationResult(
                        rule_name="evidence_spans",
                        passed=False,
                        message=f"Evidence span {i} text mismatch",
                        severity="error"
                    )
            
            return ValidationResult(
                rule_name="evidence_spans",
                passed=True,
                message=f"All {len(output.evidence_spans)} evidence spans valid",
                severity="error"
            )
            
        except Exception as e:
            return ValidationResult(
                rule_name="evidence_spans",
                passed=False,
                message=f"Evidence spans validation error: {e}",
                severity="error"
            )
    
    def _validate_content_coverage(self, output: BaseOutput, chunk_text: str) -> ValidationResult:
        """Validate content coverage."""
        try:
            # Calculate coverage based on evidence spans
            total_covered = 0
            for span in output.evidence_spans:
                total_covered += span.end - span.start
            
            coverage_ratio = total_covered / len(chunk_text) if chunk_text else 0
            
            if coverage_ratio < 0.1:
                return ValidationResult(
                    rule_name="content_coverage",
                    passed=False,
                    message=f"Low content coverage: {coverage_ratio:.2%}",
                    severity="warning"
                )
            
            return ValidationResult(
                rule_name="content_coverage",
                passed=True,
                message=f"Content coverage: {coverage_ratio:.2%}",
                severity="warning"
            )
            
        except Exception as e:
            return ValidationResult(
                rule_name="content_coverage",
                passed=False,
                message=f"Content coverage validation error: {e}",
                severity="warning"
            )

class ValidationManager:
    """Manager for validation operations."""
    
    def __init__(self):
        """Initialize validation manager."""
        self.validators: Dict[str, BaseValidator] = {}
        self.validation_history: List[ValidationReport] = []
    
    def register_validator(self, output_type: str, validator: BaseValidator) -> None:
        """
        Register a validator for an output type.
        
        Args:
            output_type: Type of output to validate
            validator: Validator instance
        """
        self.validators[output_type] = validator
        gola_logger.info(f"Registered validator for output type: {output_type}")
    
    def validate_output(self, output: BaseOutput, chunk_text: str, 
                       output_type: str = "general") -> ValidationReport:
        """
        Validate an output.
        
        Args:
            output: Output to validate
            chunk_text: Source chunk text
            output_type: Type of output
            
        Returns:
            Validation report
        """
        validator = self.validators.get(output_type)
        if not validator:
            # Use default validator
            validator = OutputValidator()
        
        report = validator.validate_output(output, chunk_text)
        
        # Store in history
        self.validation_history.append(report)
        
        # Log results
        summary = report.get_summary()
        if summary["overall_passed"]:
            gola_logger.info(f"Validation passed for {output.run_id}: {summary['pass_rate']:.1%} pass rate")
        else:
            gola_logger.warning(f"Validation failed for {output.run_id}: {summary['errors']} errors")
        
        return report
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """
        Get validation statistics.
        
        Returns:
            Validation statistics
        """
        if not self.validation_history:
            return {"total_validations": 0}
        
        total_validations = len(self.validation_history)
        passed_validations = len([r for r in self.validation_history if r.overall_passed])
        
        total_checks = sum(len(r.results) for r in self.validation_history)
        passed_checks = sum(len([res for res in r.results if res.passed]) for r in self.validation_history)
        
        return {
            "total_validations": total_validations,
            "passed_validations": passed_validations,
            "pass_rate": passed_validations / total_validations if total_validations > 0 else 0.0,
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "check_pass_rate": passed_checks / total_checks if total_checks > 0 else 0.0,
            "average_errors_per_validation": sum(r.error_count for r in self.validation_history) / total_validations,
            "average_warnings_per_validation": sum(r.warning_count for r in self.validation_history) / total_validations
        }

# Global validation manager
validation_manager = ValidationManager()

def get_validation_manager() -> ValidationManager:
    """Get the global validation manager."""
    return validation_manager

def init_validation_manager() -> ValidationManager:
    """Initialize the global validation manager."""
    global validation_manager
    
    # Register default validators
    validation_manager.register_validator("general", OutputValidator())
    
    return validation_manager 