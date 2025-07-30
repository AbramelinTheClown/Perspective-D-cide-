"""
UI Scaffold Validators for Perspective D<cide>.

Provides validation for scaffold data and generated UI components.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationIssue:
    """A validation issue found during scaffold validation."""
    
    level: ValidationLevel
    message: str
    field: Optional[str] = None
    suggestion: Optional[str] = None

@dataclass
class ValidationResult:
    """Result of scaffold validation."""
    
    is_valid: bool
    issues: List[ValidationIssue]
    warnings: List[ValidationIssue]
    errors: List[ValidationIssue]
    metadata: Dict[str, Any]

class ScaffoldValidator:
    """Validates scaffold data and generated UI components."""
    
    def __init__(self):
        self.required_fields = ["name", "type"]
        self.valid_types = ["container", "button", "input", "card", "form", "layout"]
        self.valid_frameworks = ["react", "vue", "html", "angular", "svelte"]
    
    def validate_scaffold(self, scaffold_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate scaffold data.
        
        Args:
            scaffold_data: Scaffold data to validate
            
        Returns:
            ValidationResult with validation issues
        """
        issues = []
        
        # Check required fields
        for field in self.required_fields:
            if field not in scaffold_data:
                issues.append(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    message=f"Missing required field: {field}",
                    field=field,
                    suggestion=f"Add '{field}' field to scaffold data"
                ))
        
        # Check component type
        if "type" in scaffold_data:
            component_type = scaffold_data["type"]
            if component_type not in self.valid_types:
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    message=f"Unknown component type: {component_type}",
                    field="type",
                    suggestion=f"Use one of: {', '.join(self.valid_types)}"
                ))
        
        # Check framework
        if "framework" in scaffold_data:
            framework = scaffold_data["framework"]
            if framework not in self.valid_frameworks:
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    message=f"Unknown framework: {framework}",
                    field="framework",
                    suggestion=f"Use one of: {', '.join(self.valid_frameworks)}"
                ))
        
        # Check props structure
        if "props" in scaffold_data:
            props = scaffold_data["props"]
            if not isinstance(props, dict):
                issues.append(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    message="Props must be a dictionary",
                    field="props",
                    suggestion="Convert props to a dictionary format"
                ))
        
        # Check children structure
        if "children" in scaffold_data:
            children = scaffold_data["children"]
            if not isinstance(children, (list, str)):
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    message="Children should be a list or string",
                    field="children",
                    suggestion="Convert children to list or string format"
                ))
        
        # Check styles structure
        if "styles" in scaffold_data:
            styles = scaffold_data["styles"]
            if not isinstance(styles, dict):
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    message="Styles must be a dictionary",
                    field="styles",
                    suggestion="Convert styles to a dictionary format"
                ))
        
        # Separate issues by level
        warnings = [issue for issue in issues if issue.level in [ValidationLevel.WARNING, ValidationLevel.INFO]]
        errors = [issue for issue in issues if issue.level in [ValidationLevel.ERROR, ValidationLevel.CRITICAL]]
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            warnings=warnings,
            errors=errors,
            metadata={
                "total_issues": len(issues),
                "warning_count": len(warnings),
                "error_count": len(errors)
            }
        )
    
    def validate_generated_code(self, code: str, framework: str) -> ValidationResult:
        """
        Validate generated UI code.
        
        Args:
            code: Generated code to validate
            framework: Framework the code was generated for
            
        Returns:
            ValidationResult with validation issues
        """
        issues = []
        
        if not code or not code.strip():
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                message="Generated code is empty",
                suggestion="Check scaffold data and generation process"
            ))
        
        # Framework-specific validation
        if framework == "react":
            issues.extend(self._validate_react_code(code))
        elif framework == "vue":
            issues.extend(self._validate_vue_code(code))
        elif framework == "html":
            issues.extend(self._validate_html_code(code))
        
        # Separate issues by level
        warnings = [issue for issue in issues if issue.level in [ValidationLevel.WARNING, ValidationLevel.INFO]]
        errors = [issue for issue in issues if issue.level in [ValidationLevel.ERROR, ValidationLevel.CRITICAL]]
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            warnings=warnings,
            errors=errors,
            metadata={
                "framework": framework,
                "code_length": len(code),
                "total_issues": len(issues)
            }
        )
    
    def _validate_react_code(self, code: str) -> List[ValidationIssue]:
        """Validate React/JSX code."""
        issues = []
        
        # Check for basic React structure
        if "import React" not in code and "from 'react'" not in code:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                message="Missing React import",
                suggestion="Add 'import React from \"react\";' at the top"
            ))
        
        # Check for component export
        if "export default" not in code and "export function" not in code:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                message="Missing component export",
                suggestion="Add 'export default' or 'export function'"
            ))
        
        # Check for JSX return
        if "return (" not in code:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                message="Missing JSX return statement",
                suggestion="Add 'return (...)' with JSX content"
            ))
        
        return issues
    
    def _validate_vue_code(self, code: str) -> List[ValidationIssue]:
        """Validate Vue.js code."""
        issues = []
        
        # Check for Vue template
        if "<template>" not in code:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                message="Missing Vue template",
                suggestion="Add '<template>' section"
            ))
        
        # Check for script setup
        if "<script setup" not in code:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                message="Missing script setup",
                suggestion="Add '<script setup lang=\"ts\">' section"
            ))
        
        return issues
    
    def _validate_html_code(self, code: str) -> List[ValidationIssue]:
        """Validate HTML code."""
        issues = []
        
        # Check for DOCTYPE
        if "<!DOCTYPE html>" not in code:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                message="Missing DOCTYPE declaration",
                suggestion="Add '<!DOCTYPE html>' at the beginning"
            ))
        
        # Check for html tag
        if "<html" not in code:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                message="Missing HTML tag",
                suggestion="Add '<html>' element"
            ))
        
        return issues
    
    def validate_accessibility(self, scaffold_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate accessibility features in scaffold data.
        
        Args:
            scaffold_data: Scaffold data to validate for accessibility
            
        Returns:
            ValidationResult with accessibility issues
        """
        issues = []
        
        # Check for accessibility attributes
        if "props" in scaffold_data:
            props = scaffold_data["props"]
            
            # Check for aria-label
            if "aria-label" not in props and "aria-labelledby" not in props:
                if scaffold_data.get("type") in ["button", "input", "img"]:
                    issues.append(ValidationIssue(
                        level=ValidationLevel.WARNING,
                        message="Missing accessibility label",
                        field="props",
                        suggestion="Add 'aria-label' or 'aria-labelledby' prop"
                    ))
            
            # Check for role
            if "role" not in props:
                if scaffold_data.get("type") in ["button", "link", "navigation"]:
                    issues.append(ValidationIssue(
                        level=ValidationLevel.INFO,
                        message="Consider adding explicit role",
                        field="props",
                        suggestion="Add 'role' prop for better accessibility"
                    ))
        
        # Separate issues by level
        warnings = [issue for issue in issues if issue.level in [ValidationLevel.WARNING, ValidationLevel.INFO]]
        errors = [issue for issue in issues if issue.level in [ValidationLevel.ERROR, ValidationLevel.CRITICAL]]
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            warnings=warnings,
            errors=errors,
            metadata={
                "accessibility_score": max(0, 100 - len(issues) * 10),
                "total_issues": len(issues)
            }
        ) 