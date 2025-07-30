"""
UI Scaffold Manager for Perspective D<cide>.

Provides high-level management of UI scaffold operations.
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class UIScaffoldManager:
    """High-level manager for UI scaffold operations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the UI scaffold manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.template_manager = None
        self.validator = None
        self.renderer = None
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize scaffold components."""
        try:
            from .templates import ScaffoldTemplateManager
            self.template_manager = ScaffoldTemplateManager()
        except ImportError as e:
            logger.warning(f"Template manager not available: {e}")
        
        try:
            from .validators import ScaffoldValidator
            self.validator = ScaffoldValidator()
        except ImportError as e:
            logger.warning(f"Validator not available: {e}")
        
        try:
            from .renderers import ScaffoldRenderer, RendererConfig
            renderer_config = RendererConfig(
                framework=self.config.get("framework", "react"),
                language=self.config.get("language", "tsx"),
                styling=self.config.get("styling", "tailwind"),
                components_library=self.config.get("components_library", "shadcn")
            )
            self.renderer = ScaffoldRenderer(renderer_config)
        except ImportError as e:
            logger.warning(f"Renderer not available: {e}")
    
    def create_scaffold(self, scaffold_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a complete scaffold with validation and rendering.
        
        Args:
            scaffold_data: Raw scaffold data
            
        Returns:
            Complete scaffold result with code and metadata
        """
        result = {
            "success": False,
            "scaffold_data": scaffold_data,
            "validation": None,
            "code": None,
            "errors": [],
            "warnings": []
        }
        
        # Validate scaffold data
        if self.validator:
            validation_result = self.validator.validate_scaffold(scaffold_data)
            result["validation"] = validation_result
            
            if not validation_result.is_valid:
                result["errors"].extend([issue.message for issue in validation_result.errors])
                return result
            
            result["warnings"].extend([issue.message for issue in validation_result.warnings])
        
        # Generate code
        if self.renderer:
            try:
                code_result = self.renderer.render(scaffold_data)
                result["code"] = code_result
                result["success"] = True
            except Exception as e:
                result["errors"].append(f"Code generation failed: {e}")
        else:
            result["errors"].append("Renderer not available")
        
        return result
    
    def create_from_template(self, template_name: str, customizations: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a scaffold from a template.
        
        Args:
            template_name: Name of the template to use
            customizations: Customizations to apply to the template
            
        Returns:
            Scaffold result
        """
        if not self.template_manager:
            return {
                "success": False,
                "errors": ["Template manager not available"]
            }
        
        template = self.template_manager.get_template(template_name)
        if not template:
            return {
                "success": False,
                "errors": [f"Template not found: {template_name}"]
            }
        
        # Apply customizations
        scaffold_data = template.copy()
        if customizations:
            scaffold_data.update(customizations)
        
        return self.create_scaffold(scaffold_data)
    
    def create_from_face(self, face: str, framework: str = "react") -> Dict[str, Any]:
        """
        Create a scaffold from an ASCII face.
        
        Args:
            face: ASCII face string
            framework: Target framework
            
        Returns:
            Scaffold result
        """
        try:
            from .face_parser import parse_face_to_scaffold
            scaffold_data = parse_face_to_scaffold(face, framework)
            return self.create_scaffold(scaffold_data)
        except ImportError:
            return {
                "success": False,
                "errors": ["Face parser not available"]
            }
        except Exception as e:
            return {
                "success": False,
                "errors": [f"Face parsing failed: {e}"]
            }
    
    def batch_create(self, scaffolds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create multiple scaffolds in batch.
        
        Args:
            scaffolds: List of scaffold data
            
        Returns:
            List of scaffold results
        """
        results = []
        
        for i, scaffold_data in enumerate(scaffolds):
            logger.info(f"Processing scaffold {i + 1}/{len(scaffolds)}")
            result = self.create_scaffold(scaffold_data)
            results.append(result)
        
        return results
    
    def export_scaffold(self, scaffold_data: Dict[str, Any], output_path: str) -> bool:
        """
        Export a scaffold to files.
        
        Args:
            scaffold_data: Scaffold data
            output_path: Output directory path
            
        Returns:
            True if export was successful
        """
        try:
            result = self.create_scaffold(scaffold_data)
            if not result["success"]:
                logger.error(f"Scaffold creation failed: {result['errors']}")
                return False
            
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Export scaffold data
            scaffold_file = output_dir / "scaffold.json"
            with open(scaffold_file, 'w', encoding='utf-8') as f:
                json.dump(scaffold_data, f, indent=2, ensure_ascii=False)
            
            # Export generated code
            if result["code"]:
                code_file = output_dir / f"{scaffold_data.get('name', 'component')}.{self.config.get('language', 'tsx')}"
                with open(code_file, 'w', encoding='utf-8') as f:
                    f.write(result["code"]["component"])
                
                # Export styles if available
                if result["code"]["styles"]:
                    styles_file = output_dir / f"{scaffold_data.get('name', 'component')}.css"
                    with open(styles_file, 'w', encoding='utf-8') as f:
                        f.write(result["code"]["styles"])
                
                # Export types if available
                if result["code"]["types"]:
                    types_file = output_dir / f"{scaffold_data.get('name', 'component')}.types.ts"
                    with open(types_file, 'w', encoding='utf-8') as f:
                        f.write(result["code"]["types"])
            
            logger.info(f"Exported scaffold to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
    
    def list_templates(self) -> List[str]:
        """
        List available templates.
        
        Returns:
            List of template names
        """
        if self.template_manager:
            return self.template_manager.list_templates()
        return []
    
    def search_templates(self, query: str) -> List[str]:
        """
        Search templates.
        
        Args:
            query: Search query
            
        Returns:
            List of matching template names
        """
        if self.template_manager:
            return self.template_manager.search_templates(query)
        return []
    
    def get_template_info(self, template_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a template.
        
        Args:
            template_name: Template name
            
        Returns:
            Template information or None
        """
        if self.template_manager:
            return self.template_manager.get_template(template_name)
        return None
    
    def validate_scaffold(self, scaffold_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Validate scaffold data.
        
        Args:
            scaffold_data: Scaffold data to validate
            
        Returns:
            Validation result or None
        """
        if not self.validator:
            return None
        
        validation_result = self.validator.validate_scaffold(scaffold_data)
        
        return {
            "is_valid": validation_result.is_valid,
            "issues": [
                {
                    "level": issue.level.value,
                    "message": issue.message,
                    "field": issue.field,
                    "suggestion": issue.suggestion
                }
                for issue in validation_result.issues
            ],
            "warnings": len(validation_result.warnings),
            "errors": len(validation_result.errors),
            "metadata": validation_result.metadata
        }
    
    def get_supported_frameworks(self) -> List[str]:
        """
        Get list of supported frameworks.
        
        Returns:
            List of supported framework names
        """
        return ["react", "vue", "html", "angular", "svelte"]
    
    def get_framework_info(self, framework: str) -> Dict[str, Any]:
        """
        Get information about a framework.
        
        Args:
            framework: Framework name
            
        Returns:
            Framework information
        """
        framework_info = {
            "react": {
                "name": "React",
                "language": "tsx",
                "description": "React with TypeScript and JSX",
                "features": ["Components", "Hooks", "TypeScript", "JSX"]
            },
            "vue": {
                "name": "Vue.js",
                "language": "vue",
                "description": "Vue.js with Composition API",
                "features": ["Components", "Composition API", "TypeScript", "SFC"]
            },
            "html": {
                "name": "HTML",
                "language": "html",
                "description": "Plain HTML with CSS",
                "features": ["Semantic HTML", "CSS", "Accessibility"]
            },
            "angular": {
                "name": "Angular",
                "language": "ts",
                "description": "Angular with TypeScript",
                "features": ["Components", "Services", "TypeScript", "RxJS"]
            },
            "svelte": {
                "name": "Svelte",
                "language": "svelte",
                "description": "Svelte with TypeScript",
                "features": ["Components", "Reactivity", "TypeScript", "Compiled"]
            }
        }
        
        return framework_info.get(framework, {
            "name": framework,
            "language": "unknown",
            "description": "Unknown framework",
            "features": []
        })
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the scaffold manager.
        
        Returns:
            Statistics dictionary
        """
        stats = {
            "templates_available": 0,
            "frameworks_supported": len(self.get_supported_frameworks()),
            "components_created": 0,
            "validation_enabled": self.validator is not None,
            "rendering_enabled": self.renderer is not None
        }
        
        if self.template_manager:
            stats["templates_available"] = len(self.template_manager.list_templates())
        
        return stats 