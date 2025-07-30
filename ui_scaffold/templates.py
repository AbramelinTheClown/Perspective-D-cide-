"""
UI Scaffold Templates for Perspective D<cide>.

Provides template management for UI scaffold generation.
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class ScaffoldTemplateManager:
    """Manages UI scaffold templates."""
    
    def __init__(self, templates_path: Optional[str] = None):
        """
        Initialize template manager.
        
        Args:
            templates_path: Path to templates directory
        """
        self.templates_path = templates_path
        self.templates: Dict[str, Dict[str, Any]] = {}
        self._load_default_templates()
    
    def _load_default_templates(self) -> None:
        """Load default templates."""
        self.templates = {
            "basic_form": {
                "name": "BasicForm",
                "type": "form",
                "framework": "react",
                "props": {
                    "onSubmit": "function",
                    "className": "string"
                },
                "children": [
                    {
                        "type": "input",
                        "props": {
                            "name": "email",
                            "type": "email",
                            "placeholder": "Enter email"
                        }
                    },
                    {
                        "type": "input",
                        "props": {
                            "name": "password",
                            "type": "password",
                            "placeholder": "Enter password"
                        }
                    },
                    {
                        "type": "button",
                        "props": {
                            "type": "submit",
                            "children": "Submit"
                        }
                    }
                ],
                "styles": {
                    "form": {
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "1rem",
                        "maxWidth": "400px"
                    }
                }
            },
            "card_layout": {
                "name": "CardLayout",
                "type": "layout",
                "framework": "react",
                "props": {
                    "title": "string",
                    "description": "string",
                    "className": "string"
                },
                "children": [
                    {
                        "type": "card",
                        "props": {
                            "className": "p-6"
                        },
                        "children": [
                            {
                                "type": "h2",
                                "props": {
                                    "className": "text-2xl font-bold mb-2"
                                }
                            },
                            {
                                "type": "p",
                                "props": {
                                    "className": "text-gray-600"
                                }
                            }
                        ]
                    }
                ]
            },
            "navigation_bar": {
                "name": "NavigationBar",
                "type": "navigation",
                "framework": "react",
                "props": {
                    "items": "array",
                    "className": "string"
                },
                "children": [
                    {
                        "type": "nav",
                        "props": {
                            "className": "flex items-center space-x-4"
                        }
                    }
                ],
                "styles": {
                    "nav": {
                        "backgroundColor": "#f8f9fa",
                        "padding": "1rem",
                        "borderBottom": "1px solid #e9ecef"
                    }
                }
            },
            "data_table": {
                "name": "DataTable",
                "type": "table",
                "framework": "react",
                "props": {
                    "data": "array",
                    "columns": "array",
                    "className": "string"
                },
                "children": [
                    {
                        "type": "table",
                        "props": {
                            "className": "min-w-full divide-y divide-gray-200"
                        },
                        "children": [
                            {
                                "type": "thead",
                                "children": [
                                    {
                                        "type": "tr",
                                        "children": []
                                    }
                                ]
                            },
                            {
                                "type": "tbody",
                                "children": []
                            }
                        ]
                    }
                ]
            },
            "modal_dialog": {
                "name": "ModalDialog",
                "type": "modal",
                "framework": "react",
                "props": {
                    "isOpen": "boolean",
                    "onClose": "function",
                    "title": "string",
                    "className": "string"
                },
                "children": [
                    {
                        "type": "div",
                        "props": {
                            "className": "fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center"
                        },
                        "children": [
                            {
                                "type": "div",
                                "props": {
                                    "className": "bg-white rounded-lg p-6 max-w-md w-full"
                                },
                                "children": [
                                    {
                                        "type": "h2",
                                        "props": {
                                            "className": "text-xl font-bold mb-4"
                                        }
                                    },
                                    {
                                        "type": "button",
                                        "props": {
                                            "onClick": "onClose",
                                            "className": "absolute top-4 right-4"
                                        },
                                        "children": "×"
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        }
        
        logger.info(f"Loaded {len(self.templates)} default templates")
    
    def get_template(self, template_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a template by name.
        
        Args:
            template_name: Name of the template
            
        Returns:
            Template data or None if not found
        """
        return self.templates.get(template_name)
    
    def list_templates(self) -> List[str]:
        """
        List all available template names.
        
        Returns:
            List of template names
        """
        return list(self.templates.keys())
    
    def add_template(self, name: str, template_data: Dict[str, Any]) -> None:
        """
        Add a new template.
        
        Args:
            name: Template name
            template_data: Template data
        """
        self.templates[name] = template_data
        logger.info(f"Added template: {name}")
    
    def remove_template(self, name: str) -> bool:
        """
        Remove a template.
        
        Args:
            name: Template name
            
        Returns:
            True if template was removed, False if not found
        """
        if name in self.templates:
            del self.templates[name]
            logger.info(f"Removed template: {name}")
            return True
        return False
    
    def update_template(self, name: str, template_data: Dict[str, Any]) -> bool:
        """
        Update an existing template.
        
        Args:
            name: Template name
            template_data: New template data
            
        Returns:
            True if template was updated, False if not found
        """
        if name in self.templates:
            self.templates[name] = template_data
            logger.info(f"Updated template: {name}")
            return True
        return False
    
    def search_templates(self, query: str) -> List[str]:
        """
        Search templates by name or description.
        
        Args:
            query: Search query
            
        Returns:
            List of matching template names
        """
        query_lower = query.lower()
        matches = []
        
        for name, template in self.templates.items():
            if query_lower in name.lower():
                matches.append(name)
                continue
            
            # Search in template description or type
            template_type = template.get("type", "")
            if query_lower in template_type.lower():
                matches.append(name)
                continue
            
            # Search in component name
            component_name = template.get("name", "")
            if query_lower in component_name.lower():
                matches.append(name)
                continue
        
        return matches
    
    def get_templates_by_type(self, component_type: str) -> List[str]:
        """
        Get templates by component type.
        
        Args:
            component_type: Type of component (form, layout, etc.)
            
        Returns:
            List of template names for the given type
        """
        matches = []
        
        for name, template in self.templates.items():
            if template.get("type") == component_type:
                matches.append(name)
        
        return matches
    
    def get_templates_by_framework(self, framework: str) -> List[str]:
        """
        Get templates by framework.
        
        Args:
            framework: Framework name (react, vue, etc.)
            
        Returns:
            List of template names for the given framework
        """
        matches = []
        
        for name, template in self.templates.items():
            if template.get("framework") == framework:
                matches.append(name)
        
        return matches
    
    def export_templates(self, file_path: str) -> None:
        """
        Export templates to a JSON file.
        
        Args:
            file_path: Path to export file
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.templates, f, indent=2, ensure_ascii=False)
            logger.info(f"Exported templates to: {file_path}")
        except Exception as e:
            logger.error(f"Failed to export templates: {e}")
    
    def import_templates(self, file_path: str) -> None:
        """
        Import templates from a JSON file.
        
        Args:
            file_path: Path to import file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                imported_templates = json.load(f)
            
            # Merge with existing templates
            for name, template in imported_templates.items():
                self.templates[name] = template
            
            logger.info(f"Imported {len(imported_templates)} templates from: {file_path}")
        except Exception as e:
            logger.error(f"Failed to import templates: {e}")
    
    def validate_template(self, template_data: Dict[str, Any]) -> List[str]:
        """
        Validate template data.
        
        Args:
            template_data: Template data to validate
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Check required fields
        required_fields = ["name", "type"]
        for field in required_fields:
            if field not in template_data:
                errors.append(f"Missing required field: {field}")
        
        # Check component type
        if "type" in template_data:
            valid_types = ["form", "layout", "navigation", "table", "modal", "card", "button", "input"]
            if template_data["type"] not in valid_types:
                errors.append(f"Invalid component type: {template_data['type']}")
        
        # Check framework
        if "framework" in template_data:
            valid_frameworks = ["react", "vue", "html", "angular", "svelte"]
            if template_data["framework"] not in valid_frameworks:
                errors.append(f"Invalid framework: {template_data['framework']}")
        
        return errors 