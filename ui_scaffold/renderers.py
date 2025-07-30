"""
UI Scaffold Renderers for Perspective D<cide>.

Provides renderers for different UI frameworks (React, Vue, etc.) to convert
scaffold data into actual UI code.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class RendererConfig:
    """Configuration for UI renderers."""
    
    framework: str = "react"
    language: str = "tsx"
    styling: str = "tailwind"
    components_library: str = "shadcn"
    export_type: str = "default"
    include_types: bool = True
    include_styles: bool = True

class BaseRenderer(ABC):
    """Base class for UI renderers."""
    
    def __init__(self, config: RendererConfig):
        self.config = config
    
    @abstractmethod
    def render_component(self, scaffold_data: Dict[str, Any]) -> str:
        """Render a component from scaffold data."""
        pass
    
    @abstractmethod
    def render_styles(self, scaffold_data: Dict[str, Any]) -> str:
        """Render styles for the component."""
        pass
    
    @abstractmethod
    def render_types(self, scaffold_data: Dict[str, Any]) -> str:
        """Render TypeScript types for the component."""
        pass

class ReactRenderer(BaseRenderer):
    """React/TSX renderer for UI scaffolds."""
    
    def __init__(self, config: RendererConfig):
        super().__init__(config)
        self.component_templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load React component templates."""
        return {
            "container": """
import React from 'react';
import { cn } from '@/lib/utils';

interface {component_name}Props {{
  className?: string;
  children?: React.ReactNode;
  {props}
}}

export {export_type} function {component_name}({{ className, children, {props} }}: {component_name}Props) {{
  return (
    <div className={{cn("{container_classes}", className)}}>
      {children}
    </div>
  );
}}
""",
            "button": """
import React from 'react';
import { cn } from '@/lib/utils';

interface {component_name}Props {{
  className?: string;
  children?: React.ReactNode;
  onClick?: () => void;
  variant?: 'default' | 'destructive' | 'outline' | 'secondary' | 'ghost' | 'link';
  size?: 'default' | 'sm' | 'lg' | 'icon';
  disabled?: boolean;
  {props}
}}

export {export_type} function {component_name}({{ 
  className, 
  children, 
  onClick, 
  variant = 'default', 
  size = 'default',
  disabled = false,
  {props} 
}}: {component_name}Props) {{
  return (
    <button
      className={{cn(
        "inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none ring-offset-background",
        {{
          "bg-primary text-primary-foreground hover:bg-primary/90": variant === "default",
          "bg-destructive text-destructive-foreground hover:bg-destructive/90": variant === "destructive",
          "border border-input hover:bg-accent hover:text-accent-foreground": variant === "outline",
          "bg-secondary text-secondary-foreground hover:bg-secondary/80": variant === "secondary",
          "hover:bg-accent hover:text-accent-foreground": variant === "ghost",
          "underline-offset-4 hover:underline text-primary": variant === "link",
        }},
        {{
          "h-10 py-2 px-4": size === "default",
          "h-9 px-3 rounded-md": size === "sm",
          "h-11 px-8 rounded-md": size === "lg",
          "h-10 w-10": size === "icon",
        }},
        className
      )}}
      onClick={{onClick}}
      disabled={{disabled}}
    >
      {{children}}
    </button>
  );
}}
""",
            "input": """
import React from 'react';
import { cn } from '@/lib/utils';

interface {component_name}Props {{
  className?: string;
  placeholder?: string;
  value?: string;
  onChange?: (value: string) => void;
  type?: string;
  disabled?: boolean;
  {props}
}}

export {export_type} function {component_name}({{ 
  className, 
  placeholder, 
  value, 
  onChange, 
  type = "text",
  disabled = false,
  {props} 
}}: {component_name}Props) {{
  return (
    <input
      type={{type}}
      className={{cn(
        "flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",
        className
      )}}
      placeholder={{placeholder}}
      value={{value}}
      onChange={{(e) => onChange?.(e.target.value)}}
      disabled={{disabled}}
    />
  );
}}
""",
            "card": """
import React from 'react';
import { cn } from '@/lib/utils';

interface {component_name}Props {{
  className?: string;
  children?: React.ReactNode;
  {props}
}}

export {export_type} function {component_name}({{ className, children, {props} }}: {component_name}Props) {{
  return (
    <div className={{cn("rounded-lg border bg-card text-card-foreground shadow-sm", className)}}>
      {{children}}
    </div>
  );
}}

export function {component_name}Header({{ className, children, ...props }}: React.HTMLAttributes<HTMLDivElement>) {{
  return (
    <div className={{cn("flex flex-col space-y-1.5 p-6", className)}} {{...props}}>
      {{children}}
    </div>
  );
}}

export function {component_name}Title({{ className, children, ...props }}: React.HTMLAttributes<HTMLHeadingElement>) {{
  return (
    <h3 className={{cn("text-2xl font-semibold leading-none tracking-tight", className)}} {{...props}}>
      {{children}}
    </h3>
  );
}}

export function {component_name}Description({{ className, children, ...props }}: React.HTMLAttributes<HTMLParagraphElement>) {{
  return (
    <p className={{cn("text-sm text-muted-foreground", className)}} {{...props}}>
      {{children}}
    </p>
  );
}}

export function {component_name}Content({{ className, children, ...props }}: React.HTMLAttributes<HTMLDivElement>) {{
  return (
    <div className={{cn("p-6 pt-0", className)}} {{...props}}>
      {{children}}
    </div>
  );
}}

export function {component_name}Footer({{ className, children, ...props }}: React.HTMLAttributes<HTMLDivElement>) {{
  return (
    <div className={{cn("flex items-center p-6 pt-0", className)}} {{...props}}>
      {{children}}
    </div>
  );
}}
"""
        }
    
    def render_component(self, scaffold_data: Dict[str, Any]) -> str:
        """Render a React component from scaffold data."""
        component_type = scaffold_data.get("type", "container")
        component_name = scaffold_data.get("name", "Component")
        props = scaffold_data.get("props", {})
        
        # Convert props to TypeScript interface
        props_str = ""
        if props:
            props_list = []
            for prop_name, prop_type in props.items():
                if prop_type == "string":
                    props_list.append(f"{prop_name}?: string;")
                elif prop_type == "number":
                    props_list.append(f"{prop_name}?: number;")
                elif prop_type == "boolean":
                    props_list.append(f"{prop_name}?: boolean;")
                else:
                    props_list.append(f"{prop_name}?: any;")
            props_str = "\n  " + "\n  ".join(props_list)
        
        # Get template
        template = self.component_templates.get(component_type, self.component_templates["container"])
        
        # Format template
        export_type = "default" if self.config.export_type == "default" else ""
        container_classes = scaffold_data.get("classes", "p-4")
        
        return template.format(
            component_name=component_name,
            props=props_str,
            export_type=export_type,
            container_classes=container_classes
        )
    
    def render_styles(self, scaffold_data: Dict[str, Any]) -> str:
        """Render CSS styles for the component."""
        if not self.config.include_styles:
            return ""
        
        styles = scaffold_data.get("styles", {})
        if not styles:
            return ""
        
        css = f"/* Styles for {scaffold_data.get('name', 'Component')} */\n"
        for selector, properties in styles.items():
            css += f"\n.{selector} {{\n"
            for prop, value in properties.items():
                css += f"  {prop}: {value};\n"
            css += "}\n"
        
        return css
    
    def render_types(self, scaffold_data: Dict[str, Any]) -> str:
        """Render TypeScript types for the component."""
        if not self.config.include_types:
            return ""
        
        component_name = scaffold_data.get("name", "Component")
        props = scaffold_data.get("props", {})
        
        if not props:
            return ""
        
        types = f"// Types for {component_name}\n"
        types += f"export interface {component_name}Props {{\n"
        
        for prop_name, prop_type in props.items():
            ts_type = {
                "string": "string",
                "number": "number", 
                "boolean": "boolean",
                "array": "any[]",
                "object": "Record<string, any>"
            }.get(prop_type, "any")
            
            types += f"  {prop_name}?: {ts_type};\n"
        
        types += "}\n"
        return types

class VueRenderer(BaseRenderer):
    """Vue.js renderer for UI scaffolds."""
    
    def __init__(self, config: RendererConfig):
        super().__init__(config)
        self.component_templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load Vue component templates."""
        return {
            "container": """
<template>
  <div :class="['{container_classes}', className]">
    <slot />
  </div>
</template>

<script setup lang="ts">
interface Props {{
  className?: string;
  {props}
}}

const props = withDefaults(defineProps<Props>(), {{
  className: '',
  {defaults}
}});
</script>

<style scoped>
{styles}
</style>
""",
            "button": """
<template>
  <button
    :class="[
      'inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none ring-offset-background',
      variantClasses[variant],
      sizeClasses[size],
      className
    ]"
    :disabled="disabled"
    @click="$emit('click')"
  >
    <slot />
  </button>
</template>

<script setup lang="ts">
interface Props {{
  className?: string;
  variant?: 'default' | 'destructive' | 'outline' | 'secondary' | 'ghost' | 'link';
  size?: 'default' | 'sm' | 'lg' | 'icon';
  disabled?: boolean;
  {props}
}}

const props = withDefaults(defineProps<Props>(), {{
  className: '',
  variant: 'default',
  size: 'default',
  disabled: false,
  {defaults}
}});

const variantClasses = {{
  default: 'bg-primary text-primary-foreground hover:bg-primary/90',
  destructive: 'bg-destructive text-destructive-foreground hover:bg-destructive/90',
  outline: 'border border-input hover:bg-accent hover:text-accent-foreground',
  secondary: 'bg-secondary text-secondary-foreground hover:bg-secondary/80',
  ghost: 'hover:bg-accent hover:text-accent-foreground',
  link: 'underline-offset-4 hover:underline text-primary'
}};

const sizeClasses = {{
  default: 'h-10 py-2 px-4',
  sm: 'h-9 px-3 rounded-md',
  lg: 'h-11 px-8 rounded-md',
  icon: 'h-10 w-10'
}};

defineEmits<{{
  click: []
}}>();
</script>

<style scoped>
{styles}
</style>
"""
        }
    
    def render_component(self, scaffold_data: Dict[str, Any]) -> str:
        """Render a Vue component from scaffold data."""
        component_type = scaffold_data.get("type", "container")
        component_name = scaffold_data.get("name", "Component")
        props = scaffold_data.get("props", {})
        
        # Convert props to Vue props
        props_str = ""
        defaults_str = ""
        if props:
            props_list = []
            defaults_list = []
            for prop_name, prop_type in props.items():
                if prop_type == "string":
                    props_list.append(f"{prop_name}?: string;")
                    defaults_list.append(f"{prop_name}: ''")
                elif prop_type == "number":
                    props_list.append(f"{prop_name}?: number;")
                    defaults_list.append(f"{prop_name}: 0")
                elif prop_type == "boolean":
                    props_list.append(f"{prop_name}?: boolean;")
                    defaults_list.append(f"{prop_name}: false")
                else:
                    props_list.append(f"{prop_name}?: any;")
                    defaults_list.append(f"{prop_name}: undefined")
            props_str = "\n  " + "\n  ".join(props_list)
            defaults_str = ",\n  ".join(defaults_list)
        
        # Get template
        template = self.component_templates.get(component_type, self.component_templates["container"])
        
        # Format template
        container_classes = scaffold_data.get("classes", "p-4")
        styles = self.render_styles(scaffold_data)
        
        return template.format(
            component_name=component_name,
            props=props_str,
            defaults=defaults_str,
            container_classes=container_classes,
            styles=styles
        )
    
    def render_styles(self, scaffold_data: Dict[str, Any]) -> str:
        """Render CSS styles for the component."""
        if not self.config.include_styles:
            return ""
        
        styles = scaffold_data.get("styles", {})
        if not styles:
            return ""
        
        css = ""
        for selector, properties in styles.items():
            css += f".{selector} {{\n"
            for prop, value in properties.items():
                css += f"  {prop}: {value};\n"
            css += "}\n"
        
        return css
    
    def render_types(self, scaffold_data: Dict[str, Any]) -> str:
        """Render TypeScript types for the component."""
        if not self.config.include_types:
            return ""
        
        component_name = scaffold_data.get("name", "Component")
        props = scaffold_data.get("props", {})
        
        if not props:
            return ""
        
        types = f"// Types for {component_name}\n"
        types += f"export interface {component_name}Props {{\n"
        
        for prop_name, prop_type in props.items():
            ts_type = {
                "string": "string",
                "number": "number", 
                "boolean": "boolean",
                "array": "any[]",
                "object": "Record<string, any>"
            }.get(prop_type, "any")
            
            types += f"  {prop_name}?: {ts_type};\n"
        
        types += "}\n"
        return types

class ScaffoldRenderer:
    """Main scaffold renderer that delegates to framework-specific renderers."""
    
    def __init__(self, config: RendererConfig):
        self.config = config
        self.renderers = {
            "react": ReactRenderer(config),
            "vue": VueRenderer(config)
        }
    
    def render(self, scaffold_data: Dict[str, Any]) -> Dict[str, str]:
        """Render scaffold data to UI code."""
        framework = self.config.framework.lower()
        renderer = self.renderers.get(framework)
        
        if not renderer:
            raise ValueError(f"Unsupported framework: {framework}")
        
        return {
            "component": renderer.render_component(scaffold_data),
            "styles": renderer.render_styles(scaffold_data),
            "types": renderer.render_types(scaffold_data)
        }
    
    def render_multiple(self, scaffolds: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Render multiple scaffolds."""
        components = []
        styles = []
        types = []
        
        for scaffold in scaffolds:
            result = self.render(scaffold)
            components.append(result["component"])
            if result["styles"]:
                styles.append(result["styles"])
            if result["types"]:
                types.append(result["types"])
        
        return {
            "components": components,
            "styles": styles,
            "types": types
        } 