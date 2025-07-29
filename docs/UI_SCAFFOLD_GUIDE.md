# UI Scaffold System Documentation

## 🎭 **ASCII Face to UI Scaffold Transformation**

The Perspective D<cide> framework includes a revolutionary system that transforms ASCII face glyphs into complete, responsive web interface scaffolds. Each "face" becomes a structured layout definition that can be instantiated as a full GUI/Web interface.

### **Core Philosophy**

- **Symbolic Layout**: ASCII faces encode layout structure in their visual composition
- **Five-Layer Architecture**: Each face maps to a complete UI hierarchy
- **JSONL-Driven**: Scaffolds are defined as streaming JSONL data
- **Framework Agnostic**: Can be rendered in React, Vue, Three.js, or any framework
- **Zero-Rendering Cost**: Scaffolds are pure data until instantiated

## 🏗️ **Five-Layer Architecture**

### **1. Global Shell (Bone)**
**Purpose**: Outermost container—sets up viewport, theming, grid/flex context  
**Maps to**: Arms of the face (header + footer combined)

```html
<Shell id="{left_arm + right_arm}">
  <!-- Everything else lives inside -->
</Shell>
```

### **2. Primary Containers (Blob)**
**Purpose**: Major page areas—header, aside, main, footer  
**Maps to**:
- Left arm → `<header>`
- Left ear+cheek → `<aside class="left-widget">`
- Eyes+nose+eyes → `<main>`
- Right cheek+ear → `<aside class="right-widget">`
- Right arm → `<footer>`

```html
<Header id="{left_arm}">…</Header>
<Aside class="left">{ear_left}{cheek_left}</Aside>
<Main id="{eye+nose+eye}">…core content…</Main>
<Aside class="right">{cheek_right}{ear_right}</Aside>
<Footer id="{right_arm}">…</Footer>
```

### **3. Sub-Containers & Groups (Biz)**
**Purpose**: Group related components within each primary container  
**Maps to**: Cheek (for widget grouping) and section markers inside `<main>`

```html
<Header>
  <Nav id="{widget_left}">…links…</Nav>
</Header>
<Main>
  <Section id="{core_id}">…data grids…</Section>
</Main>
<Footer>
  <Nav id="{widget_right}">…links…</Nav>
</Footer>
```

### **4. Atomic Components (Leaf)**
**Purpose**: Actual UI building blocks—buttons, inputs, cards, lists  
**Maps to**: Eyes and nose as "focus" icons—places where attention/action happens

```html
<Section id="{core_id}">
  <Card id="{eye_left}">…</Card>
  <Card id="{nose}">…highlight…</Card>
  <Card id="{eye_right}">…</Card>
</Section>
```

### **5. Overlay & Utilities (Spirit)**
**Purpose**: Modals, tooltips, loading spinners—global utilities that float above  
**Maps to**: Invisible "spirit" layer around the face

```html
<Overlay type="modal">…</Overlay>
<Tooltip target="{eye_left}">…help…</Tooltip>
<Spinner id="loading" />
```

## 📋 **JSONL Schema Definition**

### **Shell Layer Schema**
```json
{
  "type": "Shell",
  "id": "乁﴾...﴿乁",
  "props": {
    "viewport": "responsive",
    "theme": "light|dark|auto",
    "layout": "grid|flex",
    "breakpoints": {
      "mobile": "320px",
      "tablet": "768px", 
      "desktop": "1024px"
    }
  },
  "metadata": {
    "glyph_source": "ʕᵒ☯ϖ☯ᵒʔ",
    "layer": "bone",
    "description": "Global container shell"
  }
}
```

### **Container Layer Schema**
```json
{
  "type": "Header|Main|Aside|Footer",
  "id": "乁﴾",
  "class": "left-widget|right-widget",
  "props": {
    "position": "fixed|sticky|relative",
    "background": "transparent|solid|gradient",
    "border": "none|thin|thick",
    "shadow": "none|small|large"
  },
  "metadata": {
    "glyph_source": "ear_left",
    "layer": "blob",
    "description": "Primary layout container"
  }
}
```

### **Component Layer Schema**
```json
{
  "type": "Nav|Section|Card|Button|Input",
  "id": "◉",
  "props": {
    "variant": "primary|secondary|highlight",
    "size": "small|medium|large",
    "state": "default|hover|active|disabled",
    "content": "text|icon|both"
  },
  "metadata": {
    "glyph_source": "eye_left",
    "layer": "leaf",
    "description": "Atomic UI component"
  }
}
```

### **Overlay Layer Schema**
```json
{
  "type": "Overlay|Tooltip|Modal|Spinner",
  "id": "modal-root",
  "target": "◉",
  "props": {
    "position": "top|bottom|left|right|center",
    "animation": "fade|slide|scale",
    "zIndex": 1000,
    "backdrop": true
  },
  "metadata": {
    "glyph_source": "spirit",
    "layer": "spirit",
    "description": "Floating utility layer"
  }
}
```

## 🎨 **Face Parsing & Mapping**

### **ASCII Face Structure**
```
ʕᵒ☯ϖ☯ᵒʔ
│ │ │ │ │
│ │ │ │ └── Right arm (footer)
│ │ │ └──── Right ear (aside-right)
│ │ └────── Right eye (component)
│ └──────── Nose (highlight component)
└────────── Left eye (component)
```

### **Parsing Algorithm**
```python
def parse_face_to_scaffold(glyph: str) -> List[Dict]:
    """Parse ASCII face into UI scaffold JSONL."""
    
    # Parse glyph structure
    parsed = parse_glyph(glyph)
    
    scaffold = []
    
    # 1. Shell layer
    shell_id = parsed["aside_left"] + parsed["aside_right"]
    scaffold.append({
        "type": "Shell",
        "id": shell_id,
        "props": {"viewport": "responsive", "theme": "auto"},
        "metadata": {"glyph_source": glyph, "layer": "bone"}
    })
    
    # 2. Container layers
    scaffold.extend([
        {"type": "Header", "id": parsed["aside_left"], "metadata": {"layer": "blob"}},
        {"type": "Aside", "class": "left", "id": parsed["widget_left"], "metadata": {"layer": "blob"}},
        {"type": "Main", "id": parsed["content_main"], "metadata": {"layer": "blob"}},
        {"type": "Aside", "class": "right", "id": parsed["widget_right"], "metadata": {"layer": "blob"}},
        {"type": "Footer", "id": parsed["aside_right"], "metadata": {"layer": "blob"}}
    ])
    
    # 3. Component layers (from eyes and nose)
    components = parse_components(parsed["content_main"])
    for comp in components:
        scaffold.append({
            "type": "Card",
            "id": comp["id"],
            "props": {"variant": comp["variant"]},
            "metadata": {"layer": "leaf"}
        })
    
    # 4. Overlay layer
    scaffold.append({
        "type": "Overlay",
        "id": "overlay-root",
        "props": {"zIndex": 1000},
        "metadata": {"layer": "spirit"}
    })
    
    return scaffold
```

## 🔧 **Implementation Examples**

### **Complete JSONL Scaffold**
```jsonl
{"type":"Shell","id":"乁﴾...﴿乁","props":{"viewport":"responsive","theme":"auto"},"metadata":{"glyph_source":"ʕᵒ☯ϖ☯ᵒʔ","layer":"bone"}}
{"type":"Header","id":"乁﴾","props":{"position":"sticky"},"metadata":{"layer":"blob"}}
{"type":"Nav","id":"widget_left","props":{"items":5},"metadata":{"layer":"biz"}}
{"type":"Aside","class":"left","id":"ღ","props":{"width":"250px"},"metadata":{"layer":"blob"}}
{"type":"Main","id":"◉V◉","props":{"flex":"1"},"metadata":{"layer":"blob"}}
{"type":"Section","id":"content-section","props":{"padding":"2rem"},"metadata":{"layer":"biz"}}
{"type":"Card","id":"◉","props":{"variant":"highlight","size":"large"},"metadata":{"layer":"leaf"}}
{"type":"Card","id":"V","props":{"variant":"primary","size":"medium"},"metadata":{"layer":"leaf"}}
{"type":"Card","id":"◉","props":{"variant":"default","size":"large"},"metadata":{"layer":"leaf"}}
{"type":"Aside","class":"right","id":"ღ","props":{"width":"250px"},"metadata":{"layer":"blob"}}
{"type":"Footer","id":"﴿乁","props":{"position":"sticky"},"metadata":{"layer":"blob"}}
{"type":"Overlay","id":"overlay-root","props":{"zIndex":1000},"metadata":{"layer":"spirit"}}
{"type":"Tooltip","target":"◉","props":{"text":"Information","position":"top"},"metadata":{"layer":"spirit"}}
```

### **React Renderer Example**
```typescript
interface ScaffoldRenderer {
  renderShell(shell: ShellSchema): JSX.Element;
  renderContainer(container: ContainerSchema): JSX.Element;
  renderComponent(component: ComponentSchema): JSX.Element;
  renderOverlay(overlay: OverlaySchema): JSX.Element;
}

class ReactScaffoldRenderer implements ScaffoldRenderer {
  renderShell(shell: ShellSchema): JSX.Element {
    return (
      <div 
        id={shell.id}
        className={`shell ${shell.props.theme}`}
        style={{
          display: shell.props.layout === 'grid' ? 'grid' : 'flex',
          minHeight: '100vh'
        }}
      >
        {this.renderChildren(shell.children)}
      </div>
    );
  }
  
  renderContainer(container: ContainerSchema): JSX.Element {
    const Component = container.type.toLowerCase();
    return (
      <Component
        id={container.id}
        className={container.class}
        style={{
          position: container.props.position,
          background: container.props.background
        }}
      >
        {this.renderChildren(container.children)}
      </Component>
    );
  }
  
  renderComponent(component: ComponentSchema): JSX.Element {
    return (
      <div
        id={component.id}
        className={`component ${component.props.variant}`}
        data-size={component.props.size}
      >
        {component.props.content}
      </div>
    );
  }
  
  renderOverlay(overlay: OverlaySchema): JSX.Element {
    return (
      <div
        id={overlay.id}
        className={`overlay ${overlay.props.animation}`}
        style={{ zIndex: overlay.props.zIndex }}
      >
        {overlay.props.backdrop && <div className="backdrop" />}
        {this.renderChildren(overlay.children)}
      </div>
    );
  }
}
```

### **Vue Renderer Example**
```vue
<template>
  <div 
    :id="shell.id"
    :class="['shell', shell.props.theme]"
    :style="shellStyles"
  >
    <component
      v-for="item in scaffold"
      :key="item.id"
      :is="getComponentType(item.type)"
      v-bind="getComponentProps(item)"
    />
  </div>
</template>

<script>
export default {
  name: 'ScaffoldRenderer',
  props: {
    scaffold: {
      type: Array,
      required: true
    }
  },
  computed: {
    shellStyles() {
      const shell = this.scaffold.find(s => s.type === 'Shell');
      return {
        display: shell.props.layout === 'grid' ? 'grid' : 'flex',
        minHeight: '100vh'
      };
    }
  },
  methods: {
    getComponentType(type) {
      const typeMap = {
        'Shell': 'div',
        'Header': 'header',
        'Main': 'main',
        'Aside': 'aside',
        'Footer': 'footer',
        'Nav': 'nav',
        'Section': 'section',
        'Card': 'div',
        'Overlay': 'div'
      };
      return typeMap[type] || 'div';
    },
    getComponentProps(item) {
      return {
        id: item.id,
        class: item.class,
        ...item.props
      };
    }
  }
};
</script>
```

## 🎯 **Usage Patterns**

### **1. Agent-Generated Scaffolds**
```python
from perspective_dcide.symbolic import SymbolicAnalyzer
from perspective_dcide.ui_scaffold import FaceToScaffold

# Analyze content to determine appropriate face
analyzer = SymbolicAnalyzer()
content_analysis = analyzer.analyze_content(content_item)

# Generate face based on content characteristics
face_generator = FaceToScaffold()
appropriate_face = face_generator.select_face_by_content(content_analysis)

# Convert face to scaffold
scaffold = face_generator.face_to_scaffold(appropriate_face)

# Export as JSONL
with open('ui_scaffold.jsonl', 'w') as f:
    for layer in scaffold:
        f.write(json.dumps(layer) + '\n')
```

### **2. Template-Based Scaffolds**
```python
# Pre-defined scaffold templates
templates = {
    "dashboard": "ʕᵒ☯ϖ☯ᵒʔ",
    "form": "乁﴾ovo乁﴿", 
    "gallery": "୧ˇ☯-☯ˇ୨",
    "chat": "ʢᵒᴗ.ᴗᵒʡ",
    "settings": "༼⌐■∇■¬༽"
}

# Use template
scaffold = face_generator.template_to_scaffold("dashboard")
```

### **3. Dynamic Scaffold Generation**
```python
# Generate scaffold based on content analysis
def generate_adaptive_scaffold(content_analysis):
    """Generate scaffold that adapts to content characteristics."""
    
    # Determine layout based on content type
    if content_analysis.archetypal_theme == "communication":
        base_face = "ʢᵒᴗ.ᴗᵒʡ"  # Chat-like layout
    elif content_analysis.archetypal_theme == "analysis":
        base_face = "ʕᵒ☯ϖ☯ᵒʔ"  # Dashboard layout
    else:
        base_face = "乁﴾ovo乁﴿"  # Default form layout
    
    # Customize based on content complexity
    if len(content_analysis.keywords) > 10:
        # Add more component layers
        scaffold = face_generator.enhance_scaffold(base_face, "complex")
    else:
        scaffold = face_generator.face_to_scaffold(base_face)
    
    return scaffold
```

## 🔄 **Integration with Framework**

### **Framework Integration**
```python
from perspective_dcide.core import Config, initialize_framework
from perspective_dcide.ui_scaffold import UIScaffoldManager

# Initialize framework with UI scaffold support
config = Config(
    framework_name="ui-scaffold-demo",
    enable_symbolic=True,
    enable_ui_scaffold=True,  # New flag
    ui_scaffold_config={
        "default_theme": "light",
        "responsive_breakpoints": {
            "mobile": "320px",
            "tablet": "768px",
            "desktop": "1024px"
        },
        "component_library": "material-ui"  # or "bootstrap", "tailwind"
    }
)

initialize_framework(config)

# Use UI scaffold manager
scaffold_manager = UIScaffoldManager()

# Generate scaffold from content
content_item = ContentItem(
    id="demo_content",
    content="This is a sample content for UI scaffold generation"
)

scaffold = scaffold_manager.generate_scaffold(content_item)
```

### **Component Registry Integration**
```python
from perspective_dcide.core.registry import ComponentRegistry

# Register UI scaffold components
registry = ComponentRegistry.get_instance()

registry.register_component("face_parser", FaceParser())
registry.register_component("scaffold_renderer", ReactScaffoldRenderer())
registry.register_component("template_manager", ScaffoldTemplateManager())
```

## 📊 **Performance & Optimization**

### **Scaffold Caching**
```python
class ScaffoldCache:
    """Cache generated scaffolds for performance."""
    
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
    
    def get_scaffold(self, face_hash: str) -> Optional[List[Dict]]:
        """Get cached scaffold by face hash."""
        return self.cache.get(face_hash)
    
    def cache_scaffold(self, face_hash: str, scaffold: List[Dict]) -> None:
        """Cache scaffold for future use."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[face_hash] = scaffold
```

### **Lazy Rendering**
```typescript
class LazyScaffoldRenderer {
  async renderScaffold(scaffold: ScaffoldSchema[]): Promise<HTMLElement> {
    const container = document.createElement('div');
    
    // Render shell immediately
    const shell = scaffold.find(s => s.type === 'Shell');
    container.appendChild(this.renderShell(shell));
    
    // Lazy load other layers
    for (const layer of scaffold) {
      if (layer.type !== 'Shell') {
        await this.renderLayerLazy(layer, container);
      }
    }
    
    return container;
  }
  
  private async renderLayerLazy(layer: any, container: HTMLElement) {
    // Simulate async rendering
    await new Promise(resolve => setTimeout(resolve, 10));
    container.appendChild(this.renderLayer(layer));
  }
}
```

## 🧪 **Testing & Validation**

### **Scaffold Validation**
```python
class ScaffoldValidator:
    """Validate scaffold structure and properties."""
    
    def validate_scaffold(self, scaffold: List[Dict]) -> ValidationResult:
        """Validate complete scaffold."""
        errors = []
        
        # Check for required layers
        required_types = ['Shell']
        for req_type in required_types:
            if not any(s['type'] == req_type for s in scaffold):
                errors.append(f"Missing required layer: {req_type}")
        
        # Validate layer hierarchy
        for i, layer in enumerate(scaffold):
            if layer['type'] == 'Shell' and i != 0:
                errors.append("Shell must be first layer")
        
        # Validate component properties
        for layer in scaffold:
            prop_errors = self.validate_layer_props(layer)
            errors.extend(prop_errors)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors
        )
    
    def validate_layer_props(self, layer: Dict) -> List[str]:
        """Validate layer properties."""
        errors = []
        
        if 'id' not in layer:
            errors.append(f"Layer {layer.get('type', 'unknown')} missing required 'id'")
        
        if layer.get('type') == 'Card' and 'variant' not in layer.get('props', {}):
            errors.append("Card components require 'variant' property")
        
        return errors
```

### **Scaffold Testing**
```python
import pytest
from perspective_dcide.ui_scaffold import FaceToScaffold, ScaffoldValidator

class TestUIScaffold:
    def test_face_parsing(self):
        """Test ASCII face parsing to scaffold."""
        face = "ʕᵒ☯ϖ☯ᵒʔ"
        converter = FaceToScaffold()
        scaffold = converter.face_to_scaffold(face)
        
        assert len(scaffold) > 0
        assert any(s['type'] == 'Shell' for s in scaffold)
        assert any(s['type'] == 'Main' for s in scaffold)
    
    def test_scaffold_validation(self):
        """Test scaffold validation."""
        validator = ScaffoldValidator()
        
        # Valid scaffold
        valid_scaffold = [
            {"type": "Shell", "id": "test-shell", "props": {}},
            {"type": "Main", "id": "test-main", "props": {}}
        ]
        
        result = validator.validate_scaffold(valid_scaffold)
        assert result.is_valid
        
        # Invalid scaffold (missing Shell)
        invalid_scaffold = [
            {"type": "Main", "id": "test-main", "props": {}}
        ]
        
        result = validator.validate_scaffold(invalid_scaffold)
        assert not result.is_valid
        assert "Missing required layer: Shell" in result.errors
```

## 🚀 **Future Enhancements**

### **1. AI-Powered Scaffold Generation**
```python
class AIScaffoldGenerator:
    """Generate scaffolds using AI analysis of content."""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    async def generate_ai_scaffold(self, content: str) -> List[Dict]:
        """Generate scaffold using AI analysis."""
        
        prompt = f"""
        Analyze this content and generate a UI scaffold:
        {content}
        
        Return a JSONL scaffold that best represents the content structure.
        """
        
        response = await self.llm_client.generate(prompt)
        return self.parse_ai_response(response)
```

### **2. Real-time Scaffold Adaptation**
```python
class AdaptiveScaffold:
    """Scaffold that adapts to user interaction and content changes."""
    
    def __init__(self, base_scaffold: List[Dict]):
        self.base_scaffold = base_scaffold
        self.adaptation_history = []
    
    def adapt_to_interaction(self, interaction: UserInteraction) -> List[Dict]:
        """Adapt scaffold based on user interaction."""
        
        # Analyze interaction pattern
        pattern = self.analyze_interaction_pattern(interaction)
        
        # Modify scaffold accordingly
        adapted_scaffold = self.apply_adaptation(self.base_scaffold, pattern)
        
        # Record adaptation
        self.adaptation_history.append({
            'interaction': interaction,
            'adaptation': pattern,
            'timestamp': datetime.now()
        })
        
        return adapted_scaffold
```

### **3. Multi-Framework Support**
```python
class MultiFrameworkRenderer:
    """Render scaffolds in multiple frameworks."""
    
    def __init__(self):
        self.renderers = {
            'react': ReactScaffoldRenderer(),
            'vue': VueScaffoldRenderer(),
            'angular': AngularScaffoldRenderer(),
            'svelte': SvelteScaffoldRenderer(),
            'threejs': ThreeJSScaffoldRenderer()
        }
    
    def render_scaffold(self, scaffold: List[Dict], framework: str) -> str:
        """Render scaffold for specific framework."""
        
        if framework not in self.renderers:
            raise ValueError(f"Unsupported framework: {framework}")
        
        renderer = self.renderers[framework]
        return renderer.render(scaffold)
```

## 📚 **Best Practices**

### **1. Scaffold Design Principles**
- **Semantic Structure**: Use meaningful IDs and classes
- **Responsive Design**: Include breakpoint configurations
- **Accessibility**: Add ARIA labels and roles
- **Performance**: Minimize DOM depth and complexity
- **Maintainability**: Use consistent naming conventions

### **2. Content Analysis Integration**
- **Theme Matching**: Match scaffold theme to content mood
- **Complexity Scaling**: Adjust scaffold complexity based on content
- **Interaction Patterns**: Consider expected user interactions
- **Information Architecture**: Structure based on content hierarchy

### **3. Framework Integration**
- **Component Libraries**: Leverage existing UI libraries
- **State Management**: Consider state requirements in scaffold design
- **Routing**: Include routing structure in scaffold
- **Styling**: Provide theme and styling configurations

---

**UI Scaffold System** - Transforming symbolic faces into complete web interfaces through structured, streaming JSONL definitions. 