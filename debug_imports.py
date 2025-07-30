#!/usr/bin/env python3
"""
Debug script to isolate import issues.
"""

import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_basic_imports():
    """Test basic Python imports."""
    print("Testing basic imports...")
    
    try:
        from typing import Optional, Dict, List, Any
        print("✅ Basic typing imports successful")
        return True
    except Exception as e:
        print(f"❌ Basic typing imports failed: {e}")
        return False

def test_core_config():
    """Test core config import."""
    print("Testing core config...")
    
    try:
        from perspective_dcide.core.config import Config
        print("✅ Core config import successful")
        return True
    except Exception as e:
        print(f"❌ Core config import failed: {e}")
        return False

def test_core_schemas():
    """Test core schemas import."""
    print("Testing core schemas...")
    
    try:
        from perspective_dcide.core.schemas import ContentItem
        print("✅ Core schemas import successful")
        return True
    except Exception as e:
        print(f"❌ Core schemas import failed: {e}")
        return False

def test_symbolic_glyphs():
    """Test symbolic glyphs import."""
    print("Testing symbolic glyphs...")
    
    try:
        from perspective_dcide.symbolic.glyphs import GlyphLookup
        print("✅ Symbolic glyphs import successful")
        return True
    except Exception as e:
        print(f"❌ Symbolic glyphs import failed: {e}")
        return False

def test_symbolic_logic():
    """Test symbolic logic import."""
    print("Testing symbolic logic...")
    
    try:
        from perspective_dcide.symbolic.logic import SymbolicCollapse
        print("✅ Symbolic logic import successful")
        return True
    except Exception as e:
        print(f"❌ Symbolic logic import failed: {e}")
        return False

def main():
    """Run debug tests."""
    print("🔍 Debugging Import Issues")
    print("=" * 40)
    
    tests = [
        test_basic_imports,
        test_core_config,
        test_core_schemas,
        test_symbolic_glyphs,
        test_symbolic_logic
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"📊 Debug Results: {passed}/{total} tests passed")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 