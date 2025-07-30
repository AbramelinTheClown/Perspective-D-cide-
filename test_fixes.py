#!/usr/bin/env python3
"""
Test script to verify that the framework fixes are working.
"""

import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_core_imports():
    """Test core module imports."""
    print("Testing core imports...")
    
    try:
        from perspective_dcide.core.config import Config
        from perspective_dcide.core import initialize_framework
        print("✅ Core imports successful")
        return True
    except Exception as e:
        print(f"❌ Core imports failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_symbolic_imports():
    """Test symbolic module imports."""
    print("Testing symbolic imports...")
    
    try:
        from perspective_dcide.symbolic import GlyphLookup, SymbolicCollapse, TarotMapping
        print("✅ Symbolic imports successful")
        return True
    except Exception as e:
        print(f"❌ Symbolic imports failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_framework_initialization():
    """Test framework initialization."""
    print("Testing framework initialization...")
    
    try:
        from perspective_dcide.core.config import Config
        from perspective_dcide.core import initialize_framework
        
        config = Config()
        initialize_framework(config)
        print("✅ Framework initialization successful")
        return True
    except Exception as e:
        print(f"❌ Framework initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_website_scraper():
    """Test website scraper imports."""
    print("Testing website scraper...")
    
    try:
        from perspective_dcide.cli.website_scraper import WebsiteScraper
        print("✅ Website scraper imports successful")
        return True
    except Exception as e:
        print(f"❌ Website scraper imports failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Perspective D<cide> Framework Fixes")
    print("=" * 50)
    
    tests = [
        test_core_imports,
        test_symbolic_imports,
        test_framework_initialization,
        test_website_scraper
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Framework is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 