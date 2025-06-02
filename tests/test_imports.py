"""Test module imports and basic instantiation.

This test module verifies that all components of the HED Tools Integration
package can be imported and instantiated without errors.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_package_import():
    """Test that the main package can be imported."""
    import hedtools_integration
    assert hedtools_integration.__version__ == "0.1.0"
    assert hasattr(hedtools_integration, '__all__')


def test_core_imports():
    """Test that all core modules can be imported."""
    from hedtools_integration import (
        HEDServer, HEDWrapper, BIDSColumnAnalyzer, FileHandler
    )
    
    # Basic import verification
    assert HEDServer is not None
    assert HEDWrapper is not None
    assert BIDSColumnAnalyzer is not None
    assert FileHandler is not None


def test_factory_functions():
    """Test that factory functions can be imported."""
    from hedtools_integration import (
        create_server, create_hed_wrapper, 
        create_column_analyzer, create_file_handler
    )
    
    assert callable(create_server)
    assert callable(create_hed_wrapper)
    assert callable(create_column_analyzer)
    assert callable(create_file_handler)


def test_basic_instantiation():
    """Test basic class instantiation without errors."""
    from hedtools_integration import (
        HEDServer, HEDWrapper, BIDSColumnAnalyzer, FileHandler
    )
    
    # Test instantiation
    server = HEDServer()
    wrapper = HEDWrapper()
    analyzer = BIDSColumnAnalyzer()
    handler = FileHandler()
    
    assert isinstance(server, HEDServer)
    assert isinstance(wrapper, HEDWrapper)
    assert isinstance(analyzer, BIDSColumnAnalyzer)
    assert isinstance(handler, FileHandler)


def test_convenience_functions():
    """Test package-level convenience functions."""
    import hedtools_integration
    
    # Test get_package_info
    info = hedtools_integration.get_package_info()
    assert isinstance(info, dict)
    assert "name" in info
    assert "version" in info
    assert "components" in info
    assert "dependencies" in info
    
    # Test validate_installation
    validation = hedtools_integration.validate_installation()
    assert isinstance(validation, dict)
    assert "valid" in validation
    assert "errors" in validation
    assert "warnings" in validation
    assert "recommendations" in validation


def test_integration_suite():
    """Test creation of integration suite."""
    import hedtools_integration
    
    suite = hedtools_integration.create_integration_suite()
    assert isinstance(suite, dict)
    
    # Check that available components are included
    info = hedtools_integration.get_package_info()
    for component, available in info["components"].items():
        if available:
            assert component in ["server", "hed_wrapper", "column_analyzer", "file_handler"]


def test_server_module():
    """Test server module functionality."""
    from hedtools_integration.server.server import HEDServer, create_server
    
    server = create_server()
    assert isinstance(server, HEDServer)
    assert hasattr(server, 'setup')
    assert hasattr(server, 'get_server_info')
    
    # Test server info
    info = server.get_server_info()
    assert isinstance(info, dict)
    assert "name" in info
    assert "version" in info
    assert "capabilities" in info


def test_hed_wrapper_module():
    """Test HED wrapper module functionality."""
    from hedtools_integration.hed_integration.hed_wrapper import HEDWrapper, create_hed_wrapper
    
    wrapper = create_hed_wrapper()
    assert isinstance(wrapper, HEDWrapper)
    assert hasattr(wrapper, 'load_schema')
    assert hasattr(wrapper, 'validate_events')
    assert hasattr(wrapper, 'generate_sidecar_template')
    
    # Test schema info
    info = wrapper.get_schema_info()
    assert isinstance(info, dict)
    assert "loaded" in info


def test_column_analyzer_module():
    """Test column analyzer module functionality."""
    from hedtools_integration.tools.column_analyzer import BIDSColumnAnalyzer, create_column_analyzer
    
    analyzer = create_column_analyzer()
    assert isinstance(analyzer, BIDSColumnAnalyzer)
    assert hasattr(analyzer, 'analyze_events_file')
    assert hasattr(analyzer, 'suggest_hed_annotations')
    
    # Test summary
    summary = analyzer.get_summary()
    assert isinstance(summary, dict)


def test_file_handler_module():
    """Test file handler module functionality."""
    from hedtools_integration.utils.file_utils import FileHandler, create_file_handler
    
    handler = create_file_handler()
    assert isinstance(handler, FileHandler)
    assert hasattr(handler, 'load_events_file')
    assert hasattr(handler, 'save_events_file')
    assert hasattr(handler, 'load_json_file')
    assert hasattr(handler, 'save_json_file')


def test_all_modules_importable():
    """Test that all modules in the package are importable."""
    modules_to_test = [
        "hedtools_integration",
        "hedtools_integration.server",
        "hedtools_integration.server.server",
        "hedtools_integration.hed_integration",
        "hedtools_integration.hed_integration.hed_wrapper",
        "hedtools_integration.tools",
        "hedtools_integration.tools.column_analyzer",
        "hedtools_integration.utils",
        "hedtools_integration.utils.file_utils"
    ]
    
    for module_name in modules_to_test:
        try:
            __import__(module_name)
        except ImportError as e:
            pytest.fail(f"Failed to import {module_name}: {e}")


if __name__ == "__main__":
    # Run basic tests when executed directly
    print("Running basic import tests...")
    
    try:
        test_package_import()
        print("✅ Package import test passed")
        
        test_core_imports()
        print("✅ Core imports test passed")
        
        test_basic_instantiation()
        print("✅ Basic instantiation test passed")
        
        test_convenience_functions()
        print("✅ Convenience functions test passed")
        
        test_all_modules_importable()
        print("✅ All modules importable test passed")
        
        print("\n🎉 All import tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 