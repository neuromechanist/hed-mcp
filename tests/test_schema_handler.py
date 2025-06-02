"""Tests for the HED schema handling module.

This module tests the SchemaHandler class and utility functions for loading,
validating, and working with HED schemas.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from src.hed_tools.hed_integration.schema import (
    SchemaHandler, 
    SchemaManagerFacade, 
    HEDSchemaError,
    load_hed_schema,
    validate_hed_tag_simple,
    get_schema_version_info
)
from src.hed_tools.hed_integration.models import SchemaConfig, SchemaInfo


class TestSchemaHandler:
    """Test cases for the SchemaHandler class."""
    
    @pytest.fixture
    def schema_config(self):
        """Create a test schema configuration."""
        return SchemaConfig(
            version="8.3.0",
            fallback_versions=["8.2.0", "8.1.0"],
            enable_cache=True
        )
    
    @pytest.fixture
    def schema_handler(self, schema_config):
        """Create a SchemaHandler instance for testing."""
        return SchemaHandler(schema_config)
    
    def test_schema_handler_initialization(self, schema_handler, schema_config):
        """Test schema handler initialization."""
        assert schema_handler.config == schema_config
        assert schema_handler.schema is None
        assert schema_handler._schema_cache == {}
        assert schema_handler._multiple_schemas == {}
        assert schema_handler._schema_info is None
    
    @pytest.mark.asyncio
    async def test_load_schema_success(self, schema_handler):
        """Test successful schema loading."""
        mock_schema = Mock()
        mock_schema.version_number = "8.3.0"
        mock_schema.tags = {"Event": {}, "Sensory-event": {}}
        
        with patch('src.hed_tools.hed_integration.schema.load_schema_version', return_value=mock_schema):
            result = await schema_handler.load_schema()
            
            assert result.success is True
            assert schema_handler.schema == mock_schema
            assert schema_handler.is_schema_loaded() is True
            assert result.data is not None
    
    @pytest.mark.asyncio
    async def test_load_schema_with_fallback(self, schema_handler):
        """Test schema loading with fallback versions."""
        mock_schema = Mock()
        mock_schema.version_number = "8.2.0"
        mock_schema.tags = {"Event": {}}
        
        def mock_load_version(version):
            if version == "8.3.0":
                raise Exception("Version not found")
            elif version == "8.2.0":
                return mock_schema
            else:
                raise Exception("Version not found")
        
        with patch('src.hed_tools.hed_integration.schema.load_schema_version', side_effect=mock_load_version):
            result = await schema_handler.load_schema(version="8.3.0")
            
            assert result.success is True
            assert schema_handler.schema == mock_schema
    
    @pytest.mark.asyncio
    async def test_load_schema_failure(self, schema_handler):
        """Test schema loading failure."""
        with patch('src.hed_tools.hed_integration.schema.load_schema_version', side_effect=Exception("Network error")):
            result = await schema_handler.load_schema()
            
            assert result.success is False
            assert "Network error" in result.error
            assert schema_handler.schema is None
    
    @pytest.mark.asyncio
    async def test_load_multiple_schemas(self, schema_handler):
        """Test loading multiple schemas simultaneously."""
        mock_schema_83 = Mock()
        mock_schema_83.version_number = "8.3.0"
        mock_schema_83.tags = {"Event": {}}
        
        mock_schema_82 = Mock()
        mock_schema_82.version_number = "8.2.0"
        mock_schema_82.tags = {"Event": {}}
        
        def mock_load_version(version):
            if version == "8.3.0":
                return mock_schema_83
            elif version == "8.2.0":
                return mock_schema_82
            else:
                raise Exception("Version not found")
        
        with patch('src.hed_tools.hed_integration.schema.load_schema_version', side_effect=mock_load_version):
            result = await schema_handler.load_multiple_schemas(["8.3.0", "8.2.0", "8.1.0"])
            
            assert result.success is True
            assert result.data["total_requested"] == 3
            assert result.data["total_loaded"] == 2
            assert "8.3.0" in result.data["loaded_schemas"]
            assert "8.2.0" in result.data["loaded_schemas"]
            assert "8.1.0" in result.data["failed_schemas"]
    
    def test_get_schema_by_version(self, schema_handler):
        """Test retrieving specific schema versions."""
        mock_schema = Mock()
        schema_handler._multiple_schemas["8.3.0"] = mock_schema
        
        retrieved_schema = schema_handler.get_schema_by_version("8.3.0")
        assert retrieved_schema == mock_schema
        
        non_existent = schema_handler.get_schema_by_version("9.0.0")
        assert non_existent is None
    
    def test_get_loaded_schema_versions(self, schema_handler):
        """Test getting list of loaded schema versions."""
        schema_handler._multiple_schemas["8.3.0"] = Mock()
        schema_handler._multiple_schemas["8.2.0"] = Mock()
        
        versions = schema_handler.get_loaded_schema_versions()
        assert "8.3.0" in versions
        assert "8.2.0" in versions
        assert len(versions) == 2
    
    def test_get_all_schema_tags(self, schema_handler):
        """Test getting all tags from a schema."""
        mock_schema = Mock()
        mock_schema.get_all_tags.return_value = ["Event", "Sensory-event", "Agent-action"]
        schema_handler.schema = mock_schema
        
        tags = schema_handler.get_all_schema_tags()
        expected_tags = {"Event", "Sensory-event", "Agent-action"}
        assert tags == expected_tags
    
    def test_get_all_schema_tags_fallback(self, schema_handler):
        """Test getting tags when get_all_tags method is not available."""
        mock_schema = Mock()
        mock_schema.tags = {"Event": {}, "Sensory-event": {}}
        delattr(mock_schema, 'get_all_tags')  # Remove the method
        schema_handler.schema = mock_schema
        
        tags = schema_handler.get_all_schema_tags()
        expected_tags = {"Event", "Sensory-event"}
        assert tags == expected_tags
    
    def test_validate_tag(self, schema_handler):
        """Test tag validation."""
        mock_schema = Mock()
        schema_handler.schema = mock_schema
        
        # Mock HedString validation
        with patch('src.hed_tools.hed_integration.schema.HedString') as mock_hed_string:
            mock_hed_instance = Mock()
            mock_hed_instance.validate.return_value = []  # No issues
            mock_hed_string.return_value = mock_hed_instance
            
            is_valid = schema_handler.validate_tag("Event")
            assert is_valid is True
            
            # Test with validation issues
            mock_hed_instance.validate.return_value = ["Error: Invalid tag"]
            is_valid = schema_handler.validate_tag("InvalidTag")
            assert is_valid is False
    
    def test_get_tag_descendants(self, schema_handler):
        """Test getting tag descendants."""
        mock_schema = Mock()
        mock_schema.get_tag_descendants.return_value = ["Event/Sensory-event", "Event/Agent-action"]
        schema_handler.schema = mock_schema
        
        descendants = schema_handler.get_tag_descendants("Event")
        assert "Event/Sensory-event" in descendants
        assert "Event/Agent-action" in descendants
    
    def test_find_similar_tags(self, schema_handler):
        """Test finding similar tags."""
        mock_schema = Mock()
        mock_schema.get_all_tags.return_value = ["Event", "Sensory-event", "Event-context", "Agent-action"]
        schema_handler.schema = mock_schema
        
        similar_tags = schema_handler.find_similar_tags("event", max_results=5)
        assert "Event" in similar_tags
        assert "Sensory-event" in similar_tags
        assert "Event-context" in similar_tags
    
    def test_compare_schema_versions(self, schema_handler):
        """Test comparing two schema versions."""
        mock_schema_83 = Mock()
        mock_schema_83.get_all_tags.return_value = ["Event", "Sensory-event", "NewTag"]
        
        mock_schema_82 = Mock()
        mock_schema_82.get_all_tags.return_value = ["Event", "Sensory-event", "OldTag"]
        
        schema_handler._multiple_schemas["8.3.0"] = mock_schema_83
        schema_handler._multiple_schemas["8.2.0"] = mock_schema_82
        
        with patch.object(schema_handler, 'get_all_schema_tags') as mock_get_tags:
            mock_get_tags.side_effect = lambda v: (
                {"Event", "Sensory-event", "NewTag"} if v == "8.3.0" 
                else {"Event", "Sensory-event", "OldTag"}
            )
            
            comparison = schema_handler.compare_schema_versions("8.3.0", "8.2.0")
            
            assert comparison["version1"] == "8.3.0"
            assert comparison["version2"] == "8.2.0"
            assert "NewTag" in comparison["tags_in_v1_only"]
            assert "OldTag" in comparison["tags_in_v2_only"]
            assert "Event" in comparison["common_tags"]
            assert "Sensory-event" in comparison["common_tags"]
    
    @pytest.mark.asyncio
    async def test_validate_schema(self, schema_handler):
        """Test schema validation."""
        mock_schema = Mock()
        mock_schema.check_compliance.return_value = []  # No issues
        schema_handler.schema = mock_schema
        
        result = await schema_handler.validate_schema()
        assert result.success is True
        assert result.data["issue_count"] == 0
    
    def test_get_available_schemas(self, schema_handler):
        """Test getting available schemas."""
        schemas = schema_handler.get_available_schemas()
        
        assert isinstance(schemas, list)
        assert len(schemas) > 0
        
        # Check that standard schemas are included
        standard_versions = [s["version"] for s in schemas if s["type"] == "standard"]
        assert "8.3.0" in standard_versions
        assert "8.2.0" in standard_versions
    
    def test_cache_operations(self, schema_handler):
        """Test cache operations."""
        mock_schema = Mock()
        schema_handler._schema_cache["8.3.0"] = mock_schema
        schema_handler._multiple_schemas["8.3.0"] = mock_schema
        
        # Test cache info
        cache_info = schema_handler.get_cache_info()
        assert "8.3.0" in cache_info["cached_schemas"]
        assert "8.3.0" in cache_info["multiple_schemas"]
        assert cache_info["cache_size"] == 1
        
        # Test cache clearing
        schema_handler.clear_cache()
        assert len(schema_handler._schema_cache) == 0
        assert len(schema_handler._multiple_schemas) == 0
    
    @pytest.mark.asyncio
    async def test_preload_schemas(self, schema_handler):
        """Test preloading schemas."""
        with patch.object(schema_handler, 'load_multiple_schemas') as mock_load_multiple:
            mock_result = Mock()
            mock_result.success = True
            mock_result.data = {
                "loaded_schemas": ["8.3.0", "8.2.0"],
                "failed_schemas": {"8.1.0": "Error"}
            }
            mock_load_multiple.return_value = mock_result
            
            results = await schema_handler.preload_schemas(["8.3.0", "8.2.0", "8.1.0"])
            
            assert results["8.3.0"] is True
            assert results["8.2.0"] is True
            assert results["8.1.0"] is False


class TestSchemaManagerFacade:
    """Test cases for the SchemaManagerFacade class."""
    
    @pytest.fixture
    def facade(self):
        """Create a SchemaManagerFacade instance for testing."""
        return SchemaManagerFacade("8.3.0")
    
    @pytest.mark.asyncio
    async def test_initialize(self, facade):
        """Test facade initialization."""
        with patch.object(facade.handler, 'load_schema') as mock_load:
            mock_result = Mock()
            mock_result.success = True
            mock_load.return_value = mock_result
            
            success = await facade.initialize()
            assert success is True
            assert facade._initialized is True
    
    @pytest.mark.asyncio
    async def test_get_schema(self, facade):
        """Test getting schema through facade."""
        mock_schema = Mock()
        facade._initialized = True
        
        with patch.object(facade.handler, 'get_schema_info') as mock_info, \
             patch.object(facade.handler, 'get_schema') as mock_get_schema:
            
            mock_schema_info = Mock()
            mock_schema_info.version = "8.3.0"
            mock_info.return_value = mock_schema_info
            mock_get_schema.return_value = mock_schema
            
            schema = await facade.get_schema()
            assert schema == mock_schema
    
    def test_is_available(self, facade):
        """Test checking if HED functionality is available."""
        # This will depend on whether hedtools is actually installed
        result = facade.is_available()
        assert isinstance(result, bool)
    
    @pytest.mark.asyncio
    async def test_validate_schema_compatibility(self, facade):
        """Test schema compatibility validation."""
        with patch.object(facade.handler, 'load_schema') as mock_load:
            mock_result = Mock()
            mock_result.success = True
            mock_load.return_value = mock_result
            
            is_compatible = await facade.validate_schema_compatibility("8.3.0")
            assert is_compatible is True


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_load_hed_schema_success(self):
        """Test successful schema loading utility."""
        mock_schema = Mock()
        
        with patch('src.hed_tools.hed_integration.schema.load_schema_version', return_value=mock_schema):
            schema, error = load_hed_schema("8.3.0")
            
            assert schema == mock_schema
            assert error is None
    
    def test_load_hed_schema_failure(self):
        """Test schema loading utility failure."""
        with patch('src.hed_tools.hed_integration.schema.load_schema_version', side_effect=Exception("Error")):
            schema, error = load_hed_schema("8.3.0")
            
            assert schema is None
            assert "Error" in error
    
    def test_validate_hed_tag_simple_success(self):
        """Test simple tag validation utility success."""
        mock_schema = Mock()
        mock_hed_string = Mock()
        mock_hed_string.validate.return_value = []  # No issues
        
        with patch('src.hed_tools.hed_integration.schema.HedString', return_value=mock_hed_string):
            is_valid, errors = validate_hed_tag_simple("Event", mock_schema)
            
            assert is_valid is True
            assert errors == []
    
    def test_validate_hed_tag_simple_failure(self):
        """Test simple tag validation utility failure."""
        mock_schema = Mock()
        mock_hed_string = Mock()
        mock_hed_string.validate.return_value = ["Error: Invalid tag"]
        
        with patch('src.hed_tools.hed_integration.schema.HedString', return_value=mock_hed_string):
            is_valid, errors = validate_hed_tag_simple("InvalidTag", mock_schema)
            
            assert is_valid is False
            assert len(errors) == 1
            assert "Invalid tag" in errors[0]
    
    def test_get_schema_version_info(self):
        """Test schema version info extraction."""
        mock_schema = Mock()
        mock_schema.version_number = "8.3.0"
        mock_schema.name = "HED"
        mock_schema.tags = {"Event": {}, "Sensory-event": {}}
        mock_schema.units = {}
        mock_schema.attributes = {}
        mock_schema.prologue = "HED Schema prologue"
        mock_schema.epilogue = "HED Schema epilogue"
        
        info = get_schema_version_info(mock_schema)
        
        assert info["version"] == "8.3.0"
        assert info["name"] == "HED"
        assert info["tag_count"] == 2
        assert info["has_units"] is True
        assert info["has_attributes"] is True
        assert info["prologue"] == "HED Schema prologue"
        assert info["epilogue"] == "HED Schema epilogue"


class TestErrorHandling:
    """Test cases for error handling scenarios."""
    
    @pytest.fixture
    def schema_handler(self):
        """Create a SchemaHandler instance for testing."""
        return SchemaHandler()
    
    @pytest.mark.asyncio
    async def test_load_schema_no_hed_library(self, schema_handler):
        """Test schema loading when HED library is not available."""
        with patch('src.hed_tools.hed_integration.schema.load_schema_version', None):
            result = await schema_handler.load_schema()
            
            assert result.success is False
            assert "HED library not available" in result.error
    
    def test_get_all_schema_tags_no_schema(self, schema_handler):
        """Test getting tags when no schema is loaded."""
        tags = schema_handler.get_all_schema_tags()
        assert tags == set()
    
    def test_validate_tag_no_schema(self, schema_handler):
        """Test tag validation when no schema is loaded."""
        is_valid = schema_handler.validate_tag("Event")
        assert is_valid is False
    
    def test_custom_error_handling(self):
        """Test custom HEDSchemaError exception."""
        with pytest.raises(HEDSchemaError):
            raise HEDSchemaError("Test error message") 