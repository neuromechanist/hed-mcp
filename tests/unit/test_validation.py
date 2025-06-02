"""Unit tests for HED validation and sidecar generation utilities.

This module contains comprehensive tests for the validation components,
including string validation, sidecar generation, batch processing, and BIDS validation.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import pytest
import pandas as pd

from src.hed_tools.hed_integration.validation import (
    HEDValidator,
    SidecarGenerator,
    BatchValidator,
    BIDSValidator,
    ValidationError,
    create_hed_validator,
    create_sidecar_generator,
    create_batch_validator,
    create_bids_validator
)
from src.hed_tools.hed_integration.models import (
    ValidationResult,
    SidecarTemplate,
    OperationResult
)


class TestHEDValidator:
    """Test cases for HEDValidator class."""
    
    @pytest.fixture
    def mock_schema_handler(self):
        """Create mock schema handler."""
        handler = Mock()
        handler.get_schema.return_value = Mock()
        handler.current_version = "8.3.0"
        return handler
    
    @pytest.fixture
    def validator(self, mock_schema_handler):
        """Create HEDValidator instance with mocked dependencies."""
        return HEDValidator(mock_schema_handler)
    
    def test_validator_initialization(self, validator, mock_schema_handler):
        """Test validator initialization."""
        assert validator.schema_handler == mock_schema_handler
        assert validator._validator_cache == {}
    
    @patch('src.hed_tools.hed_integration.validation.HED_AVAILABLE', True)
    @patch('src.hed_tools.hed_integration.validation.HedValidator')
    def test_get_validator_cached(self, mock_hed_validator, validator):
        """Test validator caching mechanism."""
        mock_instance = Mock()
        mock_hed_validator.return_value = mock_instance
        
        # First call should create validator
        result1 = validator._get_validator("8.3.0")
        assert result1 == mock_instance
        assert mock_hed_validator.called
        
        # Second call should use cache
        mock_hed_validator.reset_mock()
        result2 = validator._get_validator("8.3.0")
        assert result2 == mock_instance
        assert not mock_hed_validator.called
    
    @patch('src.hed_tools.hed_integration.validation.HED_AVAILABLE', False)
    def test_get_validator_hed_unavailable(self, validator):
        """Test validator when HED library is not available."""
        result = validator._get_validator()
        assert result is not None  # Should return stub validator
    
    @pytest.mark.asyncio
    @patch('src.hed_tools.hed_integration.validation.HED_AVAILABLE', True)
    async def test_validate_string_success(self, validator):
        """Test successful string validation."""
        # Mock the validator to return no issues
        mock_validator = Mock()
        mock_validator.validate_string.return_value = []
        validator._get_validator = Mock(return_value=mock_validator)
        
        result = await validator.validate_string("(Label/test)")
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert result.errors == []
        assert "passed" in result.summary
    
    @pytest.mark.asyncio
    @patch('src.hed_tools.hed_integration.validation.HED_AVAILABLE', True)
    async def test_validate_string_with_errors(self, validator):
        """Test string validation with errors."""
        # Mock the validator to return issues
        mock_issues = [{"message": "Invalid tag", "code": "INVALID_TAG"}]
        mock_validator = Mock()
        mock_validator.validate_string.return_value = mock_issues
        validator._get_validator = Mock(return_value=mock_validator)
        
        result = await validator.validate_string("(InvalidTag)")
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is False
        assert result.errors == mock_issues
        assert "failed" in result.summary
    
    @pytest.mark.asyncio
    async def test_validate_string_exception(self, validator):
        """Test string validation with exception."""
        # Mock the validator to raise an exception
        mock_validator = Mock()
        mock_validator.validate_string.side_effect = Exception("Validation error")
        validator._get_validator = Mock(return_value=mock_validator)
        
        result = await validator.validate_string("(Label/test)")
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "Validation error" in result.errors[0]["message"]
    
    @pytest.mark.asyncio
    @patch('src.hed_tools.hed_integration.validation.HED_AVAILABLE', False)
    async def test_validate_events_data_hed_unavailable(self, validator):
        """Test events validation when HED is not available."""
        events_df = pd.DataFrame({
            'onset': [1.0, 2.0],
            'duration': [0.5, 0.5], 
            'trial_type': ['condition_1', 'condition_2']
        })
        
        result = await validator.validate_events_data(events_df)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert len(result.warnings) == 1
        assert "HED library not available" in result.warnings[0]["message"]
    
    @pytest.mark.asyncio
    async def test_validate_sidecar_file_success(self, validator):
        """Test successful sidecar file validation."""
        # Create temporary sidecar file
        sidecar_data = {
            "trial_type": {
                "HED": {
                    "condition_1": "(Label/condition-1)",
                    "condition_2": "(Label/condition-2)"
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sidecar_data, f)
            sidecar_path = f.name
        
        try:
            # Mock string validation to succeed
            async def mock_validate_string(hed_string, schema_version=None):
                return ValidationResult(
                    is_valid=True,
                    errors=[],
                    warnings=[],
                    summary="Validation passed"
                )
            
            validator.validate_string = mock_validate_string
            
            result = await validator.validate_sidecar_file(sidecar_path)
            
            assert isinstance(result, ValidationResult)
            assert result.is_valid is True
            assert result.errors == []
            assert "passed" in result.summary
            
        finally:
            Path(sidecar_path).unlink()
    
    @pytest.mark.asyncio
    async def test_validate_sidecar_file_with_errors(self, validator):
        """Test sidecar file validation with errors."""
        # Create temporary sidecar file
        sidecar_data = {
            "trial_type": {
                "HED": {
                    "condition_1": "(InvalidTag)"
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sidecar_data, f)
            sidecar_path = f.name
        
        try:
            # Mock string validation to fail
            async def mock_validate_string(hed_string, schema_version=None):
                return ValidationResult(
                    is_valid=False,
                    errors=[{"message": "Invalid tag", "code": "INVALID_TAG"}],
                    warnings=[],
                    summary="Validation failed"
                )
            
            validator.validate_string = mock_validate_string
            
            result = await validator.validate_sidecar_file(sidecar_path)
            
            assert isinstance(result, ValidationResult)
            assert result.is_valid is False
            assert len(result.errors) == 1
            assert "trial_type" in result.errors[0]["column"]
            
        finally:
            Path(sidecar_path).unlink()


class TestSidecarGenerator:
    """Test cases for SidecarGenerator class."""
    
    @pytest.fixture
    def mock_schema_handler(self):
        """Create mock schema handler."""
        handler = Mock()
        handler.get_schema.return_value = Mock()
        handler.current_version = "8.3.0"
        return handler
    
    @pytest.fixture
    def mock_tabular_wrapper(self):
        """Create mock TabularSummary wrapper."""
        wrapper = Mock()
        wrapper.extract_sidecar_template = AsyncMock()
        return wrapper
    
    @pytest.fixture
    def generator(self, mock_schema_handler, mock_tabular_wrapper):
        """Create SidecarGenerator instance with mocked dependencies."""
        return SidecarGenerator(mock_schema_handler, mock_tabular_wrapper)
    
    def test_generator_initialization(self, generator, mock_schema_handler, mock_tabular_wrapper):
        """Test generator initialization."""
        assert generator.schema_handler == mock_schema_handler
        assert generator.tabular_summary_wrapper == mock_tabular_wrapper
    
    @pytest.mark.asyncio
    async def test_generate_sidecar_template_success(self, generator, mock_tabular_wrapper):
        """Test successful sidecar template generation."""
        # Create sample events data
        events_df = pd.DataFrame({
            'onset': [1.0, 2.0, 3.0],
            'duration': [0.5, 0.5, 0.5],
            'trial_type': ['condition_1', 'condition_2', 'condition_1']
        })
        
        # Mock successful template extraction
        mock_template = {
            "trial_type": {
                "HED": {
                    "condition_1": "(Label/condition-1)",
                    "condition_2": "(Label/condition-2)"
                }
            }
        }
        mock_result = OperationResult(
            success=True,
            result=mock_template,
            message="Template generated successfully"
        )
        mock_tabular_wrapper.extract_sidecar_template.return_value = mock_result
        
        # Mock schema enhancement
        async def mock_enhance(template, schema_version=None):
            return template
        generator._enhance_template_with_schema = mock_enhance
        
        result = await generator.generate_sidecar_template(events_df)
        
        assert isinstance(result, SidecarTemplate)
        assert result.template == mock_template
        assert result.schema_version == "8.3.0"
        assert "trial_type" in result.generated_columns
    
    @pytest.mark.asyncio
    async def test_generate_sidecar_template_failure(self, generator, mock_tabular_wrapper):
        """Test sidecar template generation failure."""
        events_df = pd.DataFrame({
            'onset': [1.0, 2.0],
            'duration': [0.5, 0.5],
            'trial_type': ['condition_1', 'condition_2']
        })
        
        # Mock failed template extraction
        mock_result = OperationResult(
            success=False,
            error="Template generation failed",
            message="Failed to extract template"
        )
        mock_tabular_wrapper.extract_sidecar_template.return_value = mock_result
        
        with pytest.raises(ValidationError, match="Failed to generate template"):
            await generator.generate_sidecar_template(events_df)
    
    @pytest.mark.asyncio
    async def test_enhance_template_with_schema(self, generator):
        """Test template enhancement with schema."""
        template = {
            "trial_type": {
                "HED": {
                    "condition_1": "#",  # Empty placeholder
                    "condition_2": "(Label/existing)"  # Existing annotation
                }
            }
        }
        
        # Mock schema
        mock_schema = Mock()
        generator.schema_handler.get_schema.return_value = mock_schema
        
        # Mock HED suggestion
        async def mock_suggest(column, value, schema):
            if value == "condition_1":
                return "(Condition/condition-1, Label/condition-1)"
            return None
        generator._suggest_hed_annotation = mock_suggest
        
        enhanced = await generator._enhance_template_with_schema(template, "8.3.0")
        
        assert enhanced["trial_type"]["HED"]["condition_1"] == "(Condition/condition-1, Label/condition-1)"
        assert enhanced["trial_type"]["HED"]["condition_2"] == "(Label/existing)"  # Unchanged
    
    @pytest.mark.asyncio
    async def test_suggest_hed_annotation_patterns(self, generator):
        """Test HED annotation suggestion patterns."""
        mock_schema = Mock()
        
        # Test trial_type column
        result = await generator._suggest_hed_annotation("trial_type", "go_trial", mock_schema)
        assert "Condition/go_trial" in result
        assert "Label/go_trial" in result
        
        # Test response column
        result = await generator._suggest_hed_annotation("response", "left", mock_schema)
        assert "Agent-action" in result
        assert "Movement" in result
        assert "Left" in result
        
        # Test stimulus column
        result = await generator._suggest_hed_annotation("stimulus_type", "visual", mock_schema)
        assert "Sensory-event" in result
        assert "Visual-presentation" in result
        
        # Test accuracy column
        result = await generator._suggest_hed_annotation("accuracy", "1", mock_schema)
        assert "Performance" in result
        assert "Correct" in result
        
        # Test reaction time column
        result = await generator._suggest_hed_annotation("rt", "500", mock_schema)
        assert "Response-time" in result
        
        # Test unknown pattern
        result = await generator._suggest_hed_annotation("unknown_column", "value", mock_schema)
        assert "Label/value" in result
    
    @pytest.mark.asyncio
    async def test_save_sidecar_json(self, generator):
        """Test saving sidecar as JSON."""
        template = SidecarTemplate(
            template={"test": {"HED": "(Label/test)"}},
            schema_version="8.3.0",
            generated_columns=["test"],
            metadata={}
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_sidecar.json"
            
            result = await generator.save_sidecar(template, output_path, format='json')
            
            assert isinstance(result, OperationResult)
            assert result.success is True
            assert output_path.exists()
            
            # Verify content
            with open(output_path, 'r') as f:
                saved_data = json.load(f)
            assert saved_data == template.template
    
    @pytest.mark.asyncio
    async def test_save_sidecar_invalid_format(self, generator):
        """Test saving sidecar with invalid format."""
        template = SidecarTemplate(
            template={"test": {"HED": "(Label/test)"}},
            schema_version="8.3.0",
            generated_columns=["test"],
            metadata={}
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_sidecar.unknown"
            
            result = await generator.save_sidecar(template, output_path, format='unknown')
            
            assert isinstance(result, OperationResult)
            assert result.success is False
            assert "Unsupported format" in result.error


class TestBatchValidator:
    """Test cases for BatchValidator class."""
    
    @pytest.fixture
    def mock_validator(self):
        """Create mock HED validator."""
        validator = Mock()
        validator.validate_events_data = AsyncMock()
        return validator
    
    @pytest.fixture
    def batch_validator(self, mock_validator):
        """Create BatchValidator instance with mocked dependencies."""
        return BatchValidator(mock_validator, max_workers=2)
    
    def test_batch_validator_initialization(self, batch_validator, mock_validator):
        """Test batch validator initialization."""
        assert batch_validator.validator == mock_validator
        assert batch_validator.max_workers == 2
    
    @pytest.mark.asyncio
    async def test_validate_directory_success(self, batch_validator, mock_validator):
        """Test successful directory validation."""
        # Create temporary directory with test files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test events files
            events_file1 = temp_path / "sub-01_events.tsv"
            events_file2 = temp_path / "sub-02_events.tsv"
            
            events_file1.write_text("onset\tduration\ttrial_type\n1.0\t0.5\tcondition_1\n")
            events_file2.write_text("onset\tduration\ttrial_type\n2.0\t0.5\tcondition_2\n")
            
            # Create corresponding sidecar files
            sidecar_file1 = temp_path / "sub-01_events.json"
            sidecar_file2 = temp_path / "sub-02_events.json"
            
            sidecar_data = {"trial_type": {"HED": {"condition_1": "(Label/condition-1)"}}}
            sidecar_file1.write_text(json.dumps(sidecar_data))
            sidecar_file2.write_text(json.dumps(sidecar_data))
            
            # Mock successful validation
            mock_result = ValidationResult(
                is_valid=True,
                errors=[],
                warnings=[],
                summary="Validation passed"
            )
            mock_validator.validate_events_data.return_value = mock_result
            
            # Validate directory
            results = []
            async for result in batch_validator.validate_directory(temp_path, "*_events.tsv"):
                results.append(result)
            
            assert len(results) == 2
            for result in results:
                assert result["error"] is None
                assert result["validation_result"].is_valid is True
                assert "_events.tsv" in result["file"]
    
    @pytest.mark.asyncio
    async def test_validate_directory_with_exceptions(self, batch_validator, mock_validator):
        """Test directory validation with exceptions."""
        # Create temporary directory with test file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            events_file = temp_path / "test_events.tsv"
            events_file.write_text("onset\tduration\ttrial_type\n1.0\t0.5\tcondition_1\n")
            
            # Mock validation to raise exception
            mock_validator.validate_events_data.side_effect = Exception("Validation error")
            
            # Validate directory
            results = []
            async for result in batch_validator.validate_directory(temp_path):
                results.append(result)
            
            assert len(results) == 1
            result = results[0]
            assert result["error"] == "Validation error"
            assert result["validation_result"].is_valid is False


class TestBIDSValidator:
    """Test cases for BIDSValidator class."""
    
    @pytest.fixture
    def mock_hed_validator(self):
        """Create mock HED validator."""
        validator = Mock()
        return validator
    
    @pytest.fixture
    def bids_validator(self, mock_hed_validator):
        """Create BIDSValidator instance with mocked dependencies."""
        return BIDSValidator(mock_hed_validator)
    
    def test_bids_validator_initialization(self, bids_validator, mock_hed_validator):
        """Test BIDS validator initialization."""
        assert bids_validator.validator == mock_hed_validator
        assert bids_validator.batch_validator is not None
    
    @pytest.mark.asyncio
    async def test_validate_bids_dataset_no_events(self, bids_validator):
        """Test BIDS dataset validation with no events files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir)
            
            result = await bids_validator.validate_bids_dataset(dataset_path)
            
            assert result["valid"] is True
            assert result["files_validated"] == 0
            assert "No events files found" in result["summary"]
    
    @pytest.mark.asyncio
    async def test_validate_bids_structure_missing_description(self, bids_validator):
        """Test BIDS structure validation with missing dataset_description.json."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir)
            
            issues = await bids_validator._validate_bids_structure(dataset_path)
            
            assert len(issues["errors"]) == 1
            assert "Missing dataset_description.json" in issues["errors"][0]["message"]
    
    @pytest.mark.asyncio
    async def test_validate_bids_structure_invalid_naming(self, bids_validator):
        """Test BIDS structure validation with invalid file naming."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir)
            
            # Create improperly named events file
            invalid_file = dataset_path / "sub-01" / "func" / "sub-01_invalid.tsv"
            invalid_file.parent.mkdir(parents=True)
            invalid_file.write_text("onset\tduration\ttrial_type\n")
            
            issues = await bids_validator._validate_bids_structure(dataset_path)
            
            assert len(issues["warnings"]) == 1
            assert "may not follow BIDS naming" in issues["warnings"][0]["message"]


class TestFactoryFunctions:
    """Test cases for factory functions."""
    
    @pytest.mark.asyncio
    @patch('src.hed_tools.hed_integration.validation.SchemaHandler')
    async def test_create_hed_validator(self, mock_schema_handler_class):
        """Test HED validator factory function."""
        mock_handler = Mock()
        mock_handler.load_schema = AsyncMock()
        mock_schema_handler_class.return_value = mock_handler
        
        validator = await create_hed_validator("8.3.0")
        
        assert isinstance(validator, HEDValidator)
        mock_handler.load_schema.assert_called_once_with("8.3.0")
    
    @pytest.mark.asyncio
    @patch('src.hed_tools.hed_integration.validation.SchemaHandler')
    @patch('src.hed_tools.hed_integration.validation.create_tabular_summary_wrapper')
    async def test_create_sidecar_generator(self, mock_create_wrapper, mock_schema_handler_class):
        """Test sidecar generator factory function."""
        mock_handler = Mock()
        mock_handler.load_schema = AsyncMock()
        mock_schema_handler_class.return_value = mock_handler
        
        mock_wrapper = Mock()
        mock_create_wrapper.return_value = mock_wrapper
        
        generator = await create_sidecar_generator("8.3.0", data=pd.DataFrame())
        
        assert isinstance(generator, SidecarGenerator)
        mock_handler.load_schema.assert_called_once_with("8.3.0")
        mock_create_wrapper.assert_called_once_with(data=pd.DataFrame())
    
    @pytest.mark.asyncio
    @patch('src.hed_tools.hed_integration.validation.create_hed_validator')
    async def test_create_batch_validator(self, mock_create_validator):
        """Test batch validator factory function."""
        mock_validator = Mock()
        mock_create_validator.return_value = mock_validator
        
        batch_validator = await create_batch_validator("8.3.0", max_workers=8)
        
        assert isinstance(batch_validator, BatchValidator)
        assert batch_validator.max_workers == 8
        mock_create_validator.assert_called_once_with("8.3.0")
    
    @pytest.mark.asyncio
    @patch('src.hed_tools.hed_integration.validation.create_hed_validator')
    async def test_create_bids_validator(self, mock_create_validator):
        """Test BIDS validator factory function."""
        mock_validator = Mock()
        mock_create_validator.return_value = mock_validator
        
        bids_validator = await create_bids_validator("8.3.0")
        
        assert isinstance(bids_validator, BIDSValidator)
        mock_create_validator.assert_called_once_with("8.3.0")


if __name__ == "__main__":
    pytest.main([__file__]) 