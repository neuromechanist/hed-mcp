"""Unit tests for JSON-based remodeling interface."""

import json
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import AsyncMock, patch
import tempfile

from hed_tools.hed_integration.remodeling import (
    RemodelingInterface,
    RemodelingError,
    OperationRegistry,
    ExecutionContext,
    FactorHedTagsHandler,
    RemapColumnsHandler,
    FilterEventsHandler,
    SummarizeHedTagsHandler,
    MergeConsecutiveEventsHandler,
    create_remodeling_interface,
    TEMPLATE_SCHEMA,
)


class TestExecutionContext:
    """Test the ExecutionContext class."""

    def test_store_and_retrieve_data(self):
        """Test storing and retrieving data."""
        context = ExecutionContext()

        test_data = {"key": "value"}
        context.store_result("test", test_data, {"metadata": "info"})

        assert context.get_data("test") == test_data
        assert context.metadata["test"] == {"metadata": "info"}

    def test_execution_logging(self):
        """Test operation execution logging."""
        context = ExecutionContext()

        context.log_operation("test_op", "success", {"details": "completed"})
        context.log_operation("test_op2", "error", {"error": "failed"})

        summary = context.get_execution_summary()
        assert summary["total_operations"] == 2
        assert summary["successful_operations"] == 1
        assert summary["failed_operations"] == 1


class TestOperationRegistry:
    """Test the OperationRegistry functionality."""

    def test_operation_registration(self):
        """Test registering operations."""

        @OperationRegistry.register("test_operation")
        class TestHandler:
            pass

        assert OperationRegistry.get_handler("test_operation") == TestHandler
        assert "test_operation" in OperationRegistry.list_operations()

    def test_unknown_operation(self):
        """Test handling unknown operations."""
        assert OperationRegistry.get_handler("unknown_operation") is None


class TestFactorHedTagsHandler:
    """Test the FactorHedTagsHandler."""

    @pytest.fixture
    def handler(self):
        """Create a FactorHedTagsHandler instance."""
        return FactorHedTagsHandler()

    @pytest.fixture
    def sample_data(self):
        """Create sample data with HED annotations."""
        return pd.DataFrame(
            {
                "onset": [0.0, 1.0, 2.0],
                "duration": [0.5, 0.5, 0.5],
                "trial_type": ["go", "nogo", "go"],
                "HED": [
                    "Event/Category/Go, Task/Go-NoGo",
                    "Event/Category/NoGo, Condition-variable/Stop",
                    "Event/Category/Go, Task/Go-NoGo",
                ],
            }
        )

    @pytest.mark.asyncio
    async def test_factor_hed_tags(self, handler, sample_data):
        """Test factoring HED tags."""
        context = ExecutionContext()
        parameters = {"columns": ["HED"], "remove_types": ["Task"]}

        result = await handler.execute(sample_data, parameters, context)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)
        # Check that factored columns were created
        assert "HED_Event" in result.columns

    @pytest.mark.asyncio
    async def test_factor_hed_tags_invalid_input(self, handler):
        """Test with invalid input data."""
        context = ExecutionContext()
        parameters = {"columns": ["HED"]}

        with pytest.raises(RemodelingError):
            await handler.execute("invalid_data", parameters, context)

    def test_parameter_validation(self, handler):
        """Test parameter validation."""
        # Valid parameters
        errors = handler.validate_parameters(
            {"columns": ["HED"], "expand_context": True}
        )
        assert len(errors) == 0

        # Invalid parameters
        errors = handler.validate_parameters({"columns": "not_a_list"})
        assert len(errors) > 0


class TestRemapColumnsHandler:
    """Test the RemapColumnsHandler."""

    @pytest.fixture
    def handler(self):
        """Create a RemapColumnsHandler instance."""
        return RemapColumnsHandler()

    @pytest.fixture
    def sample_data(self):
        """Create sample data for remapping."""
        return pd.DataFrame(
            {
                "onset": [0.0, 1.0, 2.0],
                "old_column": ["a", "b", "c"],
                "response": ["left", "right", "left"],
            }
        )

    @pytest.mark.asyncio
    async def test_remap_columns(self, handler, sample_data):
        """Test remapping column names and values."""
        context = ExecutionContext()
        parameters = {
            "column_map": {"old_column": "new_column"},
            "value_map": {"response": {"left": "L", "right": "R"}},
        }

        result = await handler.execute(sample_data, parameters, context)

        assert "new_column" in result.columns
        assert "old_column" not in result.columns
        assert list(result["response"]) == ["L", "R", "L"]

    @pytest.mark.asyncio
    async def test_remap_columns_invalid_input(self, handler):
        """Test with invalid input data."""
        context = ExecutionContext()
        parameters = {"column_map": {}}

        with pytest.raises(RemodelingError):
            await handler.execute("invalid_data", parameters, context)


class TestFilterEventsHandler:
    """Test the FilterEventsHandler."""

    @pytest.fixture
    def handler(self):
        """Create a FilterEventsHandler instance."""
        return FilterEventsHandler()

    @pytest.fixture
    def sample_data(self):
        """Create sample data for filtering."""
        return pd.DataFrame(
            {
                "onset": [0.0, 1.0, 2.0, 3.0],
                "trial_type": ["go", "nogo", "go", "stop"],
                "accuracy": [1, 0, 1, 1],
                "response_time": [0.5, 0.8, 0.4, 0.6],
            }
        )

    @pytest.mark.asyncio
    async def test_filter_events_equals(self, handler, sample_data):
        """Test filtering with equals condition."""
        context = ExecutionContext()
        parameters = {
            "conditions": [{"column": "trial_type", "operator": "==", "value": "go"}]
        }

        result = await handler.execute(sample_data, parameters, context)

        assert len(result) == 2
        assert all(result["trial_type"] == "go")

    @pytest.mark.asyncio
    async def test_filter_events_in(self, handler, sample_data):
        """Test filtering with 'in' condition."""
        context = ExecutionContext()
        parameters = {
            "conditions": [
                {"column": "trial_type", "operator": "in", "value": ["go", "nogo"]}
            ]
        }

        result = await handler.execute(sample_data, parameters, context)

        assert len(result) == 3
        assert all(trial_type in ["go", "nogo"] for trial_type in result["trial_type"])

    @pytest.mark.asyncio
    async def test_filter_events_numeric(self, handler, sample_data):
        """Test filtering with numeric conditions."""
        context = ExecutionContext()
        parameters = {
            "conditions": [{"column": "response_time", "operator": "<", "value": 0.6}]
        }

        result = await handler.execute(sample_data, parameters, context)

        assert len(result) == 2
        assert all(rt < 0.6 for rt in result["response_time"])


class TestSummarizeHedTagsHandler:
    """Test the SummarizeHedTagsHandler."""

    @pytest.fixture
    def handler(self):
        """Create a SummarizeHedTagsHandler instance."""
        return SummarizeHedTagsHandler()

    @pytest.fixture
    def sample_data(self):
        """Create sample data with HED annotations."""
        return pd.DataFrame(
            {
                "onset": [0.0, 1.0, 2.0],
                "HED": [
                    "Event/Category/Go, Task/Go-NoGo",
                    "Event/Category/NoGo, Condition-variable/Stop",
                    "Event/Category/Go, Task/Go-NoGo",
                ],
            }
        )

    @pytest.mark.asyncio
    async def test_summarize_hed_tags(self, handler, sample_data):
        """Test summarizing HED tags."""
        context = ExecutionContext()
        parameters = {"columns": ["HED"]}

        result = await handler.execute(sample_data, parameters, context)

        assert result["total_events"] == 3
        assert "HED" in result["tag_statistics"]
        assert result["tag_statistics"]["HED"]["total_annotations"] == 3
        assert len(result["unique_tags"]) > 0


class TestMergeConsecutiveEventsHandler:
    """Test the MergeConsecutiveEventsHandler."""

    @pytest.fixture
    def handler(self):
        """Create a MergeConsecutiveEventsHandler instance."""
        return MergeConsecutiveEventsHandler()

    @pytest.fixture
    def sample_data(self):
        """Create sample data with consecutive events."""
        return pd.DataFrame(
            {
                "onset": [0.0, 0.5, 1.0, 2.0, 2.5],
                "duration": [0.5, 0.5, 1.0, 0.5, 0.5],
                "trial_type": ["go", "go", "nogo", "go", "go"],
                "HED": ["Event/Go", "Event/Go", "Event/NoGo", "Event/Go", "Event/Go"],
            }
        )

    @pytest.mark.asyncio
    async def test_merge_consecutive_events(self, handler, sample_data):
        """Test merging consecutive events."""
        context = ExecutionContext()
        parameters = {"merge_columns": ["trial_type"], "tolerance": 0.0}

        result = await handler.execute(sample_data, parameters, context)

        # Should merge the first two 'go' events and the last two 'go' events
        assert len(result) == 3

        # Check first merged event
        first_event = result.iloc[0]
        assert first_event["trial_type"] == "go"
        assert first_event["onset"] == 0.0
        assert first_event["duration"] == 1.0  # 0.5 + 0.5


class TestRemodelingInterface:
    """Test the main RemodelingInterface class."""

    @pytest.fixture
    def interface(self):
        """Create a RemodelingInterface instance."""
        return RemodelingInterface()

    @pytest.fixture
    def sample_template(self):
        """Create a sample JSON template."""
        return {
            "name": "test_template",
            "description": "Test template for unit tests",
            "version": "1.0",
            "operations": [
                {
                    "operation": "filter_events",
                    "parameters": {
                        "conditions": [
                            {"column": "trial_type", "operator": "==", "value": "go"}
                        ]
                    },
                    "required_inputs": ["input_data"],
                    "output_name": "filtered_data",
                    "description": "Filter only go trials",
                }
            ],
        }

    @pytest.fixture
    def sample_data(self):
        """Create sample events data."""
        return pd.DataFrame(
            {
                "onset": [0.0, 1.0, 2.0, 3.0],
                "duration": [0.5, 0.5, 0.5, 0.5],
                "trial_type": ["go", "nogo", "go", "stop"],
            }
        )

    def test_load_template_valid(self, interface, sample_template):
        """Test loading a valid template."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_template, f)
            template_path = f.name

        try:
            loaded_template = interface.load_template(template_path)
            assert loaded_template == sample_template
        finally:
            Path(template_path).unlink()

    def test_load_template_invalid_json(self, interface):
        """Test loading invalid JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content")
            template_path = f.name

        try:
            with pytest.raises(RemodelingError):
                interface.load_template(template_path)
        finally:
            Path(template_path).unlink()

    def test_load_template_missing_file(self, interface):
        """Test loading non-existent template file."""
        with pytest.raises(FileNotFoundError):
            interface.load_template("non_existent_file.json")

    @pytest.mark.asyncio
    async def test_execute_operations(self, interface, sample_template, sample_data):
        """Test executing operations from a template."""
        results = await interface.execute_operations(sample_template, sample_data)

        assert "results" in results
        assert "filtered_data" in results["results"]
        assert "execution_summary" in results

        # Check that filtering worked
        filtered_data = results["results"]["filtered_data"]
        assert len(filtered_data) == 2  # Only 'go' trials
        assert all(trial_type == "go" for trial_type in filtered_data["trial_type"])

    @pytest.mark.asyncio
    async def test_execute_operations_unknown_operation(self, interface, sample_data):
        """Test executing template with unknown operation."""
        template = {
            "name": "invalid_template",
            "operations": [
                {
                    "operation": "unknown_operation",
                    "parameters": {},
                    "required_inputs": ["input_data"],
                    "output_name": "output",
                }
            ],
        }

        with pytest.raises(RemodelingError):
            await interface.execute_operations(template, sample_data)

    def test_save_results_json(self, interface):
        """Test saving results in JSON format."""
        results = {
            "results": {"output": pd.DataFrame({"a": [1, 2, 3]})},
            "metadata": {"test": "data"},
        }

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            result = interface.save_results(results, output_path, "json")
            assert result.success

            # Verify file was created and is valid JSON
            with open(output_path, "r") as f:
                saved_data = json.load(f)
            assert "results" in saved_data
        finally:
            Path(output_path).unlink()

    def test_save_results_pickle(self, interface):
        """Test saving results in pickle format."""
        results = {
            "results": {"output": pd.DataFrame({"a": [1, 2, 3]})},
            "metadata": {"test": "data"},
        }

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            output_path = f.name

        try:
            result = interface.save_results(results, output_path, "pickle")
            assert result.success

            # Verify file was created
            assert Path(output_path).exists()
        finally:
            Path(output_path).unlink()

    def test_generate_factor_template(self, interface):
        """Test generating factor operation template."""
        template = interface.generate_factor_template(["HED"], "hed_tags")

        assert template["name"] == "factor_hed_tags_template"
        assert len(template["operations"]) == 1
        assert template["operations"][0]["operation"] == "factor_hed_tags"

    def test_generate_pipeline_template(self, interface):
        """Test generating pipeline template."""
        operations_sequence = [
            {"operation": "filter_events", "parameters": {"conditions": []}},
            {"operation": "remap_columns", "parameters": {"column_map": {}}},
        ]

        template = interface.generate_pipeline_template(operations_sequence)

        assert template["name"] == "pipeline_template"
        assert len(template["operations"]) == 2
        assert template["operations"][0]["required_inputs"] == ["input_data"]
        assert template["operations"][1]["required_inputs"] == ["output_0"]


@pytest.mark.asyncio
async def test_create_remodeling_interface():
    """Test the factory function for creating remodeling interface."""
    with patch("hed_tools.hed_integration.remodeling.SchemaHandler") as mock_schema:
        mock_schema_instance = AsyncMock()
        mock_schema.return_value = mock_schema_instance

        interface = await create_remodeling_interface("8.1.0")

        assert isinstance(interface, RemodelingInterface)
        mock_schema_instance.load_schema.assert_called_once_with("8.1.0")


class TestTemplateValidation:
    """Test JSON template validation."""

    def test_valid_template_schema(self):
        """Test that the template schema validates correct templates."""
        from jsonschema import validate

        valid_template = {
            "name": "test_template",
            "operations": [
                {
                    "operation": "filter_events",
                    "parameters": {"conditions": []},
                    "required_inputs": ["input_data"],
                    "output_name": "filtered",
                }
            ],
        }

        # Should not raise an exception
        validate(valid_template, TEMPLATE_SCHEMA)

    def test_invalid_template_schema(self):
        """Test that the template schema rejects invalid templates."""
        from jsonschema import ValidationError, validate

        invalid_template = {
            "name": "test_template",
            # Missing required 'operations' field
        }

        with pytest.raises(ValidationError):
            validate(invalid_template, TEMPLATE_SCHEMA)
