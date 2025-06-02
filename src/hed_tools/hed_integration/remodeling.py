"""JSON-based remodeling interface for HED operations.

This module provides a JSON-driven approach to HED data remodeling operations,
allowing users to define complex data transformation pipelines through
declarative JSON templates rather than imperative code.
"""

import hashlib
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from jsonschema import ValidationError as JSONValidationError
from jsonschema import validate

from .models import OperationResult
from .schema import SchemaHandler

logger = logging.getLogger(__name__)

# JSON template schema for validation
TEMPLATE_SCHEMA = {
    "type": "object",
    "required": ["name", "operations"],
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "version": {"type": "string"},
        "author": {"type": "string"},
        "operations": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["operation", "parameters"],
                "properties": {
                    "operation": {
                        "enum": [
                            "factor_hed_tags",
                            "factor_hed_type",
                            "remap_columns",
                            "summarize_hed_tags",
                            "remove_rows",
                            "merge_consecutive_events",
                            "extract_events",
                            "filter_events",
                            "transform_columns",
                        ]
                    },
                    "parameters": {"type": "object"},
                    "required_inputs": {"type": "array", "items": {"type": "string"}},
                    "output_name": {"type": "string"},
                    "description": {"type": "string"},
                },
            },
        },
        "metadata": {"type": "object"},
    },
}


class RemodelingError(Exception):
    """Exception raised during remodeling operations."""

    pass


class ExecutionContext:
    """Context manager for operation execution state.

    Maintains data store, metadata, and execution logs for pipeline operations.
    """

    def __init__(self):
        """Initialize execution context."""
        self.data_store: Dict[str, Any] = {}
        self.metadata: Dict[str, Dict] = {}
        self.execution_log: List[Dict] = []

    def store_result(self, name: str, data: Any, metadata: Optional[Dict] = None):
        """Store operation result with optional metadata.

        Args:
            name: Name identifier for the result
            data: The result data to store
            metadata: Optional metadata about the result
        """
        self.data_store[name] = data
        if metadata:
            self.metadata[name] = metadata

    def get_data(self, name: str) -> Any:
        """Retrieve stored data by name.

        Args:
            name: Name identifier for the data

        Returns:
            The stored data or None if not found
        """
        return self.data_store.get(name)

    def log_operation(
        self, operation: str, status: str, details: Optional[Dict] = None
    ):
        """Log operation execution details.

        Args:
            operation: Name of the operation
            status: Execution status (success, error, warning)
            details: Optional additional details about the operation
        """
        self.execution_log.append(
            {
                "operation": operation,
                "status": status,
                "timestamp": datetime.now().isoformat(),
                "details": details or {},
            }
        )

    def get_execution_summary(self) -> Dict:
        """Get summary of execution context.

        Returns:
            Dictionary with execution summary statistics
        """
        return {
            "total_operations": len(self.execution_log),
            "successful_operations": len(
                [log for log in self.execution_log if log["status"] == "success"]
            ),
            "failed_operations": len(
                [log for log in self.execution_log if log["status"] == "error"]
            ),
            "data_outputs": list(self.data_store.keys()),
            "execution_log": self.execution_log,
        }


class OperationRegistry:
    """Registry for remodeling operation handlers."""

    _handlers: Dict[str, type] = {}

    @classmethod
    def register(cls, operation_name: str):
        """Decorator to register operation handlers.

        Args:
            operation_name: Name of the operation to register

        Returns:
            Decorator function
        """

        def decorator(handler_class):
            cls._handlers[operation_name] = handler_class
            return handler_class

        return decorator

    @classmethod
    def get_handler(cls, operation_name: str):
        """Get handler class for operation.

        Args:
            operation_name: Name of the operation

        Returns:
            Handler class or None if not found
        """
        return cls._handlers.get(operation_name)

    @classmethod
    def list_operations(cls) -> List[str]:
        """List all registered operations.

        Returns:
            List of operation names
        """
        return list(cls._handlers.keys())


class BaseOperationHandler:
    """Base class for operation handlers."""

    def __init__(self, schema_handler: Optional[SchemaHandler] = None):
        """Initialize operation handler.

        Args:
            schema_handler: Optional schema handler for HED operations
        """
        self.schema_handler = schema_handler or SchemaHandler()
        self.logger = logging.getLogger(self.__class__.__name__)

    async def execute(
        self, data: Any, parameters: Dict, context: ExecutionContext
    ) -> Any:
        """Execute the operation.

        Args:
            data: Input data for the operation
            parameters: Operation parameters
            context: Execution context

        Returns:
            Operation result

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Operation handlers must implement execute method")

    def validate_parameters(self, parameters: Dict) -> List[str]:
        """Validate operation parameters.

        Args:
            parameters: Parameters to validate

        Returns:
            List of validation error messages
        """
        return []  # Default implementation - no validation errors


class RemodelingInterface:
    """Main interface for JSON-based HED remodeling operations.

    Provides methods for loading JSON templates, executing operation pipelines,
    and managing results with comprehensive error handling and validation.
    """

    def __init__(
        self,
        schema_handler: Optional[SchemaHandler] = None,
        cache_dir: Optional[Path] = None,
    ):
        """Initialize the remodeling interface.

        Args:
            schema_handler: Optional schema handler for HED operations
            cache_dir: Optional directory for caching intermediate results
        """
        self.schema_handler = schema_handler or SchemaHandler()
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(__name__)

    def load_template(self, template_path: Union[str, Path]) -> Dict[str, Any]:
        """Load and validate a JSON remodeling template.

        Args:
            template_path: Path to the JSON template file

        Returns:
            Validated template dictionary

        Raises:
            FileNotFoundError: If template file doesn't exist
            JSONValidationError: If template format is invalid
            RemodelingError: If template content is invalid
        """
        try:
            template_path = Path(template_path)
            if not template_path.exists():
                raise FileNotFoundError(f"Template file not found: {template_path}")

            with open(template_path, "r", encoding="utf-8") as f:
                template = json.load(f)

            # Validate against schema
            validate(template, TEMPLATE_SCHEMA)

            # Additional validation
            validation_errors = self._validate_template_dependencies(template)
            if validation_errors:
                raise RemodelingError(
                    f"Template validation failed: {'; '.join(validation_errors)}"
                )

            self.logger.info(f"Successfully loaded template: {template.get('name')}")
            return template

        except json.JSONDecodeError as e:
            raise RemodelingError(f"Invalid JSON in template file: {e}")
        except JSONValidationError as e:
            raise RemodelingError(f"Template schema validation failed: {e.message}")

    def _validate_template_dependencies(self, template: Dict) -> List[str]:
        """Validate that operation dependencies are satisfied.

        Args:
            template: Template to validate

        Returns:
            List of validation error messages
        """
        errors = []
        available_outputs = {"input_data"}  # Always available

        for i, operation in enumerate(template["operations"]):
            # Check required inputs
            required_inputs = operation.get("required_inputs", [])
            for input_name in required_inputs:
                if input_name not in available_outputs:
                    errors.append(
                        f"Operation {i} ({operation['operation']}): "
                        f"Missing required input '{input_name}'"
                    )

            # Check if operation is registered
            operation_name = operation["operation"]
            if not OperationRegistry.get_handler(operation_name):
                errors.append(f"Operation {i}: Unknown operation '{operation_name}'")

            # Add output to available outputs
            output_name = operation.get("output_name", f"output_{i}")
            available_outputs.add(output_name)

        return errors

    async def execute_operations(
        self, template: Dict[str, Any], data: Any
    ) -> Dict[str, Any]:
        """Execute all operations in a template.

        Args:
            template: Template containing operation definitions
            data: Input data for the operations

        Returns:
            Dictionary containing execution results and context

        Raises:
            RemodelingError: If execution fails
        """
        try:
            context = ExecutionContext()
            context.store_result("input_data", data)

            self.logger.info(f"Starting execution of template: {template.get('name')}")

            for i, operation_def in enumerate(template["operations"]):
                operation_name = operation_def["operation"]
                parameters = operation_def["parameters"]
                output_name = operation_def.get("output_name", f"output_{i}")

                self.logger.info(f"Executing operation {i}: {operation_name}")

                try:
                    # Get handler
                    handler_class = OperationRegistry.get_handler(operation_name)
                    if not handler_class:
                        raise RemodelingError(
                            f"No handler found for operation: {operation_name}"
                        )

                    # Validate parameters
                    handler = handler_class(self.schema_handler)
                    param_errors = handler.validate_parameters(parameters)
                    if param_errors:
                        raise RemodelingError(
                            f"Parameter validation failed: {'; '.join(param_errors)}"
                        )

                    # Get input data
                    required_inputs = operation_def.get(
                        "required_inputs", ["input_data"]
                    )
                    if len(required_inputs) == 1:
                        input_data = context.get_data(required_inputs[0])
                    else:
                        input_data = {
                            name: context.get_data(name) for name in required_inputs
                        }

                    # Execute operation
                    result = await handler.execute(input_data, parameters, context)

                    # Store result
                    context.store_result(
                        output_name,
                        result,
                        {
                            "operation": operation_name,
                            "parameters": parameters,
                            "timestamp": datetime.now().isoformat(),
                        },
                    )

                    context.log_operation(
                        operation_name, "success", {"output_name": output_name}
                    )

                    self.logger.info(
                        f"Operation {i} ({operation_name}) completed successfully"
                    )

                except Exception as e:
                    context.log_operation(
                        operation_name, "error", {"error": str(e), "operation_index": i}
                    )
                    self.logger.error(f"Operation {i} ({operation_name}) failed: {e}")
                    raise RemodelingError(
                        f"Operation {i} ({operation_name}) failed: {e}"
                    ) from e

            return {
                "results": context.data_store,
                "metadata": context.metadata,
                "execution_summary": context.get_execution_summary(),
            }

        except Exception as e:
            self.logger.error(f"Template execution failed: {e}")
            raise RemodelingError(f"Template execution failed: {e}") from e

    async def execute_with_caching(
        self, template: Dict[str, Any], data: Any
    ) -> Dict[str, Any]:
        """Execute template with intermediate result caching.

        Args:
            template: Template containing operation definitions
            data: Input data for the operations

        Returns:
            Dictionary containing execution results and context
        """
        if not self.cache_dir:
            return await self.execute_operations(template, data)

        # Generate cache key from template
        template_str = json.dumps(template, sort_keys=True)
        cache_key = hashlib.md5(template_str.encode()).hexdigest()
        cache_file = self.cache_dir / f"remodeling_cache_{cache_key}.pkl"

        # Check cache
        if cache_file.exists():
            try:
                self.logger.info(f"Loading cached results for template: {cache_key}")
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache file: {e}")

        # Execute and cache results
        results = await self.execute_operations(template, data)

        # Save to cache
        if self.cache_dir:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                with open(cache_file, "wb") as f:
                    pickle.dump(results, f)
                self.logger.info(f"Cached results for template: {cache_key}")
            except Exception as e:
                self.logger.warning(f"Failed to save cache file: {e}")

        return results

    def save_results(
        self,
        results: Dict[str, Any],
        output_path: Union[str, Path],
        format: str = "json",
    ) -> OperationResult:
        """Save operation results to file.

        Args:
            results: Results to save
            output_path: Path where to save the results
            format: Output format ('json', 'pickle', 'csv' for DataFrames)

        Returns:
            OperationResult indicating success or failure

        Raises:
            RemodelingError: If saving fails
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if format == "json":
                # Convert DataFrames to serializable format
                serializable_results = self._make_json_serializable(results)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(serializable_results, f, indent=2)

            elif format == "pickle":
                with open(output_path, "wb") as f:
                    pickle.dump(results, f)

            elif format == "csv":
                # Save DataFrames as CSV files
                for name, data in results.get("results", {}).items():
                    if isinstance(data, pd.DataFrame):
                        csv_path = output_path.parent / f"{output_path.stem}_{name}.csv"
                        data.to_csv(csv_path, index=False)

            else:
                raise RemodelingError(f"Unsupported output format: {format}")

            self.logger.info(f"Results saved to: {output_path}")
            return OperationResult(
                success=True,
                message=f"Results saved successfully to {output_path}",
                metadata={"output_path": str(output_path), "format": format},
            )

        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            return OperationResult(
                success=False,
                message=f"Failed to save results: {e}",
                errors=[{"code": "SAVE_ERROR", "message": str(e)}],
            )

    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format.

        Args:
            obj: Object to convert

        Returns:
            JSON-serializable representation
        """
        if isinstance(obj, pd.DataFrame):
            return {
                "_type": "DataFrame",
                "data": obj.to_dict(orient="records"),
                "columns": list(obj.columns),
                "shape": obj.shape,
            }
        elif isinstance(obj, dict):
            return {
                key: self._make_json_serializable(value) for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (datetime,)):
            return obj.isoformat()
        elif hasattr(obj, "to_dict"):
            return obj.to_dict()
        else:
            return obj

    # Template generation utilities
    def generate_factor_template(
        self, columns: List[str], factor_type: str = "hed_tags"
    ) -> Dict:
        """Generate template for factoring operations.

        Args:
            columns: Columns to factor
            factor_type: Type of factoring ('hed_tags', 'hed_type')

        Returns:
            Generated template dictionary
        """
        return {
            "name": f"factor_{factor_type}_template",
            "description": f"Template for factoring {factor_type}",
            "version": "1.0",
            "operations": [
                {
                    "operation": f"factor_{factor_type}",
                    "parameters": {
                        "columns": columns,
                        "remove_types": ["Condition-variable", "Task"],
                    },
                    "required_inputs": ["input_data"],
                    "output_name": f"factored_{factor_type}",
                    "description": f"Factor {factor_type} from specified columns",
                }
            ],
        }

    def generate_pipeline_template(self, operations_sequence: List[Dict]) -> Dict:
        """Generate multi-step pipeline template.

        Args:
            operations_sequence: List of operation definitions

        Returns:
            Generated pipeline template
        """
        operations = []
        for i, op_def in enumerate(operations_sequence):
            operation = {
                "operation": op_def["operation"],
                "parameters": op_def.get("parameters", {}),
                "required_inputs": ["input_data"] if i == 0 else [f"output_{i - 1}"],
                "output_name": f"output_{i}",
                "description": op_def.get("description", f"Step {i + 1}"),
            }
            operations.append(operation)

        return {
            "name": "pipeline_template",
            "description": "Multi-step remodeling pipeline",
            "version": "1.0",
            "operations": operations,
        }


# Factory function for easy instantiation
async def create_remodeling_interface(
    schema_version: Optional[str] = None, cache_dir: Optional[Path] = None
) -> RemodelingInterface:
    """Create a remodeling interface with optional schema version.

    Args:
        schema_version: HED schema version to use
        cache_dir: Optional cache directory for intermediate results

    Returns:
        Configured RemodelingInterface instance
    """
    schema_handler = SchemaHandler()
    if schema_version:
        await schema_handler.load_schema(schema_version)

    return RemodelingInterface(schema_handler=schema_handler, cache_dir=cache_dir)


@OperationRegistry.register("factor_hed_tags")
class FactorHedTagsHandler(BaseOperationHandler):
    """Handler for factoring HED tags from events."""

    async def execute(
        self, data: Any, parameters: Dict, context: ExecutionContext
    ) -> pd.DataFrame:
        """Factor HED tags from events data.

        Args:
            data: Input DataFrame with HED annotations
            parameters: Operation parameters
            context: Execution context

        Returns:
            DataFrame with factored HED tags as separate columns
        """
        if not isinstance(data, pd.DataFrame):
            raise RemodelingError("factor_hed_tags requires DataFrame input")

        df = data.copy()
        columns = parameters.get("columns", ["HED"])
        remove_types = parameters.get("remove_types", [])

        for column in columns:
            if column not in df.columns:
                self.logger.warning(f"Column {column} not found in data")
                continue

            # Extract HED tags and create new columns
            # This is a simplified implementation - real implementation would use HED library
            for idx, hed_string in df[column].fillna("").items():
                if hed_string:
                    # Parse HED string and extract factored components
                    factored_tags = self._extract_hed_factors(hed_string, remove_types)
                    for tag_type, tag_value in factored_tags.items():
                        factor_column = f"{column}_{tag_type}"
                        if factor_column not in df.columns:
                            df[factor_column] = ""
                        df.loc[idx, factor_column] = tag_value

        return df

    def _extract_hed_factors(
        self, hed_string: str, remove_types: List[str]
    ) -> Dict[str, str]:
        """Extract factored HED tags from a HED string.

        Args:
            hed_string: HED annotation string
            remove_types: Tag types to remove during factoring

        Returns:
            Dictionary of factored tag types and values
        """
        # Simplified extraction logic - real implementation would use HED parsing
        factors = {}

        # Basic pattern matching for common HED structures
        if "Event" in hed_string:
            factors["Event"] = "Event"
        if (
            "Condition-variable" in hed_string
            and "Condition-variable" not in remove_types
        ):
            factors["Condition"] = "Variable"
        if "Task" in hed_string and "Task" not in remove_types:
            factors["Task"] = "Task"

        return factors

    def validate_parameters(self, parameters: Dict) -> List[str]:
        """Validate parameters for factor_hed_tags operation."""
        errors = []

        if "columns" in parameters and not isinstance(parameters["columns"], list):
            errors.append("'columns' parameter must be a list")

        if "remove_types" in parameters and not isinstance(
            parameters["remove_types"], list
        ):
            errors.append("'remove_types' parameter must be a list")

        return errors


@OperationRegistry.register("remap_columns")
class RemapColumnsHandler(BaseOperationHandler):
    """Handler for remapping column values."""

    async def execute(
        self, data: Any, parameters: Dict, context: ExecutionContext
    ) -> pd.DataFrame:
        """Remap column values according to mapping rules.

        Args:
            data: Input DataFrame
            parameters: Operation parameters including column mappings
            context: Execution context

        Returns:
            DataFrame with remapped column values
        """
        if not isinstance(data, pd.DataFrame):
            raise RemodelingError("remap_columns requires DataFrame input")

        df = data.copy()
        column_map = parameters.get("column_map", {})
        value_map = parameters.get("value_map", {})

        # Remap column names
        if column_map:
            df = df.rename(columns=column_map)

        # Remap column values
        for column, mapping in value_map.items():
            if column in df.columns:
                df[column] = df[column].map(mapping).fillna(df[column])

        return df

    def validate_parameters(self, parameters: Dict) -> List[str]:
        """Validate parameters for remap_columns operation."""
        errors = []

        if "column_map" in parameters and not isinstance(
            parameters["column_map"], dict
        ):
            errors.append("'column_map' parameter must be a dictionary")

        if "value_map" in parameters and not isinstance(parameters["value_map"], dict):
            errors.append("'value_map' parameter must be a dictionary")

        return errors


@OperationRegistry.register("filter_events")
class FilterEventsHandler(BaseOperationHandler):
    """Handler for filtering events based on conditions."""

    async def execute(
        self, data: Any, parameters: Dict, context: ExecutionContext
    ) -> pd.DataFrame:
        """Filter events based on specified conditions.

        Args:
            data: Input DataFrame
            parameters: Operation parameters including filter conditions
            context: Execution context

        Returns:
            Filtered DataFrame
        """
        if not isinstance(data, pd.DataFrame):
            raise RemodelingError("filter_events requires DataFrame input")

        df = data.copy()
        conditions = parameters.get("conditions", [])

        for condition in conditions:
            column = condition.get("column")
            operator = condition.get("operator", "==")
            value = condition.get("value")

            if column not in df.columns:
                self.logger.warning(f"Filter column {column} not found in data")
                continue

            # Apply filter condition
            if operator == "==":
                mask = df[column] == value
            elif operator == "!=":
                mask = df[column] != value
            elif operator == "in":
                mask = df[column].isin(value if isinstance(value, list) else [value])
            elif operator == "not_in":
                mask = ~df[column].isin(value if isinstance(value, list) else [value])
            elif operator == ">":
                mask = df[column] > value
            elif operator == "<":
                mask = df[column] < value
            elif operator == ">=":
                mask = df[column] >= value
            elif operator == "<=":
                mask = df[column] <= value
            elif operator == "contains":
                mask = df[column].astype(str).str.contains(str(value), na=False)
            else:
                self.logger.warning(f"Unknown operator: {operator}")
                continue

            df = df[mask]

        return df

    def validate_parameters(self, parameters: Dict) -> List[str]:
        """Validate parameters for filter_events operation."""
        errors = []

        conditions = parameters.get("conditions", [])
        if not isinstance(conditions, list):
            errors.append("'conditions' parameter must be a list")
        else:
            for i, condition in enumerate(conditions):
                if not isinstance(condition, dict):
                    errors.append(f"Condition {i} must be a dictionary")
                    continue

                if "column" not in condition:
                    errors.append(f"Condition {i} missing required 'column' field")

                if "value" not in condition:
                    errors.append(f"Condition {i} missing required 'value' field")

        return errors


@OperationRegistry.register("summarize_hed_tags")
class SummarizeHedTagsHandler(BaseOperationHandler):
    """Handler for summarizing HED tags."""

    async def execute(
        self, data: Any, parameters: Dict, context: ExecutionContext
    ) -> Dict[str, Any]:
        """Summarize HED tags from events data.

        Args:
            data: Input DataFrame with HED annotations
            parameters: Operation parameters
            context: Execution context

        Returns:
            Dictionary containing HED tag summary statistics
        """
        if not isinstance(data, pd.DataFrame):
            raise RemodelingError("summarize_hed_tags requires DataFrame input")

        columns = parameters.get("columns", ["HED"])
        summary = {
            "total_events": len(data),
            "hed_columns": columns,
            "tag_statistics": {},
            "unique_tags": set(),
        }

        for column in columns:
            if column not in data.columns:
                continue

            column_summary = {
                "total_annotations": data[column].notna().sum(),
                "unique_annotations": data[column].nunique(),
                "most_common": data[column].value_counts().head(10).to_dict(),
            }

            # Extract individual HED tags
            all_tags = []
            for hed_string in data[column].fillna(""):
                if hed_string:
                    tags = self._extract_tags(hed_string)
                    all_tags.extend(tags)
                    summary["unique_tags"].update(tags)

            column_summary["total_tags"] = len(all_tags)
            column_summary["unique_tag_count"] = len(set(all_tags))

            summary["tag_statistics"][column] = column_summary

        # Convert set to list for JSON serialization
        summary["unique_tags"] = list(summary["unique_tags"])

        return summary

    def _extract_tags(self, hed_string: str) -> List[str]:
        """Extract individual HED tags from a HED string.

        Args:
            hed_string: HED annotation string

        Returns:
            List of individual HED tags
        """
        # Simplified tag extraction - real implementation would use HED parsing
        # Remove parentheses and split by commas
        cleaned = hed_string.replace("(", "").replace(")", "")
        tags = [tag.strip() for tag in cleaned.split(",") if tag.strip()]
        return tags

    def validate_parameters(self, parameters: Dict) -> List[str]:
        """Validate parameters for summarize_hed_tags operation."""
        errors = []

        if "columns" in parameters and not isinstance(parameters["columns"], list):
            errors.append("'columns' parameter must be a list")

        return errors


@OperationRegistry.register("merge_consecutive_events")
class MergeConsecutiveEventsHandler(BaseOperationHandler):
    """Handler for merging consecutive events with same properties."""

    async def execute(
        self, data: Any, parameters: Dict, context: ExecutionContext
    ) -> pd.DataFrame:
        """Merge consecutive events that have the same specified properties.

        Args:
            data: Input DataFrame with events
            parameters: Operation parameters
            context: Execution context

        Returns:
            DataFrame with merged consecutive events
        """
        if not isinstance(data, pd.DataFrame):
            raise RemodelingError("merge_consecutive_events requires DataFrame input")

        df = data.copy().sort_values("onset")
        merge_columns = parameters.get("merge_columns", ["trial_type"])
        tolerance = parameters.get("tolerance", 0.0)

        if not all(col in df.columns for col in merge_columns):
            missing = [col for col in merge_columns if col not in df.columns]
            raise RemodelingError(f"Missing columns for merging: {missing}")

        merged_rows = []
        current_group = None

        for _, row in df.iterrows():
            if current_group is None:
                current_group = [row]
            else:
                # Check if this row can be merged with current group
                last_row = current_group[-1]

                # Check if properties match
                properties_match = all(
                    row[col] == last_row[col] for col in merge_columns
                )

                # Check if events are consecutive (within tolerance)
                time_consecutive = (
                    row["onset"] <= last_row["onset"] + last_row["duration"] + tolerance
                )

                if properties_match and time_consecutive:
                    current_group.append(row)
                else:
                    # Finalize current group and start new one
                    merged_rows.append(self._merge_group(current_group))
                    current_group = [row]

        # Don't forget the last group
        if current_group:
            merged_rows.append(self._merge_group(current_group))

        return pd.DataFrame(merged_rows)

    def _merge_group(self, group: List[pd.Series]) -> Dict:
        """Merge a group of consecutive events.

        Args:
            group: List of pandas Series representing events to merge

        Returns:
            Dictionary representing the merged event
        """
        if len(group) == 1:
            return group[0].to_dict()

        merged = group[0].copy()

        # Update onset and duration
        merged["onset"] = group[0]["onset"]
        merged["duration"] = (
            group[-1]["onset"] + group[-1]["duration"] - group[0]["onset"]
        )

        # Merge HED annotations if present
        if "HED" in merged.index:
            hed_strings = [row["HED"] for row in group if pd.notna(row["HED"])]
            if hed_strings:
                merged["HED"] = ", ".join(hed_strings)

        return merged.to_dict()

    def validate_parameters(self, parameters: Dict) -> List[str]:
        """Validate parameters for merge_consecutive_events operation."""
        errors = []

        if "merge_columns" in parameters and not isinstance(
            parameters["merge_columns"], list
        ):
            errors.append("'merge_columns' parameter must be a list")

        if "tolerance" in parameters and not isinstance(
            parameters["tolerance"], (int, float)
        ):
            errors.append("'tolerance' parameter must be a number")

        return errors
