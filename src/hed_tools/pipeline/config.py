"""Pipeline configuration management.

This module provides configuration classes for the HED sidecar generation pipeline,
including support for MCP request parameter mapping and stage-specific configurations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class StageConfig:
    """Configuration for individual pipeline stages."""

    # General stage configuration
    enabled: bool = True
    timeout_seconds: int = 0  # 0 means no timeout
    retry_count: int = 0
    continue_on_error: bool = False

    # Stage-specific parameters
    stage_params: Dict[str, Any] = field(default_factory=dict)

    def get_param(self, key: str, default: Any = None) -> Any:
        """Get a stage-specific parameter."""
        return self.stage_params.get(key, default)

    def set_param(self, key: str, value: Any) -> None:
        """Set a stage-specific parameter."""
        self.stage_params[key] = value


@dataclass
class PipelineConfig:
    """Main configuration for the HED sidecar generation pipeline.

    This class centralizes all configuration options and provides methods
    to create configurations from MCP requests and other sources.
    """

    # Schema and validation settings
    hed_schema_version: str = "8.3.0"
    validation_enabled: bool = True
    schema_validation_strict: bool = False

    # Execution settings
    async_execution: bool = True
    timeout_seconds: int = 300  # 5 minutes default
    max_retries: int = 2
    continue_on_error: bool = False

    # Performance settings
    max_workers: int = 4
    batch_size: int = 10
    memory_limit_mb: int = 1024  # 1GB default
    cache_enabled: bool = True

    # Output settings
    output_format: str = "json"  # json, yaml
    include_descriptions: bool = True
    include_examples: bool = False
    include_metadata: bool = True

    # Debugging and logging
    debug_mode: bool = False
    log_level: str = "INFO"
    save_intermediate_results: bool = False

    # Stage-specific configurations
    stage_configs: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "data_ingestion": {
                "validate_file_format": True,
                "max_file_size_mb": 100,
                "encoding": "utf-8",
                "delimiter": "auto",  # auto-detect, or specify "\t", ","
            },
            "column_classification": {
                "confidence_threshold": 0.7,
                "use_llm_enhancement": True,
                "fallback_to_heuristics": True,
                "sample_size": 100,
            },
            "hed_mapping": {
                "auto_suggest_tags": True,
                "include_tag_descriptions": True,
                "use_tabular_summary": True,
                "mapping_strategy": "intelligent",  # basic, intelligent, comprehensive
            },
            "sidecar_generation": {
                "template_format": "json",
                "include_column_metadata": True,
                "generate_value_mapping": True,
                "sort_columns": True,
            },
            "validation": {
                "validate_schema_compatibility": True,
                "validate_tag_syntax": True,
                "allow_warnings": True,
                "strict_mode": False,
            },
        }
    )

    @classmethod
    def from_mcp_request(cls, request_data: Dict[str, Any]) -> "PipelineConfig":
        """Create pipeline configuration from MCP tool request parameters.

        Args:
            request_data: Dictionary of parameters from MCP tool call

        Returns:
            PipelineConfig instance with mapped parameters
        """
        config = cls()

        # Map common MCP parameters to config fields
        config.hed_schema_version = request_data.get(
            "schema_version", config.hed_schema_version
        )
        config.validation_enabled = request_data.get(
            "validate_schema_compatibility", config.validation_enabled
        )
        config.output_format = request_data.get(
            "output_format", config.output_format
        ).lower()
        config.include_descriptions = request_data.get(
            "include_descriptions", config.include_descriptions
        )
        config.include_examples = request_data.get(
            "include_examples", config.include_examples
        )

        # Map advanced parameters
        config.async_execution = request_data.get(
            "async_execution", config.async_execution
        )
        config.timeout_seconds = request_data.get(
            "timeout_seconds", config.timeout_seconds
        )
        config.debug_mode = request_data.get("debug", config.debug_mode)

        # Update stage configurations with request parameters
        config._update_stage_configs_from_request(request_data)

        logger.info(
            f"Created pipeline config from MCP request: schema={config.hed_schema_version}, "
            f"format={config.output_format}, validation={config.validation_enabled}"
        )

        return config

    def _update_stage_configs_from_request(self, request_data: Dict[str, Any]) -> None:
        """Update stage configurations with parameters from MCP request."""

        # Data ingestion parameters
        if "max_file_size_mb" in request_data:
            self.stage_configs["data_ingestion"]["max_file_size_mb"] = request_data[
                "max_file_size_mb"
            ]

        if "file_encoding" in request_data:
            self.stage_configs["data_ingestion"]["encoding"] = request_data[
                "file_encoding"
            ]

        # Column classification parameters
        if "confidence_threshold" in request_data:
            self.stage_configs["column_classification"]["confidence_threshold"] = (
                request_data["confidence_threshold"]
            )

        if "use_llm_enhancement" in request_data:
            self.stage_configs["column_classification"]["use_llm_enhancement"] = (
                request_data["use_llm_enhancement"]
            )

        # HED mapping parameters
        if "auto_suggest_tags" in request_data:
            self.stage_configs["hed_mapping"]["auto_suggest_tags"] = request_data[
                "auto_suggest_tags"
            ]

        if "mapping_strategy" in request_data:
            self.stage_configs["hed_mapping"]["mapping_strategy"] = request_data[
                "mapping_strategy"
            ]

        # Sidecar generation parameters
        if "include_column_metadata" in request_data:
            self.stage_configs["sidecar_generation"]["include_column_metadata"] = (
                request_data["include_column_metadata"]
            )

        # Validation parameters
        if "strict_validation" in request_data:
            self.stage_configs["validation"]["strict_mode"] = request_data[
                "strict_validation"
            ]

    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "PipelineConfig":
        """Load configuration from a file.

        Args:
            config_path: Path to configuration file (JSON or YAML)

        Returns:
            PipelineConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file format is invalid
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            if config_path.suffix.lower() == ".json":
                import json

                with open(config_path, "r") as f:
                    data = json.load(f)
            elif config_path.suffix.lower() in [".yaml", ".yml"]:
                import yaml

                with open(config_path, "r") as f:
                    data = yaml.safe_load(f)
            else:
                raise ValueError(
                    f"Unsupported config file format: {config_path.suffix}"
                )

            return cls.from_dict(data)

        except Exception as e:
            raise ValueError(f"Failed to load configuration from {config_path}: {e}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Create configuration from dictionary.

        Args:
            data: Dictionary containing configuration data

        Returns:
            PipelineConfig instance
        """
        config = cls()

        # Update fields that exist in the dataclass
        for field_name in config.__dataclass_fields__:
            if field_name in data:
                setattr(config, field_name, data[field_name])

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration
        """
        result = {}
        for field_name in self.__dataclass_fields__:
            result[field_name] = getattr(self, field_name)
        return result

    def save_to_file(self, config_path: Union[str, Path]) -> None:
        """Save configuration to file.

        Args:
            config_path: Path where to save the configuration
        """
        config_path = Path(config_path)
        data = self.to_dict()

        if config_path.suffix.lower() == ".json":
            import json

            with open(config_path, "w") as f:
                json.dump(data, f, indent=2)
        elif config_path.suffix.lower() in [".yaml", ".yml"]:
            import yaml

            with open(config_path, "w") as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

        logger.info(f"Saved pipeline configuration to {config_path}")

    def get_stage_config(self, stage_name: str) -> StageConfig:
        """Get configuration for a specific stage.

        Args:
            stage_name: Name of the stage

        Returns:
            StageConfig instance
        """
        stage_data = self.stage_configs.get(stage_name, {})

        # Extract general stage settings
        general_settings = {
            "enabled": stage_data.get("enabled", True),
            "timeout_seconds": stage_data.get("timeout_seconds", 0),
            "retry_count": stage_data.get("retry_count", 0),
            "continue_on_error": stage_data.get("continue_on_error", False),
        }

        # Everything else goes into stage_params
        stage_params = {
            k: v for k, v in stage_data.items() if k not in general_settings
        }

        return StageConfig(**general_settings, stage_params=stage_params)

    def update_stage_config(self, stage_name: str, updates: Dict[str, Any]) -> None:
        """Update configuration for a specific stage.

        Args:
            stage_name: Name of the stage to update
            updates: Dictionary of updates to apply
        """
        if stage_name not in self.stage_configs:
            self.stage_configs[stage_name] = {}

        self.stage_configs[stage_name].update(updates)
        logger.debug(f"Updated configuration for stage '{stage_name}': {updates}")

    def validate(self) -> List[str]:
        """Validate the configuration and return any errors.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Validate basic settings
        if self.timeout_seconds < 0:
            errors.append("timeout_seconds must be non-negative")

        if self.max_retries < 0:
            errors.append("max_retries must be non-negative")

        if self.max_workers < 1:
            errors.append("max_workers must be at least 1")

        if self.batch_size < 1:
            errors.append("batch_size must be at least 1")

        if self.memory_limit_mb < 1:
            errors.append("memory_limit_mb must be at least 1")

        # Validate output format
        if self.output_format not in ["json", "yaml"]:
            errors.append(
                f"output_format must be 'json' or 'yaml', got '{self.output_format}'"
            )

        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            errors.append(
                f"log_level must be one of {valid_log_levels}, got '{self.log_level}'"
            )

        # Validate HED schema version format (basic check)
        if not self.hed_schema_version or not isinstance(self.hed_schema_version, str):
            errors.append("hed_schema_version must be a non-empty string")

        return errors

    def is_valid(self) -> bool:
        """Check if the configuration is valid.

        Returns:
            True if configuration is valid, False otherwise
        """
        return len(self.validate()) == 0

    def __post_init__(self):
        """Post-initialization validation."""
        errors = self.validate()
        if errors:
            logger.warning(f"Configuration validation warnings: {errors}")


def create_default_config() -> PipelineConfig:
    """Create a default pipeline configuration.

    Returns:
        PipelineConfig with sensible defaults
    """
    return PipelineConfig()


def create_development_config() -> PipelineConfig:
    """Create a configuration optimized for development/testing.

    Returns:
        PipelineConfig with development-friendly settings
    """
    config = PipelineConfig()
    config.debug_mode = True
    config.log_level = "DEBUG"
    config.save_intermediate_results = True
    config.timeout_seconds = 60  # Shorter timeout for dev
    config.validation_enabled = True  # Always validate in dev
    config.schema_validation_strict = True  # Strict validation in dev

    return config


def create_production_config() -> PipelineConfig:
    """Create a configuration optimized for production use.

    Returns:
        PipelineConfig with production-ready settings
    """
    config = PipelineConfig()
    config.debug_mode = False
    config.log_level = "INFO"
    config.save_intermediate_results = False
    config.timeout_seconds = 300  # Longer timeout for production
    config.max_retries = 3  # More retries in production
    config.cache_enabled = True  # Enable caching in production
    config.memory_limit_mb = 2048  # More memory in production

    return config
