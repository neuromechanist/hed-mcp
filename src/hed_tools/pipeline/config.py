"""Configuration management for HED sidecar generation pipeline.

This module provides configuration classes for:
- Pipeline-wide settings
- Stage-specific configurations
- Performance tuning parameters
- Validation rules and defaults
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Exception raised for configuration-related errors."""

    pass


@dataclass
class StageConfig:
    """Configuration for a specific pipeline stage.

    Each stage can have its own configuration parameters that control
    how it processes data and interacts with other stages.
    """

    # Core configuration
    enabled: bool = True
    timeout: float = 30.0
    retries: int = 3
    cache_enabled: bool = True

    # Stage-specific parameters (flexible dict)
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Dependencies and ordering
    dependencies: List[str] = field(default_factory=list)
    parallel_compatible: bool = False

    # Resource limits
    memory_limit_mb: Optional[int] = None
    cpu_threads: Optional[int] = None

    def validate(self) -> None:
        """Validate stage configuration."""
        if self.timeout <= 0:
            raise ConfigurationError(
                f"Stage timeout must be positive, got {self.timeout}"
            )

        if self.retries < 0:
            raise ConfigurationError(
                f"Stage retries must be non-negative, got {self.retries}"
            )

        if self.memory_limit_mb is not None and self.memory_limit_mb <= 0:
            raise ConfigurationError(
                f"Memory limit must be positive, got {self.memory_limit_mb}"
            )

        if self.cpu_threads is not None and self.cpu_threads <= 0:
            raise ConfigurationError(
                f"CPU threads must be positive, got {self.cpu_threads}"
            )

    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get stage-specific parameter."""
        return self.parameters.get(key, default)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value (dict-like interface)."""
        # First check if it's a known field
        if hasattr(self, key):
            return getattr(self, key)
        # Then check parameters
        return self.parameters.get(key, default)

    def set_parameter(self, key: str, value: Any) -> None:
        """Set stage-specific parameter."""
        self.parameters[key] = value

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "StageConfig":
        """Create StageConfig from dictionary."""
        # Extract known fields
        known_fields = {
            "enabled",
            "timeout",
            "retries",
            "cache_enabled",
            "dependencies",
            "parallel_compatible",
            "memory_limit_mb",
            "cpu_threads",
        }

        stage_config = cls()
        parameters = {}

        for key, value in config_dict.items():
            if key in known_fields:
                setattr(stage_config, key, value)
            else:
                parameters[key] = value

        stage_config.parameters = parameters
        stage_config.validate()

        return stage_config


@dataclass
class PipelineConfig:
    """Configuration for the entire sidecar generation pipeline.

    This class manages all configuration aspects including performance
    settings, stage configurations, and system limits.
    """

    # Performance settings
    target_execution_time: float = 10.0  # Maximum execution time target
    max_memory_usage_mb: int = 500
    enable_parallel_processing: bool = True
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600

    # Pipeline behavior
    fail_fast: bool = True  # Stop on first stage failure
    validation_enabled: bool = True
    detailed_logging: bool = False

    # Stage configurations
    stage_configs: Dict[str, StageConfig] = field(default_factory=dict)

    # HED-specific settings
    default_hed_version: str = "8.3.0"
    hed_library_path: Optional[str] = None

    # File handling
    max_file_size_mb: int = 100
    supported_file_extensions: List[str] = field(
        default_factory=lambda: [".tsv", ".csv"]
    )

    # Output settings
    output_format: str = "json"  # json, yaml
    include_metadata: bool = True

    def __post_init__(self):
        """Post-initialization validation and setup."""
        self.validate()
        self._setup_default_stage_configs()

    def validate(self) -> None:
        """Validate pipeline configuration."""
        if self.target_execution_time <= 0:
            raise ConfigurationError(
                f"Target execution time must be positive, got {self.target_execution_time}"
            )

        if self.max_memory_usage_mb <= 0:
            raise ConfigurationError(
                f"Max memory usage must be positive, got {self.max_memory_usage_mb}"
            )

        if self.cache_ttl_seconds <= 0:
            raise ConfigurationError(
                f"Cache TTL must be positive, got {self.cache_ttl_seconds}"
            )

        if self.max_file_size_mb <= 0:
            raise ConfigurationError(
                f"Max file size must be positive, got {self.max_file_size_mb}"
            )

        if self.output_format not in ["json", "yaml"]:
            raise ConfigurationError(
                f"Output format must be 'json' or 'yaml', got {self.output_format}"
            )

        # Validate stage configs
        for stage_name, stage_config in self.stage_configs.items():
            try:
                stage_config.validate()
            except ConfigurationError as e:
                raise ConfigurationError(
                    f"Stage '{stage_name}' configuration error: {e}"
                )

    def _setup_default_stage_configs(self) -> None:
        """Setup default configurations for standard stages."""
        default_configs = {
            "data_ingestion": StageConfig(
                timeout=15.0,
                parameters={
                    "max_file_size_mb": self.max_file_size_mb,
                    "supported_extensions": self.supported_file_extensions,
                    "encoding": "utf-8",
                },
            ),
            "column_classification": StageConfig(
                timeout=10.0,
                cache_enabled=True,
                parallel_compatible=True,
                parameters={
                    "classification_threshold": 0.8,
                    "enable_type_inference": True,
                },
            ),
            "hed_mapping": StageConfig(
                timeout=20.0,
                cache_enabled=True,
                parameters={
                    "hed_version": self.default_hed_version,
                    "validation_level": "strict",
                    "auto_completion": True,
                },
            ),
            "sidecar_generation": StageConfig(
                timeout=10.0,
                parameters={
                    "output_format": self.output_format,
                    "include_metadata": self.include_metadata,
                    "include_descriptions": True,
                },
            ),
            "validation": StageConfig(
                enabled=self.validation_enabled,
                timeout=5.0,
                parameters={"strict_validation": True, "check_completeness": True},
            ),
        }

        # Merge with existing configs (user configs take precedence)
        for stage_name, default_config in default_configs.items():
            if stage_name not in self.stage_configs:
                self.stage_configs[stage_name] = default_config
            else:
                # Merge parameters while preserving user settings
                user_config = self.stage_configs[stage_name]
                for param_key, param_value in default_config.parameters.items():
                    if param_key not in user_config.parameters:
                        user_config.parameters[param_key] = param_value

    def get_stage_config(self, stage_name: str) -> StageConfig:
        """Get configuration for a specific stage."""
        if stage_name not in self.stage_configs:
            # Return default stage config
            return StageConfig()
        return self.stage_configs[stage_name]

    def set_stage_config(
        self, stage_name: str, config: Union[StageConfig, Dict[str, Any]]
    ) -> None:
        """Set configuration for a specific stage."""
        if isinstance(config, dict):
            config = StageConfig.from_dict(config)

        config.validate()
        self.stage_configs[stage_name] = config

    def enable_debug_mode(self) -> None:
        """Enable debug mode with detailed logging and extended timeouts."""
        self.detailed_logging = True
        self.fail_fast = False

        # Extend timeouts for debugging
        for stage_config in self.stage_configs.values():
            stage_config.timeout *= 2

    def optimize_for_performance(self) -> None:
        """Optimize configuration for best performance."""
        self.enable_parallel_processing = True
        self.enable_caching = True
        self.detailed_logging = False

        # Enable parallel processing where possible
        for stage_name, stage_config in self.stage_configs.items():
            if stage_name in ["column_classification", "hed_mapping"]:
                stage_config.parallel_compatible = True

    def optimize_for_memory(self) -> None:
        """Optimize configuration for low memory usage."""
        self.enable_caching = False
        self.max_memory_usage_mb = min(self.max_memory_usage_mb, 200)

        # Reduce memory limits for stages
        for stage_config in self.stage_configs.values():
            stage_config.memory_limit_mb = 50
            stage_config.cache_enabled = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {
            "target_execution_time": self.target_execution_time,
            "max_memory_usage_mb": self.max_memory_usage_mb,
            "enable_parallel_processing": self.enable_parallel_processing,
            "enable_caching": self.enable_caching,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "fail_fast": self.fail_fast,
            "validation_enabled": self.validation_enabled,
            "detailed_logging": self.detailed_logging,
            "default_hed_version": self.default_hed_version,
            "hed_library_path": self.hed_library_path,
            "max_file_size_mb": self.max_file_size_mb,
            "supported_file_extensions": self.supported_file_extensions,
            "output_format": self.output_format,
            "include_metadata": self.include_metadata,
            "stage_configs": {},
        }

        # Convert stage configs
        for stage_name, stage_config in self.stage_configs.items():
            result["stage_configs"][stage_name] = {
                "enabled": stage_config.enabled,
                "timeout": stage_config.timeout,
                "retries": stage_config.retries,
                "cache_enabled": stage_config.cache_enabled,
                "dependencies": stage_config.dependencies,
                "parallel_compatible": stage_config.parallel_compatible,
                "memory_limit_mb": stage_config.memory_limit_mb,
                "cpu_threads": stage_config.cpu_threads,
                **stage_config.parameters,
            }

        return result

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PipelineConfig":
        """Create PipelineConfig from dictionary."""
        # Extract stage configs
        stage_configs_dict = config_dict.pop("stage_configs", {})

        # Create pipeline config with remaining parameters
        config = cls(**config_dict)

        # Add stage configs
        for stage_name, stage_config_dict in stage_configs_dict.items():
            config.set_stage_config(stage_name, stage_config_dict)

        return config

    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "PipelineConfig":
        """Load configuration from file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                if config_path.suffix.lower() == ".json":
                    config_dict = json.load(f)
                elif config_path.suffix.lower() in [".yml", ".yaml"]:
                    try:
                        import yaml

                        config_dict = yaml.safe_load(f)
                    except ImportError:
                        raise ConfigurationError(
                            "PyYAML required for YAML configuration files"
                        )
                else:
                    raise ConfigurationError(
                        f"Unsupported configuration file format: {config_path.suffix}"
                    )

            return cls.from_dict(config_dict)

        except Exception as e:
            raise ConfigurationError(
                f"Error loading configuration from {config_path}: {e}"
            )

    def save_to_file(self, config_path: Union[str, Path]) -> None:
        """Save configuration to file."""
        config_path = Path(config_path)
        config_dict = self.to_dict()

        try:
            with open(config_path, "w", encoding="utf-8") as f:
                if config_path.suffix.lower() == ".json":
                    json.dump(config_dict, f, indent=2)
                elif config_path.suffix.lower() in [".yml", ".yaml"]:
                    try:
                        import yaml

                        yaml.safe_dump(
                            config_dict, f, default_flow_style=False, indent=2
                        )
                    except ImportError:
                        raise ConfigurationError(
                            "PyYAML required for YAML configuration files"
                        )
                else:
                    raise ConfigurationError(
                        f"Unsupported configuration file format: {config_path.suffix}"
                    )

        except Exception as e:
            raise ConfigurationError(
                f"Error saving configuration to {config_path}: {e}"
            )


def create_default_config() -> PipelineConfig:
    """Create pipeline configuration with sensible defaults."""
    return PipelineConfig()


def create_development_config() -> PipelineConfig:
    """Create pipeline configuration optimized for development."""
    config = PipelineConfig()
    config.enable_debug_mode()
    return config


def create_production_config() -> PipelineConfig:
    """Create pipeline configuration optimized for production."""
    config = PipelineConfig()
    config.optimize_for_performance()
    return config


def create_memory_constrained_config() -> PipelineConfig:
    """Create pipeline configuration for memory-constrained environments."""
    config = PipelineConfig()
    config.optimize_for_memory()
    return config
