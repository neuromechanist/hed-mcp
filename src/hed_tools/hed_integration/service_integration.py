"""Service integration module for HED tools.

This module provides multiple integration options for HED tools:
- Direct Python API integration using hedtools library
- Web service client for HED online tools
- Common interface that abstracts implementation details
- Performance comparison and automatic selection
- Feature detection and fallback mechanisms
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Callable
from urllib.parse import urljoin
import aiohttp
import hashlib

try:
    from hed import schema as hed_schema
    from hed.tools.analysis.tabular_summary import TabularSummary
    from hed.errors.exceptions import HedFileError

    HED_AVAILABLE = True
except ImportError:
    hed_schema = None
    TabularSummary = None
    HedFileError = Exception
    HED_AVAILABLE = False

logger = logging.getLogger(__name__)


class IntegrationMethod(Enum):
    """Available HED integration methods."""

    DIRECT_API = "direct_api"
    WEB_SERVICE = "web_service"
    AUTO_SELECT = "auto_select"


@dataclass
class IntegrationResult:
    """Result from HED integration call."""

    success: bool
    data: Any = None
    error: Optional[str] = None
    method_used: Optional[IntegrationMethod] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics for integration methods."""

    method: IntegrationMethod
    average_latency: float = 0.0
    success_rate: float = 0.0
    error_count: int = 0
    total_calls: int = 0
    last_updated: float = field(default_factory=time.time)

    def update(self, latency: float, success: bool):
        """Update metrics with new measurement."""
        self.total_calls += 1
        if not success:
            self.error_count += 1

        # Running average
        self.average_latency = (
            self.average_latency * (self.total_calls - 1) + latency
        ) / self.total_calls
        self.success_rate = (self.total_calls - self.error_count) / self.total_calls
        self.last_updated = time.time()


class HEDIntegrationBase(ABC):
    """Abstract base class for HED integrations."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.metrics = PerformanceMetrics(method=self.get_method())

    @abstractmethod
    def get_method(self) -> IntegrationMethod:
        """Get the integration method."""
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if this integration method is available."""
        pass

    @abstractmethod
    async def validate_hed_string(
        self, hed_string: str, schema_version: str = "8.3.0"
    ) -> IntegrationResult:
        """Validate a HED string."""
        pass

    @abstractmethod
    async def generate_sidecar_template(
        self,
        event_files: List[str],
        value_columns: List[str] = None,
        skip_columns: List[str] = None,
    ) -> IntegrationResult:
        """Generate HED sidecar template."""
        pass

    @abstractmethod
    async def load_schema(self, version: str = "8.3.0") -> IntegrationResult:
        """Load HED schema."""
        pass

    async def _measure_performance(self, operation: Callable) -> IntegrationResult:
        """Measure performance of an operation."""
        start_time = time.perf_counter()
        success = True
        error = None

        try:
            result = await operation()
            return result
        except Exception as e:
            success = False
            error = str(e)
            return IntegrationResult(
                success=False, error=error, method_used=self.get_method()
            )
        finally:
            execution_time = time.perf_counter() - start_time
            self.metrics.update(execution_time, success)


class DirectAPIIntegration(HEDIntegrationBase):
    """Direct Python API integration using hedtools library."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.loaded_schemas: Dict[str, Any] = {}
        self.tabular_summary_cache: Dict[str, TabularSummary] = {}

    def get_method(self) -> IntegrationMethod:
        return IntegrationMethod.DIRECT_API

    async def is_available(self) -> bool:
        """Check if HED Python library is available."""
        if not HED_AVAILABLE:
            self.logger.warning("HED Python library not available")
            return False

        try:
            # Test basic functionality
            await self.load_schema("8.3.0")
            return True
        except Exception as e:
            self.logger.error(f"Direct API availability check failed: {e}")
            return False

    async def validate_hed_string(
        self, hed_string: str, schema_version: str = "8.3.0"
    ) -> IntegrationResult:
        """Validate HED string using direct API."""

        async def _validate():
            if not HED_AVAILABLE:
                raise RuntimeError("HED library not available")

            # Load schema if not cached
            schema_result = await self.load_schema(schema_version)
            if not schema_result.success:
                raise RuntimeError(f"Failed to load schema: {schema_result.error}")

            schema = self.loaded_schemas[schema_version]

            # Import HED validation components
            from hed.models.hed_string import HedString
            from hed.validator.hed_validator import HedValidator

            # Create HED string object
            hed_obj = HedString(hed_string)

            # Validate
            validator = HedValidator(schema)
            issues = validator.validate(hed_obj)

            return IntegrationResult(
                success=len(issues) == 0,
                data={
                    "valid": len(issues) == 0,
                    "issues": [str(issue) for issue in issues],
                    "hed_string": hed_string,
                    "schema_version": schema_version,
                },
                method_used=self.get_method(),
            )

        return await self._measure_performance(_validate)

    async def generate_sidecar_template(
        self,
        event_files: List[str],
        value_columns: List[str] = None,
        skip_columns: List[str] = None,
    ) -> IntegrationResult:
        """Generate sidecar template using TabularSummary."""

        async def _generate():
            if not HED_AVAILABLE:
                raise RuntimeError("HED library not available")

            # Create cache key
            cache_key = hashlib.md5(
                json.dumps(
                    {
                        "files": sorted(event_files),
                        "value_cols": sorted(value_columns or []),
                        "skip_cols": sorted(skip_columns or []),
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()

            # Check cache
            if cache_key in self.tabular_summary_cache:
                summary = self.tabular_summary_cache[cache_key]
            else:
                # Create TabularSummary
                summary = TabularSummary(
                    value_cols=value_columns or [],
                    skip_cols=skip_columns or [],
                    name=self.config.get("dataset_name", "Dataset"),
                )

                # Update with event files
                await asyncio.get_event_loop().run_in_executor(
                    None, summary.update, event_files
                )

                # Cache the summary
                self.tabular_summary_cache[cache_key] = summary

            # Extract sidecar template
            template = await asyncio.get_event_loop().run_in_executor(
                None, summary.extract_sidecar_template
            )

            return IntegrationResult(
                success=True,
                data={
                    "template": template,
                    "summary_info": {
                        "total_columns": len(summary.col_map)
                        if hasattr(summary, "col_map")
                        else 0,
                        "value_columns": len(value_columns or []),
                        "skip_columns": len(skip_columns or []),
                    },
                },
                method_used=self.get_method(),
            )

        return await self._measure_performance(_generate)

    async def load_schema(self, version: str = "8.3.0") -> IntegrationResult:
        """Load HED schema directly."""

        async def _load():
            if not HED_AVAILABLE:
                raise RuntimeError("HED library not available")

            if version in self.loaded_schemas:
                return IntegrationResult(
                    success=True,
                    data={"cached": True, "version": version},
                    method_used=self.get_method(),
                )

            # Load schema in executor to avoid blocking
            schema = await asyncio.get_event_loop().run_in_executor(
                None, hed_schema.load_schema, version
            )

            self.loaded_schemas[version] = schema

            return IntegrationResult(
                success=True,
                data={
                    "cached": False,
                    "version": version,
                    "schema_info": {
                        "version": schema.version_number
                        if hasattr(schema, "version_number")
                        else version,
                        "tags_count": len(schema.tags)
                        if hasattr(schema, "tags")
                        else 0,
                    },
                },
                method_used=self.get_method(),
            )

        return await self._measure_performance(_load)


class WebServiceIntegration(HEDIntegrationBase):
    """Web service integration for HED online tools."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.base_url = self.config.get("base_url", "https://hedtools.ucsd.edu/hed")
        self.timeout = self.config.get("timeout", 30.0)
        self.session = None
        self.max_retries = self.config.get("max_retries", 3)

    def get_method(self) -> IntegrationMethod:
        return IntegrationMethod.WEB_SERVICE

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self.session is None or self.session.closed:
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={"User-Agent": "HED-MCP-Client/1.0"},
            )
        return self.session

    async def is_available(self) -> bool:
        """Check if web service is available."""
        try:
            session = await self._get_session()
            url = urljoin(self.base_url, "/heartbeat")

            async with session.get(url) as response:
                if response.status == 200:
                    return True
                elif response.status == 404:
                    # Try alternative endpoint
                    url = urljoin(self.base_url, "/")
                    async with session.get(url) as alt_response:
                        return alt_response.status == 200
                return False

        except Exception as e:
            self.logger.error(f"Web service availability check failed: {e}")
            return False

    async def validate_hed_string(
        self, hed_string: str, schema_version: str = "8.3.0"
    ) -> IntegrationResult:
        """Validate HED string via web service."""

        async def _validate():
            session = await self._get_session()
            url = urljoin(self.base_url, "/validate")

            payload = {
                "hed_string": hed_string,
                "schema_version": schema_version,
                "check_for_warnings": True,
            }

            for attempt in range(self.max_retries):
                try:
                    async with session.post(url, json=payload) as response:
                        if response.status == 200:
                            result_data = await response.json()

                            return IntegrationResult(
                                success=result_data.get("valid", False),
                                data=result_data,
                                method_used=self.get_method(),
                            )
                        else:
                            error_text = await response.text()
                            if attempt == self.max_retries - 1:
                                raise RuntimeError(
                                    f"HTTP {response.status}: {error_text}"
                                )

                except asyncio.TimeoutError:
                    if attempt == self.max_retries - 1:
                        raise RuntimeError("Request timed out after retries")
                    await asyncio.sleep(2**attempt)  # Exponential backoff

                except Exception:
                    if attempt == self.max_retries - 1:
                        raise
                    await asyncio.sleep(2**attempt)

        return await self._measure_performance(_validate)

    async def generate_sidecar_template(
        self,
        event_files: List[str],
        value_columns: List[str] = None,
        skip_columns: List[str] = None,
    ) -> IntegrationResult:
        """Generate sidecar template via web service."""

        async def _generate():
            session = await self._get_session()
            url = urljoin(self.base_url, "/generate_sidecar")

            # For web service, we need to upload files or provide file URLs
            # This is a simplified implementation - real service might require file upload
            payload = {
                "value_columns": value_columns or [],
                "skip_columns": skip_columns or [],
                "dataset_name": self.config.get("dataset_name", "Dataset"),
                "files": event_files,  # Assuming service accepts file paths/URLs
            }

            for attempt in range(self.max_retries):
                try:
                    async with session.post(url, json=payload) as response:
                        if response.status == 200:
                            result_data = await response.json()

                            return IntegrationResult(
                                success=True,
                                data=result_data,
                                method_used=self.get_method(),
                            )
                        else:
                            error_text = await response.text()
                            if attempt == self.max_retries - 1:
                                raise RuntimeError(
                                    f"HTTP {response.status}: {error_text}"
                                )

                except asyncio.TimeoutError:
                    if attempt == self.max_retries - 1:
                        raise RuntimeError("Request timed out after retries")
                    await asyncio.sleep(2**attempt)

                except Exception:
                    if attempt == self.max_retries - 1:
                        raise
                    await asyncio.sleep(2**attempt)

        return await self._measure_performance(_generate)

    async def load_schema(self, version: str = "8.3.0") -> IntegrationResult:
        """Load schema information via web service."""

        async def _load():
            session = await self._get_session()
            url = urljoin(self.base_url, f"/schema/{version}")

            for attempt in range(self.max_retries):
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            schema_data = await response.json()

                            return IntegrationResult(
                                success=True,
                                data=schema_data,
                                method_used=self.get_method(),
                            )
                        else:
                            error_text = await response.text()
                            if attempt == self.max_retries - 1:
                                raise RuntimeError(
                                    f"HTTP {response.status}: {error_text}"
                                )

                except asyncio.TimeoutError:
                    if attempt == self.max_retries - 1:
                        raise RuntimeError("Request timed out after retries")
                    await asyncio.sleep(2**attempt)

                except Exception:
                    if attempt == self.max_retries - 1:
                        raise
                    await asyncio.sleep(2**attempt)

        return await self._measure_performance(_load)

    async def close(self):
        """Close HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()


class HEDIntegrationManager:
    """Manager that handles multiple integration methods with automatic selection."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.HEDIntegrationManager")

        # Initialize integrations
        self.integrations = {
            IntegrationMethod.DIRECT_API: DirectAPIIntegration(
                config.get("direct_api", {})
            ),
            IntegrationMethod.WEB_SERVICE: WebServiceIntegration(
                config.get("web_service", {})
            ),
        }

        self.preferred_method = IntegrationMethod(
            config.get("preferred_method", "auto_select")
        )

        self.availability_cache: Dict[
            IntegrationMethod, tuple
        ] = {}  # (available, timestamp)
        self.cache_ttl = config.get("availability_cache_ttl", 300)  # 5 minutes

        # Performance comparison
        self.performance_history: List[Dict[str, Any]] = []
        self.comparison_enabled = config.get("enable_performance_comparison", False)

    async def _check_availability(self, method: IntegrationMethod) -> bool:
        """Check if integration method is available with caching."""
        current_time = time.time()

        # Check cache
        if method in self.availability_cache:
            available, timestamp = self.availability_cache[method]
            if current_time - timestamp < self.cache_ttl:
                return available

        # Check availability
        integration = self.integrations[method]
        available = await integration.is_available()

        # Cache result
        self.availability_cache[method] = (available, current_time)

        return available

    async def _select_integration_method(self) -> Optional[IntegrationMethod]:
        """Select the best available integration method."""
        if self.preferred_method != IntegrationMethod.AUTO_SELECT:
            if await self._check_availability(self.preferred_method):
                return self.preferred_method
            else:
                self.logger.warning(
                    f"Preferred method {self.preferred_method} not available, falling back"
                )

        # Auto-select based on availability and performance
        available_methods = []
        for method in [IntegrationMethod.DIRECT_API, IntegrationMethod.WEB_SERVICE]:
            if await self._check_availability(method):
                available_methods.append(method)

        if not available_methods:
            return None

        # If only one method available, use it
        if len(available_methods) == 1:
            return available_methods[0]

        # Select based on performance metrics
        best_method = min(
            available_methods,
            key=lambda m: self.integrations[m].metrics.average_latency,
        )

        return best_method

    async def validate_hed_string(
        self, hed_string: str, schema_version: str = "8.3.0"
    ) -> IntegrationResult:
        """Validate HED string using best available method."""
        method = await self._select_integration_method()

        if method is None:
            return IntegrationResult(
                success=False,
                error="No integration methods available",
                method_used=None,
            )

        integration = self.integrations[method]
        result = await integration.validate_hed_string(hed_string, schema_version)

        if self.comparison_enabled and len(self.integrations) > 1:
            await self._perform_comparison(
                "validate_hed_string", hed_string, schema_version
            )

        return result

    async def generate_sidecar_template(
        self,
        event_files: List[str],
        value_columns: List[str] = None,
        skip_columns: List[str] = None,
    ) -> IntegrationResult:
        """Generate sidecar template using best available method."""
        method = await self._select_integration_method()

        if method is None:
            return IntegrationResult(
                success=False,
                error="No integration methods available",
                method_used=None,
            )

        integration = self.integrations[method]
        result = await integration.generate_sidecar_template(
            event_files, value_columns, skip_columns
        )

        if self.comparison_enabled and len(self.integrations) > 1:
            await self._perform_comparison(
                "generate_sidecar_template", event_files, value_columns, skip_columns
            )

        return result

    async def load_schema(self, version: str = "8.3.0") -> IntegrationResult:
        """Load schema using best available method."""
        method = await self._select_integration_method()

        if method is None:
            return IntegrationResult(
                success=False,
                error="No integration methods available",
                method_used=None,
            )

        integration = self.integrations[method]
        return await integration.load_schema(version)

    async def _perform_comparison(self, operation: str, *args, **kwargs):
        """Perform operation with all available methods for comparison."""
        comparison_results = {}

        for method, integration in self.integrations.items():
            if await self._check_availability(method):
                try:
                    start_time = time.perf_counter()

                    if operation == "validate_hed_string":
                        result = await integration.validate_hed_string(*args, **kwargs)
                    elif operation == "generate_sidecar_template":
                        result = await integration.generate_sidecar_template(
                            *args, **kwargs
                        )
                    else:
                        continue

                    execution_time = time.perf_counter() - start_time

                    comparison_results[method] = {
                        "success": result.success,
                        "execution_time": execution_time,
                        "data_size": len(str(result.data)) if result.data else 0,
                        "error": result.error,
                    }

                except Exception as e:
                    comparison_results[method] = {
                        "success": False,
                        "execution_time": 0,
                        "error": str(e),
                    }

        # Store comparison results
        self.performance_history.append(
            {
                "timestamp": time.time(),
                "operation": operation,
                "results": comparison_results,
            }
        )

        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]

    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance comparison report."""
        report = {
            "integration_methods": {},
            "performance_comparison": {},
            "recommendations": [],
        }

        # Individual method metrics
        for method, integration in self.integrations.items():
            metrics = integration.metrics
            report["integration_methods"][method.value] = {
                "average_latency": metrics.average_latency,
                "success_rate": metrics.success_rate,
                "total_calls": metrics.total_calls,
                "error_count": metrics.error_count,
                "last_updated": metrics.last_updated,
            }

        # Performance comparison from history
        if self.performance_history:
            operations = {}
            for entry in self.performance_history[-50:]:  # Last 50 operations
                op = entry["operation"]
                if op not in operations:
                    operations[op] = {"direct_api": [], "web_service": []}

                for method, result in entry["results"].items():
                    if result["success"]:
                        operations[op][method.value].append(result["execution_time"])

            for op, times in operations.items():
                if times["direct_api"] and times["web_service"]:
                    avg_direct = sum(times["direct_api"]) / len(times["direct_api"])
                    avg_web = sum(times["web_service"]) / len(times["web_service"])

                    report["performance_comparison"][op] = {
                        "direct_api_avg": avg_direct,
                        "web_service_avg": avg_web,
                        "faster_method": "direct_api"
                        if avg_direct < avg_web
                        else "web_service",
                        "speedup_ratio": max(avg_direct, avg_web)
                        / min(avg_direct, avg_web),
                    }

        # Generate recommendations
        if report["integration_methods"]:
            best_latency = min(
                m["average_latency"]
                for m in report["integration_methods"].values()
                if m["total_calls"] > 0
            )
            best_success = max(
                m["success_rate"]
                for m in report["integration_methods"].values()
                if m["total_calls"] > 0
            )

            for method, metrics in report["integration_methods"].items():
                if metrics["total_calls"] > 0:
                    if metrics["average_latency"] == best_latency:
                        report["recommendations"].append(
                            f"{method} has the best latency performance"
                        )
                    if metrics["success_rate"] == best_success:
                        report["recommendations"].append(
                            f"{method} has the best reliability"
                        )

        return report

    async def close(self):
        """Close all integrations."""
        for integration in self.integrations.values():
            if hasattr(integration, "close"):
                await integration.close()


# Factory function for easy instantiation
def create_hed_integration_manager(
    config: Dict[str, Any] = None,
) -> HEDIntegrationManager:
    """Create HED integration manager with default configuration."""
    default_config = {
        "preferred_method": "auto_select",
        "enable_performance_comparison": False,
        "availability_cache_ttl": 300,
        "direct_api": {"dataset_name": "HED Dataset"},
        "web_service": {
            "base_url": "https://hedtools.ucsd.edu/hed",
            "timeout": 30.0,
            "max_retries": 3,
        },
    }

    if config:
        # Deep merge configs
        def merge_dicts(base, override):
            result = base.copy()
            for key, value in override.items():
                if (
                    key in result
                    and isinstance(result[key], dict)
                    and isinstance(value, dict)
                ):
                    result[key] = merge_dicts(result[key], value)
                else:
                    result[key] = value
            return result

        default_config = merge_dicts(default_config, config)

    return HEDIntegrationManager(default_config)
