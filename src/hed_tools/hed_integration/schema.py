"""HED schema handling module.

This module provides utilities for loading, validating, and working with HED schemas
in a consistent manner, abstracting the complexity of the underlying hed.schema implementation.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, List, Set, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

try:
    from hed.schema.hed_schema_io import load_schema_version, load_schema
    from hed.schema.hed_schema import HedSchema
    from hed.schema.hed_schema_group import HedSchemaGroup
    from hed.errors.exceptions import HedFileError, HedExceptions
    from hed.models import HedString
except ImportError:
    # Graceful fallback if hed is not available
    load_schema_version = None
    load_schema = None
    HedSchema = None
    HedSchemaGroup = None
    HedFileError = Exception
    HedExceptions = Exception
    HedString = None

from .models import SchemaConfig, SchemaInfo, OperationResult

logger = logging.getLogger(__name__)


class HEDSchemaError(Exception):
    """Custom exception for HED schema-related errors."""

    pass


class SchemaHandler:
    """Handler for HED schema operations with caching and fallback support."""

    def __init__(self, config: Optional[SchemaConfig] = None):
        """Initialize the schema handler.

        Args:
            config: Schema configuration object
        """
        self.config = config or SchemaConfig()
        self.schema: Optional[HedSchema] = None
        self._schema_cache: Dict[str, HedSchema] = {}
        self._schema_info: Optional[SchemaInfo] = None
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._multiple_schemas: Dict[str, HedSchema] = {}

        if load_schema_version is None:
            logger.warning(
                "HED library not available - schema handler will run in stub mode"
            )

    async def load_schema(
        self,
        version: Optional[str] = None,
        custom_path: Optional[Path] = None,
        force_reload: bool = False,
    ) -> OperationResult:
        """Load HED schema asynchronously with fallback support.

        Args:
            version: HED schema version to load (overrides config)
            custom_path: Custom schema file path (overrides config)
            force_reload: Force reload even if schema is cached

        Returns:
            OperationResult indicating success/failure with timing info
        """
        start_time = time.time()

        if load_schema_version is None:
            return OperationResult(
                success=False,
                error="HED library not available - please install hedtools package",
                processing_time=time.time() - start_time,
            )

        try:
            schema_source = custom_path or self.config.custom_path
            schema_version = version or self.config.version
            cache_key = str(schema_source) if schema_source else schema_version

            # Check cache first
            if not force_reload and cache_key in self._schema_cache:
                self.schema = self._schema_cache[cache_key]
                self._update_schema_info()
                logger.info(f"Using cached HED schema: {cache_key}")
                return OperationResult(
                    success=True,
                    data=self._schema_info,
                    processing_time=time.time() - start_time,
                )

            # Load schema in thread pool to avoid blocking
            loop = asyncio.get_event_loop()

            if schema_source:
                logger.info(f"Loading custom HED schema from: {schema_source}")
                self.schema = await loop.run_in_executor(
                    self._executor, self._load_schema_from_file, schema_source
                )
            else:
                logger.info(f"Loading HED schema version: {schema_version}")
                self.schema = await loop.run_in_executor(
                    self._executor, self._load_schema_with_fallback, schema_version
                )

            # Cache the loaded schema
            self._schema_cache[cache_key] = self.schema
            self._update_schema_info()

            logger.info(f"HED schema loaded successfully: {cache_key}")
            return OperationResult(
                success=True,
                data=self._schema_info,
                processing_time=time.time() - start_time,
            )

        except HedFileError as e:
            error_msg = f"Failed to load HED schema: {e}. Please check the schema version or file path."
            logger.error(error_msg)
            return OperationResult(
                success=False, error=error_msg, processing_time=time.time() - start_time
            )
        except Exception as e:
            error_msg = f"Unexpected error loading HED schema: {e}"
            logger.error(error_msg)
            return OperationResult(
                success=False, error=error_msg, processing_time=time.time() - start_time
            )

    async def load_multiple_schemas(self, versions: List[str]) -> OperationResult:
        """Load multiple HED schema versions simultaneously.

        Args:
            versions: List of HED schema versions to load

        Returns:
            OperationResult with information about loaded schemas
        """
        start_time = time.time()

        if load_schema_version is None:
            return OperationResult(
                success=False,
                error="HED library not available",
                processing_time=time.time() - start_time,
            )

        try:
            loaded_schemas = {}
            failed_schemas = {}

            # Load schemas in parallel
            loop = asyncio.get_event_loop()
            tasks = []

            for version in versions:
                task = loop.run_in_executor(
                    self._executor, self._load_single_schema_version, version
                )
                tasks.append((version, task))

            for version, task in tasks:
                try:
                    schema = await task
                    loaded_schemas[version] = schema
                    self._multiple_schemas[version] = schema
                    logger.info(f"Successfully loaded HED schema version: {version}")
                except Exception as e:
                    failed_schemas[version] = str(e)
                    logger.warning(f"Failed to load HED schema version {version}: {e}")

            success = len(loaded_schemas) > 0
            result_data = {
                "loaded_schemas": list(loaded_schemas.keys()),
                "failed_schemas": failed_schemas,
                "total_requested": len(versions),
                "total_loaded": len(loaded_schemas),
            }

            return OperationResult(
                success=success,
                data=result_data,
                error=None if success else "Failed to load any schemas",
                processing_time=time.time() - start_time,
            )

        except Exception as e:
            error_msg = f"Error during multiple schema loading: {e}"
            logger.error(error_msg)
            return OperationResult(
                success=False, error=error_msg, processing_time=time.time() - start_time
            )

    def _load_single_schema_version(self, version: str) -> HedSchema:
        """Load a single schema version (blocking operation for thread pool)."""
        try:
            return load_schema_version(version)
        except Exception as e:
            raise HEDSchemaError(f"Failed to load schema version {version}: {e}")

    def _load_schema_from_file(self, file_path: Path) -> HedSchema:
        """Load schema from file (blocking operation for thread pool)."""
        try:
            if not file_path.exists():
                raise HEDSchemaError(f"Schema file not found: {file_path}")

            return load_schema(str(file_path))
        except Exception as e:
            raise HEDSchemaError(f"Failed to load schema from {file_path}: {e}")

    def _load_schema_with_fallback(self, version: str) -> HedSchema:
        """Load schema with fallback versions (blocking operation for thread pool)."""
        versions_to_try = [version] + [
            v for v in self.config.fallback_versions if v != version
        ]

        last_error = None
        for ver in versions_to_try:
            try:
                logger.info(f"Attempting to load HED schema version: {ver}")
                return load_schema_version(ver)
            except Exception as e:
                last_error = e
                logger.warning(f"Failed to load schema version {ver}: {e}")
                continue

        raise HEDSchemaError(
            f"Failed to load any schema versions from {versions_to_try}. Last error: {last_error}"
        )

    def get_schema_by_version(self, version: str) -> Optional[HedSchema]:
        """Get a specific schema version from the loaded schemas.

        Args:
            version: Schema version to retrieve

        Returns:
            HedSchema object or None if not loaded
        """
        return self._multiple_schemas.get(version)

    def get_loaded_schema_versions(self) -> List[str]:
        """Get list of currently loaded schema versions.

        Returns:
            List of schema version strings
        """
        return list(self._multiple_schemas.keys())

    def get_all_schema_tags(self, version: Optional[str] = None) -> Set[str]:
        """Get all available tags from a schema.

        Args:
            version: Schema version (uses current schema if None)

        Returns:
            Set of all tag names
        """
        try:
            schema = self.get_schema_by_version(version) if version else self.schema

            if schema is None:
                logger.warning(
                    f"Schema {'version ' + version if version else ''} not loaded"
                )
                return set()

            if hasattr(schema, "get_all_tags"):
                return set(schema.get_all_tags())
            elif hasattr(schema, "tags"):
                return set(schema.tags.keys())
            else:
                logger.warning("Schema object does not support tag enumeration")
                return set()

        except Exception as e:
            logger.error(f"Error getting schema tags: {e}")
            return set()

    def validate_tag(self, tag: str, version: Optional[str] = None) -> bool:
        """Validate if a tag exists in the schema.

        Args:
            tag: HED tag to validate
            version: Schema version (uses current schema if None)

        Returns:
            True if tag is valid
        """
        try:
            schema = self.get_schema_by_version(version) if version else self.schema

            if schema is None:
                logger.warning(
                    f"Schema {'version ' + version if version else ''} not loaded"
                )
                return False

            if hasattr(schema, "check_compliance"):
                # Try to validate using HedString if available
                if HedString:
                    hed_string = HedString(tag)
                    issues = hed_string.validate(schema)
                    return len(issues) == 0

            # Fallback to checking if tag exists in schema
            all_tags = self.get_all_schema_tags(version)
            return tag in all_tags

        except Exception as e:
            logger.error(f"Error validating tag '{tag}': {e}")
            return False

    def get_tag_descendants(self, tag: str, version: Optional[str] = None) -> List[str]:
        """Get all descendant tags of a given tag.

        Args:
            tag: Parent tag to find descendants for
            version: Schema version (uses current schema if None)

        Returns:
            List of descendant tag names
        """
        try:
            schema = self.get_schema_by_version(version) if version else self.schema

            if schema is None:
                logger.warning(
                    f"Schema {'version ' + version if version else ''} not loaded"
                )
                return []

            if hasattr(schema, "get_tag_descendants"):
                return schema.get_tag_descendants(tag)
            elif hasattr(schema, "tags"):
                # Manual implementation for schemas without direct support
                descendants = []
                tag_lower = tag.lower()

                for schema_tag in schema.tags:
                    if schema_tag.lower().startswith(tag_lower + "/"):
                        descendants.append(schema_tag)

                return descendants

            return []

        except Exception as e:
            logger.error(f"Error getting descendants for tag '{tag}': {e}")
            return []

    def find_similar_tags(
        self, partial_tag: str, version: Optional[str] = None, max_results: int = 10
    ) -> List[str]:
        """Find tags similar to a partial tag string.

        Args:
            partial_tag: Partial tag to search for
            version: Schema version (uses current schema if None)
            max_results: Maximum number of results to return

        Returns:
            List of similar tag names
        """
        try:
            all_tags = self.get_all_schema_tags(version)
            partial_lower = partial_tag.lower()

            # Find exact matches first, then partial matches
            exact_matches = [tag for tag in all_tags if tag.lower() == partial_lower]
            partial_matches = [
                tag
                for tag in all_tags
                if partial_lower in tag.lower() and tag not in exact_matches
            ]

            # Combine and limit results
            results = exact_matches + partial_matches
            return results[:max_results]

        except Exception as e:
            logger.error(f"Error finding similar tags for '{partial_tag}': {e}")
            return []

    def compare_schema_versions(self, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two schema versions and return differences.

        Args:
            version1: First schema version
            version2: Second schema version

        Returns:
            Dictionary with comparison results
        """
        try:
            schema1 = self.get_schema_by_version(version1)
            schema2 = self.get_schema_by_version(version2)

            if schema1 is None or schema2 is None:
                missing = []
                if schema1 is None:
                    missing.append(version1)
                if schema2 is None:
                    missing.append(version2)
                return {
                    "error": f"Schema versions not loaded: {missing}",
                    "version1": version1,
                    "version2": version2,
                }

            tags1 = self.get_all_schema_tags(version1)
            tags2 = self.get_all_schema_tags(version2)

            comparison = {
                "version1": version1,
                "version2": version2,
                "tags_in_v1_only": list(tags1 - tags2),
                "tags_in_v2_only": list(tags2 - tags1),
                "common_tags": list(tags1 & tags2),
                "total_tags_v1": len(tags1),
                "total_tags_v2": len(tags2),
                "similarity_percentage": (len(tags1 & tags2) / len(tags1 | tags2)) * 100
                if tags1 | tags2
                else 100,
            }

            return comparison

        except Exception as e:
            logger.error(
                f"Error comparing schema versions {version1} and {version2}: {e}"
            )
            return {"error": str(e), "version1": version1, "version2": version2}

    def _update_schema_info(self):
        """Update cached schema information."""
        if self.schema is None:
            self._schema_info = SchemaInfo(version="unknown", loaded=False)
            return

        try:
            # Extract schema information
            version = getattr(self.schema, "version_number", "unknown")
            tag_count = len(getattr(self.schema, "tags", {}))
            library_schemas = []

            # Check if this is a schema group with multiple schemas
            if hasattr(self.schema, "schema_versions"):
                library_schemas = self.schema.schema_versions

            self._schema_info = SchemaInfo(
                version=version,
                loaded=True,
                path=self.config.custom_path,
                tag_count=tag_count,
                description=f"HED schema version {version}",
                library_schemas=library_schemas,
            )

        except Exception as e:
            logger.warning(f"Failed to extract schema info: {e}")
            self._schema_info = SchemaInfo(
                version="unknown",
                loaded=True,
                description="Schema loaded but info unavailable",
            )

    def get_schema_info(self) -> SchemaInfo:
        """Get information about the currently loaded schema.

        Returns:
            Schema information object
        """
        if self._schema_info is None:
            return SchemaInfo(version="none", loaded=False)
        return self._schema_info

    def is_schema_loaded(self) -> bool:
        """Check if a schema is currently loaded.

        Returns:
            True if schema is loaded
        """
        return self.schema is not None

    def get_schema(self) -> Optional[HedSchema]:
        """Get the currently loaded schema.

        Returns:
            The loaded HED schema or None
        """
        return self.schema

    async def validate_schema(self) -> OperationResult:
        """Validate the currently loaded schema.

        Returns:
            OperationResult with validation details
        """
        start_time = time.time()

        if self.schema is None:
            return OperationResult(
                success=False,
                error="No schema loaded for validation",
                processing_time=time.time() - start_time,
            )

        try:
            # Run schema validation in thread pool
            loop = asyncio.get_event_loop()
            validation_issues = await loop.run_in_executor(
                self._executor, self._validate_schema_blocking
            )

            success = len(validation_issues) == 0
            result_data = {
                "issues": validation_issues,
                "issue_count": len(validation_issues),
                "schema_version": self._schema_info.version
                if self._schema_info
                else "unknown",
            }

            return OperationResult(
                success=success,
                data=result_data,
                error=None
                if success
                else f"Schema validation failed with {len(validation_issues)} issues",
                processing_time=time.time() - start_time,
            )

        except Exception as e:
            error_msg = f"Schema validation error: {e}"
            logger.error(error_msg)
            return OperationResult(
                success=False, error=error_msg, processing_time=time.time() - start_time
            )

    def _validate_schema_blocking(self) -> List[Dict[str, Any]]:
        """Validate schema (blocking operation for thread pool)."""
        try:
            if hasattr(self.schema, "check_compliance"):
                issues = self.schema.check_compliance()
                return [
                    {"message": str(issue), "type": "compliance"} for issue in issues
                ]
            return []
        except Exception as e:
            return [{"message": f"Validation error: {e}", "type": "error"}]

    def get_available_schemas(self) -> List[Dict[str, str]]:
        """Get list of available HED schema versions.

        Returns:
            List of schema information dictionaries
        """
        # This would ideally query the HED schema repository
        # For now, return known versions
        known_schemas = [
            {
                "version": "8.3.0",
                "description": "Latest stable HED schema",
                "type": "standard",
            },
            {
                "version": "8.2.0",
                "description": "Previous stable HED schema",
                "type": "standard",
            },
            {
                "version": "8.1.0",
                "description": "Older HED schema version",
                "type": "standard",
            },
            {"version": "8.0.0", "description": "HED schema 8.0.0", "type": "standard"},
            {
                "version": "score_1.1.0",
                "description": "SCORE library schema",
                "type": "library",
            },
            {
                "version": "testlib_1.0.0",
                "description": "Test library schema",
                "type": "library",
            },
        ]

        return known_schemas

    def clear_cache(self):
        """Clear the schema cache."""
        self._schema_cache.clear()
        self._multiple_schemas.clear()
        logger.info("Schema cache cleared")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the schema cache.

        Returns:
            Cache statistics and contents
        """
        return {
            "cached_schemas": list(self._schema_cache.keys()),
            "cache_size": len(self._schema_cache),
            "multiple_schemas": list(self._multiple_schemas.keys()),
            "current_schema": self._schema_info.version if self._schema_info else None,
        }

    async def preload_schemas(self, versions: List[str]) -> Dict[str, bool]:
        """Preload multiple schema versions into cache.

        Args:
            versions: List of schema versions to preload

        Returns:
            Dictionary mapping versions to load success status
        """
        result = await self.load_multiple_schemas(versions)

        if result.success and result.data:
            results = {}
            loaded_schemas = result.data.get("loaded_schemas", [])

            for version in versions:
                results[version] = version in loaded_schemas

            return results
        else:
            return {version: False for version in versions}

    def close(self):
        """Clean up resources."""
        if self._executor:
            self._executor.shutdown(wait=True)
            logger.info("Schema handler executor shut down")


# Utility functions for common schema operations
def load_hed_schema(
    version: str = "8.3.0",
) -> Tuple[Optional[HedSchema], Optional[str]]:
    """Utility function to quickly load a HED schema.

    Args:
        version: HED schema version to load

    Returns:
        Tuple of (schema, error_message)
    """
    try:
        if load_schema_version is None:
            return None, "HED library not available"

        schema = load_schema_version(version)
        return schema, None
    except Exception as e:
        return None, str(e)


def validate_hed_tag_simple(tag: str, schema: HedSchema) -> Tuple[bool, List[str]]:
    """Simple utility to validate a HED tag against a schema.

    Args:
        tag: HED tag to validate
        schema: HED schema to validate against

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    try:
        if HedString is None:
            return False, ["HED library not available"]

        hed_string = HedString(tag)
        issues = hed_string.validate(schema)

        if issues:
            errors = [str(issue) for issue in issues]
            return False, errors
        else:
            return True, []

    except Exception as e:
        return False, [str(e)]


def get_schema_version_info(schema: HedSchema) -> Dict[str, Any]:
    """Extract version information from a HED schema.

    Args:
        schema: HED schema object

    Returns:
        Dictionary with schema version information
    """
    try:
        info = {
            "version": getattr(schema, "version_number", "unknown"),
            "name": getattr(schema, "name", "unknown"),
            "tag_count": len(getattr(schema, "tags", {})),
            "has_units": hasattr(schema, "units"),
            "has_attributes": hasattr(schema, "attributes"),
        }

        if hasattr(schema, "prologue"):
            info["prologue"] = schema.prologue

        if hasattr(schema, "epilogue"):
            info["epilogue"] = schema.epilogue

        return info
    except Exception as e:
        return {"error": str(e)}


class SchemaManagerFacade:
    """Facade class that simplifies HED schema operations."""

    def __init__(self, schema_version: str = "8.3.0"):
        """Initialize with default schema version.

        Args:
            schema_version: Default HED schema version
        """
        config = SchemaConfig(version=schema_version)
        self.handler = SchemaHandler(config)
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize the schema manager by loading the default schema.

        Returns:
            True if initialization successful
        """
        result = await self.handler.load_schema()
        self._initialized = result.success
        return result.success

    async def get_schema(self, version: Optional[str] = None) -> Optional[HedSchema]:
        """Get a schema, loading it if necessary.

        Args:
            version: Schema version (None for default)

        Returns:
            Loaded HED schema or None if failed
        """
        if not self._initialized:
            await self.initialize()

        if version and version != self.handler.get_schema_info().version:
            result = await self.handler.load_schema(version=version)
            if not result.success:
                return None

        return self.handler.get_schema()

    def is_available(self) -> bool:
        """Check if HED functionality is available.

        Returns:
            True if HED library is available
        """
        return load_schema_version is not None

    async def validate_schema_compatibility(self, version: str) -> bool:
        """Check if a schema version is compatible.

        Args:
            version: Schema version to check

        Returns:
            True if version is compatible
        """
        try:
            result = await self.handler.load_schema(version=version)
            return result.success
        except Exception:
            return False
