"""HED schema handling module.

This module provides utilities for loading, validating, and working with HED schemas
in a consistent manner, abstracting the complexity of the underlying hed.schema implementation.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

try:
    from hed.schema.hed_schema_io import load_schema_version, load_schema
    from hed.schema.hed_schema import HedSchema
    from hed.schema.hed_schema_group import HedSchemaGroup
    from hed.errors.exceptions import HedFileError, HedExceptions
except ImportError:
    # Graceful fallback if hed is not available
    load_schema_version = None
    load_schema = None
    HedSchema = None
    HedSchemaGroup = None
    HedFileError = Exception
    HedExceptions = Exception

from .models import SchemaConfig, SchemaInfo, OperationResult

logger = logging.getLogger(__name__)


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
        
        if load_schema_version is None:
            logger.warning("HED library not available - schema handler will run in stub mode")
    
    async def load_schema(self, version: Optional[str] = None, 
                         custom_path: Optional[Path] = None,
                         force_reload: bool = False) -> OperationResult:
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
                error="HED library not available",
                processing_time=time.time() - start_time
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
                    processing_time=time.time() - start_time
                )
            
            # Load schema in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            if schema_source:
                logger.info(f"Loading custom HED schema from: {schema_source}")
                self.schema = await loop.run_in_executor(
                    self._executor, 
                    self._load_schema_from_file, 
                    schema_source
                )
            else:
                logger.info(f"Loading HED schema version: {schema_version}")
                self.schema = await loop.run_in_executor(
                    self._executor,
                    self._load_schema_with_fallback,
                    schema_version
                )
            
            # Cache the loaded schema
            self._schema_cache[cache_key] = self.schema
            self._update_schema_info()
            
            logger.info(f"HED schema loaded successfully: {cache_key}")
            return OperationResult(
                success=True,
                data=self._schema_info,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Failed to load HED schema: {e}")
            return OperationResult(
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            )
    
    def _load_schema_from_file(self, file_path: Path) -> HedSchema:
        """Load schema from file (blocking operation for thread pool)."""
        return load_schema(str(file_path))
    
    def _load_schema_with_fallback(self, version: str) -> HedSchema:
        """Load schema with fallback versions (blocking operation for thread pool)."""
        versions_to_try = [version] + [v for v in self.config.fallback_versions if v != version]
        
        for ver in versions_to_try:
            try:
                logger.info(f"Attempting to load HED schema version: {ver}")
                return load_schema_version(ver)
            except Exception as e:
                logger.warning(f"Failed to load schema version {ver}: {e}")
                continue
        
        raise HedFileError(f"Failed to load any schema versions: {versions_to_try}")
    
    def _update_schema_info(self):
        """Update cached schema information."""
        if self.schema is None:
            self._schema_info = SchemaInfo(
                version="unknown",
                loaded=False
            )
            return
        
        try:
            # Extract schema information
            version = getattr(self.schema, 'version_number', 'unknown')
            tag_count = len(getattr(self.schema, 'tags', {}))
            library_schemas = []
            
            # Check if this is a schema group with multiple schemas
            if hasattr(self.schema, 'schema_versions'):
                library_schemas = self.schema.schema_versions
            
            self._schema_info = SchemaInfo(
                version=version,
                loaded=True,
                path=self.config.custom_path,
                tag_count=tag_count,
                description=f"HED schema version {version}",
                library_schemas=library_schemas
            )
            
        except Exception as e:
            logger.warning(f"Failed to extract schema info: {e}")
            self._schema_info = SchemaInfo(
                version="unknown",
                loaded=True,
                description="Schema loaded but info unavailable"
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
                error="No schema loaded",
                processing_time=time.time() - start_time
            )
        
        try:
            # Run schema validation in thread pool
            loop = asyncio.get_event_loop()
            validation_issues = await loop.run_in_executor(
                self._executor,
                self._validate_schema_blocking
            )
            
            success = len(validation_issues) == 0
            result_data = {
                "issues": validation_issues,
                "issue_count": len(validation_issues),
                "schema_version": self._schema_info.version if self._schema_info else "unknown"
            }
            
            return OperationResult(
                success=success,
                data=result_data,
                error=None if success else f"Schema validation failed with {len(validation_issues)} issues",
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Schema validation error: {e}")
            return OperationResult(
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            )
    
    def _validate_schema_blocking(self) -> List[Dict[str, Any]]:
        """Validate schema (blocking operation for thread pool)."""
        if hasattr(self.schema, 'check_compliance'):
            issues = self.schema.check_compliance()
            return [{"message": str(issue), "type": "compliance"} for issue in issues]
        return []
    
    def get_available_schemas(self) -> List[Dict[str, str]]:
        """Get list of available HED schema versions.
        
        Returns:
            List of schema information dictionaries
        """
        # This would ideally query the HED schema repository
        # For now, return known versions
        known_schemas = [
            {"version": "8.3.0", "description": "Latest stable HED schema", "type": "standard"},
            {"version": "8.2.0", "description": "Previous stable HED schema", "type": "standard"},
            {"version": "8.1.0", "description": "Older HED schema version", "type": "standard"},
            {"version": "8.0.0", "description": "HED schema 8.0.0", "type": "standard"},
            {"version": "score_1.1.0", "description": "SCORE library schema", "type": "library"},
            {"version": "testlib_1.0.0", "description": "Test library schema", "type": "library"}
        ]
        
        return known_schemas
    
    def clear_cache(self):
        """Clear the schema cache."""
        self._schema_cache.clear()
        logger.info("Schema cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the schema cache.
        
        Returns:
            Cache statistics and contents
        """
        return {
            "cached_schemas": list(self._schema_cache.keys()),
            "cache_size": len(self._schema_cache),
            "current_schema": self._schema_info.version if self._schema_info else None
        }
    
    async def preload_schemas(self, versions: List[str]) -> Dict[str, bool]:
        """Preload multiple schema versions into cache.
        
        Args:
            versions: List of schema versions to preload
            
        Returns:
            Dictionary mapping versions to load success status
        """
        results = {}
        
        for version in versions:
            try:
                result = await self.load_schema(version=version)
                results[version] = result.success
                if result.success:
                    logger.info(f"Preloaded schema version: {version}")
                else:
                    logger.warning(f"Failed to preload schema version {version}: {result.error}")
            except Exception as e:
                logger.error(f"Error preloading schema {version}: {e}")
                results[version] = False
        
        return results
    
    def close(self):
        """Clean up resources."""
        if self._executor:
            self._executor.shutdown(wait=True)
            logger.info("Schema handler executor shut down")


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