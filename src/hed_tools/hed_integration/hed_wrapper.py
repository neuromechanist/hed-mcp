"""Main interface wrapper for HED library operations.

This module provides a simplified interface to the HED (Hierarchical Event Descriptor)
Python tools library, focusing on schema loading, validation, and TabularSummary operations.
"""

from typing import Optional, Dict, Any, List, Union, Tuple
from pathlib import Path
import logging
import pandas as pd

try:
    import hed
    from hed import TabularSummary
    from hed.models import HedString, HedTag
    from hed.schema import HedSchema
except ImportError:
    # Graceful fallback if hed is not available yet
    hed = None
    TabularSummary = None
    HedString = None
    HedTag = None
    HedSchema = None

logger = logging.getLogger(__name__)


class HEDWrapper:
    """Wrapper class for HED library functionality.
    
    This class provides a simplified interface to common HED operations including:
    - Schema loading and management
    - Event validation and processing
    - TabularSummary operations for sidecar generation
    - HED string parsing and manipulation
    """
    
    def __init__(self, schema_path: Optional[Path] = None, schema_version: Optional[str] = None):
        """Initialize the HED wrapper.
        
        Args:
            schema_path: Optional path to custom HED schema file
            schema_version: HED schema version to load (e.g., "8.2.0")
        """
        self.schema_path = schema_path
        self.schema_version = schema_version or "latest"
        self.schema: Optional[HedSchema] = None
        self._schema_loaded = False
        
        if hed is None:
            logger.warning("HED library not available - wrapper will run in stub mode")
    
    async def load_schema(self, version: Optional[str] = None, 
                         custom_path: Optional[Path] = None) -> bool:
        """Load HED schema asynchronously.
        
        Args:
            version: HED schema version to load (overrides instance setting)
            custom_path: Custom schema file path (overrides instance setting)
            
        Returns:
            True if schema loaded successfully, False otherwise
        """
        if hed is None:
            logger.error("HED library not available - cannot load schema")
            return False
        
        try:
            schema_source = custom_path or self.schema_path
            schema_ver = version or self.schema_version
            
            if schema_source:
                logger.info(f"Loading custom HED schema from: {schema_source}")
                self.schema = HedSchema(schema_source)
            else:
                logger.info(f"Loading HED schema version: {schema_ver}")
                # TODO: Implement version-specific schema loading from hedtools
                self.schema = HedSchema()  # Default schema for now
            
            self._schema_loaded = True
            logger.info("HED schema loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load HED schema: {e}")
            self._schema_loaded = False
            return False
    
    async def validate_events(self, events_data: Union[pd.DataFrame, Dict[str, Any]], 
                            sidecar: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Validate HED annotations in events data.
        
        Args:
            events_data: BIDS events DataFrame or dictionary
            sidecar: Optional HED sidecar with column definitions
            
        Returns:
            Validation results with errors, warnings, and statistics
        """
        if not self._schema_loaded:
            await self.load_schema()
        
        if hed is None or self.schema is None:
            return {
                "valid": False,
                "errors": ["HED library or schema not available"],
                "warnings": [],
                "statistics": {}
            }
        
        try:
            # TODO: Implement actual HED validation using hedtools
            logger.info("Validating HED annotations")
            
            # Placeholder validation logic
            validation_result = {
                "valid": True,
                "errors": [],
                "warnings": [],
                "statistics": {
                    "total_events": len(events_data) if hasattr(events_data, '__len__') else 0,
                    "validated_columns": [],
                    "hed_tags_found": 0
                }
            }
            
            return validation_result
            
        except Exception as e:
            logger.error(f"HED validation failed: {e}")
            return {
                "valid": False,
                "errors": [str(e)],
                "warnings": [],
                "statistics": {}
            }
    
    async def generate_sidecar_template(self, events_df: pd.DataFrame, 
                                      columns_to_process: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate HED sidecar template using TabularSummary.
        
        Args:
            events_df: BIDS events DataFrame
            columns_to_process: Specific columns to include in sidecar (None for all)
            
        Returns:
            Generated HED sidecar template dictionary
        """
        if not self._schema_loaded:
            await self.load_schema()
        
        if hed is None or TabularSummary is None:
            logger.error("HED TabularSummary not available")
            return {"error": "HED library not available"}
        
        try:
            logger.info("Generating HED sidecar template using TabularSummary")
            
            # TODO: Implement actual TabularSummary operations
            # This should follow the pattern from extract_json_template.ipynb
            
            # Placeholder sidecar template
            sidecar_template = {
                "onset": {
                    "Description": "Onset time of the event",
                    "Units": "s"
                },
                "duration": {
                    "Description": "Duration of the event", 
                    "Units": "s"
                }
            }
            
            # Add templates for data columns
            if columns_to_process:
                for col in columns_to_process:
                    if col not in ["onset", "duration"]:
                        sidecar_template[col] = {
                            "Description": f"Description for {col}",
                            "HED": {
                                # Placeholder HED template structure
                                f"{col}_value": "Event/Category/Placeholder"
                            }
                        }
            
            logger.info(f"Generated sidecar template for {len(sidecar_template)} columns")
            return sidecar_template
            
        except Exception as e:
            logger.error(f"Sidecar generation failed: {e}")
            return {"error": str(e)}
    
    def get_available_schemas(self) -> List[Dict[str, str]]:
        """Get list of available HED schema versions.
        
        Returns:
            List of schema information dictionaries
        """
        # TODO: Implement actual schema version discovery
        # This should query available schemas from hedtools
        
        return [
            {"version": "8.2.0", "description": "Latest HED schema"},
            {"version": "8.1.0", "description": "Previous HED schema"},
            {"version": "8.0.0", "description": "HED schema 8.0.0"}
        ]
    
    def parse_hed_string(self, hed_string: str) -> Dict[str, Any]:
        """Parse a HED string and return structured information.
        
        Args:
            hed_string: HED annotation string to parse
            
        Returns:
            Parsed HED information including tags, structure, and validation status
        """
        if hed is None or HedString is None:
            return {"error": "HED library not available"}
        
        try:
            # TODO: Implement actual HED string parsing
            logger.info(f"Parsing HED string: {hed_string[:50]}...")
            
            # Placeholder parsing result
            return {
                "original": hed_string,
                "valid": True,
                "tags": [],
                "structure": {},
                "errors": []
            }
            
        except Exception as e:
            logger.error(f"HED string parsing failed: {e}")
            return {
                "original": hed_string,
                "valid": False,
                "tags": [],
                "structure": {},
                "errors": [str(e)]
            }
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded schema.
        
        Returns:
            Schema metadata and statistics
        """
        if not self._schema_loaded or self.schema is None:
            return {"loaded": False, "error": "No schema loaded"}
        
        return {
            "loaded": True,
            "version": self.schema_version,
            "path": str(self.schema_path) if self.schema_path else None,
            "tag_count": 0,  # TODO: Get actual tag count from schema
            "description": "HED schema for event annotation"
        }


def create_hed_wrapper(schema_version: str = "latest") -> HEDWrapper:
    """Factory function to create and initialize a HED wrapper.
    
    Args:
        schema_version: HED schema version to load
        
    Returns:
        Initialized HEDWrapper instance
    """
    wrapper = HEDWrapper(schema_version=schema_version)
    return wrapper 