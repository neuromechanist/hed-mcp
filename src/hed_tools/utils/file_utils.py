"""File handling utilities for various formats.

This module provides utilities for loading, saving, and validating files
in formats commonly used in BIDS and HED workflows.
"""

from typing import Union, Dict, Any, Optional, List
from pathlib import Path
import logging
import json
import pandas as pd
import asyncio
import aiofiles

logger = logging.getLogger(__name__)


class FileHandler:
    """Generic file handler for multiple formats.
    
    Supports common file operations for:
    - BIDS events files (TSV/CSV)
    - HED sidecar files (JSON)
    - Configuration files
    - Data validation and format conversion
    """
    
    def __init__(self):
        """Initialize the file handler."""
        self.supported_events_formats = ['.tsv', '.csv']
        self.supported_sidecar_formats = ['.json']
        self.encoding = 'utf-8'
    
    @staticmethod
    async def load_events_file(file_path: Path) -> Optional[pd.DataFrame]:
        """Load events file (TSV/CSV) asynchronously.
        
        Args:
            file_path: Path to events file
            
        Returns:
            Loaded DataFrame or None if loading failed
        """
        try:
            logger.info(f"Loading events file: {file_path}")
            
            if not file_path.exists():
                logger.error(f"Events file not found: {file_path}")
                return None
            
            # Determine format and load appropriately
            if file_path.suffix.lower() == '.tsv':
                df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
            elif file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path, encoding='utf-8')
            else:
                # Try to auto-detect format
                logger.warning(f"Unknown format {file_path.suffix}, attempting auto-detection")
                try:
                    # Try TSV first (more common in BIDS)
                    df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
                except:
                    # Fall back to CSV
                    df = pd.read_csv(file_path, encoding='utf-8')
            
            logger.info(f"Loaded events file: {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load events file {file_path}: {e}")
            return None
    
    @staticmethod
    async def save_events_file(df: pd.DataFrame, file_path: Path, 
                              format: str = 'tsv') -> bool:
        """Save events DataFrame to file.
        
        Args:
            df: DataFrame to save
            file_path: Output file path
            format: File format ('tsv' or 'csv')
            
        Returns:
            True if save was successful, False otherwise
        """
        try:
            logger.info(f"Saving events file: {file_path} (format: {format})")
            
            # Ensure output directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == 'tsv':
                df.to_csv(file_path, sep='\t', index=False, encoding='utf-8')
            elif format.lower() == 'csv':
                df.to_csv(file_path, index=False, encoding='utf-8')
            else:
                logger.error(f"Unsupported format: {format}")
                return False
            
            logger.info(f"Successfully saved events file: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save events file {file_path}: {e}")
            return False
    
    @staticmethod
    async def load_json_file(file_path: Path) -> Optional[Dict[str, Any]]:
        """Load JSON file asynchronously.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Loaded JSON data or None if loading failed
        """
        try:
            logger.info(f"Loading JSON file: {file_path}")
            
            if not file_path.exists():
                logger.error(f"JSON file not found: {file_path}")
                return None
            
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                data = json.loads(content)
            
            logger.info(f"Loaded JSON file: {len(data)} top-level keys")
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load JSON file {file_path}: {e}")
            return None
    
    @staticmethod
    async def save_json_file(data: Dict[str, Any], file_path: Path, 
                           indent: int = 2) -> bool:
        """Save data to JSON file asynchronously.
        
        Args:
            data: Data to save
            file_path: Output file path
            indent: JSON indentation level
            
        Returns:
            True if save was successful, False otherwise
        """
        try:
            logger.info(f"Saving JSON file: {file_path}")
            
            # Ensure output directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert numpy types to native Python types for JSON serialization
            json_data = FileHandler._convert_numpy_types(data)
            
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                json_str = json.dumps(json_data, indent=indent, ensure_ascii=False)
                await f.write(json_str)
            
            logger.info(f"Successfully saved JSON file: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save JSON file {file_path}: {e}")
            return False
    
    @staticmethod
    def _convert_numpy_types(obj: Any) -> Any:
        """Convert numpy types to native Python types for JSON serialization.
        
        Args:
            obj: Object that may contain numpy types
            
        Returns:
            Object with numpy types converted to native Python types
        """
        import numpy as np
        
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: FileHandler._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [FileHandler._convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    @staticmethod
    def validate_file_format(file_path: Path, 
                           expected_formats: List[str]) -> bool:
        """Validate file format against expected types.
        
        Args:
            file_path: Path to file to validate
            expected_formats: List of expected file extensions (e.g., ['.tsv', '.csv'])
            
        Returns:
            True if file format is valid, False otherwise
        """
        try:
            if not file_path.exists():
                logger.error(f"File does not exist: {file_path}")
                return False
            
            file_extension = file_path.suffix.lower()
            expected_extensions = [fmt.lower() for fmt in expected_formats]
            
            if file_extension not in expected_extensions:
                logger.error(f"Invalid file format {file_extension}. Expected: {expected_formats}")
                return False
            
            logger.info(f"File format validation passed: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"File format validation failed for {file_path}: {e}")
            return False
    
    @staticmethod
    async def validate_bids_events_structure(file_path: Path) -> Dict[str, Any]:
        """Validate BIDS events file structure and content.
        
        Args:
            file_path: Path to BIDS events file
            
        Returns:
            Validation report with errors, warnings, and statistics
        """
        try:
            validation_result = {
                "valid": True,
                "errors": [],
                "warnings": [],
                "file_info": {},
                "column_info": {}
            }
            
            # Check file existence and format
            if not file_path.exists():
                validation_result["valid"] = False
                validation_result["errors"].append(f"File not found: {file_path}")
                return validation_result
            
            # Check file extension
            if file_path.suffix.lower() not in ['.tsv', '.csv']:
                validation_result["warnings"].append(f"BIDS events files should be .tsv format, found: {file_path.suffix}")
            
            # Load and validate content
            df = await FileHandler.load_events_file(file_path)
            if df is None:
                validation_result["valid"] = False
                validation_result["errors"].append("Failed to load events file")
                return validation_result
            
            # Basic file statistics
            validation_result["file_info"] = {
                "rows": len(df),
                "columns": len(df.columns),
                "size_bytes": file_path.stat().st_size
            }
            
            # Check required BIDS columns
            required_columns = ['onset', 'duration']
            missing_required = [col for col in required_columns if col not in df.columns]
            if missing_required:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Missing required BIDS columns: {missing_required}")
            
            # Validate onset column
            if 'onset' in df.columns:
                if not pd.api.types.is_numeric_dtype(df['onset']):
                    validation_result["errors"].append("'onset' column must be numeric")
                elif df['onset'].isnull().any():
                    validation_result["errors"].append("'onset' column contains null values")
                elif (df['onset'] < 0).any():
                    validation_result["warnings"].append("'onset' column contains negative values")
            
            # Validate duration column
            if 'duration' in df.columns:
                if not pd.api.types.is_numeric_dtype(df['duration']):
                    validation_result["errors"].append("'duration' column must be numeric")
                elif (df['duration'] < 0).any():
                    validation_result["warnings"].append("'duration' column contains negative values")
            
            # Check column naming conventions
            for col in df.columns:
                if not col.islower():
                    validation_result["warnings"].append(f"Column '{col}' should be lowercase")
                if ' ' in col:
                    validation_result["warnings"].append(f"Column '{col}' should not contain spaces")
            
            # Column-specific information
            validation_result["column_info"] = {
                col: {
                    "type": str(df[col].dtype),
                    "null_count": int(df[col].isnull().sum()),
                    "unique_count": int(df[col].nunique())
                }
                for col in df.columns
            }
            
            if validation_result["errors"]:
                validation_result["valid"] = False
            
            logger.info(f"BIDS validation completed for {file_path}: {'VALID' if validation_result['valid'] else 'INVALID'}")
            return validation_result
            
        except Exception as e:
            logger.error(f"BIDS validation failed for {file_path}: {e}")
            return {
                "valid": False,
                "errors": [str(e)],
                "warnings": [],
                "file_info": {},
                "column_info": {}
            }
    
    @staticmethod
    async def create_backup(file_path: Path, backup_dir: Optional[Path] = None) -> Optional[Path]:
        """Create a backup copy of a file.
        
        Args:
            file_path: Path to file to backup
            backup_dir: Optional backup directory (defaults to file's directory)
            
        Returns:
            Path to backup file or None if backup failed
        """
        try:
            if not file_path.exists():
                logger.error(f"Cannot backup non-existent file: {file_path}")
                return None
            
            # Determine backup location
            if backup_dir is None:
                backup_dir = file_path.parent
            
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Create backup filename with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"
            backup_path = backup_dir / backup_name
            
            # Copy file content
            import shutil
            shutil.copy2(file_path, backup_path)
            
            logger.info(f"Created backup: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Failed to create backup for {file_path}: {e}")
            return None
    
    @staticmethod
    def get_file_info(file_path: Path) -> Dict[str, Any]:
        """Get comprehensive file information.
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary with file metadata and statistics
        """
        try:
            if not file_path.exists():
                return {"error": f"File not found: {file_path}"}
            
            stat = file_path.stat()
            return {
                "path": str(file_path),
                "name": file_path.name,
                "extension": file_path.suffix,
                "size_bytes": stat.st_size,
                "size_mb": stat.st_size / (1024 * 1024),
                "created": stat.st_ctime,
                "modified": stat.st_mtime,
                "is_file": file_path.is_file(),
                "is_readable": file_path.exists() and stat.st_mode & 0o444,
                "is_writable": file_path.exists() and stat.st_mode & 0o200
            }
            
        except Exception as e:
            logger.error(f"Failed to get file info for {file_path}: {e}")
            return {"error": str(e)}


def create_file_handler() -> FileHandler:
    """Factory function to create a file handler.
    
    Returns:
        Initialized FileHandler instance
    """
    return FileHandler() 