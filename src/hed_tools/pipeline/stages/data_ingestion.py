"""Data ingestion stage for the HED sidecar generation pipeline.

This stage handles loading and initial validation of BIDS event files,
extracting basic metadata and preparing data for subsequent processing stages.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

from ..core import PipelineStage, PipelineContext


class DataIngestionStage(PipelineStage):
    """Stage for loading and validating input data files.

    This stage:
    1. Validates file path and accessibility
    2. Detects file format and encoding
    3. Loads data into a pandas DataFrame
    4. Extracts basic metadata about the file
    5. Validates file size and format requirements
    """

    async def validate_prerequisites(self, context: PipelineContext) -> bool:
        """Validate that required input parameters are present."""
        file_path = context.input_data.get("file_path")
        if not file_path:
            context.add_error("file_path parameter is required", self.name)
            return False

        if not isinstance(file_path, str) or not file_path.strip():
            context.add_error("file_path must be a non-empty string", self.name)
            return False

        return True

    async def execute(self, context: PipelineContext) -> bool:
        """Execute data ingestion and validation."""
        try:
            file_path = context.input_data["file_path"]
            self.logger.info(f"Loading data from: {file_path}")

            # Validate file exists and is accessible
            if not await self._validate_file_access(file_path, context):
                return False

            # Detect file format and load data
            df = await self._load_data_file(file_path, context)
            if df is None:
                return False

            # Extract and validate metadata
            metadata = await self._extract_file_metadata(file_path, df, context)

            # Store results in context
            context.processed_data["dataframe"] = df
            context.metadata.update(metadata)
            context.set_stage_result(
                self.name,
                {
                    "file_loaded": True,
                    "row_count": len(df),
                    "column_count": len(df.columns),
                    "columns": df.columns.tolist(),
                    "file_size_mb": metadata.get("file_size_mb", 0),
                    "encoding": metadata.get("encoding", "unknown"),
                },
            )

            self.logger.info(
                f"Successfully loaded {len(df)} rows with {len(df.columns)} columns"
            )
            return True

        except Exception as e:
            context.add_error(f"Data ingestion failed: {str(e)}", self.name)
            self.logger.error(f"Data ingestion error: {e}", exc_info=True)
            return False

    async def _validate_file_access(
        self, file_path: str, context: PipelineContext
    ) -> bool:
        """Validate file exists and is accessible."""
        path = Path(file_path)

        if not path.exists():
            context.add_error(f"File not found: {file_path}", self.name)
            return False

        if not path.is_file():
            context.add_error(f"Path is not a file: {file_path}", self.name)
            return False

        if not os.access(file_path, os.R_OK):
            context.add_error(f"File not readable: {file_path}", self.name)
            return False

        # Check file size limits
        max_size_mb = self.config.get("max_file_size_mb", 100)
        file_size_mb = path.stat().st_size / (1024 * 1024)

        if file_size_mb > max_size_mb:
            context.add_error(
                f"File too large: {file_size_mb:.1f}MB exceeds limit of {max_size_mb}MB",
                self.name,
            )
            return False

        return True

    async def _load_data_file(
        self, file_path: str, context: PipelineContext
    ) -> Optional[pd.DataFrame]:
        """Load data file into pandas DataFrame."""
        path = Path(file_path)
        encoding = self.config.get("encoding", "utf-8")
        delimiter = self.config.get("delimiter", "auto")

        try:
            # Auto-detect delimiter if needed
            if delimiter == "auto":
                delimiter = self._detect_delimiter(file_path)

            # Handle different file formats
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(file_path, encoding=encoding, sep=delimiter)
            elif path.suffix.lower() == ".tsv":
                df = pd.read_csv(file_path, encoding=encoding, sep="\t")
            elif path.suffix.lower() in [".txt"]:
                # Try tab first, then comma
                try:
                    df = pd.read_csv(file_path, encoding=encoding, sep="\t")
                except (pd.errors.ParserError, pd.errors.EmptyDataError):
                    df = pd.read_csv(file_path, encoding=encoding, sep=",")
            else:
                # Default to pandas auto-detection
                df = pd.read_csv(file_path, encoding=encoding, sep=delimiter)

            # Basic validation
            if df.empty:
                context.add_error("Loaded file contains no data", self.name)
                return None

            if len(df.columns) == 0:
                context.add_error("Loaded file contains no columns", self.name)
                return None

            # Log any columns with all NaN values
            empty_cols = df.columns[df.isnull().all()].tolist()
            if empty_cols:
                context.add_warning(
                    f"Found completely empty columns: {empty_cols}", self.name
                )

            return df

        except UnicodeDecodeError as e:
            context.add_error(f"Encoding error loading file: {str(e)}", self.name)
            return None
        except pd.errors.EmptyDataError:
            context.add_error("File contains no data", self.name)
            return None
        except Exception as e:
            context.add_error(f"Error loading file: {str(e)}", self.name)
            return None

    def _detect_delimiter(self, file_path: str) -> str:
        """Detect the delimiter used in the file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                first_line = f.readline()

            # Count potential delimiters
            tab_count = first_line.count("\t")
            comma_count = first_line.count(",")
            semicolon_count = first_line.count(";")

            # Return the most common delimiter
            if tab_count >= comma_count and tab_count >= semicolon_count:
                return "\t"
            elif comma_count >= semicolon_count:
                return ","
            else:
                return ";"

        except Exception:
            # Default to tab for BIDS event files
            return "\t"

    async def _extract_file_metadata(
        self, file_path: str, df: pd.DataFrame, context: PipelineContext
    ) -> Dict[str, Any]:
        """Extract metadata about the loaded file."""
        path = Path(file_path)

        metadata = {
            "source_file": str(path.absolute()),
            "file_name": path.name,
            "file_size_mb": path.stat().st_size / (1024 * 1024),
            "encoding": self.config.get("encoding", "utf-8"),
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
        }

        # Add basic statistics about columns
        metadata["column_stats"] = {}
        for col in df.columns:
            col_stats = {
                "dtype": str(df[col].dtype),
                "null_count": df[col].isnull().sum(),
                "unique_count": df[col].nunique(),
            }

            # Add type-specific stats
            if df[col].dtype in ["int64", "float64"]:
                col_stats.update(
                    {
                        "min": df[col].min() if not df[col].empty else None,
                        "max": df[col].max() if not df[col].empty else None,
                        "mean": df[col].mean() if not df[col].empty else None,
                    }
                )

            metadata["column_stats"][col] = col_stats

        return metadata

    async def cleanup(self, context: PipelineContext) -> None:
        """Cleanup resources after data ingestion."""
        # No specific cleanup needed for this stage
        pass
