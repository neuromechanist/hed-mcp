"""BIDS event file parser for handling TSV format and metadata extraction.

This module provides functionality to parse BIDS event files, validate them against
BIDS specifications, and extract metadata from accompanying JSON sidecar files.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)


class BIDSValidationError(Exception):
    """Exception raised for BIDS validation errors."""

    pass


class BIDSEventParser:
    """Parser for BIDS event files with validation and metadata extraction.

    This class handles the parsing of BIDS-compliant event files in TSV format,
    validates required columns according to BIDS specifications, and extracts
    metadata from accompanying JSON sidecar files.
    """

    # Required columns according to BIDS specification
    REQUIRED_COLUMNS = ["onset", "duration"]

    # Common BIDS event columns for reference
    COMMON_COLUMNS = [
        "onset",
        "duration",
        "trial_type",
        "response_time",
        "stim_file",
        "HED",
        "sample",
        "value",
        "response_hand",
        "correct",
    ]

    def __init__(self, validate_bids: bool = True, encoding: str = "utf-8"):
        """Initialize the BIDS event parser.

        Args:
            validate_bids: Whether to perform BIDS validation
            encoding: File encoding to use when reading files
        """
        self.validate_bids = validate_bids
        self.encoding = encoding
        self.logger = logging.getLogger(__name__)

    def parse_events_file(
        self,
        file_path: Union[str, Path],
        sidecar_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Parse a BIDS events file and extract metadata.

        Args:
            file_path: Path to the events TSV file
            sidecar_path: Optional path to JSON sidecar file

        Returns:
            Tuple of (events_dataframe, metadata_dict)

        Raises:
            BIDSValidationError: If file doesn't meet BIDS specifications
            FileNotFoundError: If required files are not found
            ValueError: If file format is invalid
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Events file not found: {file_path}")

        # Load the TSV file
        try:
            events_df = self._load_tsv_file(file_path)
        except Exception as e:
            raise ValueError(f"Failed to load TSV file {file_path}: {e}")

        # Validate BIDS compliance if requested
        if self.validate_bids:
            self._validate_bids_columns(events_df, file_path)

        # Load sidecar metadata if available
        metadata = {}
        if sidecar_path:
            metadata = self._load_sidecar_metadata(Path(sidecar_path))
        else:
            # Try to find sidecar automatically
            auto_sidecar_path = self._find_sidecar_file(file_path)
            if auto_sidecar_path:
                metadata = self._load_sidecar_metadata(auto_sidecar_path)

        # Add file-level metadata
        metadata.update(self._extract_file_metadata(file_path, events_df))

        self.logger.info(f"Successfully parsed events file: {file_path}")
        self.logger.debug(
            f"Events shape: {events_df.shape}, Columns: {list(events_df.columns)}"
        )

        return events_df, metadata

    def _load_tsv_file(self, file_path: Path) -> pd.DataFrame:
        """Load TSV file with proper error handling.

        Args:
            file_path: Path to the TSV file

        Returns:
            DataFrame containing the events data

        Raises:
            ValueError: If file cannot be parsed as TSV
        """
        try:
            # Try TSV format first (tab-separated)
            events_df = pd.read_csv(
                file_path,
                sep="\t",
                encoding=self.encoding,
                na_values=["n/a", "N/A", "NA", ""],
            )
        except Exception as e:
            self.logger.warning(f"Failed to read as TSV, trying CSV format: {e}")
            try:
                # Fallback to CSV format
                events_df = pd.read_csv(
                    file_path,
                    encoding=self.encoding,
                    na_values=["n/a", "N/A", "NA", ""],
                )
            except Exception as csv_error:
                raise ValueError(
                    f"Could not parse file as TSV or CSV. TSV error: {e}, CSV error: {csv_error}"
                )

        if events_df.empty:
            raise ValueError("Events file is empty")

        return events_df

    def _validate_bids_columns(self, events_df: pd.DataFrame, file_path: Path) -> None:
        """Validate that required BIDS columns are present.

        Args:
            events_df: Events DataFrame to validate
            file_path: Path to the events file (for error reporting)

        Raises:
            BIDSValidationError: If required columns are missing
        """
        missing_columns = []
        for col in self.REQUIRED_COLUMNS:
            if col not in events_df.columns:
                missing_columns.append(col)

        if missing_columns:
            raise BIDSValidationError(
                f"Missing required BIDS columns in {file_path}: {missing_columns}. "
                f"Required columns: {self.REQUIRED_COLUMNS}"
            )

        # Validate onset column (must be numeric and non-negative)
        if "onset" in events_df.columns:
            if not pd.api.types.is_numeric_dtype(events_df["onset"]):
                raise BIDSValidationError(
                    f"'onset' column must be numeric in {file_path}"
                )

            if (events_df["onset"] < 0).any():
                raise BIDSValidationError(
                    f"'onset' column contains negative values in {file_path}"
                )

        # Validate duration column (must be numeric and non-negative)
        if "duration" in events_df.columns:
            if not pd.api.types.is_numeric_dtype(events_df["duration"]):
                raise BIDSValidationError(
                    f"'duration' column must be numeric in {file_path}"
                )

            if (events_df["duration"] < 0).any():
                raise BIDSValidationError(
                    f"'duration' column contains negative values in {file_path}"
                )

    def _find_sidecar_file(self, events_file_path: Path) -> Optional[Path]:
        """Automatically find the corresponding JSON sidecar file.

        Args:
            events_file_path: Path to the events TSV file

        Returns:
            Path to sidecar file if found, None otherwise
        """
        # Replace .tsv extension with .json
        sidecar_path = events_file_path.with_suffix(".json")

        if sidecar_path.exists():
            return sidecar_path

        # Try looking for task-level sidecar (without run number)
        if "run-" in events_file_path.name:
            # Remove run-XX part from filename
            base_name = events_file_path.name
            parts = base_name.split("_")
            filtered_parts = [part for part in parts if not part.startswith("run-")]
            task_sidecar_name = "_".join(filtered_parts).replace(".tsv", ".json")
            task_sidecar_path = events_file_path.parent / task_sidecar_name

            if task_sidecar_path.exists():
                return task_sidecar_path

        return None

    def _load_sidecar_metadata(self, sidecar_path: Path) -> Dict[str, Any]:
        """Load metadata from JSON sidecar file.

        Args:
            sidecar_path: Path to the JSON sidecar file

        Returns:
            Dictionary containing sidecar metadata

        Raises:
            ValueError: If JSON file cannot be parsed
        """
        try:
            with open(sidecar_path, "r", encoding=self.encoding) as f:
                metadata = json.load(f)

            self.logger.debug(f"Loaded sidecar metadata from: {sidecar_path}")
            return metadata

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in sidecar file {sidecar_path}: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load sidecar file {sidecar_path}: {e}")

    def _extract_file_metadata(
        self, file_path: Path, events_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Extract metadata from the file and DataFrame.

        Args:
            file_path: Path to the events file
            events_df: Events DataFrame

        Returns:
            Dictionary containing file metadata
        """
        return {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "n_events": len(events_df),
            "columns": list(events_df.columns),
            "n_columns": len(events_df.columns),
            "file_size_bytes": file_path.stat().st_size,
            "duration_total": events_df["duration"].sum()
            if "duration" in events_df.columns
            else None,
            "onset_range": [events_df["onset"].min(), events_df["onset"].max()]
            if "onset" in events_df.columns
            else None,
        }

    def parse_multiple_files(
        self,
        file_paths: List[Union[str, Path]],
        sidecar_paths: Optional[List[Union[str, Path]]] = None,
    ) -> List[Tuple[pd.DataFrame, Dict[str, Any]]]:
        """Parse multiple BIDS events files.

        Args:
            file_paths: List of paths to events files
            sidecar_paths: Optional list of paths to sidecar files

        Returns:
            List of tuples (events_dataframe, metadata_dict)
        """
        results = []
        sidecar_paths = sidecar_paths or [None] * len(file_paths)

        for i, file_path in enumerate(file_paths):
            try:
                sidecar_path = sidecar_paths[i] if i < len(sidecar_paths) else None
                result = self.parse_events_file(file_path, sidecar_path)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to parse {file_path}: {e}")
                # Continue with other files
                continue

        return results

    def get_column_info(self, events_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Extract detailed information about each column.

        Args:
            events_df: Events DataFrame

        Returns:
            Dictionary with column names as keys and column info as values
        """
        column_info = {}

        for col in events_df.columns:
            series = events_df[col]

            info = {
                "dtype": str(series.dtype),
                "non_null_count": series.count(),
                "null_count": series.isnull().sum(),
                "unique_count": series.nunique(),
            }

            # Add type-specific information
            if pd.api.types.is_numeric_dtype(series):
                info.update(
                    {
                        "min": series.min(),
                        "max": series.max(),
                        "mean": series.mean(),
                        "std": series.std(),
                    }
                )
            elif pd.api.types.is_object_dtype(series):
                # Get unique values (limited to first 20 for large datasets)
                unique_vals = series.dropna().unique()
                info.update(
                    {
                        "unique_values": unique_vals[:20].tolist(),
                        "sample_values": series.dropna().head(10).tolist(),
                    }
                )

            column_info[col] = info

        return column_info


def parse_bids_events(
    file_path: Union[str, Path],
    sidecar_path: Optional[Union[str, Path]] = None,
    validate_bids: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Convenience function to parse a single BIDS events file.

    Args:
        file_path: Path to the events TSV file
        sidecar_path: Optional path to JSON sidecar file
        validate_bids: Whether to perform BIDS validation

    Returns:
        Tuple of (events_dataframe, metadata_dict)
    """
    parser = BIDSEventParser(validate_bids=validate_bids)
    return parser.parse_events_file(file_path, sidecar_path)
