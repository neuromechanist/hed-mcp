"""Data ingestion stage for HED sidecar generation pipeline.

This stage handles:
- File loading and format validation
- Data preprocessing and cleaning
- Initial metadata extraction
- Error detection and reporting
"""

import csv
import io
import logging
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Union

from . import (
    PipelineStage,
    StageInput,
    StageOutput,
    create_stage_output,
    register_stage,
)

logger = logging.getLogger(__name__)


class DataIngestionStage(PipelineStage):
    """Data ingestion stage for loading and validating input data.

    Supports multiple input formats and provides comprehensive validation.
    """

    def __init__(self, config: Dict[str, Any] = None):
        # Set a default name for this stage
        super().__init__("DataIngestion", config)

        # Configuration parameters
        self.max_file_size_mb = self.get_config_value("max_file_size_mb", 100)
        self.supported_extensions = self.get_config_value(
            "supported_extensions", [".csv", ".tsv", ".xlsx", ".xls"]
        )
        self.default_encoding = self.get_config_value("encoding", "utf-8")
        self.validate_headers = self.get_config_value("validate_headers", True)
        self.clean_data = self.get_config_value("clean_data", True)
        self.sample_size = self.get_config_value("sample_size", 1000)

    async def _initialize_implementation(self) -> None:
        """Initialize the data ingestion stage."""
        self.logger.info(
            f"Initializing data ingestion: max_size={self.max_file_size_mb}MB, "
            f"extensions={self.supported_extensions}"
        )

    async def _execute_implementation(self, stage_input: StageInput) -> StageOutput:
        """Execute data ingestion for the given input."""
        input_data = stage_input.get_data()

        # Handle different input types
        if isinstance(input_data, (str, Path)):
            # File path input
            return await self._load_from_file(
                file_path=input_data,
                metadata=stage_input.metadata,
                context=stage_input.context,
            )
        elif isinstance(input_data, (str, bytes)) and len(input_data) > 0:
            # Raw data input (CSV/TSV content)
            return await self._load_from_string(
                data_content=input_data,
                metadata=stage_input.metadata,
                context=stage_input.context,
            )
        elif isinstance(input_data, pd.DataFrame):
            # DataFrame input (already loaded)
            return await self._process_dataframe(
                dataframe=input_data,
                metadata=stage_input.metadata,
                context=stage_input.context,
            )
        else:
            return create_stage_output(
                data=None,
                metadata=stage_input.metadata,
                context=stage_input.context,
                errors=[f"Unsupported input type: {type(input_data)}"],
            )

    async def _load_from_file(
        self,
        file_path: Union[str, Path],
        metadata: Dict[str, Any],
        context: Dict[str, Any],
    ) -> StageOutput:
        """Load data from a file path."""
        file_path = Path(file_path)

        # Validate file existence
        if not file_path.exists():
            return create_stage_output(
                data=None,
                metadata=metadata,
                context=context,
                errors=[f"File not found: {file_path}"],
            )

        # Validate file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            return create_stage_output(
                data=None,
                metadata=metadata,
                context=context,
                errors=[
                    f"File too large: {file_size_mb:.1f}MB exceeds limit of {self.max_file_size_mb}MB"
                ],
            )

        # Validate file extension
        if file_path.suffix.lower() not in self.supported_extensions:
            return create_stage_output(
                data=None,
                metadata=metadata,
                context=context,
                errors=[
                    f"Unsupported file type: {file_path.suffix}. "
                    f"Supported: {', '.join(self.supported_extensions)}"
                ],
            )

        try:
            # Load based on file extension
            if file_path.suffix.lower() in [".csv", ".tsv"]:
                dataframe = await self._load_csv_file(file_path)
            elif file_path.suffix.lower() in [".xlsx", ".xls"]:
                dataframe = await self._load_excel_file(file_path)
            else:
                return create_stage_output(
                    data=None,
                    metadata=metadata,
                    context=context,
                    errors=[f"Cannot load file type: {file_path.suffix}"],
                )

            # Update metadata with file information
            file_metadata = {
                "source_file": str(file_path),
                "file_size_mb": file_size_mb,
                "file_type": file_path.suffix.lower(),
                "file_name": file_path.name,
            }
            metadata.update(file_metadata)

            return await self._process_dataframe(dataframe, metadata, context)

        except Exception as e:
            self.logger.error(f"Failed to load file {file_path}: {e}")
            return create_stage_output(
                data=None,
                metadata=metadata,
                context=context,
                errors=[f"Failed to load file: {str(e)}"],
            )

    async def _load_csv_file(self, file_path: Path) -> pd.DataFrame:
        """Load CSV or TSV file."""
        # Detect delimiter
        delimiter = "\t" if file_path.suffix.lower() == ".tsv" else ","

        # Try different encodings if default fails
        encodings_to_try = [self.default_encoding, "utf-8-sig", "latin-1", "iso-8859-1"]

        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(
                    file_path,
                    delimiter=delimiter,
                    encoding=encoding,
                    na_values=["", "NA", "N/A", "null", "NULL"],
                    keep_default_na=True,
                )
                self.logger.debug(
                    f"Successfully loaded {file_path} with encoding {encoding}"
                )
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                if encoding == encodings_to_try[-1]:  # Last encoding to try
                    raise e
                continue

        raise ValueError(
            f"Could not decode file with any supported encoding: {encodings_to_try}"
        )

    async def _load_excel_file(self, file_path: Path) -> pd.DataFrame:
        """Load Excel file."""
        try:
            # Load first sheet by default
            df = pd.read_excel(
                file_path, sheet_name=0, na_values=["", "NA", "N/A", "null", "NULL"]
            )
            return df
        except Exception as e:
            raise ValueError(f"Failed to load Excel file: {str(e)}")

    async def _load_from_string(
        self,
        data_content: Union[str, bytes],
        metadata: Dict[str, Any],
        context: Dict[str, Any],
    ) -> StageOutput:
        """Load data from string content."""
        try:
            # Convert bytes to string if necessary
            if isinstance(data_content, bytes):
                data_content = data_content.decode(self.default_encoding)

            # Detect delimiter by examining first few lines
            delimiter = self._detect_delimiter(data_content)

            # Load using pandas
            df = pd.read_csv(
                io.StringIO(data_content),
                delimiter=delimiter,
                na_values=["", "NA", "N/A", "null", "NULL"],
                keep_default_na=True,
            )

            # Update metadata
            string_metadata = {
                "source_type": "string_content",
                "delimiter": delimiter,
                "content_length": len(data_content),
            }
            metadata.update(string_metadata)

            return await self._process_dataframe(df, metadata, context)

        except Exception as e:
            self.logger.error(f"Failed to load from string content: {e}")
            return create_stage_output(
                data=None,
                metadata=metadata,
                context=context,
                errors=[f"Failed to parse content: {str(e)}"],
            )

    def _detect_delimiter(self, content: str) -> str:
        """Detect the delimiter used in CSV content."""
        # Sample first few lines
        lines = content.split("\n")[:5]
        sample = "\n".join(lines)

        # Use csv.Sniffer to detect delimiter
        try:
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample, delimiters=",\t;|").delimiter
            return delimiter
        except Exception:
            # Fallback: count occurrences of common delimiters
            tab_count = sample.count("\t")
            comma_count = sample.count(",")
            semicolon_count = sample.count(";")

            if tab_count > comma_count and tab_count > semicolon_count:
                return "\t"
            elif semicolon_count > comma_count:
                return ";"
            else:
                return ","

    async def _process_dataframe(
        self, dataframe: pd.DataFrame, metadata: Dict[str, Any], context: Dict[str, Any]
    ) -> StageOutput:
        """Process and validate loaded DataFrame."""
        warnings = []
        errors = []

        # Basic validation
        if dataframe.empty:
            errors.append("Dataset is empty")
            return create_stage_output(
                data=None, metadata=metadata, context=context, errors=errors
            )

        # Sample data if requested
        original_rows = len(dataframe)
        if self.sample_size and self.sample_size < original_rows:
            dataframe = dataframe.sample(n=self.sample_size, random_state=42)
            warnings.append(
                f"Sampled {self.sample_size} rows from {original_rows} total rows"
            )

        # Data cleaning
        if self.clean_data:
            dataframe, cleaning_warnings = self._clean_dataframe(dataframe)
            warnings.extend(cleaning_warnings)

        # Header validation
        if self.validate_headers:
            header_warnings = self._validate_headers(dataframe)
            warnings.extend(header_warnings)

        # Extract data characteristics
        data_info = self._extract_data_info(dataframe)

        # Update metadata with processing information
        processing_metadata = {
            "rows_processed": len(dataframe),
            "columns_processed": len(dataframe.columns),
            "original_rows": original_rows,
            "processing_applied": {
                "cleaned": self.clean_data,
                "sampled": self.sample_size is not None
                and self.sample_size < original_rows,
                "headers_validated": self.validate_headers,
            },
            "data_characteristics": data_info,
        }
        metadata.update(processing_metadata)

        # Set context for next stages
        context.update(
            {
                "dataframe_shape": dataframe.shape,
                "column_names": list(dataframe.columns),
                "data_types": {
                    col: str(dtype) for col, dtype in dataframe.dtypes.items()
                },
            }
        )

        self.logger.info(
            f"Successfully ingested data: {len(dataframe)} rows, {len(dataframe.columns)} columns"
        )

        return create_stage_output(
            data=dataframe,
            metadata=metadata,
            context=context,
            warnings=warnings,
            errors=errors,
        )

    def _clean_dataframe(self, df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
        """Clean the DataFrame and return warnings."""
        warnings = []
        cleaned_df = df.copy()

        # Remove completely empty rows
        empty_rows_before = len(cleaned_df)
        cleaned_df = cleaned_df.dropna(how="all")
        empty_rows_removed = empty_rows_before - len(cleaned_df)

        if empty_rows_removed > 0:
            warnings.append(f"Removed {empty_rows_removed} completely empty rows")

        # Strip whitespace from string columns
        string_columns = cleaned_df.select_dtypes(include=["object"]).columns
        for col in string_columns:
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip()

        # Replace common null representations
        null_values = ["n/a", "N/A", "null", "NULL", "none", "None", "-", "--"]
        cleaned_df = cleaned_df.replace(null_values, pd.NA)

        return cleaned_df, warnings

    def _validate_headers(self, df: pd.DataFrame) -> List[str]:
        """Validate DataFrame headers and return warnings."""
        warnings = []

        # Check for duplicate column names
        if df.columns.duplicated().any():
            duplicates = df.columns[df.columns.duplicated()].tolist()
            warnings.append(f"Found duplicate column names: {duplicates}")

        # Check for empty column names
        empty_cols = [i for i, col in enumerate(df.columns) if not col or pd.isna(col)]
        if empty_cols:
            warnings.append(f"Found empty column names at positions: {empty_cols}")

        # Check for very long column names
        long_cols = [col for col in df.columns if len(str(col)) > 100]
        if long_cols:
            warnings.append(
                f"Found very long column names (>100 chars): {len(long_cols)} columns"
            )

        # Check for special characters in column names
        special_char_cols = []
        for col in df.columns:
            col_str = str(col)
            if any(
                char in col_str
                for char in ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]
            ):
                special_char_cols.append(col)

        if special_char_cols:
            warnings.append(
                f"Column names contain special characters: {len(special_char_cols)} columns"
            )

        return warnings

    def _extract_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract characteristics and statistics about the data."""
        info = {
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
            "null_counts": df.isnull().sum().to_dict(),
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "numeric_columns": list(df.select_dtypes(include=["number"]).columns),
            "string_columns": list(df.select_dtypes(include=["object"]).columns),
            "datetime_columns": list(df.select_dtypes(include=["datetime"]).columns),
        }

        # Calculate null percentage
        info["null_percentages"] = {
            col: (count / len(df) * 100) for col, count in info["null_counts"].items()
        }

        # Identify columns with high null rates
        info["high_null_columns"] = [
            col for col, pct in info["null_percentages"].items() if pct > 50
        ]

        return info

    async def _validate_input(self, stage_input: StageInput) -> None:
        """Validate input for data ingestion stage."""
        await super()._validate_input(stage_input)

        input_data = stage_input.get_data()

        # Check for valid input types
        valid_types = (str, Path, bytes, pd.DataFrame)
        if not isinstance(input_data, valid_types):
            raise ValueError(
                f"Invalid input type for data ingestion: {type(input_data)}. "
                f"Expected one of: {valid_types}"
            )

    async def _cleanup_implementation(self) -> None:
        """Clean up data ingestion stage resources."""
        # No specific cleanup needed for this stage
        pass


# Register the stage
register_stage("data_ingestion", DataIngestionStage)
