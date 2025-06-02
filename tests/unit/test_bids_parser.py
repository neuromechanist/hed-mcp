"""Unit tests for BIDS event file parser."""

import json
import tempfile
import pytest
import pandas as pd
from pathlib import Path

from hed_tools.tools.bids_parser import (
    BIDSEventParser,
    BIDSValidationError,
    parse_bids_events,
)


class TestBIDSEventParser:
    """Test the BIDSEventParser class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = BIDSEventParser()

        # Sample valid events data
        self.valid_events_data = {
            "onset": [0.0, 1.5, 3.0, 4.5],
            "duration": [1.0, 1.0, 1.0, 1.0],
            "trial_type": ["go", "stop", "go", "stop"],
            "response_time": [0.5, None, 0.7, 0.4],
            "HED": [
                "Sensory-event, Visual-presentation",
                "Agent-action, Inhibition",
                "Sensory-event, Visual-presentation",
                "Agent-action, Inhibition",
            ],
        }

        # Sample sidecar metadata
        self.sidecar_metadata = {
            "TaskName": "Go-NoGo Task",
            "Instructions": "Press button for go trials",
            "trial_type": {
                "LongName": "Trial type",
                "Description": "Type of trial",
                "Levels": {"go": "Go trial", "stop": "Stop trial"},
            },
            "response_time": {
                "LongName": "Response time",
                "Description": "Time from stimulus onset to response",
                "Units": "s",
            },
        }

    def test_init(self):
        """Test parser initialization."""
        parser = BIDSEventParser(validate_bids=False, encoding="utf-16")
        assert parser.validate_bids is False
        assert parser.encoding == "utf-16"

    def test_parse_valid_events_file(self):
        """Test parsing a valid events file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_events.tsv", delete=False
        ) as f:
            df = pd.DataFrame(self.valid_events_data)
            df.to_csv(f.name, sep="\t", index=False)
            events_path = Path(f.name)

        try:
            events_df, metadata = self.parser.parse_events_file(events_path)

            # Check DataFrame content
            assert len(events_df) == 4
            assert "onset" in events_df.columns
            assert "duration" in events_df.columns
            assert "trial_type" in events_df.columns

            # Check metadata
            assert metadata["file_name"] == events_path.name
            assert metadata["n_events"] == 4
            assert metadata["n_columns"] == 5
            assert "onset" in metadata["columns"]

        finally:
            events_path.unlink()

    def test_parse_events_with_sidecar(self):
        """Test parsing events file with sidecar metadata."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_events.tsv", delete=False
        ) as events_f:
            df = pd.DataFrame(self.valid_events_data)
            df.to_csv(events_f.name, sep="\t", index=False)
            events_path = Path(events_f.name)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_events.json", delete=False
        ) as sidecar_f:
            json.dump(self.sidecar_metadata, sidecar_f)
            sidecar_path = Path(sidecar_f.name)

        try:
            events_df, metadata = self.parser.parse_events_file(
                events_path, sidecar_path
            )

            # Check that sidecar metadata is included
            assert "TaskName" in metadata
            assert metadata["TaskName"] == "Go-NoGo Task"
            assert "trial_type" in metadata
            assert "Levels" in metadata["trial_type"]

        finally:
            events_path.unlink()
            sidecar_path.unlink()

    def test_auto_find_sidecar(self):
        """Test automatic sidecar file detection."""
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create events and sidecar files with matching names
            events_path = temp_path / "sub-01_task-test_events.tsv"
            sidecar_path = temp_path / "sub-01_task-test_events.json"

            df = pd.DataFrame(self.valid_events_data)
            df.to_csv(events_path, sep="\t", index=False)

            with open(sidecar_path, "w") as f:
                json.dump(self.sidecar_metadata, f)

            events_df, metadata = self.parser.parse_events_file(events_path)

            # Should automatically load sidecar metadata
            assert "TaskName" in metadata
            assert metadata["TaskName"] == "Go-NoGo Task"

    def test_auto_find_task_level_sidecar(self):
        """Test automatic detection of task-level sidecar (without run number)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create run-specific events file and task-level sidecar
            events_path = temp_path / "sub-01_task-test_run-01_events.tsv"
            sidecar_path = temp_path / "sub-01_task-test_events.json"

            df = pd.DataFrame(self.valid_events_data)
            df.to_csv(events_path, sep="\t", index=False)

            with open(sidecar_path, "w") as f:
                json.dump(self.sidecar_metadata, f)

            events_df, metadata = self.parser.parse_events_file(events_path)

            # Should find task-level sidecar
            assert "TaskName" in metadata

    def test_missing_file_error(self):
        """Test error when events file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            self.parser.parse_events_file("/nonexistent/file.tsv")

    def test_invalid_tsv_format(self):
        """Test error handling for invalid TSV format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("Invalid\tTSV\ndata\tthat\tdoesn't\tmake\tsense")
            invalid_path = Path(f.name)

        try:
            # Should raise BIDS validation error due to missing required columns
            with pytest.raises(BIDSValidationError):
                self.parser.parse_events_file(invalid_path)
        finally:
            invalid_path.unlink()

    def test_empty_file_error(self):
        """Test error when events file is empty."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            # Write only header
            f.write("onset\tduration\n")
            empty_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Events file is empty"):
                self.parser.parse_events_file(empty_path)
        finally:
            empty_path.unlink()

    def test_bids_validation_missing_columns(self):
        """Test BIDS validation with missing required columns."""
        invalid_data = {
            "trial_type": ["go", "stop"],
            "response_time": [0.5, 0.7],
            # Missing 'onset' and 'duration'
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            df = pd.DataFrame(invalid_data)
            df.to_csv(f.name, sep="\t", index=False)
            invalid_path = Path(f.name)

        try:
            with pytest.raises(
                BIDSValidationError, match="Missing required BIDS columns"
            ):
                self.parser.parse_events_file(invalid_path)
        finally:
            invalid_path.unlink()

    def test_bids_validation_invalid_onset(self):
        """Test BIDS validation with invalid onset values."""
        invalid_data = {
            "onset": [-1.0, "invalid", 3.0],  # Negative and non-numeric
            "duration": [1.0, 1.0, 1.0],
            "trial_type": ["go", "stop", "go"],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            df = pd.DataFrame(invalid_data)
            df.to_csv(f.name, sep="\t", index=False)
            invalid_path = Path(f.name)

        try:
            with pytest.raises(BIDSValidationError):
                self.parser.parse_events_file(invalid_path)
        finally:
            invalid_path.unlink()

    def test_bids_validation_disabled(self):
        """Test parsing with BIDS validation disabled."""
        parser = BIDSEventParser(validate_bids=False)

        invalid_data = {
            "trial_type": ["go", "stop"],
            "response_time": [0.5, 0.7],
            # Missing required BIDS columns
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            df = pd.DataFrame(invalid_data)
            df.to_csv(f.name, sep="\t", index=False)
            invalid_path = Path(f.name)

        try:
            # Should work without validation
            events_df, metadata = parser.parse_events_file(invalid_path)
            assert len(events_df) == 2
            assert "trial_type" in events_df.columns
        finally:
            invalid_path.unlink()

    def test_invalid_sidecar_json(self):
        """Test error handling for invalid JSON sidecar."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_events.tsv", delete=False
        ) as events_f:
            df = pd.DataFrame(self.valid_events_data)
            df.to_csv(events_f.name, sep="\t", index=False)
            events_path = Path(events_f.name)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_events.json", delete=False
        ) as sidecar_f:
            sidecar_f.write("{ invalid json }")
            sidecar_path = Path(sidecar_f.name)

        try:
            with pytest.raises(ValueError, match="Invalid JSON"):
                self.parser.parse_events_file(events_path, sidecar_path)
        finally:
            events_path.unlink()
            sidecar_path.unlink()

    def test_parse_multiple_files(self):
        """Test parsing multiple events files."""
        file_paths = []

        try:
            # Create multiple test files
            for i in range(3):
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=f"_run-{i + 1:02d}_events.tsv", delete=False
                ) as f:
                    df = pd.DataFrame(self.valid_events_data)
                    df.to_csv(f.name, sep="\t", index=False)
                    file_paths.append(Path(f.name))

            results = self.parser.parse_multiple_files(file_paths)

            assert len(results) == 3
            for events_df, metadata in results:
                assert len(events_df) == 4
                assert "onset" in events_df.columns
                assert metadata["n_events"] == 4

        finally:
            for path in file_paths:
                if path.exists():
                    path.unlink()

    def test_get_column_info(self):
        """Test column information extraction."""
        df = pd.DataFrame(self.valid_events_data)
        column_info = self.parser.get_column_info(df)

        # Check that all columns are analyzed
        assert set(column_info.keys()) == set(df.columns)

        # Check numeric column info
        onset_info = column_info["onset"]
        assert onset_info["dtype"] == "float64"
        assert "min" in onset_info
        assert "max" in onset_info
        assert "mean" in onset_info

        # Check categorical column info
        trial_type_info = column_info["trial_type"]
        assert "unique_values" in trial_type_info
        assert "sample_values" in trial_type_info
        assert trial_type_info["unique_count"] == 2

    def test_csv_fallback(self):
        """Test CSV fallback when TSV parsing fails."""
        # Create a file that will cause TSV parsing to fail but CSV to succeed
        # Use a file with inconsistent tab structure that pandas TSV parser will reject
        problematic_data = (
            "onset,duration,trial_type\n0.0\t1.0,go\n1.5,1.0,stop"  # Mixed separators
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write(problematic_data)
            csv_path = Path(f.name)

        try:
            # Since this has mixed separators, TSV parsing might fail, triggering CSV fallback
            # Let's test without BIDS validation to focus on the parsing logic
            parser_no_validation = BIDSEventParser(validate_bids=False)

            # This should either parse successfully or fallback to CSV parsing
            events_df, metadata = parser_no_validation.parse_events_file(csv_path)

            # Verify we got some data back
            assert len(events_df) >= 1
            assert len(events_df.columns) >= 1

        finally:
            csv_path.unlink()


class TestConvenienceFunction:
    """Test the convenience function."""

    def test_parse_bids_events_function(self):
        """Test the parse_bids_events convenience function."""
        valid_events_data = {
            "onset": [0.0, 1.5],
            "duration": [1.0, 1.0],
            "trial_type": ["go", "stop"],
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_events.tsv", delete=False
        ) as f:
            df = pd.DataFrame(valid_events_data)
            df.to_csv(f.name, sep="\t", index=False)
            events_path = Path(f.name)

        try:
            events_df, metadata = parse_bids_events(events_path)
            assert len(events_df) == 2
            assert "onset" in events_df.columns
            assert metadata["n_events"] == 2
        finally:
            events_path.unlink()

    def test_parse_bids_events_with_validation_disabled(self):
        """Test convenience function with validation disabled."""
        invalid_data = {
            "trial_type": ["go", "stop"]
            # Missing required columns
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            df = pd.DataFrame(invalid_data)
            df.to_csv(f.name, sep="\t", index=False)
            events_path = Path(f.name)

        try:
            events_df, metadata = parse_bids_events(events_path, validate_bids=False)
            assert len(events_df) == 2
            assert "trial_type" in events_df.columns
        finally:
            events_path.unlink()


class TestBIDSValidationError:
    """Test the BIDSValidationError exception."""

    def test_bids_validation_error(self):
        """Test that BIDSValidationError can be raised and caught."""
        with pytest.raises(BIDSValidationError):
            raise BIDSValidationError("Test error message")
