#!/usr/bin/env python3
"""Debug script to see the structure of column analysis results."""

import asyncio
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, "src")

try:
    from hed_tools.tools.column_analyzer import ColumnAnalyzer
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


async def debug_column_analysis():
    """Debug column analysis structure."""
    print("=== Debugging Column Analysis Structure ===")

    try:
        analyzer = ColumnAnalyzer()
        file_path = "examples/contrast_change_detection_events.tsv"

        if Path(file_path).exists():
            print(f"Analyzing {file_path}...")
            result = await analyzer.analyze_events_file(file_path)

            print("Result structure:")
            print(json.dumps(result, indent=2, default=str))

        else:
            print(f"File not found: {file_path}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(debug_column_analysis())
