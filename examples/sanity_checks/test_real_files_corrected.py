#!/usr/bin/env python3
"""Test script for HED MCP tools with real-world event files - corrected imports."""

import asyncio
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, "src")

try:
    from hed_tools.tools.column_analyzer import ColumnAnalyzer
    from hed_tools.hed_integration.validation import SidecarGenerator
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


async def test_column_analysis():
    """Test column analysis with real event files."""
    print("=== Testing Column Analysis ===")

    try:
        analyzer = ColumnAnalyzer()

        # Test both event files
        files = [
            "examples/contrast_change_detection_events.tsv",
            "examples/surround_supp_events.tsv",
        ]

        results = {}

        for file_path in files:
            if Path(file_path).exists():
                print(f"\n--- Analyzing {file_path} ---")
                result = await analyzer.analyze_events_file(file_path)

                print(f"File: {result['source_file']}")
                print(
                    f"Rows: {result['file_info']['total_rows']}, Columns: {result['file_info']['total_columns']}"
                )
                print("Columns found:", list(result["columns"].keys()))

                # Extract skip and value columns based on analysis
                likely_skip_cols = []
                likely_value_cols = []

                # Get skip columns from BIDS patterns (timing columns)
                for pattern in result.get("patterns_detected", []):
                    if pattern["pattern_name"] in ["onset", "duration", "sample"]:
                        likely_skip_cols.append(pattern["column_name"])

                # Get value columns from HED candidates
                for candidate in result.get("hed_candidates", []):
                    if candidate["type"] == "categorical":
                        likely_value_cols.append(candidate["column"])

                print("Suggested skip_cols:", likely_skip_cols)
                print("Suggested value_cols:", likely_value_cols)

                # Create a simplified result structure for compatibility
                simplified_result = {
                    "file_path": result["source_file"],
                    "total_rows": result["file_info"]["total_rows"],
                    "total_columns": result["file_info"]["total_columns"],
                    "columns": result["columns"],
                    "suggestions": {
                        "likely_skip_cols": likely_skip_cols,
                        "likely_value_cols": likely_value_cols,
                        "reasoning": f"Identified {len(likely_skip_cols)} timing columns and {len(likely_value_cols)} categorical columns suitable for HED annotation",
                    },
                }

                results[file_path] = simplified_result
            else:
                print(f"File not found: {file_path}")

        return results

    except Exception as e:
        print(f"Error in column analysis: {e}")
        import traceback

        traceback.print_exc()
        return None


async def test_sidecar_generation(analysis_results):
    """Test sidecar generation with analyzed results."""
    print("\n\n=== Testing Sidecar Generation ===")

    try:
        generator = SidecarGenerator()

        sidecar_results = {}

        for file_path, analysis in analysis_results.items():
            print(f"\n--- Generating sidecar for {file_path} ---")

            skip_cols = analysis["suggestions"]["likely_skip_cols"]
            value_cols = analysis["suggestions"]["likely_value_cols"]

            if not value_cols:
                print("No value columns suggested, skipping sidecar generation")
                continue

            print(f"Using skip_cols: {skip_cols}")
            print(f"Using value_cols: {value_cols}")

            # Generate sidecar template
            template = await generator.generate_sidecar_template(
                events_data=file_path,
                skip_columns=skip_cols,
                value_columns=value_cols,
                schema_version="8.3.0",
            )

            if template.is_valid:
                print("✅ Sidecar generation successful!")

                # Save the sidecar to a file
                output_file = file_path.replace(".tsv", "_sidecar.json")
                result = await generator.save_sidecar(
                    template, output_file, format="json"
                )

                if result.success:
                    print(f"Sidecar saved to: {output_file}")
                else:
                    print(f"Failed to save sidecar: {result.error}")

                # Show sample of the sidecar
                print("Sample sidecar content:")
                for col_name, col_data in list(template.template.items())[:2]:
                    if not col_name.startswith("_"):  # Skip metadata
                        print(f"  {col_name}: {json.dumps(col_data, indent=4)}")

            else:
                print("❌ Sidecar generation failed")
                print("Generation errors:", template.errors)

            sidecar_results[file_path] = template

        return sidecar_results

    except Exception as e:
        print(f"Error in sidecar generation: {e}")
        import traceback

        traceback.print_exc()
        return None


async def main():
    """Main test function."""
    print("Testing HED MCP Tools with Real-World Event Files")
    print("=" * 60)

    # Test column analysis
    analysis_results = await test_column_analysis()

    if analysis_results:
        # Test sidecar generation
        sidecar_results = await test_sidecar_generation(analysis_results)

        print("\n\n=== Summary ===")
        print(f"Successfully analyzed {len(analysis_results)} files")
        if sidecar_results:
            successful_sidecars = sum(
                1 for r in sidecar_results.values() if r and r.is_valid
            )
            print(f"Successfully generated {successful_sidecars} sidecars")
        print("Test completed!")
    else:
        print("Column analysis failed, skipping sidecar generation")


if __name__ == "__main__":
    asyncio.run(main())
