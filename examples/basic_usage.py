#!/usr/bin/env python3
"""Basic Usage Example for HED MCP Server

This script demonstrates the core functionality of the HED MCP Server package,
including BIDS events file analysis and HED sidecar generation.

Usage:
    python examples/basic_usage.py [events_file.tsv]

Requirements:
    - hed_tools package installed
    - Sample BIDS events file (TSV format)
"""

import asyncio
import sys
from pathlib import Path
import pandas as pd

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import hed_tools
from hed_tools import (
    create_integration_suite,
    create_column_analyzer,
    create_hed_wrapper,
    create_file_handler,
)


def print_banner(title: str):
    """Print a formatted banner for section headers."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n--- {title} ---")


async def create_sample_events_file(output_path: Path) -> bool:
    """Create a sample BIDS events file for demonstration."""
    print_section("Creating Sample Events File")

    # Create sample BIDS events data
    events_data = {
        "onset": [0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5],
        "duration": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        "trial_type": [
            "face",
            "house",
            "face",
            "house",
            "scrambled",
            "face",
            "house",
            "scrambled",
            "face",
            "house",
        ],
        "condition": [
            "happy",
            "modern",
            "sad",
            "old",
            "pattern",
            "neutral",
            "modern",
            "pattern",
            "happy",
            "old",
        ],
        "response": [
            "left",
            "right",
            "left",
            "left",
            "right",
            "right",
            "left",
            "right",
            "left",
            "right",
        ],
        "accuracy": [1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
        "response_time": [0.85, 1.23, 1.56, 0.92, 1.78, 1.12, 1.34, 1.45, 0.89, 1.67],
    }

    # Create DataFrame
    df = pd.DataFrame(events_data)

    # Save as TSV
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, sep="\t", index=False)
        print(f"✅ Sample events file created: {output_path}")
        print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
        return True
    except Exception as e:
        print(f"❌ Failed to create sample file: {e}")
        return False


async def demonstrate_package_info():
    """Demonstrate package information and validation."""
    print_banner("Package Information")

    # Get package info
    info = hed_tools.get_package_info()
    print(f"Package: {info['name']} v{info['version']}")
    print(f"Description: {info['description']}")

    print("\nComponent Availability:")
    for component, available in info["components"].items():
        status = "✅ Available" if available else "❌ Not Available"
        print(f"  {component}: {status}")

    print("\nDependency Status:")
    for dep, available in info["dependencies"].items():
        status = "✅ Installed" if available else "❌ Not Installed"
        print(f"  {dep}: {status}")

    # Validate installation
    print("\nInstallation Validation:")
    validation = hed_tools.validate_installation()

    if validation["valid"]:
        print("✅ Installation is complete and ready to use!")
    else:
        print("❌ Installation has issues:")
        for error in validation["errors"]:
            print(f"   Error: {error}")
        for warning in validation["warnings"]:
            print(f"   Warning: {warning}")

    if validation["recommendations"]:
        print("\nRecommendations:")
        for rec in validation["recommendations"]:
            print(f"  • {rec}")


async def demonstrate_file_analysis(events_path: Path):
    """Demonstrate BIDS events file analysis."""
    print_banner("BIDS Events File Analysis")

    # Create column analyzer
    analyzer = create_column_analyzer()

    print(f"📂 Analyzing events file: {events_path}")

    try:
        # Analyze the events file
        analysis = await analyzer.analyze_events_file(events_path)

        # Display file information
        print_section("File Information")
        file_info = analysis["file_info"]
        print(f"Total rows: {file_info['total_rows']}")
        print(f"Total columns: {file_info['total_columns']}")
        print(f"File size: {file_info['file_size_mb']:.2f} MB")

        # Display BIDS compliance
        print_section("BIDS Compliance")
        bids_info = analysis["bids_compliance"]
        compliance_status = "✅ Valid" if bids_info["valid"] else "❌ Invalid"
        print(f"BIDS compliant: {compliance_status}")

        if bids_info["errors"]:
            print("Errors:")
            for error in bids_info["errors"]:
                print(f"  - {error}")

        if bids_info["warnings"]:
            print("Warnings:")
            for warning in bids_info["warnings"]:
                print(f"  - {warning}")

        # Display column analysis
        print_section("Column Analysis")
        columns = analysis["columns"]
        for col_name, col_info in columns.items():
            suitable = "✅" if col_info["hed_suitable"] else "❌"
            print(f"{suitable} {col_name}:")
            print(f"    Type: {col_info['type']}")
            print(f"    Unique values: {col_info['unique_count']}")
            print(f"    Null count: {col_info['null_count']}")

            if col_info["unique_values"] and len(col_info["unique_values"]) <= 10:
                print(f"    Values: {col_info['unique_values']}")

            if not col_info["hed_suitable"]:
                print(f"    Skip reason: {col_info['skip_reason']}")

        # Display HED candidates
        print_section("HED Annotation Candidates")
        hed_candidates = analysis["hed_candidates"]

        if hed_candidates:
            print(f"Found {len(hed_candidates)} columns suitable for HED annotation:")

            for candidate in hed_candidates:
                priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                    candidate["priority"], "⚪"
                )
                print(
                    f"\n{priority_icon} {candidate['column']} ({candidate['priority']} priority)"
                )
                print(f"    Type: {candidate['type']}")
                print(f"    Unique values: {candidate['unique_values'][:5]}...")
        else:
            print("❌ No columns suitable for HED annotation found")

        # Display recommendations
        print_section("Recommendations")
        recommendations = analysis["recommendations"]
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")
        else:
            print("No specific recommendations")

        return analysis

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return None


async def demonstrate_hed_integration(events_path: Path, analysis: dict):
    """Demonstrate HED integration capabilities."""
    print_banner("HED Integration Demo")

    # Create HED wrapper
    wrapper = create_hed_wrapper()

    print_section("Schema Management")

    # Get available schemas
    print("Available HED schemas:")
    schemas = wrapper.get_available_schemas()
    for schema in schemas:
        print(f"  • {schema['version']}: {schema['description']}")

    # Load schema
    print("\n📋 Loading HED schema...")
    schema_loaded = await wrapper.load_schema()

    if schema_loaded:
        print("✅ Schema loaded successfully")

        # Get schema info
        schema_info = wrapper.get_schema_info()
        print(f"   Version: {schema_info['version']}")
        print(f"   Loaded: {schema_info['loaded']}")
    else:
        print("❌ Schema loading failed")
        return

    print_section("HED String Validation")

    # Test HED string parsing
    test_strings = [
        "Event/Category/Experimental-stimulus",
        "Sensory-event/Visual/Face, Red",
        "Agent-action/Move/Press, Participant-response/Manual",
        "Invalid/Tag/Structure/That/Does/Not/Exist",
    ]

    for hed_string in test_strings:
        result = wrapper.parse_hed_string(hed_string)
        status = "✅ Valid" if result["valid"] else "❌ Invalid"
        print(f"{status} {hed_string[:50]}...")

        if not result["valid"] and result["errors"]:
            print(f"         Errors: {result['errors'][:2]}")

    # Generate HED suggestions for identified candidates
    if analysis and analysis["hed_candidates"]:
        print_section("HED Annotation Suggestions")

        # Load events data
        handler = create_file_handler()
        events_df = await handler.load_events_file(events_path)

        if events_df is not None:
            analyzer = create_column_analyzer()

            # Show suggestions for each HED candidate
            for candidate in analysis["hed_candidates"][
                :3
            ]:  # Limit to first 3 for demo
                column_name = candidate["column"]
                column_data = events_df[column_name]

                print(f"\n🎯 Suggestions for '{column_name}':")

                suggestions = await analyzer.suggest_hed_annotations(
                    column_data, column_name
                )

                for suggestion in suggestions[:5]:  # Show first 5 suggestions
                    confidence_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                        suggestion["confidence"], "⚪"
                    )
                    print(
                        f"   {confidence_icon} {suggestion['value']} → {suggestion['suggested_hed']}"
                    )


async def demonstrate_sidecar_generation(events_path: Path, analysis: dict):
    """Demonstrate HED sidecar generation."""
    print_banner("HED Sidecar Generation")

    if not analysis or not analysis["hed_candidates"]:
        print("❌ No HED candidates available for sidecar generation")
        return

    # Create integration suite
    suite = create_integration_suite()
    wrapper = suite["hed_wrapper"]
    handler = suite["file_handler"]

    # Load events data
    print("📂 Loading events data...")
    events_df = await handler.load_events_file(events_path)

    if events_df is None:
        print("❌ Failed to load events data")
        return

    # Load schema
    print("📋 Loading HED schema...")
    await wrapper.load_schema()

    # Extract columns for processing
    target_columns = [c["column"] for c in analysis["hed_candidates"]]
    print(f"🎯 Processing columns: {target_columns}")

    # Generate sidecar template
    print("🏗️  Generating HED sidecar template...")
    sidecar = await wrapper.generate_sidecar_template(events_df, target_columns)

    if sidecar:
        print("✅ Sidecar generated successfully!")

        # Display sidecar structure
        print_section("Generated Sidecar Structure")

        for column, content in sidecar.items():
            print(f"\n📋 {column}:")
            if "Description" in content:
                print(f"   Description: {content['Description']}")

            if "HED" in content:
                print("   HED annotations:")
                hed_content = content["HED"]
                if isinstance(hed_content, dict):
                    for value, hed_string in list(hed_content.items())[
                        :3
                    ]:  # Show first 3
                        print(f"     {value} → {hed_string}")
                    if len(hed_content) > 3:
                        print(f"     ... and {len(hed_content) - 3} more")
                else:
                    print(f"     {hed_content}")

        # Save sidecar
        sidecar_path = events_path.with_suffix(".json")
        print(f"\n💾 Saving sidecar to {sidecar_path}...")

        success = await handler.save_json_file(sidecar, sidecar_path)

        if success:
            print("✅ Sidecar saved successfully!")

            # Validate the generated sidecar
            print_section("Sidecar Validation")
            validation = await wrapper.validate_events(events_df, sidecar)

            status = "✅ Valid" if validation["valid"] else "❌ Invalid"
            print(f"HED validation: {status}")

            stats = validation["statistics"]
            print(f"Events processed: {stats['total_events']}")
            print(f"HED tags found: {stats['hed_tags_found']}")

            if validation["errors"]:
                print(f"Validation errors: {len(validation['errors'])}")
                for error in validation["errors"][:3]:
                    print(f"  - {error}")

            if validation["warnings"]:
                print(f"Validation warnings: {len(validation['warnings'])}")
        else:
            print("❌ Failed to save sidecar")
    else:
        print("❌ Sidecar generation failed")


async def demonstrate_quick_analysis(events_path: Path):
    """Demonstrate the quick analysis convenience function."""
    print_banner("Quick Analysis Function")

    print("🚀 Using convenience function for quick analysis...")

    # Use the package-level convenience function
    try:
        results = hed_tools.quick_analyze_events(
            str(events_path), str(events_path.with_suffix("_analysis.json"))
        )

        print("✅ Quick analysis completed!")
        print(f"   File: {results.get('file_path', 'Unknown')}")
        print(f"   HED candidates: {len(results.get('hed_candidates', []))}")
        print(
            f"   BIDS compliant: {results.get('bids_compliance', {}).get('valid', False)}"
        )

        analysis_file = events_path.with_suffix("_analysis.json")
        print(f"📄 Analysis results saved to: {analysis_file}")

    except Exception as e:
        print(f"❌ Quick analysis failed: {e}")


async def main():
    """Main demonstration function."""
    print_banner("HED MCP Server - Basic Usage Demo")
    print("This demo showcases the core functionality of the HED MCP Server package.")

    # Handle command line arguments
    if len(sys.argv) > 1:
        events_path = Path(sys.argv[1])
        if not events_path.exists():
            print(f"❌ Events file not found: {events_path}")
            return
    else:
        # Create sample events file
        events_path = Path("examples/sample_events.tsv")
        sample_created = await create_sample_events_file(events_path)
        if not sample_created:
            print("❌ Failed to create sample events file")
            return

    print(f"\n🎯 Using events file: {events_path}")

    try:
        # 1. Package information
        await demonstrate_package_info()

        # 2. File analysis
        analysis = await demonstrate_file_analysis(events_path)

        if analysis:
            # 3. HED integration
            await demonstrate_hed_integration(events_path, analysis)

            # 4. Sidecar generation
            await demonstrate_sidecar_generation(events_path, analysis)

            # 5. Quick analysis function
            await demonstrate_quick_analysis(events_path)

        print_banner("Demo Complete")
        print("✅ All demonstrations completed successfully!")
        print("\nNext steps:")
        print("1. Review the generated sidecar file")
        print("2. Explore the analysis results JSON")
        print("3. Try with your own BIDS events files")
        print("4. Check the documentation for advanced features")

    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback

        print("\nFull traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
