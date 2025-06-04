#!/usr/bin/env python3
"""Test script to verify HED sidecar generation functionality."""

import asyncio
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from hed_tools.server.server import generate_hed_sidecar

    print("✅ Successfully imported generate_hed_sidecar")
except ImportError as e:
    print(f"❌ Failed to import generate_hed_sidecar: {e}")
    sys.exit(1)


async def test_sidecar_generation():
    """Test the sidecar generation with the events file."""

    events_file = "sub-NDARBU098PJT_task-contrastChangeDetection_run-2_events.tsv"
    output_file = "test_sidecar_output.json"

    if not Path(events_file).exists():
        print(f"❌ Events file not found: {events_file}")
        return False

    print(f"📂 Testing sidecar generation for: {events_file}")

    try:
        # Generate the sidecar template
        sidecar_template = await generate_hed_sidecar(
            events_file=events_file,
            output_path=output_file,
            skip_columns="onset,duration,sample",  # Explicitly specify skip columns
            value_columns="",  # Let TabularSummary auto-detect categorical columns
            include_validation=True,
        )

        print(f"📝 Result:\n{sidecar_template}")

        # Check if output file was created
        if Path(output_file).exists():
            print(f"✅ Output file created: {output_file}")
            with open(output_file, "r") as f:
                content = f.read()
                print(f"📄 Content preview (first 500 chars):\n{content[:500]}...")
            return True
        else:
            print(f"❌ Output file not created: {output_file}")
            return False

    except Exception as e:
        print(f"❌ Error during sidecar generation: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🧪 Testing HED sidecar generation...")
    success = asyncio.run(test_sidecar_generation())
    if success:
        print("✅ Test completed successfully!")
    else:
        print("❌ Test failed!")
        sys.exit(1)
