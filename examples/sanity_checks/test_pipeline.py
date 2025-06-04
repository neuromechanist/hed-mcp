#!/usr/bin/env python3
"""Simple test script for the HED pipeline."""

import asyncio
import pandas as pd
from src.hed_tools.pipeline import create_default_pipeline


async def test_pipeline():
    """Test the pipeline with simple data."""
    print("Creating pipeline...")
    pipeline = create_default_pipeline()
    print(f"Pipeline created with {len(pipeline.stages)} stages")

    # Create simple test data
    test_data = pd.DataFrame(
        {
            "trial": [1, 2, 3],
            "event_type": ["stimulus", "response", "stimulus"],
            "duration": [1.5, 0.5, 2.0],
        }
    )

    print("Executing pipeline...")
    result = await pipeline.execute(test_data, {"source": "test"})

    print(f"✅ Pipeline executed: {result['success']}")
    print(f"Stages executed: {len(result['execution_info']['stages_executed'])}")

    if result["execution_info"]["errors"]:
        print("Errors:", result["execution_info"]["errors"])

    if result["execution_info"]["warnings"]:
        print("Warnings:", result["execution_info"]["warnings"])

    return result


if __name__ == "__main__":
    result = asyncio.run(test_pipeline())
