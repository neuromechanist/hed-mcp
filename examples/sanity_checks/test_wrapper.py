import pandas as pd
from src.hed_tools.hed_integration.tabular_summary import (
    TabularSummaryWrapper,
    TabularSummaryConfig,
)
import asyncio


async def test():
    # Load the test file
    df = pd.read_csv(
        "sub-NDARBU098PJT_task-contrastChangeDetection_run-2_events.tsv", sep="\t"
    )
    print("DataFrame columns:", df.columns.tolist())
    print("DataFrame shape:", df.shape)
    print("Sample data:")
    print(df.head())

    # Test TabularSummary wrapper
    config = TabularSummaryConfig()
    wrapper = TabularSummaryWrapper(config)

    try:
        result = await wrapper.extract_sidecar_template(
            data=df, skip_columns=["onset", "duration", "sample"], use_cache=False
        )
        print("Success! Template keys:", list(result.template.keys()))
        print("Template structure:")
        for key, value in result.template.items():
            print(f"{key}:", value)
    except Exception as e:
        print("Error:", str(e))
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test())
