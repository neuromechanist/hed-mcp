try:
    from hed.tools.analysis.tabular_summary import TabularSummary

    print("HED TabularSummary available")
    HED_AVAILABLE = True
except ImportError as e:
    print("HED import failed:", e)
    HED_AVAILABLE = False

print("HED_AVAILABLE:", HED_AVAILABLE)

if HED_AVAILABLE:
    import pandas as pd

    # Test creating TabularSummary
    try:
        summary = TabularSummary()
        print("TabularSummary created successfully")

        # Test with data
        data = {"onset": [1, 2, 3], "value": ["a", "b", "a"], "event_code": [1, 2, 1]}
        df = pd.DataFrame(data)

        summary.update(df)
        template = summary.extract_sidecar_template()
        print("Template:", template)

    except Exception as e:
        print("TabularSummary test failed:", e)
        import traceback

        traceback.print_exc()
