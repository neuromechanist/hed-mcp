#!/usr/bin/env python3
"""
Utility script to run all HED MCP sanity check scripts.

This script runs all sanity check scripts in the examples/sanity_checks/ directory
and provides a summary report of results.
"""

import sys
import subprocess
import time
from pathlib import Path
from typing import List, Tuple


def run_script(script_path: Path) -> Tuple[str, bool, str]:
    """
    Run a single sanity check script.

    Returns:
        Tuple of (script_name, success, output)
    """
    script_name = script_path.name

    try:
        print(f"🔄 Running {script_name}...")

        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
            cwd=Path.cwd(),
        )

        success = result.returncode == 0
        output = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

        if success:
            print(f"✅ {script_name} - PASSED")
        else:
            print(f"❌ {script_name} - FAILED (exit code: {result.returncode})")

        return script_name, success, output

    except subprocess.TimeoutExpired:
        print(f"⏰ {script_name} - TIMEOUT")
        return script_name, False, "Script timed out after 2 minutes"

    except Exception as e:
        print(f"💥 {script_name} - ERROR: {e}")
        return script_name, False, f"Exception: {str(e)}"


def main():
    """Run all sanity check scripts and generate a report."""

    # Find all sanity check scripts
    sanity_checks_dir = Path("examples/sanity_checks")

    if not sanity_checks_dir.exists():
        print(f"❌ Sanity checks directory not found: {sanity_checks_dir}")
        sys.exit(1)

    scripts = list(sanity_checks_dir.glob("test_*.py"))

    if not scripts:
        print(f"❌ No test scripts found in {sanity_checks_dir}")
        sys.exit(1)

    print(f"🧪 Found {len(scripts)} sanity check scripts")
    print("=" * 60)

    # Run all scripts
    results: List[Tuple[str, bool, str]] = []
    start_time = time.time()

    for script in sorted(scripts):
        result = run_script(script)
        results.append(result)
        print()  # Add space between scripts

    # Generate summary report
    total_time = time.time() - start_time
    passed = sum(1 for _, success, _ in results if success)
    failed = len(results) - passed

    print("=" * 60)
    print("📊 SANITY CHECK SUMMARY")
    print("=" * 60)
    print(f"Total scripts: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {passed / len(results) * 100:.1f}%")
    print(f"Total time: {total_time:.1f}s")
    print()

    # Show failed scripts
    if failed > 0:
        print("❌ FAILED SCRIPTS:")
        for script_name, success, output in results:
            if not success:
                print(f"  - {script_name}")
        print()

    # Detailed output (only for failed scripts by default)
    show_all = "--verbose" in sys.argv or "-v" in sys.argv

    if failed > 0 or show_all:
        print("📝 DETAILED OUTPUT:")
        print("-" * 60)

        for script_name, success, output in results:
            if not success or show_all:
                print(f"\n=== {script_name} ===")
                print(output)
                print("-" * 40)

    # Exit with appropriate code
    if failed > 0:
        print(f"💥 {failed} scripts failed")
        sys.exit(1)
    else:
        print("🎉 All sanity checks passed!")
        sys.exit(0)


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        print("\nUsage:")
        print("  python scripts/run_sanity_checks.py [--verbose|-v]")
        print("\nOptions:")
        print("  --verbose, -v    Show detailed output for all scripts")
        print("  --help, -h       Show this help message")
        sys.exit(0)

    main()
