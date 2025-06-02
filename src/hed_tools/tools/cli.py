"""Command-line interface for BIDS Column Analysis Engine."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from .column_analysis_engine import (
    BIDSColumnAnalysisEngine,
    AnalysisConfig,
    analyze_bids_directory,
    analyze_bids_files,
)
from .llm_preprocessor import SamplingConfig


def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def create_config_from_args(args) -> AnalysisConfig:
    """Create AnalysisConfig from command line arguments."""
    sampling_config = SamplingConfig(
        max_tokens=args.max_tokens,
        max_samples_per_column=args.max_samples,
        random_seed=args.seed,
    )

    return AnalysisConfig(
        max_workers=args.workers,
        enable_caching=args.cache,
        enable_statistical_analysis=args.stats,
        enable_pattern_recognition=args.patterns,
        enable_hed_detection=args.hed,
        output_format=args.format,
        include_detailed_stats=args.detailed,
        include_sample_data=args.samples,
        sampling_config=sampling_config,
        enable_progress_tracking=args.progress,
    )


async def progress_callback(current: int, total: int, filename: str) -> None:
    """Progress callback for batch processing."""
    percentage = (current / total) * 100
    print(f"Progress: {current}/{total} ({percentage:.1f}%) - Processing: {filename}")


async def analyze_files_command(args) -> int:
    """Handle the analyze-files command."""
    try:
        config = create_config_from_args(args)

        if args.progress:
            config.progress_callback = progress_callback

        file_paths = [Path(p) for p in args.files]

        # Validate files exist
        for file_path in file_paths:
            if not file_path.exists():
                print(f"Error: File not found: {file_path}", file=sys.stderr)
                return 1

        print(f"Analyzing {len(file_paths)} files...")
        result = await analyze_bids_files(file_paths, config)

        print("\nAnalysis complete!")
        print(f"Files processed: {result.successful_files}/{result.total_files}")
        print(f"Total processing time: {result.total_processing_time:.2f}s")
        print(f"Average time per file: {result.average_processing_time:.2f}s")

        if args.output:
            engine = BIDSColumnAnalysisEngine(config)
            engine.save_results(result, args.output)
            print(f"Results saved to: {args.output}")

        if args.summary:
            engine = BIDSColumnAnalysisEngine(config)
            summary = engine.get_analysis_summary(result)
            print("\n" + "=" * 50)
            print("ANALYSIS SUMMARY")
            print("=" * 50)

            print("\nOverview:")
            print(f"  Success rate: {summary['overview']['success_rate']:.1%}")
            print(
                f"  Total columns analyzed: {summary['data_statistics']['total_columns_analyzed']}"
            )
            print(
                f"  HED candidate columns found: {summary['data_statistics']['hed_candidate_columns']}"
            )

            print("\nColumn type distribution:")
            for col_type, count in summary["data_statistics"][
                "column_type_distribution"
            ].items():
                print(f"  {col_type}: {count}")

            print("\nPerformance:")
            print(
                f"  Files per second: {summary['performance']['files_per_second']:.2f}"
            )
            if (
                summary["cache_performance"]["cache_hits"]
                + summary["cache_performance"]["cache_misses"]
                > 0
            ):
                print(
                    f"  Cache hit rate: {summary['cache_performance']['cache_hit_rate']:.1%}"
                )

        return 0

    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        return 1


async def analyze_directory_command(args) -> int:
    """Handle the analyze-directory command."""
    try:
        config = create_config_from_args(args)

        if args.progress:
            config.progress_callback = progress_callback

        directory_path = Path(args.directory)

        if not directory_path.exists():
            print(f"Error: Directory not found: {directory_path}", file=sys.stderr)
            return 1

        if not directory_path.is_dir():
            print(f"Error: Path is not a directory: {directory_path}", file=sys.stderr)
            return 1

        print(f"Scanning directory: {directory_path}")
        print(f"Pattern: {args.pattern}")

        result = await analyze_bids_directory(
            directory_path, pattern=args.pattern, config=config
        )

        print("\nAnalysis complete!")
        print(f"Files found: {result.total_files}")
        print(f"Files processed successfully: {result.successful_files}")
        print(f"Files failed: {result.failed_files}")
        print(f"Total processing time: {result.total_processing_time:.2f}s")

        if result.total_files > 0:
            print(f"Average time per file: {result.average_processing_time:.2f}s")

        if args.output:
            engine = BIDSColumnAnalysisEngine(config)
            engine.save_results(result, args.output)
            print(f"Results saved to: {args.output}")

        if args.summary:
            engine = BIDSColumnAnalysisEngine(config)
            summary = engine.get_analysis_summary(result)
            print("\n" + "=" * 50)
            print("ANALYSIS SUMMARY")
            print("=" * 50)

            print("\nOverview:")
            print(f"  Success rate: {summary['overview']['success_rate']:.1%}")
            print(
                f"  Total columns analyzed: {summary['data_statistics']['total_columns_analyzed']}"
            )
            print(
                f"  HED candidate columns found: {summary['data_statistics']['hed_candidate_columns']}"
            )

            if summary["data_statistics"]["column_type_distribution"]:
                print("\nColumn type distribution:")
                for col_type, count in summary["data_statistics"][
                    "column_type_distribution"
                ].items():
                    print(f"  {col_type}: {count}")

            print("\nQuality metrics:")
            print(
                f"  BIDS compliant files: {summary['quality_metrics']['bids_compliant_files']}"
            )
            print(
                f"  Files with HED candidates: {summary['quality_metrics']['files_with_hed_candidates']}"
            )

            print("\nPerformance:")
            print(
                f"  Files per second: {summary['performance']['files_per_second']:.2f}"
            )
            if (
                summary["cache_performance"]["cache_hits"]
                + summary["cache_performance"]["cache_misses"]
                > 0
            ):
                print(
                    f"  Cache hit rate: {summary['cache_performance']['cache_hit_rate']:.1%}"
                )

        if args.list_failures and result.failed_files > 0:
            print("\nFailed files:")
            for file_result in result.file_results:
                if not file_result.success:
                    print(f"  {file_result.file_path}: {file_result.error_message}")

        return 0

    except Exception as e:
        print(f"Error during directory analysis: {e}", file=sys.stderr)
        return 1


def create_parser() -> argparse.ArgumentParser:
    """Create the command line argument parser."""
    parser = argparse.ArgumentParser(
        prog="hed-column-analysis",
        description=(
            "BIDS Column Analysis Engine - Analyze BIDS event files for "
            "column types, patterns, and HED candidates"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze specific files
  hed-column-analysis analyze-files file1_events.tsv file2_events.tsv -o results.json

  # Analyze all event files in a BIDS dataset
  hed-column-analysis analyze-directory /path/to/bids/dataset -s --progress

  # Analyze with custom settings
  hed-column-analysis analyze-directory /data --workers 8 --format csv --detailed
        """,
    )

    # Global options
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)",
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Common analysis options
    def add_analysis_options(subparser):
        """Add common analysis options to a subparser."""
        # Processing options
        proc_group = subparser.add_argument_group("Processing Options")
        proc_group.add_argument(
            "--workers",
            "-w",
            type=int,
            default=4,
            help="Number of parallel workers (default: 4)",
        )
        proc_group.add_argument(
            "--cache",
            action="store_true",
            default=True,
            help="Enable result caching (default: enabled)",
        )
        proc_group.add_argument(
            "--no-cache",
            dest="cache",
            action="store_false",
            help="Disable result caching",
        )

        # Analysis options
        analysis_group = subparser.add_argument_group("Analysis Options")
        analysis_group.add_argument(
            "--stats",
            action="store_true",
            default=True,
            help="Enable statistical analysis (default: enabled)",
        )
        analysis_group.add_argument(
            "--no-stats",
            dest="stats",
            action="store_false",
            help="Disable statistical analysis",
        )
        analysis_group.add_argument(
            "--patterns",
            action="store_true",
            default=True,
            help="Enable pattern recognition (default: enabled)",
        )
        analysis_group.add_argument(
            "--no-patterns",
            dest="patterns",
            action="store_false",
            help="Disable pattern recognition",
        )
        analysis_group.add_argument(
            "--hed",
            action="store_true",
            default=True,
            help="Enable HED candidate detection (default: enabled)",
        )
        analysis_group.add_argument(
            "--no-hed",
            dest="hed",
            action="store_false",
            help="Disable HED candidate detection",
        )

        # Output options
        output_group = subparser.add_argument_group("Output Options")
        output_group.add_argument(
            "--format",
            "-f",
            choices=["json", "csv", "pickle"],
            default="json",
            help="Output format (default: json)",
        )
        output_group.add_argument(
            "--output", "-o", type=str, help="Output file path (without extension)"
        )
        output_group.add_argument(
            "--detailed",
            action="store_true",
            default=True,
            help="Include detailed statistics (default: enabled)",
        )
        output_group.add_argument(
            "--no-detailed",
            dest="detailed",
            action="store_false",
            help="Exclude detailed statistics",
        )
        output_group.add_argument(
            "--samples",
            action="store_true",
            default=True,
            help="Include sample data (default: enabled)",
        )
        output_group.add_argument(
            "--no-samples",
            dest="samples",
            action="store_false",
            help="Exclude sample data",
        )
        output_group.add_argument(
            "--summary",
            "-s",
            action="store_true",
            help="Print analysis summary to console",
        )

        # LLM preprocessing options
        llm_group = subparser.add_argument_group("LLM Preprocessing Options")
        llm_group.add_argument(
            "--max-tokens",
            type=int,
            default=512,
            help="Maximum tokens per column sample (default: 512)",
        )
        llm_group.add_argument(
            "--max-samples",
            type=int,
            default=50,
            help="Maximum samples per column (default: 50)",
        )
        llm_group.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed for sampling (default: 42)",
        )

        # UI options
        ui_group = subparser.add_argument_group("Interface Options")
        ui_group.add_argument(
            "--progress", action="store_true", help="Show progress during analysis"
        )

    # analyze-files command
    files_parser = subparsers.add_parser(
        "analyze-files", help="Analyze specific BIDS event files"
    )
    files_parser.add_argument("files", nargs="+", help="BIDS event files to analyze")
    add_analysis_options(files_parser)

    # analyze-directory command
    dir_parser = subparsers.add_parser(
        "analyze-directory", help="Analyze all BIDS event files in a directory"
    )
    dir_parser.add_argument("directory", help="Directory containing BIDS event files")
    dir_parser.add_argument(
        "--pattern",
        default="**/*_events.tsv",
        help="Glob pattern for finding event files (default: **/*_events.tsv)",
    )
    dir_parser.add_argument(
        "--list-failures", action="store_true", help="List files that failed to process"
    )
    add_analysis_options(dir_parser)

    return parser


async def main() -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Set up logging
    setup_logging(args.log_level)

    # Execute command
    if args.command == "analyze-files":
        return await analyze_files_command(args)
    elif args.command == "analyze-directory":
        return await analyze_directory_command(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1


def cli_main() -> None:
    """Entry point for setuptools console script."""
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    cli_main()
