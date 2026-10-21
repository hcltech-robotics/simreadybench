"""Run USD reconstruction benchmarks."""

import argparse
import sys
from pathlib import Path


def main(argv=None):
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="simready-bench")
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("setup", help="Prepare a benchmark runtime")
    targets = setup.add_subparsers(dest="setup_target", required=True)
    daytona = targets.add_parser("daytona", help="Import the published runtime into Daytona")
    daytona.add_argument("--output", type=Path, required=True, help="Path for runtime-images.json")
    daytona.add_argument("--release-manifest", type=Path)
    daytona.add_argument("--snapshot", help="Organisation snapshot name")
    daytona.add_argument("--cpu", type=int, default=2)
    daytona.add_argument("--memory", type=int, default=12, help="Memory in GiB (default: 12)")
    daytona.add_argument("--disk", type=int, default=10, help="Disk in GiB")
    daytona.add_argument("--timeout", type=int, default=600, help="Import timeout in seconds")
    command = commands.add_parser(
        "benchmark", help="Reconstruct and score reference USD assets"
    )
    command.add_argument(
        "reference",
        nargs="?",
        default="NV25",
        help="the NV25 public reference set, a USD file or a JSON asset list",
    )
    command.add_argument("--model", required=True)
    command.add_argument("--harness", default="pi")
    command.add_argument("--deploy", choices=("local", "daytona"), default="local")
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--skills", nargs="+", action="extend", type=Path)
    command.add_argument("--repetitions", type=int, default=4)
    command.add_argument(
        "--authoring-turns", type=int, choices=range(1, 129), default=None,
        metavar="1..128", help="Maximum inference turns per reconstruction (default: 64)",
    )
    command.add_argument(
        "--workers",
        type=int,
        choices=range(1, 17),
        default=1,
        metavar="1..16",
        help="Number of simultaneous Daytona workers",
    )
    command.add_argument(
        "--reasoning", choices=("low", "medium", "high", "xhigh", "max")
    )
    command.add_argument(
        "--max-retries",
        type=int,
        choices=range(11),
        default=2,
        metavar="0..10",
        help="Automatic retries for failed reconstructions",
    )
    command.add_argument("--retry-delay-seconds", type=float, default=5)
    command.add_argument("--maximum-retry-delay-seconds", type=float, default=60)
    command.add_argument(
        "--supervise", action="store_true",
        help="Restart a failed Daytona coordinator on Linux or WSL",
    )
    command.add_argument(
        "--coordinator-restarts", type=int, choices=range(21), default=5,
        metavar="0..20", help="Maximum coordinator restarts with --supervise (default: 5)",
    )
    command.add_argument("--max-spend-usd", dest="maximum_spend_usd", type=float)
    command.add_argument("--provider", choices=("openai", "huggingface"))
    command.add_argument("--api-key-env")
    command.add_argument("--model-metadata", type=Path)
    command.add_argument("--policy", type=Path)
    command.add_argument("--assets-root", type=Path)
    command.add_argument("--runtime-images", type=Path)
    command.add_argument("--cpu-snapshot")
    command.add_argument(
        "--hf-repo", help="Hugging Face dataset receiving completed reconstructions"
    )
    command.add_argument("--hf-run", help="Run name within the Hugging Face dataset")
    command.add_argument(
        "--resume",
        action="store_true",
        help="Continue the saved benchmark in this output directory",
    )
    arguments = parser.parse_args(raw_arguments)
    if arguments.command == "setup":
        from simready_bench.runtime_release import setup_daytona

        try:
            result = setup_daytona(**{
                name: getattr(arguments, name)
                for name in ("output", "release_manifest", "snapshot", "cpu", "memory", "disk", "timeout")
            })
        except (ValueError, OSError, ImportError, TypeError, RuntimeError) as error:
            sys.stderr.write(str(error) + "\n")
            return 2
        sys.stdout.write("Snapshot ready: " + result["snapshot"] + "\n")
        sys.stdout.write("Runtime images: " + result["runtime_images"] + "\n")
        return 0
    if arguments.supervise:
        if arguments.deploy != "daytona":
            parser.error("--supervise requires --deploy daytona")
        from simready_bench.supervision import SupervisorBusyError, supervise

        try:
            return supervise(
                raw_arguments, output_dir=arguments.output_dir,
                maximum_restarts=arguments.coordinator_restarts,
            )
        except SupervisorBusyError as error:
            sys.stderr.write(str(error) + "\n")
            return 75
        except (ValueError, OSError, RuntimeError) as error:
            sys.stderr.write(str(error) + "\n")
            return 2
    from simready_bench.benchmark import benchmark

    options = {
        name: getattr(arguments, name)
        for name in (
            "model",
            "harness",
            "deploy",
            "output_dir",
            "skills",
            "repetitions",
            "authoring_turns",
            "workers",
            "max_retries",
            "retry_delay_seconds",
            "maximum_retry_delay_seconds",
            "reasoning",
            "maximum_spend_usd",
            "provider",
            "api_key_env",
            "model_metadata",
            "policy",
            "assets_root",
            "runtime_images",
            "cpu_snapshot",
            "hf_repo",
            "hf_run",
            "resume",
        )
    }
    try:
        results = benchmark(arguments.reference, **options)
    except (
        ValueError,
        OSError,
        KeyError,
        ImportError,
        TypeError,
        RuntimeError,
    ) as error:
        report = (getattr(error, "result", None) or {}).get("report", {})
        if report.get("html"):
            sys.stderr.write("Results: " + report["html"] + "\n")
        sys.stderr.write(str(error) + "\n")
        from simready_bench.supervision import coordinator_exit_code

        return coordinator_exit_code(error, deploy=arguments.deploy)
    completed = results.get("completed_conditions", 0)
    planned = results.get("planned_conditions", len(results.get("results", [])))
    sys.stdout.write(
        str(completed) + " of " + str(planned) + " reconstruction attempts completed.\n"
    )
    report = results.get("report", {})
    if report.get("html"):
        sys.stdout.write("Report: " + report["html"] + "\n")
    if report.get("scores"):
        sys.stdout.write("Scores: " + report["scores"] + "\n")
    return (
        0
        if completed == planned
        and all(row.get("status") == "completed" for row in results.get("results", []))
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
