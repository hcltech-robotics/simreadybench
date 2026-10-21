"""Create a Daytona snapshot from the prepared CPU runtime images."""

from pathlib import Path
import argparse
import json
import os


def main():
    from daytona import CreateSnapshotParams, Daytona, DaytonaConfig, Image, Resources

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--name", default="simready-bench-cpu")
    parser.add_argument("--memory", type=int, default=8)
    parser.add_argument("--disk", type=int, default=10)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    root = args.context.absolute()
    expected = {
        "images.tar.gz",
        "images.txt",
        "bootstrap.sh",
        "snapshot.Dockerfile",
        "runtime-images.json",
    }
    if {path.name for path in root.iterdir()} not in (
        expected, expected | {"local-runtime-images.json"}
    ) or any(
        not path.is_file() or path.is_symlink() for path in root.iterdir()
    ):
        raise ValueError("Use the context created by prepare_snapshot.py")
    if args.receipt.exists() or args.receipt.is_symlink():
        raise FileExistsError(args.receipt)
    key = os.environ.get("DAYTONA_API_KEY")
    if not key:
        raise ValueError("Set DAYTONA_API_KEY")
    client = Daytona(DaytonaConfig(api_key=key))
    image = Image.from_dockerfile(root / "snapshot.Dockerfile", strict_context=True)
    snapshot = client.snapshot.create(
        CreateSnapshotParams(
            name=args.name,
            image=image,
            resources=Resources(cpu=2, memory=args.memory, disk=args.disk),
        ),
        timeout=600,
    )
    value = {
        "name": args.name,
        "id": snapshot.id,
        "state": str(snapshot.state),
        "images": json.loads(
            (root / "runtime-images.json").read_text(encoding="utf-8")
        ),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    args.receipt.chmod(0o600)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
