"""Prepare the locked OpenUSD imaging packages for the CPU image."""

from pathlib import Path
import argparse, concurrent.futures, hashlib, json, urllib.request


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = args.output.absolute()
    if root.exists() or root.parent.resolve() != root.parent:
        raise ValueError("Fresh output below a regular parent required")
    lock = json.loads(
        (Path(__file__).parents[2] / "locks/storm-linux-64.json").read_text()
    )
    root.mkdir(mode=0o700)
    packages = root / "packages"
    packages.mkdir()

    def download(item, path):
        if not item["url"].startswith("https://conda.anaconda.org/conda-forge/"):
            raise ValueError("Pinned public package URL required")
        with (
            urllib.request.urlopen(item["url"], timeout=180) as source,
            path.open("xb") as output,
        ):
            digest = hashlib.sha256()
            while block := source.read(1024**2):
                output.write(block)
                digest.update(block)
        if digest.hexdigest() != item["sha256"]:
            raise ValueError("Public package checksum differs: " + path.name)

    def fetch(item):
        if Path(item["fn"]).name != item["fn"]:
            raise ValueError("Literal package filename required")
        download(item, packages / item["fn"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(fetch, lock["packages"]))
    download(lock["micromamba"], root / "micromamba-linux-64.tar.bz2")
    (packages / "explicit.txt").write_text(
        "@EXPLICIT\n"
        + "".join(
            "file:///packages/" + i["fn"] + "#" + i["md5"] + "\n"
            for i in lock["packages"]
        )
    )
    (packages / "SHA256SUMS").write_text(
        "".join(i["sha256"] + "  " + i["fn"] + "\n" for i in lock["packages"])
    )
    (root / "Dockerfile").write_bytes(
        (Path(__file__).parent / "Dockerfile.storm").read_bytes()
    )
    print(json.dumps({"output": str(root)}))


if __name__ == "__main__":
    main()
