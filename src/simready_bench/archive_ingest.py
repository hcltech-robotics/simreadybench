"""Private, content-addressed reference archive ingestion. No actor or provider access."""

from pathlib import Path, PurePosixPath
import hashlib, json, os, re, shutil, stat, tarfile, tempfile, zipfile

INDEX = "REFERENCE_ARCHIVE_INDEX.json"


def _path(name):
    if (
        not isinstance(name, str)
        or not name
        or len(name) > 512
        or "\\" in name
        or ":" in name
        or "\0" in name
    ):
        raise ValueError("Invalid archive member")
    p = PurePosixPath(name)
    if (
        p.is_absolute()
        or ".." in p.parts
        or p.as_posix() != name
        or name in (".", "..")
    ):
        raise ValueError("Archive traversal or ambiguous path")
    return p


def _unique(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            raise ValueError("Duplicate JSON key")
        result[k] = v
    return result


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ingest_archive(
    archive,
    destination,
    *,
    sha256,
    max_archive_bytes=512 * 1024**2,
    max_expanded_bytes=512 * 1024**2,
    max_members=10000,
):
    """Require a pinned local archive and exact per-file index; publish a new private tree."""
    archive = Path(archive).absolute()
    destination = Path(destination).absolute()
    if not re.fullmatch("[a-f0-9]{64}", sha256):
        raise ValueError("Pinned archive SHA-256 required")
    if (
        archive.resolve() != archive
        or not archive.is_file()
        or archive.stat().st_nlink != 1
    ):
        raise ValueError("Regular non-symlink archive required")
    if archive.stat().st_size > max_archive_bytes or _digest(archive) != sha256:
        raise ValueError("Archive hash/size differs")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Preserve existing reference tree")
    if (
        destination.parent.resolve() != destination.parent
        or not destination.parent.is_dir()
    ):
        raise ValueError("Existing nonsymlink destination parent required")
    staging = Path(
        tempfile.mkdtemp(prefix=".private-reference-", dir=destination.parent)
    )
    staging.chmod(0o700)
    try:
        iszip = zipfile.is_zipfile(archive)
        container = zipfile.ZipFile(archive) if iszip else tarfile.open(archive, "r:*")
        with container:
            rows = {}
            for member in container.infolist() if iszip else container:
                name = member.filename if iszip else member.name
                directory = member.is_dir() if iszip else member.isdir()
                _path(name.rstrip("/") if directory else name)
                if iszip:
                    mode = member.external_attr >> 16
                    if (
                        stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)
                        or member.flag_bits & 1
                    ):
                        raise ValueError(
                            "Archive links/special/encrypted members forbidden"
                        )
                    size = member.file_size
                else:
                    if not (member.isfile() or member.isdir()):
                        raise ValueError("Archive links/special members forbidden")
                    size = member.size
                if directory:
                    continue
                if name in rows or name.casefold() in {x.casefold() for x in rows}:
                    raise ValueError("Duplicate archive member")
                rows[name] = (member, size)
                if (
                    len(rows) > max_members
                    or sum(x[1] for x in rows.values()) > max_expanded_bytes
                ):
                    raise ValueError("Expanded archive exceeds bound")
            if INDEX not in rows or rows[INDEX][1] > 1024**2:
                raise ValueError("Bounded reference index required")

            def open_member(name):
                return (
                    container.open(rows[name][0])
                    if iszip
                    else container.extractfile(rows[name][0])
                )

            with open_member(INDEX) as stream:
                index_raw = stream.read(1024**2 + 1)
            index = json.loads(index_raw, object_pairs_hook=_unique)
            if (
                set(index) - {"schema", "files", "assets"}
                or index.get("schema") != "simready-reference-archive-v1"
                or type(index.get("files")) is not dict
            ):
                raise ValueError("Exact reference index required")
            declared = index["files"]
            if (
                set(declared) != set(rows) - {INDEX}
                or "ingest-receipt.json" in declared
            ):
                raise ValueError("Closed-world archive index differs")
            for name, row in declared.items():
                _path(name)
                if (
                    type(row) is not dict
                    or set(row) != {"sha256", "bytes"}
                    or not re.fullmatch("[a-f0-9]{64}", str(row["sha256"]))
                    or type(row["bytes"]) is not int
                    or row["bytes"] != rows[name][1]
                ):
                    raise ValueError("Archive index member differs")
                out = staging.joinpath(*PurePosixPath(name).parts)
                out.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                h = hashlib.sha256()
                size = 0
                with open_member(name) as source, out.open("xb") as target:
                    out.chmod(0o600)
                    while block := source.read(1024**2):
                        size += len(block)
                        if size > row["bytes"]:
                            raise ValueError("Expanded member exceeds declared size")
                        h.update(block)
                        target.write(block)
                if size != row["bytes"] or h.hexdigest() != row["sha256"]:
                    raise ValueError("Archive dependency hash differs")
            (staging / INDEX).write_bytes(index_raw)
            (staging / INDEX).chmod(0o600)
        if _digest(archive) != sha256:
            raise ValueError("Archive changed during ingestion")
        result = {
            "schema": "simready-private-reference-ingest-v1",
            "archive_sha256": sha256,
            "index_sha256": hashlib.sha256(index_raw).hexdigest(),
            "files": declared,
            "assets": index.get("assets", []),
            "visibility": "private_ground_truth",
            "provider_calls": 0,
        }
        (staging / "ingest-receipt.json").write_text(
            json.dumps(result, sort_keys=True, indent=2) + "\n"
        )
        (staging / "ingest-receipt.json").chmod(0o600)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("Preserve existing destination")
        os.rename(staging, destination)
        return result
    finally:
        if staging.exists():
            shutil.rmtree(staging)
