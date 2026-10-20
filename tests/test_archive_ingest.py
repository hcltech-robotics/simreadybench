import hashlib, io, json, tarfile, zipfile, stat
import pytest


def make(path, rows):
    with tarfile.open(path, "w:gz") as t:
        for name, data, kind in rows:
            m = tarfile.TarInfo(name)
            m.type = kind
            m.size = len(data)
            if kind == tarfile.SYMTYPE:
                m.linkname = "/etc/passwd"
                m.size = 0
            t.addfile(m, io.BytesIO(data) if m.size else None)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_archive_ingest_hashes_closure_and_keeps_private_modes(tmp_path):
    from simready_bench.archive_ingest import ingest_archive

    data = b"#usda 1.0\n"
    files = {
        "asset/asset.usda": {
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }
    }
    index = json.dumps(
        {"schema": "simready-reference-archive-v1", "files": files}
    ).encode()
    p = tmp_path / "refs.tar.gz"
    h = make(
        p,
        [
            ("asset/asset.usda", data, tarfile.REGTYPE),
            ("REFERENCE_ARCHIVE_INDEX.json", index, tarfile.REGTYPE),
        ],
    )
    out = tmp_path / "private"
    result = ingest_archive(p, out, sha256=h)
    assert (out / "asset/asset.usda").read_bytes() == data and result["files"] == files
    assert stat.S_IMODE(out.stat().st_mode) == 0o700
    assert stat.S_IMODE((out / "asset/asset.usda").stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        ingest_archive(p, out, sha256=h)


@pytest.mark.parametrize(
    "name,kind",
    [
        ("../outside", tarfile.REGTYPE),
        ("/absolute", tarfile.REGTYPE),
        ("alias", tarfile.SYMTYPE),
        ("a\\b", tarfile.REGTYPE),
    ],
)
def test_archive_rejects_escape_or_link_before_publishing(tmp_path, name, kind):
    from simready_bench.archive_ingest import ingest_archive

    p = tmp_path / "refs.tar.gz"
    h = make(p, [(name, b"x", kind)])
    with pytest.raises(ValueError):
        ingest_archive(p, tmp_path / "private", sha256=h)
    assert not (tmp_path / "private").exists() and not (tmp_path / "outside").exists()


def test_archive_rejects_unknown_hash_and_undeclared_member(tmp_path):
    from simready_bench.archive_ingest import ingest_archive

    p = tmp_path / "refs.tar.gz"
    h = make(p, [("asset.usda", b"x", tarfile.REGTYPE)])
    with pytest.raises(ValueError):
        ingest_archive(p, tmp_path / "private", sha256="0" * 64)
    with pytest.raises(ValueError):
        ingest_archive(p, tmp_path / "private", sha256=h)
    assert not (tmp_path / "private").exists()


def test_nvidia_dot_metadata_is_a_valid_private_dependency(tmp_path):
    from simready_bench.archive_ingest import ingest_archive

    data = b'{"license":"CC-BY-4.0"}'
    name = "asset/.metadata/com.nvidia.simready.packaging.bom.json"
    files = {name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}}
    index = json.dumps(
        {"schema": "simready-reference-archive-v1", "files": files}
    ).encode()
    p = tmp_path / "refs.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr(name, data)
        z.writestr("REFERENCE_ARCHIVE_INDEX.json", index)
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    out = tmp_path / "private"
    assert ingest_archive(p, out, sha256=h)["files"] == files
    assert (out / name).read_bytes() == data
