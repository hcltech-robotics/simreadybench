"""Plan or apply an additive results/ copy of a materialized local HF snapshot.

No Hub client, network access, login state, remote mutation, or deletion is used.
The revision argument attests which snapshot the operator materialized; this tool
checks local bytes, not their remote provenance. Keep the original checkout and
receipt until the additive commit and consumer revision have been recorded.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil

import yaml


ORIGINAL_REVISION = '0d19347b23ed31a077f045ca3454bacc5bed1d26'
REPOSITORY = 'hcltech-robotics/simreadybench'
_SCHEMA = 'simreadybench-results-migration/v1'
_SHARED = {'.gitattributes', '.gitignore', 'LICENSE', 'LICENSE.md', 'LICENSE.txt'}


def _relative(value):
    if not isinstance(value, str) or not value or '\\' in value or ':' in value:
        raise ValueError('Expected a confined repository-relative path')
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or path.as_posix() != value:
        raise ValueError('Expected a confined repository-relative path')
    return path


def _fingerprint(path):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return path.stat().st_size, digest


def _prefixed(value):
    _relative(value)
    return value if value.startswith('results/') else 'results/' + value


def _result_url(value):
    """Only move this repository's mutable main links; immutable URLs survive."""
    if not isinstance(value, str):
        return value
    base = f'https://huggingface.co/datasets/{REPOSITORY}/'
    for kind in ('blob', 'resolve'):
        prefix = base + kind + '/main/'
        if value.startswith(prefix):
            relative = value[len(prefix):]
            if relative.startswith(('runs/', 'viewer/', 'media/', 'gallery/')):
                return prefix + 'results/' + relative
    return value


def _index_paths(index):
    """Yield owned data paths, including missing CSV sources in the old index."""
    if not isinstance(index, dict) or not isinstance(index.get('runs'), dict):
        raise ValueError('A historical viewer index with retained runs is required')
    for run in index['runs'].values():
        for key in ('scores', 'viewer_scores'):
            if key in run:
                value = run[key]
                _relative(value)
                if not value.startswith(('runs/', 'viewer/')):
                    raise ValueError('Unexpected historical viewer source path')
                yield value
        for groups in run.get('groups', {}).values():
            for group in groups:
                value = group['path']
                _relative(value)
                if not value.startswith(('runs/', 'viewer/')):
                    raise ValueError('Unexpected historical viewer group path')
                yield value


def _transform(name, path):
    """Change routing metadata only; score tables and provenance stay byte exact."""
    if name == 'viewer-index.json':
        value = json.loads(path.read_text())
        for run in value['runs'].values():
            for key in ('scores', 'viewer_scores'):
                if key in run:
                    run[key] = _prefixed(run[key])
            for groups in run.get('groups', {}).values():
                for group in groups:
                    group['path'] = _prefixed(group['path'])
        return (json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + '\n').encode()
    if name == 'leaderboard.json':
        value = json.loads(path.read_text())
        for entry in value['entries']:
            if 'results_url' in entry:
                entry['results_url'] = _result_url(entry['results_url'])
        return (json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + '\n').encode()
    if name == 'README.md':
        text = path.read_text()
        if text.startswith('---\n'):
            end = text.index('\n---\n', 4)
            metadata, body = yaml.safe_load(text[4:end]) or {}, text[end + 5:]
            def move_paths(value):
                if isinstance(value, str):
                    return _prefixed(value)
                if isinstance(value, list):
                    return [move_paths(item) for item in value]
                if isinstance(value, dict):
                    return {key: move_paths(item) if key == 'path' else item for key, item in value.items()}
                raise ValueError('Invalid historical data_files configuration')
            for config in metadata.get('configs', []):
                if 'data_files' in config:
                    config['data_files'] = move_paths(config['data_files'])
            text = '---\n' + yaml.safe_dump(metadata, sort_keys=False) + '---\n' + body
        return _move_text_urls(text).encode()
    if path.suffix.lower() in ('.html', '.htm', '.md'):
        text = path.read_text()
        moved = _move_text_urls(text)
        if moved != text:
            return moved.encode()
    return None


def _move_text_urls(text):
    # Moving the entire tree preserves relative gallery/media links automatically.
    # Absolute links to main need a prefix; SHA-pinned explorer links are untouched.
    base = f'https://huggingface.co/datasets/{REPOSITORY}/'
    for kind in ('blob', 'resolve'):
        for folder in ('runs', 'viewer', 'media', 'gallery'):
            before = base + kind + '/main/' + folder + '/'
            text = text.replace(before, base + kind + '/main/results/' + folder + '/')
    return text


def migrate_results(root, *, source_revision, apply=False, expected_shards=11,
                    expected_rows=606, tracked_paths=None, additive_copy_revision=None,
                    consumer_revision=None):
    """Return a checksum/count/rollback receipt, copying only with ``apply=True``.

    ``root`` must be a fully materialized local historical snapshot (no symlinks).
    Optional ``tracked_paths`` is the original Hub's complete tracked-path list;
    when supplied every original path must be present. Already missing *indexed*
    CSVs are instead recorded as missing and are never fabricated. Count overrides
    support small synthetic fixtures. Original files are never overwritten.
    """
    if source_revision != ORIGINAL_REVISION:
        raise ValueError('Only the immutable historical migration revision is supported')
    for revision in (additive_copy_revision, consumer_revision):
        if revision is not None and not re.fullmatch('[a-f0-9]{40}', revision):
            raise ValueError('Checkpoint revisions must be full immutable SHAs')
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('A materialized local archive directory is required')
    local = {}
    for path in root.rglob('*'):
        name = path.relative_to(root).as_posix()
        if '.git' in path.relative_to(root).parts:
            continue
        if path.is_symlink():
            raise ValueError('Materialize symlinks before local migration')
        if path.is_file() and not name.startswith('results/'):
            local[name] = path
    if tracked_paths is not None:
        tracked = {_relative(name).as_posix() for name in tracked_paths}
        if tracked != set(local):
            raise ValueError('The local archive does not match the complete tracked-path inventory')
    index_path = local.get('viewer-index.json')
    if index_path is None or 'leaderboard.json' not in local:
        raise ValueError('Historical viewer index and leaderboard are required')
    index = json.loads(index_path.read_text())
    indexed = set(_index_paths(index))
    missing = sorted(indexed - local.keys())
    if any(not name.endswith('.csv') for name in missing):
        raise ValueError('A retained indexed viewer shard is missing')
    # Count the canonical per-run shards, not the extra grouped projections.
    shards = sorted(name for name in local if re.fullmatch(r'viewer/[^/]+/scores\.parquet', name))
    import pyarrow.parquet as pq
    rows = sum(pq.read_metadata(local[name]).num_rows for name in shards)
    if len(shards) != expected_shards or rows != expected_rows:
        raise ValueError(f'Historical shard/row counts differ: {len(shards)} shards, {rows} rows')
    entries, transformed = [], {}
    for name, path in sorted(local.items()):
        size, digest = _fingerprint(path)
        shared = name in _SHARED or name.startswith('LICENSES/')
        legacy = name.startswith('benchmark/')
        destination = None if shared or legacy else 'results/' + name
        entry = {'source': name, 'source_bytes': size, 'source_sha256': digest,
                 'classification': 'shared_metadata' if shared else 'unchanged_legacy' if legacy else 'verified_results_copy',
                 'destination': destination}
        if destination:
            raw = _transform(name, path)
            if raw is not None:
                transformed[name] = raw
            entry.update(destination_bytes=len(raw) if raw is not None else size,
                         destination_sha256=hashlib.sha256(raw).hexdigest() if raw is not None else digest,
                         transformed=raw is not None and hashlib.sha256(raw).hexdigest() != digest)
            target = root / destination
            if target.resolve() != target or target.exists() and (
                not target.is_file() or _fingerprint(target) != (entry['destination_bytes'], entry['destination_sha256'])
            ):
                raise ValueError(f'Conflicting or unsafe additive destination: {destination}')
            if any(parent.exists() and not parent.is_dir() for parent in target.parents if parent != root):
                raise ValueError(f'Invalid additive destination parent: {destination}')
        entries.append(entry)
    # All conflicts, counts, indexes and paths have been checked before any writes.
    if apply:
        for entry in entries:
            if not entry['destination']:
                continue
            source, target = root / entry['source'], root / entry['destination']
            if _fingerprint(source) != (entry['source_bytes'], entry['source_sha256']):
                raise ValueError('Historical source changed during migration')
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                with target.open('xb') as outgoing:
                    if entry['source'] in transformed:
                        outgoing.write(transformed[entry['source']])
                    else:
                        with source.open('rb') as incoming:
                            shutil.copyfileobj(incoming, outgoing, 1024**2)
            if _fingerprint(target) != (entry['destination_bytes'], entry['destination_sha256']):
                raise ValueError('Additive copy failed checksum verification; originals remain intact')
    return {'schema': _SCHEMA, 'repository': REPOSITORY, 'mode': 'apply' if apply else 'dry-run',
            'source_revision': source_revision, 'source_revision_verification': 'operator-attested local snapshot; local bytes SHA-256 verified',
            'additive_copy_revision': additive_copy_revision, 'consumer_revision': consumer_revision,
            'viewer_shards': len(shards), 'viewer_rows': rows, 'missing_indexed_sources': missing,
            'inventory_complete_against_tracked_paths': tracked_paths is not None,
            'files': entries, 'rollback': {'original_revision': source_revision,
                'reader_writer_prefix': '', 'consumer_revision_to_revert': consumer_revision,
                'delete_paths': [], 'instructions': 'Restore legacy reader/writer configuration or read the original pinned revision; retain both trees.'}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='Materialized local snapshot; no download is performed')
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--apply', action='store_true', help='Copy locally; default is a read-only dry run')
    parser.add_argument('--receipt', type=Path, help='New JSON receipt outside the archive; otherwise stdout')
    parser.add_argument('--tracked-paths', type=Path, help='JSON list of every original tracked repository path')
    parser.add_argument('--expected-shards', type=int, default=11)
    parser.add_argument('--expected-rows', type=int, default=606)
    parser.add_argument('--additive-copy-revision')
    parser.add_argument('--consumer-revision')
    args = parser.parse_args(argv)
    if args.receipt and (args.receipt.exists() or args.receipt.resolve().is_relative_to(args.root.resolve())):
        parser.error('--receipt must be a new path outside the source archive')
    try:
        receipt = migrate_results(args.root, source_revision=args.source_revision, apply=args.apply,
            expected_shards=args.expected_shards, expected_rows=args.expected_rows,
            tracked_paths=json.loads(args.tracked_paths.read_text()) if args.tracked_paths else None,
            additive_copy_revision=args.additive_copy_revision, consumer_revision=args.consumer_revision)
    except ValueError as error:
        parser.error(str(error))
    text = json.dumps(receipt, indent=2, allow_nan=False) + '\n'
    if args.receipt:
        with args.receipt.open('x') as stream:
            stream.write(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
