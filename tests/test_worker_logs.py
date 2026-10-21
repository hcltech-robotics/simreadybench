import json
import sqlite3

from simready_bench.worker_logs import RECORD_LIMIT, SessionFollower, request_summary, safe_text, summarise_record


def event(content, role='assistant', **extra):
    return {'type': 'message', 'timestamp': '2026-10-21T14:00:00Z',
            'message': {'role': role, 'content': content, **extra}}


def test_partial_records_wait_for_newline_and_do_not_repeat(tmp_path):
    path = tmp_path / 'session.jsonl'
    encoded = json.dumps(event([{'type': 'toolCall', 'name': 'bash',
                                 'arguments': {'command': 'python build.py'}}])).encode()
    path.write_bytes(encoded[:50])
    follower = SessionFollower(path)
    assert follower.poll() == []
    with path.open('ab') as stream:
        stream.write(encoded[50:] + b'\n')
    assert follower.poll() == ['2026-10-21T14:00:00Z tool bash: python build.py']
    assert follower.poll() == []


def test_reasoning_images_and_user_prompts_are_omitted():
    assert summarise_record(event([{'type': 'thinking', 'thinking': 'private reasoning'},
                                   {'type': 'image', 'data': 'base64 image'}])) == []
    assert summarise_record(event([{'type': 'text', 'text': 'full prompt'}], role='user')) == []
    assert summarise_record(event([{'type': 'text', 'text': 'Saved candidate.usda'}],
                                   role='toolResult', toolName='bash')) == [
                                       'result bash: Saved candidate.usda']


def test_credentials_blobs_controls_and_long_output_are_bounded():
    text = safe_text('HF_TOKEN=hf_abcdefghijklmnopqrst API_KEY="secret value" '
                     'Authorization: Bearer abcdefghijklmno \x1b[31m' + 'A' * 500)
    assert 'abcdefghijklmnopqrst' not in text
    assert 'secret value' not in text
    assert 'abcdefghijklmno' not in text
    assert '\x1b' not in text
    assert 'A' * 100 not in text
    assert len(safe_text('many words ' * 100)) <= 300
    assert 'private-value' not in safe_text('DAYTONA_API_KEY=private-value')
    assert 'private-value' not in safe_text('--api-key private-value')
    assert 'private-value' not in safe_text('{"OPENAI_API_KEY":"private-value"}')


def test_oversized_record_is_discarded_and_next_record_is_read(tmp_path):
    path = tmp_path / 'session.jsonl'
    path.write_bytes(b'')
    follower = SessionFollower(path)
    follower.poll()
    with path.open('ab') as stream:
        stream.write(b'A' * (RECORD_LIMIT + 100))
    for _ in range(6):
        assert follower.poll() == []
        assert len(follower.pending) <= RECORD_LIMIT
    with path.open('ab') as stream:
        stream.write(b'\n' + json.dumps(event([{'type': 'text', 'text': 'Recovered'}])).encode() + b'\n')
    assert follower.poll() == ['2026-10-21T14:00:00Z assistant: Recovered']


def test_request_summary_reads_counts_without_response_contents(tmp_path):
    path = tmp_path / 'inference-gateway.sqlite3'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE requests(record TEXT)')
        for status in ('completed', 'completed', 'admitted', 'failed'):
            db.execute('INSERT INTO requests VALUES (?)', (json.dumps({'status':status, 'prompt':'private prompt'}),))
    assert request_summary(tmp_path) == 'model requests: admitted=1, completed=2, failed=1'
