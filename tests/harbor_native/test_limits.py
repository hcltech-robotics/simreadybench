"""One artifact budget is enforced across transport, parser and final freezing."""

from simready_eval import cli, export, feedback, limits, package, prepare


def test_shared_candidate_budget_and_public_contract_are_consistent():
    assert limits.CANDIDATE_BYTES == cli.MAX_BYTES == package.LIMIT_BYTES == feedback.CANDIDATE_BYTES == 1024**3
    assert package.REFERENCE_LIMIT_BYTES == 2 * 1024**3
    assert cli.MAX_FILES == package.LIMIT_FILES == feedback.MAX_FILES == limits.MAX_ENTRIES == 10000
    assert feedback.MAX_ARCHIVE == limits.TAR_BYTES > limits.CANDIDATE_BYTES
    assert 'at most 1 GiB' in prepare.public_instruction([1, 2, 3])
    assert '/tmp:size=2147483648,mode=1777' in export._compose()['services']['feedback']['tmpfs']
    assert export._compose()['services']['feedback']['mem_limit'] == '8g'
    assert cli.RENDER_ADDRESS_SPACE_BYTES == 12 * 1024**3


def test_final_freeze_rejects_one_byte_above_shared_budget_before_reading(tmp_path):
    root = tmp_path / 'submission'
    root.mkdir()
    with (root / 'asset.usd').open('wb') as stream:
        stream.truncate(limits.CANDIDATE_BYTES + 1)
    result = cli.freeze_candidate(root, tmp_path / 'frozen')
    assert result['status'] == 'invalid' and result['reason'] == 'package_too_large'
