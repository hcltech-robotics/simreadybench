import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def manager(monkeypatch):
    sdk = SimpleNamespace(Daytona=Mock(), SessionExecuteRequest=SimpleNamespace)
    monkeypatch.setitem(sys.modules, 'daytona', sdk)
    path = Path(__file__).resolve().parents[1] / 'scripts/daytona_live_logs.py'
    spec = importlib.util.spec_from_file_location('daytona_live_logs_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('labels', [
    {'project':'unrelated', 'role':'benchmark-executor'},
    {'project':'simreadybench', 'role':'unrelated'},
    {},
])
def test_install_rejects_other_owners_before_any_upload(manager, labels):
    box = SimpleNamespace(id='sandbox-1', labels=labels, fs=Mock(), process=Mock())
    client = SimpleNamespace(get=Mock(return_value=box))
    with pytest.raises(ValueError, match='ownership'):
        manager.install(client, {'sandbox_id':'sandbox-1'}, b'viewer source')
    box.fs.upload_file.assert_not_called()
    assert box.process.mock_calls == []


def test_install_reuses_saved_running_session_without_duplicate_creation(manager):
    process = Mock()
    process.get_session_command.return_value = SimpleNamespace(exit_code=None)
    box = SimpleNamespace(id='sandbox-1', labels={'project':'simreadybench',
                          'role':'benchmark-executor'}, fs=Mock(), process=process)
    client = SimpleNamespace(get=Mock(return_value=box))
    previous = {'sandbox_id':'sandbox-1', 'command_id':'command-1', 'status':'running'}
    result = manager.install(client, {'sandbox_id':'sandbox-1'}, b'viewer source', previous)
    assert result is previous
    process.get_session_command.assert_called_once_with('simready-live-logs', 'command-1')
    process.create_session.assert_not_called()
    process.execute_session_command.assert_not_called()
    box.fs.upload_file.assert_not_called()


def test_install_refuses_unrecorded_existing_session_before_upload(manager):
    process = Mock()
    process.list_sessions.return_value = [SimpleNamespace(session_id='simready-live-logs')]
    box = SimpleNamespace(id='sandbox-1', labels={'project':'simreadybench',
                          'role':'benchmark-executor'}, fs=Mock(), process=process)
    client = SimpleNamespace(get=Mock(return_value=box))
    with pytest.raises(RuntimeError, match='reconciliation'):
        manager.install(client, {'sandbox_id':'sandbox-1'}, b'viewer source')
    box.fs.upload_file.assert_not_called()
    process.create_session.assert_not_called()
