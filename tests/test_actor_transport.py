"""Preserve active actor transport while bounding queued and idle connections."""
import asyncio
import os
from types import SimpleNamespace

import pytest

from simready_bench.engine.actor_relay import RelayServer, _Front, _Peer


class Transport:
    def __init__(self):
        self.data = bytearray()
        self.closed = False

    def write(self, data):
        self.data.extend(data)

    def close(self):
        self.closed = True

    abort = close

    def set_write_buffer_limits(self, **kwargs):
        pass

    def resume_reading(self):
        pass


def test_first_tcp_packet_waits_for_unix_connection():
    async def exercise():
        server = SimpleNamespace(connection_timeout=1, _connections=set())
        front = _Front(server)
        front.transport = Transport()
        front.data_received(b'POST /v1/chat/completions HTTP/1.1\r\n')
        assert not front.closed
        upstream = Transport()
        _Peer(front).connection_made(upstream)
        assert bytes(upstream.data) == b'POST /v1/chat/completions HTTP/1.1\r\n'
        assert not front.pending
        front.abort()

    asyncio.run(exercise())


def test_unconnected_first_packet_remains_bounded():
    async def exercise():
        front = _Front(SimpleNamespace(connection_timeout=1, _connections=set()))
        front.transport = Transport()
        front.data_received(b'x' * 262144)
        assert not front.closed
        front.data_received(b'x')
        assert front.closed
        assert not front.pending
        front.abort()

    asyncio.run(exercise())


@pytest.mark.skipif(os.name == 'nt', reason='Unix actor service transport')
def test_busy_connection_survives_its_original_age_then_expires_when_idle(tmp_path):
    async def exercise():
        socket = tmp_path / 'service.sock'
        clients = set()

        async def echo(reader, writer):
            clients.add(writer)
            try:
                while data := await reader.read(1024):
                    writer.write(data)
                    await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()
                clients.discard(writer)

        upstream = await asyncio.start_unix_server(echo, path=str(socket))
        relay = RelayServer(socket, port=0, connection_timeout=0.3)
        await relay.start()
        reader, writer = await asyncio.open_connection('127.0.0.1', relay.port)
        try:
            for _ in range(8):
                writer.write(b'x')
                await writer.drain()
                assert await asyncio.wait_for(reader.readexactly(1), 1) == b'x'
                await asyncio.sleep(0.08)
            assert await asyncio.wait_for(reader.read(1), 1) == b''
        finally:
            writer.close()
            await writer.wait_closed()
            await relay.close()
            upstream.close()
            await upstream.wait_closed()
            for client in tuple(clients):
                client.close()

    asyncio.run(exercise())


def test_upstream_stream_activity_renews_idle_deadline():
    async def exercise():
        front = _Front(SimpleNamespace(connection_timeout=0.15, _connections=set()))
        front.transport = Transport()
        peer = _Peer(front)
        peer.connection_made(Transport())
        try:
            for _ in range(5):
                await asyncio.sleep(0.06)
                peer.data_received(b'data: token\n\n')
                assert not front.closed
            await asyncio.sleep(0.22)
            assert front.closed
        finally:
            front.abort()

    asyncio.run(exercise())
