"""Actor-only loopback to private Unix transport. Standard library; no secrets.

This forwards raw bytes and has no controller, provider URL or credentials.
The trusted service behind the mounted socket still authenticates every request.
"""

from __future__ import annotations
import argparse
import asyncio
from pathlib import Path
import signal
import stat


class _Reject(asyncio.Protocol):
    def connection_made(self, transport):
        transport.close()


class _Peer(asyncio.Protocol):
    def __init__(self, front):
        self.front = front
        self.transport = None

    def connection_made(self, transport):
        self.transport = transport
        transport.set_write_buffer_limits(high=65536, low=16384)
        self.front.peer = self
        if self.front.transport is None or self.front.closed:
            transport.close()
        else:
            if self.front.pending:
                transport.write(self.front.pending)
                self.front.pending.clear()
            self.front.transport.resume_reading()

    def data_received(self, data):
        if not self.front.closed:
            self.front.touch()
            self.front.transport.write(data)

    def pause_writing(self):
        if self.front.transport:
            self.front.transport.pause_reading()

    def resume_writing(self):
        if self.front.transport and not self.front.closed:
            self.front.transport.resume_reading()

    def connection_lost(self, exc):
        self.front.finish()


class _Front(asyncio.Protocol):
    def __init__(self, server):
        self.server = server
        self.transport = None
        self.peer = None
        self.task = None
        self.closed = False
        self.pending = bytearray()
        self.timer = None
        self.touch()

    def touch(self):
        if self.closed:
            return
        if self.timer is not None:
            self.timer.cancel()
        self.timer = asyncio.get_running_loop().call_later(
            self.server.connection_timeout, self.finish
        )

    def connection_made(self, transport):
        self.transport = transport
        transport.set_write_buffer_limits(high=65536, low=16384)
        transport.pause_reading()
        self.task = asyncio.create_task(self.connect())

    async def connect(self):
        try:
            await asyncio.wait_for(
                asyncio.get_running_loop().create_unix_connection(
                    lambda: _Peer(self), path=str(self.server.socket_path)
                ),
                5,
            )
        except (OSError, asyncio.TimeoutError):
            self.finish()

    def data_received(self, data):
        if self.closed:
            return
        self.touch()
        if self.peer and self.peer.transport:
            self.peer.transport.write(data)
        elif len(self.pending) + len(data) <= 262144:
            # TCP may deliver an already queued read after pause_reading().
            # Retain that bounded first packet until the Unix peer connects.
            self.pending.extend(data)
        else:
            self.finish()

    def pause_writing(self):
        if self.peer and self.peer.transport:
            self.peer.transport.pause_reading()

    def resume_writing(self):
        if self.peer and self.peer.transport and not self.closed:
            self.peer.transport.resume_reading()

    def connection_lost(self, exc):
        self.finish()
        self.release()

    def release(self):
        self.timer.cancel()
        self.server._connections.discard(self)

    def abort(self):
        self.closed = True
        self.pending.clear()
        if self.transport:
            self.transport.abort()
        if self.peer and self.peer.transport:
            self.peer.transport.abort()
        if self.task and not self.task.done():
            self.task.cancel()
        self.release()

    def finish(self):
        if self.closed:
            return
        self.closed = True
        self.pending.clear()
        self.timer.cancel()
        if self.transport:
            self.transport.close()
        if self.peer and self.peer.transport:
            self.peer.transport.close()
        if (
            self.task
            and not self.task.done()
            and self.task is not asyncio.current_task()
        ):
            self.task.cancel()
        # Retain the slot while TCP output drains; a slow reader cannot cause
        # unlimited closed-but-buffered transports after new admissions.
        self.timer = asyncio.get_running_loop().call_later(5, self.abort)
        if self.transport is None:
            self.abort()


class RelayServer:
    def __init__(
        self, socket_path, *, port=32190, max_connections=16, connection_timeout=1800
    ):
        self.socket_path = Path(socket_path).absolute()
        if (
            not 0 <= port <= 65535
            or type(max_connections) is not int
            or max_connections <= 0
        ):
            raise ValueError("Valid fixed loopback relay limits required")
        if not 0 < connection_timeout <= 3600:
            raise ValueError("Bounded relay idle timeout required")
        self.port = port
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self._connections = set()
        self._server = None

    @property
    def connections_in_use(self):
        return len(self._connections)

    def _factory(self):
        if len(self._connections) >= self.max_connections:
            return _Reject()
        front = _Front(self)
        self._connections.add(front)
        return front

    async def start(self):
        if self.socket_path.resolve() != self.socket_path or not stat.S_ISSOCK(
            self.socket_path.stat().st_mode
        ):
            raise ValueError("Explicit ordinary Unix service socket required")
        self._server = await asyncio.get_running_loop().create_server(
            self._factory,
            host="127.0.0.1",
            port=self.port,
            backlog=self.max_connections,
        )
        self.port = self._server.sockets[0].getsockname()[1]

    async def close(self):
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        for front in list(self._connections):
            front.abort()


async def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unix-socket", required=True)
    parser.add_argument("--port", type=int, default=32190)
    parser.add_argument("--ready-file")
    args = parser.parse_args()
    relay = RelayServer(args.unix_socket, port=args.port)
    await relay.start()
    if args.ready_file:
        path = Path(args.ready_file)
        with path.open("x") as stream:
            stream.write(str(relay.port))
        path.chmod(0o600)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    try:
        await stop.wait()
    finally:
        await relay.close()


if __name__ == "__main__":
    asyncio.run(_main())
