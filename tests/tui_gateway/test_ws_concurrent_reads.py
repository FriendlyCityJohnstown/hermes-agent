"""The WS concurrent-read lane (#134889 GW-18 review): waiting context reads stay bounded, a server
shutdown or client hang-up never waits for reads that have not started, and a reply the read lane
cannot send never cancels the control handler the serial lane is running.

The real ``handle_ws`` dispatcher; only ``_dispatch_request`` is a gated async stand-in.
"""
import asyncio
import json

import pytest

from tui_gateway import server
from tui_gateway import ws as ws_mod


class Socket:
    def __init__(self):
        self.inbound = asyncio.Queue()
        self.received = 0
        self.sent = []
        self.fail_ids = set()

    async def accept(self):
        pass

    async def send_text(self, line):
        frame = json.loads(line)
        if frame.get("id") in self.fail_ids:
            raise RuntimeError("peer gone")
        self.sent.append(frame)

    async def receive_text(self):
        item = await self.inbound.get()
        if item is None:
            raise ws_mod._WebSocketDisconnect()
        self.received += 1
        return json.dumps(item)

    async def close(self):
        pass


@pytest.fixture
def rig(monkeypatch):
    gates, log = {}, []

    async def dispatch(authority, req, method, transport):
        log.append(f"start:{req['id']}")
        try:
            gate = gates.get(method)
            if gate is not None:
                await gate.wait()
        except asyncio.CancelledError:
            log.append(f"cancelled:{req['id']}")
            raise
        log.append(f"done:{req['id']}")
        return {"jsonrpc": "2.0", "id": req["id"], "result": {}}

    monkeypatch.setattr(ws_mod, "_dispatch_request", dispatch)
    monkeypatch.setattr(ws_mod, "_start_backend_liveness", lambda: None)
    monkeypatch.setattr(server, "_ensure_skin_watcher", lambda: None)
    monkeypatch.setattr(server, "_ensure_lease_watcher", lambda: None)
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0)
    monkeypatch.setattr(server, "_close_sessions_for_transport", lambda transport, end_reason: (log.append("teardown"), (0, 0))[1])
    return Socket(), gates, log


def rpc(req_id, method="session.context_breakdown"):
    return {"jsonrpc": "2.0", "id": req_id, "method": method, "params": {}}


async def settle(predicate, timeout=5.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            return False
        await asyncio.sleep(0.01)
    return True


@pytest.mark.asyncio
async def test_waiting_context_reads_keep_reader_backpressure(rig):
    sock, gates, log = rig
    gates["session.context_breakdown"] = release = asyncio.Event()
    task = asyncio.create_task(ws_mod.handle_ws(sock))
    burst = ws_mod._WS_DISPATCH_QUEUE_MAX * 4
    for n in range(burst):
        sock.inbound.put_nowait(rpc(n))
    try:
        # A bounded reader parks with frames unread; give it every chance to drain them all.
        assert not await settle(lambda: sock.inbound.empty(), timeout=2.0)
        bound = ws_mod._WS_DISPATCH_QUEUE_MAX + ws_mod._WS_CONCURRENT_READ_LIMIT + ws_mod._WS_WAITING_READ_LIMIT + 2
        assert sock.received <= bound, (sock.received, bound)
        assert sum(entry.startswith("start:") for entry in log) == ws_mod._WS_CONCURRENT_READ_LIMIT
    finally:
        release.set()
    assert await settle(lambda: sock.inbound.empty())
    sock.inbound.put_nowait(None)
    await asyncio.wait_for(task, 30)
    assert len(sock.sent) == burst + 1  # gateway.ready + one reply per read


@pytest.mark.asyncio
async def test_server_shutdown_does_not_wait_for_a_running_context_read(rig):
    sock, gates, log = rig
    gates["session.context_breakdown"] = release = asyncio.Event()
    task = asyncio.create_task(ws_mod.handle_ws(sock))
    sock.inbound.put_nowait(rpc(1))
    try:
        assert await settle(lambda: "start:1" in log)
        task.cancel()
        done, _ = await asyncio.wait({task}, timeout=5)
        assert task in done, "socket shutdown still waits for the slow context read"
        assert "cancelled:1" in log
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_hang_up_drops_reads_that_never_started(rig):
    sock, gates, log = rig
    gates["session.context_breakdown"] = release = asyncio.Event()
    task = asyncio.create_task(ws_mod.handle_ws(sock))
    for n in range(6):
        sock.inbound.put_nowait(rpc(n))
    assert await settle(lambda: sock.received == 6 and sum(e.startswith("start:") for e in log) == 2)
    sock.inbound.put_nowait(None)
    assert not await settle(lambda: "teardown" in log, timeout=2.0), "teardown ran under a started read"
    release.set()
    await asyncio.wait_for(task, 10)
    assert sorted(e for e in log if e.startswith("start:")) == ["start:0", "start:1"]
    assert log[-1] == "teardown"


@pytest.mark.asyncio
async def test_a_failed_read_reply_does_not_cancel_the_running_control(rig):
    """The serial lane's handler (a submit, an interrupt, a mutate) finishes; then the link ends."""
    sock, gates, log = rig
    gates["prompt.submit"] = release = asyncio.Event()
    gates["session.context_breakdown"] = read_gate = asyncio.Event()
    sock.fail_ids.add("read")
    task = asyncio.create_task(ws_mod.handle_ws(sock))
    sock.inbound.put_nowait(rpc("read"))
    assert await settle(lambda: "start:read" in log)
    sock.inbound.put_nowait(rpc("submit", "prompt.submit"))
    assert await settle(lambda: "start:submit" in log)
    read_gate.set()  # the read's reply now fails while the submit is mid-await
    assert await settle(lambda: "done:read" in log)
    assert not await settle(lambda: task.done(), timeout=2.0), "the link ended under the running control"
    release.set()
    await asyncio.wait_for(task, 10)
    assert "cancelled:submit" not in log and "done:submit" in log
    assert log[-1] == "teardown"


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["session.transcript"])
async def test_a_slow_stored_read_never_holds_an_interrupt(rig, method):
    """A picker paging a long stored transcript must not queue the socket's interrupt behind it (#150091)."""
    sock, gates, log = rig
    gates[method] = release = asyncio.Event()
    task = asyncio.create_task(ws_mod.handle_ws(sock))
    sock.inbound.put_nowait(rpc(1, method))
    sock.inbound.put_nowait(rpc(2, "session.interrupt"))
    try:
        assert await settle(lambda: "done:2" in log), log
        assert "done:1" not in log
    finally:
        release.set()
    sock.inbound.put_nowait(None)
    await asyncio.wait_for(task, 30)


@pytest.mark.asyncio
async def test_post_disconnect_uvicorn_close_error_does_not_warn_or_clobber_reason(rig, caplog):
    """If a client disconnects while session.context_breakdown is running, Uvicorn's post-close
    send RuntimeError must not emit WARNINGs or overwrite disconnect_reason/send_failures."""
    import logging

    sock, gates, log = rig
    gates["session.context_breakdown"] = release = asyncio.Event()
    real_send = type(sock).send_text

    async def post_close_send(self, line):
        if json.loads(line).get("id") == "ctx":
            raise RuntimeError(
                "Unexpected ASGI message 'websocket.send', after sending 'websocket.close' or response already completed."
            )
        await real_send(self, line)

    type(sock).send_text = post_close_send
    try:
        with caplog.at_level(logging.INFO, logger="tui_gateway.ws"):
            task = asyncio.create_task(ws_mod.handle_ws(sock))
            sock.inbound.put_nowait(rpc("ctx", "session.context_breakdown"))
            assert await settle(lambda: "start:ctx" in log)
            sock.inbound.put_nowait(None)
            assert await settle(lambda: sock.received == 1)
            await asyncio.sleep(0.05)
            release.set()
            await asyncio.wait_for(task, 10)
    finally:
        type(sock).send_text = real_send

    warnings = [r for r in caplog.records if r.name == "tui_gateway.ws" and r.levelno >= logging.WARNING]
    assert warnings == [], [r.getMessage() for r in warnings]
    closed_msgs = [r.getMessage() for r in caplog.records if r.name == "tui_gateway.ws" and "ws closed" in r.getMessage()]
    assert len(closed_msgs) == 1
    assert "reason=client_disconnect(" in closed_msgs[0]
    assert "send_failures=0" in closed_msgs[0]


@pytest.mark.asyncio
async def test_unexpected_send_error_while_connected_still_warns(rig, caplog):
    """While the client is still connected, an unexpected send error still logs WARNINGs and
    records send_failures."""
    import logging

    sock, _gates, _log = rig
    sock.fail_ids.add("ctx")

    with caplog.at_level(logging.WARNING, logger="tui_gateway.ws"):
        task = asyncio.create_task(ws_mod.handle_ws(sock))
        sock.inbound.put_nowait(rpc("ctx", "session.context_breakdown"))
        await asyncio.wait_for(task, 10)

    warn_msgs = [r.getMessage() for r in caplog.records if r.name == "tui_gateway.ws" and r.levelno >= logging.WARNING]
    assert any("ws send failed" in m for m in warn_msgs), warn_msgs
    assert any("ws response send failed" in m for m in warn_msgs), warn_msgs
