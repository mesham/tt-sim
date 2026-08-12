"""Stop the tt-metal host when the simulator cannot complete its side of a rendezvous.

Why this exists
---------------

UMD's simulation wire protocol is strictly host-initiated: the host sends
``WRITE``/``READ``/``RESET_*`` and blocks in ``recv_from_device`` for the reply.
There is no "the simulator gave up" message, and UMD sets no receive timeout
(``simulation_host.cpp`` binds an nng ``pair1`` listener and calls
``nng_recvmsg`` with no deadline). So when the simulator stops answering — for
any reason — **the host waits forever**. It is not a tt-sim bug and it cannot
be fixed from the tt-sim side of the wire.

It can be fixed from the *process* side. UMD spawns the simulator
(``rtl_sim_communicator.cpp`` → ``uv_spawn`` of ``run.sh``), so the simulator is
in a position to end the host it was started by, which is exactly what the
runbook has so far told users to do by hand ("interrupt and re-run with the
coord added"). :func:`stop_host` does it for them.

Identifying the host
--------------------

**The host is the process holding the listening socket at the address we were
told to dial.** UMD generates ``tcp://<hostname>:<random 50000-59999 port>``,
binds it, exports it as ``NNG_SOCKET_ADDR`` and only then spawns us — so the
owner of that ``LISTEN`` socket *is* our wire peer, by definition of a bound
TCP port. That is an exact identification, not a heuristic, and it is what this
module uses: parse ``/proc/net/tcp{,6}`` for a listener on the port, map its
inode to a pid via ``/proc/*/fd``, and require the answer to be a single pid
that is neither us nor init.

Deliberately *not* used: the parent pid on its own. It happens to be correct
today (``uv_spawn`` is a plain fork/exec and ``run.sh`` ``exec``s python, so
``getppid()`` is the host), but a server started by hand from a shell has the
user's shell as its parent, and "signal your parent" would then kill the shell.
The port owner is right in both cases and wrong in neither.

Set ``TT_SIM_NO_HOST_STOP`` truthy to disable — the host then hangs as before
and must be interrupted by hand.
"""

import contextlib
import glob
import os
import re
import signal
import sys
import time

# How long to let the host act on SIGTERM before escalating to SIGKILL.
_TERM_GRACE_S = 3.0
_POLL_S = 0.05

# Only ``tcp://host:port`` is resolvable this way, and only that form is
# accepted: an ``ipc://`` path ending in digits would otherwise be read as a
# port number and could name a listener belonging to an unrelated process.
_ADDR_PORT_RE = re.compile(r"^tcp://[^/]+:(\d+)$")

DISABLE_ENV = "TT_SIM_NO_HOST_STOP"


def _truthy(val):
    return val is not None and val.strip().lower() in {"1", "true", "yes", "on"}


def wire_port(addr):
    """Return the TCP port in an ``NNG_SOCKET_ADDR``-style address, or None.

    ``tcp://devvm:54812`` -> ``54812``. Anything else (an ``ipc://`` path, say)
    is not resolvable this way and gives ``None``.
    """
    if not addr:
        return None
    match = _ADDR_PORT_RE.match(addr.strip())
    if match is None:
        return None
    port = int(match.group(1))
    return port if 0 < port < 65536 else None


def _listening_inodes(port, proc_net=("/proc/net/tcp", "/proc/net/tcp6")):
    """Socket inodes of every TCP listener bound to ``port`` on this host."""
    inodes = set()
    for path in proc_net:
        try:
            with open(path) as handle:
                lines = handle.read().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            # sl local_address rem_address st ... inode ; st 0A == TCP_LISTEN.
            if len(fields) < 10 or fields[3] != "0A":
                continue
            try:
                local_port = int(fields[1].rsplit(":", 1)[1], 16)
            except (IndexError, ValueError):
                continue
            if local_port == port:
                inodes.add(fields[9])
    return inodes


def _pids_owning(inodes):
    """Pids holding an fd for any of ``inodes``."""
    if not inodes:
        return set()
    wanted = {f"socket:[{inode}]" for inode in inodes}
    pids = set()
    for entry in glob.glob("/proc/[0-9]*/fd/*"):
        try:
            if os.readlink(entry) in wanted:
                pids.add(int(entry.split("/")[2]))
        except (OSError, ValueError):
            # The process exited mid-scan, or its fds aren't ours to read.
            continue
    return pids


def find_wire_peer(addr):
    """Return the pid listening at ``addr``, or None if it is not unambiguous.

    ``None`` is returned rather than a guess whenever the answer is anything
    other than exactly one live process that is not this one: no port in the
    address, no listener, several owners (a forked host sharing the fd), or
    only ourselves.
    """
    port = wire_port(addr)
    if port is None:
        return None
    pids = _pids_owning(_listening_inodes(port))
    pids.discard(os.getpid())
    pids.discard(1)
    if len(pids) != 1:
        return None
    return pids.pop()


def _alive(pid):
    """True while ``pid`` is a running process — a zombie does not count.

    ``kill(pid, 0)`` alone is not enough: a process that has died but not yet
    been reaped by its parent still answers it, so the grace loop would time
    out and escalate to SIGKILL against a corpse. That is exactly the shape of
    the case where the *host* is a child of the thing waiting on it.
    """
    try:
        with open(f"/proc/{pid}/stat") as handle:
            stat = handle.read()
    except OSError:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    # "pid (comm) STATE ..." — comm may itself contain spaces and parens, so
    # the state is the first field after the *last* closing paren.
    tail = stat.rpartition(")")[2].split()
    return bool(tail) and tail[0] != "Z"


def stop_host(reason, *, addr=None, env=None, stream=None):
    """End the tt-metal host stranded by ``reason``. True if one was stopped.

    ``reason`` is a short phrase completing "the host is waiting for ..." and is
    echoed in the message, so the line that appears on the user's terminal says
    both what is being killed and why. SIGTERM first (the host gets to run its
    own teardown), SIGKILL only if it is still there after a short grace.

    Returns False — having said so on stderr — when no peer can be identified
    or stopping is disabled. Callers must treat that as "the host is still
    live", because the alternative (exit anyway) is the infinite wait this
    module exists to remove.
    """
    if env is None:
        env = os.environ
    if stream is None:
        stream = sys.stderr
    if addr is None:
        addr = env.get("NNG_SOCKET_ADDR")

    def say(msg):
        print(f"[server] {msg}", file=stream, flush=True)

    if _truthy(env.get(DISABLE_ENV)):
        say(
            f"{DISABLE_ENV} is set — leaving the tt-metal host running; it will wait forever, interrupt it with Ctrl-C."
        )
        return False

    pid = find_wire_peer(addr)
    if pid is None:
        say(
            f"could not identify the tt-metal host on the other end of "
            f"{addr!r} — it is waiting for {reason} and nothing will arrive; "
            f"interrupt it with Ctrl-C."
        )
        return False

    say(
        f"stopping the tt-metal host (pid {pid}): it is waiting for {reason}, which no simulated tile will ever send."
    )
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        say(
            f"could not signal the tt-metal host (pid {pid}): {exc}; interrupt it with Ctrl-C."
        )
        return False

    deadline = time.monotonic() + _TERM_GRACE_S
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(_POLL_S)

    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
    return True


@contextlib.contextmanager
def host_not_stranded(addr, *, stopper=stop_host):
    """Don't leave the host waiting when the simulator dies of an exception.

    Same infinite wait as the launch guard, reached a different way: a
    ``NoCAlignmentError`` (or any other simulator-side exception) escapes
    ``Transport.serve``, the server process unwinds and exits, and the host —
    which has no timeout — blocks in ``recv`` for ever, with the traceback that
    explains it scrolled off the top. Only exceptions get this treatment;
    ``KeyboardInterrupt`` and a clean ``EXIT`` shutdown do not, since in both of
    those the run is already ending under someone's control.
    """
    try:
        yield
    except Exception as exc:
        stopper(
            f"a reply the simulator can no longer send ({type(exc).__name__}: {exc})",
            addr=addr,
        )
        raise
