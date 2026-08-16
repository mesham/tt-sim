import sys
import threading
from enum import IntEnum

from tt_sim.device.clock import Clockable
from tt_sim.memory.mem_mapable import MemMapable
from tt_sim.network.alignment import (
    L1_CONGRUENCE,
    check_congruence,
    congruence_for_read,
)
from tt_sim.network.noc_coords import WormholeNocCoords
from tt_sim.perf.model import noc_cost_model
from tt_sim.trace import EventCategory, NoCEvent, get_bus
from tt_sim.util.bits import clear_bit, extract_bits, replace_bits
from tt_sim.util.conversion import (
    conv_to_bytes,
    conv_to_uint32,
)


class NoCOverlay(MemMapable):
    NOC_NUM_STREAMS = 64
    NOC_STREAM_REG_SPACE_SIZE = 0x1000
    STREAM_MSG_DATA_CLEAR_REG_INDEX = 22

    def __init__(self):
        self.stream_regs = [
            [0 for i in range(NoCOverlay.NOC_STREAM_REG_SPACE_SIZE >> 2)]
            for i in range(NoCOverlay.NOC_NUM_STREAMS)
        ]

    def read(self, addr, size):
        stream_id, reg_id = self.getStreamAndRegisterFromAddress(addr)
        return conv_to_bytes(self.stream_regs[stream_id][reg_id])

    def write(self, addr, value, size=None):
        stream_id, reg_id = self.getStreamAndRegisterFromAddress(addr)
        self.stream_regs[stream_id][reg_id] = conv_to_uint32(value)

    def getStreamAndRegisterFromAddress(self, addr):
        stream_id = int(addr / NoCOverlay.NOC_STREAM_REG_SPACE_SIZE)
        reg_id = int(addr % NoCOverlay.NOC_STREAM_REG_SPACE_SIZE) >> 2
        return stream_id, reg_id

    def getStreamRegisterAddress(self, stream_id, reg_id):
        return (stream_id * NoCOverlay.NOC_STREAM_REG_SPACE_SIZE) + (reg_id << 2)

    def getSize(self):
        return 0x3FFFF


class NoCCoordinateError(ValueError):
    """A coordinate handed to the hop model names no cell on the grid.

    Both distance functions below walk a torus of ``grid_x`` x ``grid_y``
    routers, so a coordinate outside it is not a longer journey — it is not a
    journey at all. :func:`noc_hop_count` would return a modular fiction and
    :func:`noc_route_links` would never terminate (its walk steps
    ``x = (x + 1) % grid_x``, which can never equal an out-of-grid ``dst[0]``),
    spinning in pure Python with the device clock stopped and no deadlock
    detector able to fire. There is no legitimate caller with an out-of-grid
    coordinate, so both refuse instead — the same choice
    :class:`~tt_sim.network.alignment.NoCAlignmentError` and
    :class:`NoCResponseError` make, and for the same reason.

    The message names the coordinate, the grid and which end of the journey it
    came from. Which *endpoint object* it came from is added by
    :meth:`NUI.flight_cycles_to` and :meth:`NUI.route_links_to`, in an
    ``except`` clause rather than as an argument: both are per-packet calls, so
    the description must cost nothing on the path where nothing is wrong.
    """


def _check_in_grid(coord, grid_x, grid_y, role):
    """Raise :class:`NoCCoordinateError` unless ``coord`` is a cell of the grid."""
    x, y = coord
    if 0 <= x < grid_x and 0 <= y < grid_y:
        return
    raise NoCCoordinateError(
        f"NoC {role} coordinate {(x, y)} is outside the {grid_x}x{grid_y} grid "
        "this NoC routes on, so the distance to it is undefined"
    )


def noc_hop_count(
    src, dst, grid_x, grid_y, *, src_role="source", dst_role="destination"
):
    """Router-to-router hops from ``src`` to ``dst``, **in one NoC's own space**.

    Each NoC is a torus: every row is a ring and every column is a ring, and a
    packet only ever travels in the increasing direction of that NoC's own
    coordinates, wrapping past the edge rather than turning round. So the hop
    count on each axis is the *forward* distance modulo the grid dimension, and
    the total is their sum (dimension-ordered routing, X then Y — the order
    does not change the count).

    Two consequences worth stating, because both are easy to get wrong and
    there are tests for each:

    * **It is not symmetric.** ``hops(a, b) + hops(b, a)`` is ``grid_x`` on any
      axis where the two differ, not ``2 * |dx|`` — a request and its response
      travel different distances on the same NoC, and a round trip between two
      tiles differing on both axes always costs exactly ``grid_x + grid_y``
      hops however close together they are.
    * **NoC 1 is the same formula in a different space.** NoC 1's origin is the
      opposite corner, so an ``NUI`` on NoC 1 holds the mirrored coord
      ``(grid_x-1-x, grid_y-1-y)`` in :attr:`NUI.x_coord` / :attr:`NUI.y_coord`.
      Mirroring both endpoints negates ``dx`` and ``dy``, which is exactly the
      reversal of routing direction that distinguishes the two NoCs — so
      feeding this function each endpoint's *per-NoC* coord gives NoC 1's
      (opposite) hop count with no special case, and
      ``hops_noc1(a, b) == hops_noc0(b, a)``.

    Callers must pass coords in the same space, which on this NoC means
    ``NUI.x_coord`` / ``NUI.y_coord`` (per-NoC) and **never** ``id_pair``
    (canonical NoC 0 on both NoCs). ``src_role`` / ``dst_role`` name the
    endpoint each coord came from, for :class:`NoCCoordinateError`'s message.
    """
    _check_in_grid(src, grid_x, grid_y, src_role)
    _check_in_grid(dst, grid_x, grid_y, dst_role)
    return (dst[0] - src[0]) % grid_x + (dst[1] - src[1]) % grid_y


def noc_route_links(
    src, dst, grid_x, grid_y, *, src_role="source", dst_role="destination"
):
    """The router-to-router links a packet crosses, **in order**.

    :func:`noc_hop_count` is this function's length, and for a long time the
    length was all the network layer committed to: a hop *count* decides a
    flight time, and a flight time is all the model spent. A link's *identity*
    needs an order as well, and this is where the network layer commits to one.

    Dimension-ordered X then Y on a directional torus — the same routing the
    hop count already assumes (it sums the two axes' forward distances, which
    is only the distance travelled if the packet turns exactly once). A link is
    named by the router it leaves and the axis it leaves on, so ``("X", y, x)``
    is the link from ``(x, y)`` to ``((x + 1) % grid_x, y)`` and ``("Y", x, y)``
    the link from ``(x, y)`` to ``(x, (y + 1) % grid_y)``. Two packets cross the
    same link exactly when they produce the same tuple, which is what makes the
    link a shareable resource rather than a per-packet fact.

    Coords are in one NoC's own space, exactly as for :func:`noc_hop_count`, so
    NoC 1's mirrored coords give NoC 1's (opposite) route with no special case.
    ``tt_sim.perf.noc_congestion_plan.route_links`` **is** this function — the
    experiment planner and the simulator have to name links identically or a
    measured shared-link count describes a different machine from the modelled
    one.
    """
    _check_in_grid(src, grid_x, grid_y, src_role)
    _check_in_grid(dst, grid_x, grid_y, dst_role)
    links = []
    x, y = src
    while x != dst[0]:
        links.append(("X", y, x))
        x = (x + 1) % grid_x
    while y != dst[1]:
        links.append(("Y", x, y))
        y = (y + 1) % grid_y
    return tuple(links)


class NocLinkRegistry:
    """One free-cycle watermark per router-to-router link, on one NoC.

    The shared half of the bandwidth model. ``NUI._tx_free_cycle`` is the same
    object one level up — "the cycle this link finishes injecting what is
    already on it" — and the only structural difference is *ownership*: an
    injection port belongs to one NIU, while a router-to-router link is crossed
    by every tile whose route passes through it. So this lives on the **device**
    (one instance per NoC, handed to every NUI as it is registered) rather than
    on an NIU, because two NIUs contending on it is the entire point.

    That is also the whole mechanism. No arbitration policy, no flow census, no
    notion of which flows are "saturating": a packet takes the link when it is
    free and waits when it is not, and a flow that under-uses a link simply
    never finds it busy. That is why a 64-byte packet — one flit against a
    ~40-cycle issue loop — measures zero effect on silicon and costs zero here,
    with no threshold anybody had to choose.

    Cross-thread safe because ``MultiTileClock`` can tick heavy tiles on worker
    threads (opt-in; the default pump is sequential and therefore
    deterministic). Under threading the *order* two tiles claim a link in is
    the scheduler's, which is the same non-determinism inbound packet ordering
    already has.
    """

    def __init__(self):
        #: ``{link: cycle}`` — when each link finishes carrying what is on it.
        #: Sparse: a link nothing has crossed has no entry and reads 0.
        self._free_cycle = {}
        self._lock = threading.Lock()
        #: Links claimed, and how many of those found the link busy. Diagnostic
        #: only — but ``waits == 0`` is the property that says "this workload
        #: has no link contention", which is what makes the term's effect on a
        #: single-flow workload checkable rather than assumed.
        self.claims = 0
        self.waits = 0
        self.cycles_waited = 0

    def free_cycle(self, link):
        """The cycle ``link`` next comes free; 0 for one nothing has crossed."""
        return self._free_cycle.get(link, 0)

    def claim(self, links, start_cycle, occupancy):
        """Push ``occupancy`` cycles of traffic along ``links`` from ``start_cycle``.

        Returns the cycles the packet spent waiting for a busy link — the delay
        to add to its arrival. The head flit walks the route in order, waiting
        at each link until it is free and then holding it for the whole packet:

        * the wait is **cumulative** along the route (a packet held up at the
          first link reaches the second one later), which is what makes a
          contended route cost the arrival, not just the link;
        * each link is claimed **once per packet**, so a multicast that the
          routers fan out must pass its whole (de-duplicated) tree here in one
          call rather than one call per modelled destination.

        Propagation between links is deliberately *not* added to the walk: the
        per-hop latency is already charged, once, by ``NocCostModel``. Adding it
        again here would double-count it, and it cannot change the steady-state
        answer — which is a throughput, not a phase.
        """
        if occupancy is None or not links:
            return 0
        waited = 0
        at = start_cycle
        with self._lock:
            for link in links:
                free = self._free_cycle.get(link, 0)
                if free > at:
                    waited += free - at
                    at = free
                    self.waits += 1
                self._free_cycle[link] = at + occupancy
            self.claims += len(links)
            self.cycles_waited += waited
        return waited


class NoCResponseError(RuntimeError):
    """A response arrived that no outstanding request accounts for.

    Raised rather than guessed at. Every response carries the issue sequence
    number of the request it answers (``NoCDataRequest.seq``), and the issuing
    NIU looks the state it saved up by that number — the L1 address a read's
    data belongs at, or the flags a write's ACK needs. If the number is not
    there, the alternative to failing is to hand back *some other* request's
    state, which silently writes the wrong data to the wrong address. A loud
    failure is the only honest option, and this is the same choice
    ``NoCAlignmentError`` makes for a violated congruence rule.
    """


def _payload_bytes(packet):
    """Bytes of data ``packet`` actually carries on the wire.

    Not ``data_length_bytes``, which is the *transaction* size and is set on
    both legs of a read: a read request is a header that names 8 KiB and
    carries none of it, and the 8 KiB travels on the response. What is on the
    wire is whatever ``data`` holds — ``None`` for a read request, a write ACK
    and an atomic request, whose operand rides in the header.
    """
    data = packet.data
    return 0 if data is None else len(data)


def _endpoint_noc_coord(endpoint):
    """An endpoint's coordinate in the coordinate space of its own NoC.

    A :class:`NUI` knows its own per-NoC coord. An :class:`AliasedEndpoint`
    carries the coord of the *cell the packet was addressed to*, which is the
    whole reason it exists. A :class:`NullEndpoint` has only the directory key
    it was looked up under, which is assumed to be in this NoC's space because
    it came straight out of the initiator's command registers.

    **That assumption is exactly true on NoC 0 and only partly true on NoC 1.**
    NoC 1's directory is keyed in two conventions at once (see
    ``TT_Device._register_tile_internals``): a tile's canonical SoC-physical
    coord, and the ``(GRID-1-x, GRID-1-y)`` mirror tt-metal's bank-to-noc table
    emits. Only the mirror is a NoC 1 coordinate, so a ``NullEndpoint`` built
    from a canonically keyed cell holds a coord in the *other* space and is
    costed a flight that is not the one it makes. That is a real defect; it is
    the two-convention keying itself, and it goes away when that keying does,
    not by patching the coordinate here. It is measured rather than left to
    this docstring — see ``tt_sim/network/noc_endpoint_consistency_test.py``,
    which pins how many entries are affected on each architecture.
    """
    coord = getattr(endpoint, "coord", None)
    return coord if coord is not None else (endpoint.x_coord, endpoint.y_coord)


class AliasedEndpoint:
    """A NIU addressed at a NoC cell that is not the one its NUI sits on.

    A DRAM channel is one controller behind *several* NoC interfaces, at
    genuinely different grid cells. Wormhole exposes each of its six channels
    at two worker-visible endpoints on opposite ends of a column —
    ``(0, 11)`` and ``(0, 1)`` are the same channel — and Blackhole's NoC 1
    view of a channel is a different subchannel from its NoC 0 one. tt-sim
    models the controller once, with one NUI per NoC standing at the *primary*
    endpoint, and registers the other cells as extra directory keys. Timing a
    packet from the NUI's own coord therefore charged it the flight to a
    different physical NIU: on Wormhole, wrong for **all 480** worker/endpoint
    pairs on each NoC, mean 34 cycles, worst 90.

    So a directory key that names a cell the NUI does not stand on gets one of
    these instead, holding that cell's coord *in this NoC's own space* (the
    grid mirror of the canonical coord on NoC 1, exactly as
    :attr:`NUI.x_coord` is). Everything else delegates: the endpoint is the NUI
    for every purpose except where it stands on the grid.

    The stamp in :meth:`transmit` is the other half. A response leaves from the
    interface the request arrived at, not from the controller's primary one, so
    the arrival cell rides on the request and :meth:`NUI.send_response` times
    the return leg from it. Without that the two legs would be measured between
    different pairs of points, which on a directional torus is not a small
    error in one direction — it is a round trip that does not close.
    """

    __slots__ = ("endpoint", "coord")

    def __init__(self, endpoint, coord):
        self.endpoint = endpoint
        self.coord = coord

    def transmit(self, request, delay=None):
        request.arrived_at = self.coord
        self.endpoint.transmit(request, delay)

    def __getattr__(self, name):
        return getattr(self.endpoint, name)

    def __repr__(self):
        return f"<AliasedEndpoint {self.coord} -> {self.endpoint!r}>"


def resolved_nui(endpoint):
    """The NUI behind ``endpoint`` — itself, unless it is an alias view.

    For callers that key off endpoint *identity* (the device's
    ``_tile_of_nui`` map, the NoC 1 shadow census) rather than off where the
    endpoint sits.
    """
    return endpoint.endpoint if isinstance(endpoint, AliasedEndpoint) else endpoint


class NullEndpoint:
    """Stand-in for an unregistered NoC destination.

    Real Wormhole NoC routers forward to whichever tile sits at the target
    coord. If a kernel addresses a coord whose tile we haven't modelled (e.g.
    an Ethernet core, or the hardcoded ``(1, 0)`` in tt-metal's
    ``hello_world_datatypes_kernel`` example), this endpoint completes the
    transaction the way the requester expects — reads come back zero-filled,
    marked writes get acknowledged, posted writes are silently dropped — so
    the simulated NoC doesn't deadlock the calling kernel.

    **Acknowledging is deliberate, including for a coord the caller got
    wrong.** This stands for *a tile tt-sim does not model*, not *a tile that
    is not there*: eth, pcie, arc, router-only and the DRAM channels outside
    the profile are all real NIUs on silicon, and they all ACK. Staying silent
    instead would make a multicast whose rectangle overruns the worker columns
    — which over-ACKs and hangs on hardware too — pass in the simulator with
    two destinations never written, trading a hang for a wrong answer. The
    diagnosability complaint is real and is answered where it belongs, in
    :meth:`NUI.report_multicast_gaps`, which names the offending cells.
    """

    def __init__(self, coord):
        self.coord = coord

    def transmit(self, request, delay=None):
        if request.action == NUI.NoCDataRequest.DataRequestAction.READ:
            self._respond(
                request,
                delay,
                NUI.NoCDataRequest(
                    None,
                    NUI.NoCDataRequest.DataRequestAction.RESPONSE_READ,
                    request.data_length_bytes,
                    self.coord,
                    request.request_id,
                    bytes(request.data_length_bytes),
                    seq=request.seq,
                ),
            )
        elif request.action == NUI.NoCDataRequest.DataRequestAction.WRITE:
            if request.noc_cmd_resp_marked:
                self._respond(
                    request,
                    delay,
                    NUI.NoCDataRequest(
                        None,
                        NUI.NoCDataRequest.DataRequestAction.ACK,
                        request.data_length_bytes,
                        self.coord,
                        request.request_id,
                        seq=request.seq,
                    ),
                )
        elif request.action == NUI.NoCDataRequest.DataRequestAction.ATOMIC:
            if request.noc_cmd_resp_marked:
                self._respond(
                    request,
                    delay,
                    NUI.NoCDataRequest(
                        None,
                        NUI.NoCDataRequest.DataRequestAction.RESPONSE_ATOMIC,
                        request.data_length_bytes,
                        self.coord,
                        request.request_id,
                        data=bytes(request.data_length_bytes),
                        seq=request.seq,
                    ),
                )

    def _respond(self, request, delay, response):
        # Responses go back to the endpoint that issued the request, never via
        # a coordinate lookup — see ``NUI.send_response``. The return flight is
        # timed by the requester (which owns the latency model and the grid
        # dims) from this endpoint's coord, so a null-routed transaction costs
        # the same as a real one at the same distance.
        source = request.reply_to
        back = source.flight_cycles_from(self.coord)
        if back is not None:
            # This endpoint has no clock, so it cannot hold the request for its
            # outbound flight the way a real NIU does. Both legs are charged to
            # the response instead: the same total time, without a queue. For
            # the same reason it has no injection port to occupy, so only the
            # response's own tail is charged, never a queueing delay.
            back += (delay or 0) + source.tail_cycles_for(response)
        source.transmit(response, back)


# Maximum NoC packet payload per Wormhole's
# ``tt_metal/hw/inc/wormhole/noc/noc_parameters.h``
# (``NOC_MAX_BURST_WORDS * NOC_WORD_BYTES = 256 * 32``). Reads or writes
# larger than this need the NIU to split the request into multiple flits;
# each lands on its own slice of L1 and produces its own response.
# tt-metal's ``ncrisc_noc_fast_read_any_len`` / ``_fast_write_any_len``
# chunk on the kernel side, so in practice the NIU only sees ``<=`` this
# size from real kernels — but modelling the split at the NIU level
# preserves the semantics for any caller that does write a single larger
# request directly to the NIU registers.
NOC_MAX_BURST_SIZE = 8192


def _split_burst(
    total_size: int, max_burst: int = NOC_MAX_BURST_SIZE
) -> list[tuple[int, int]]:
    """Return ``[(offset, size), ...]`` chunks covering ``total_size`` bytes,
    each at most ``max_burst`` bytes (defaulting to :data:`NOC_MAX_BURST_SIZE`,
    the Wormhole limit; Blackhole's is larger)."""
    if total_size <= max_burst:
        return [(0, total_size)]
    chunks = []
    offset = 0
    while offset < total_size:
        chunk = min(max_burst, total_size - offset)
        chunks.append((offset, chunk))
        offset += chunk
    return chunks


class NUI(MemMapable, Clockable):
    class NoCDataRequest:
        class DataRequestAction(IntEnum):
            READ = 0
            WRITE = 1
            RESPONSE_READ = 2
            ACK = 3
            ATOMIC = 4
            RESPONSE_ATOMIC = 5

        def __init__(
            self,
            tgt_address,
            action,
            data_length_bytes,
            source_coord,
            request_id,
            data=None,
            noc_cmd_resp_marked=True,
            at_data=0,
            reply_to=None,
            seq=None,
        ):
            """A packet in flight on one NoC.

            ``source_coord`` is the *coordinate* of the sender, carried for
            diagnostics and trace only. It is deliberately not a routing key:
            it is the sender's canonical (SoC-physical NoC 0) coord on both
            NoCs, whereas NoC 1's directory is keyed in NoC 1's mirrored
            space, so resolving it against ``noc_directory`` on NoC 1 can hand
            back a completely different tile (see ``NUI.send_response``).

            ``reply_to`` is the *endpoint object* that issued the request —
            the only thing a response is ever routed by.

            ``seq`` is the issuing NIU's own monotonic request number, echoed
            unchanged by whatever answers the request. ``request_id`` (the
            transaction ID) cannot do this job: kernels reuse one trid for many
            in-flight requests on purpose, which is why the outstanding-request
            store is per-trid in the first place. ``seq`` is what makes a
            response identify *which* of them it answers, so the store does not
            have to assume responses come back in issue order — an assumption
            per-destination flight times can break. See
            :meth:`NUI.take_outstanding_noc_request`.
            """
            self.tgt_address = tgt_address
            self.action = action
            self.request_id = request_id
            self.data_length_bytes = data_length_bytes
            self.source_coord = source_coord
            self.reply_to = reply_to
            self.seq = seq
            self.data = data
            self.noc_cmd_resp_marked = noc_cmd_resp_marked
            # Immediate operand for ATOMIC requests (increment value for
            # atomic add). Unused for non-atomic ops.
            self.at_data = at_data
            # Cycle the sending NIU handed this packet to the wire, stamped by
            # ``NUI.transmit``. Trace-only (``NoCEvent.issue_cycle``); nothing
            # routes or schedules on it. -1 until transmitted, and for a NIU
            # with no owning tile clock, which cannot know the absolute cycle.
            self.issue_cycle = -1
            # The NoC cell this request entered its destination tile at, in
            # that NoC's own space — set only when the tile answers to more
            # than one cell (see ``AliasedEndpoint``), so that its response
            # leaves from the same interface. ``None`` means "the destination
            # NIU's own coord", which is every ordinary tile.
            self.arrived_at = None

    class RequestInitiator:
        def __init__(self, nui):
            self.target_addr_low = 0
            self.target_addr_mid = 0
            # Dedicated coordinate register (Blackhole NOC_TARG/RET_ADDR_HI);
            # unused by Wormhole, whose coord lives in the MID register.
            self.target_addr_hi = 0
            self.ret_addr_low = 0
            self.ret_addr_mid = 0
            self.ret_addr_hi = 0
            self.packet_tag = 0
            self.ctrl = 0
            self.at_len_be = 0
            self.at_data = 0
            self.cmd_ctrl = 0
            self.nui = nui

        @staticmethod
        def _kind_name(endpoint):
            """Human-readable name of a NoC endpoint's memory kind, for messages."""
            return {"D": "DRAM", "T": "L1", "E": "L1"}.get(
                getattr(endpoint, "tile_kind", None), "unmodelled tile"
            )

        def _check_alignment(self, modulus, src_addr, dst_addr, *, path):
            check_congruence(
                modulus,
                src_addr,
                dst_addr,
                path=path,
                noc_number=self.nui.noc_number,
                initiator=self.nui.id_pair,
            )

        def handle_read_transfer(self):
            noc_packet_transaction_id = extract_bits(self.packet_tag, 4, 10)
            total_size = self.at_len_be
            # Split into NOC_MAX_BURST_SIZE chunks; each becomes its own
            # read request + response, sharing the trid. The per-trid
            # FIFO carries the L1 destination offset per chunk so each
            # response writes into the right slice. OUTSTANDING bumps by
            # the chunk count so ``noc_async_read_barrier`` waits for all
            # responses.
            chunks = _split_burst(total_size, self.nui.noc_max_burst_size)
            num_chunks = len(chunks)

            self.nui.nui_counters.increment(
                [
                    NUI.NUICounters.CounterNames.NIU_MST_CMD_ACCEPTED,
                    NUI.NUICounters.CounterNames.NIU_MST_RD_REQ_STARTED,
                ]
            )
            self.nui.nui_counters.increment(
                NUI.NUICounters.CounterNames.NIU_MST_REQS_OUTSTANDING_ID_0
                + noc_packet_transaction_id,
                num_chunks,
            )
            self.cmd_ctrl = 0

            target_tile_x, target_tile_y = self.nui.noc_coord_strategy.target_coord(
                self
            )
            destination = self.nui.resolve_destination((target_tile_x, target_tile_y))

            # Source is the remote tile, destination is this tile's L1. Which
            # congruence applies depends on what we are reading *from*.
            self._check_alignment(
                congruence_for_read(
                    self.target_addr_low,
                    getattr(destination, "tile_kind", None) == "D",
                    self.nui.noc_dram_read_congruence,
                ),
                self.target_addr_low,
                self.ret_addr_low,
                path=f"{self._kind_name(destination)} -> L1 read",
            )

            for chunk_offset, chunk_size in chunks:
                seq = self.nui.next_request_seq()
                read_req = NUI.NoCDataRequest(
                    self.target_addr_low + chunk_offset,
                    NUI.NoCDataRequest.DataRequestAction.READ,
                    chunk_size,
                    self.nui.id_pair,
                    noc_packet_transaction_id,
                    reply_to=self.nui,
                    seq=seq,
                )
                self.nui.add_outstanding_noc_request(
                    noc_packet_transaction_id, self.ret_addr_low + chunk_offset, seq
                )
                self.nui.send_to(destination, read_req)

            self.nui.nui_counters.increment(
                NUI.NUICounters.CounterNames.NIU_MST_RD_REQ_SENT, num_chunks
            )

            if self.nui.snoop:
                print(
                    f"[NoC{self.nui.noc_number} {self.nui.id_pair}]: Issue read request id "
                    f"{noc_packet_transaction_id} to NUI "
                    f"{(target_tile_x, target_tile_y)}, reading at "
                    f"{hex(self.target_addr_low)} of total size "
                    f"{hex(total_size)} ({num_chunks} chunk(s)) and store in "
                    f"{hex(self.ret_addr_low)}"
                )

        def handle_inline_write_transfer(
            self, noc_cmd_wr_be, noc_cmd_wr_inline, noc_cmd_resp_marked
        ):
            noc_packet_transaction_id = extract_bits(self.packet_tag, 4, 10)

            if noc_cmd_resp_marked:
                self.nui.nui_counters.increment(
                    NUI.NUICounters.CounterNames.NIU_MST_REQS_OUTSTANDING_ID_0
                    + noc_packet_transaction_id
                )

            self.nui.nui_counters.increment(
                NUI.NUICounters.CounterNames.NIU_MST_CMD_ACCEPTED
            )

            if noc_cmd_resp_marked:
                self.nui.nui_counters.increment(
                    [
                        NUI.NUICounters.CounterNames.NIU_MST_NONPOSTED_WR_REQ_STARTED,
                        NUI.NUICounters.CounterNames.NIU_MST_NONPOSTED_WR_REQ_SENT,
                    ]
                )
            else:
                self.nui.nui_counters.increment(
                    [
                        NUI.NUICounters.CounterNames.NIU_MST_POSTED_WR_REQ_STARTED,
                        NUI.NUICounters.CounterNames.NIU_MST_POSTED_WR_REQ_SENT,
                    ]
                )
            self.cmd_ctrl = 0

            # Send write request
            ret_tile_x, ret_tile_y = self.nui.noc_coord_strategy.ret_coord(self)
            destination = self.nui.resolve_destination((ret_tile_x, ret_tile_y))

            data = self.nui.attached_memory.read(self.target_addr_low, self.at_len_be)

            seq = self.nui.next_request_seq()
            write_req = NUI.NoCDataRequest(
                self.ret_addr_low,
                NUI.NoCDataRequest.DataRequestAction.WRITE,
                self.at_len_be,
                self.nui.id_pair,
                noc_packet_transaction_id,
                data,
                noc_cmd_resp_marked,
                reply_to=self.nui,
                seq=seq,
            )
            self.nui.add_outstanding_noc_request(
                noc_packet_transaction_id,
                (noc_cmd_wr_inline, noc_cmd_resp_marked),
                seq,
            )
            self.nui.send_to(destination, write_req)

            if self.nui.snoop:
                print(
                    f"[NoC{self.nui.noc_number} {self.nui.id_pair}]: Issue write request id {write_req.request_id} to NUI "
                    f"{(ret_tile_x, ret_tile_y)}, writing from {hex(self.target_addr_low)} "
                    f" to {hex(write_req.tgt_address)} of size {hex(write_req.data_length_bytes)}"
                )

        def handle_none_inline_write(
            self, noc_cmd_wr_be, noc_cmd_wr_inline, noc_cmd_resp_marked
        ):
            noc_packet_transaction_id = extract_bits(self.packet_tag, 4, 10)
            total_size = self.at_len_be
            chunks = _split_burst(total_size, self.nui.noc_max_burst_size)
            num_chunks = len(chunks)

            if noc_cmd_resp_marked:
                # One outstanding entry per chunk — ``noc_async_write_barrier``
                # polls OUTSTANDING and waits for 0, so it must match the
                # ACK count exactly. (The pre-split code multiplied by
                # noc_cmd_wr_be here, but that's the byte-enable bit and
                # is 0 in tt-metal's noc_async_write — a latent bug that
                # was masked because tt-sim resolves responses within the
                # same cycle pump as the request.)
                self.nui.nui_counters.increment(
                    NUI.NUICounters.CounterNames.NIU_MST_REQS_OUTSTANDING_ID_0
                    + noc_packet_transaction_id,
                    num_chunks,
                )

            self.nui.nui_counters.increment(
                NUI.NUICounters.CounterNames.NIU_MST_WRITE_REQS_OUTGOING_ID_0
            )

            self.nui.nui_counters.increment(
                NUI.NUICounters.CounterNames.NIU_MST_CMD_ACCEPTED
            )

            if noc_cmd_resp_marked:
                self.nui.nui_counters.increment(
                    NUI.NUICounters.CounterNames.NIU_MST_NONPOSTED_WR_REQ_STARTED
                )
            else:
                self.nui.nui_counters.increment(
                    NUI.NUICounters.CounterNames.NIU_MST_POSTED_WR_REQ_STARTED
                )
            self.cmd_ctrl = 0

            # Send N write requests, one per chunk. Each carries its slice
            # of the source data + writes into a contiguous slice of the
            # destination. The per-trid FIFO gains N entries so each ACK
            # decrements OUTSTANDING by one.
            ret_tile_x, ret_tile_y = self.nui.noc_coord_strategy.ret_coord(self)
            destination = self.nui.resolve_destination((ret_tile_x, ret_tile_y))

            # Source is this tile's L1, destination is the remote tile. The
            # length-mode table gives C16 for L1 -> L1 and L1 -> Other alike, so
            # the destination kind does not change the rule here.
            self._check_alignment(
                L1_CONGRUENCE,
                self.target_addr_low,
                self.ret_addr_low,
                path=f"L1 -> {self._kind_name(destination)} write",
            )

            for chunk_offset, chunk_size in chunks:
                data = self.nui.attached_memory.read(
                    self.target_addr_low + chunk_offset, chunk_size
                )
                seq = self.nui.next_request_seq()
                write_req = NUI.NoCDataRequest(
                    self.ret_addr_low + chunk_offset,
                    NUI.NoCDataRequest.DataRequestAction.WRITE,
                    chunk_size,
                    self.nui.id_pair,
                    noc_packet_transaction_id,
                    data,
                    noc_cmd_resp_marked,
                    reply_to=self.nui,
                    seq=seq,
                )
                self.nui.add_outstanding_noc_request(
                    noc_packet_transaction_id,
                    (noc_cmd_wr_inline, noc_cmd_resp_marked),
                    seq,
                )
                self.nui.send_to(destination, write_req)

            if noc_cmd_resp_marked:
                self.nui.nui_counters.increment(
                    NUI.NUICounters.CounterNames.NIU_MST_NONPOSTED_WR_REQ_SENT,
                    num_chunks,
                )
                self.nui.nui_counters.increment(
                    NUI.NUICounters.CounterNames.NIU_MST_NONPOSTED_WR_DATA_WORD_SENT,
                    total_size / 4,
                )
            else:
                self.nui.nui_counters.increment(
                    NUI.NUICounters.CounterNames.NIU_MST_POSTED_WR_REQ_SENT,
                    num_chunks,
                )
                self.nui.nui_counters.increment(
                    NUI.NUICounters.CounterNames.NIU_MST_POSTED_WR_DATA_WORD_SENT,
                    total_size / 4,
                )
            self.nui.nui_counters.decrement(
                NUI.NUICounters.CounterNames.NIU_MST_WRITE_REQS_OUTGOING_ID_0
            )

            if self.nui.snoop:
                print(
                    f"[NoC{self.nui.noc_number} {self.nui.id_pair}]: Issue write request id "
                    f"{noc_packet_transaction_id} to NUI "
                    f"{(ret_tile_x, ret_tile_y)}, writing from "
                    f"{hex(self.target_addr_low)} to "
                    f"{hex(self.ret_addr_low)} of total size "
                    f"{hex(total_size)} ({num_chunks} chunk(s))"
                )

        def handle_write_transfer(self):
            noc_cmd_wr_be = extract_bits(self.ctrl, 1, 2)
            noc_cmd_wr_inline = extract_bits(self.ctrl, 1, 3)
            noc_cmd_resp_marked = extract_bits(self.ctrl, 1, 4)
            noc_cmd_brcst_packet = extract_bits(self.ctrl, 1, 5)

            if noc_cmd_brcst_packet:
                # noc_async_write_multicast / noc_semaphore_set_multicast.
                # Real hardware fans the packet out internally; we model it
                # as N back-to-back unicast writes, one per destination in
                # the (x_start..x_end, y_start..y_end) rectangle.
                self.handle_multicast_write_transfer(
                    noc_cmd_wr_be, noc_cmd_wr_inline, noc_cmd_resp_marked
                )
            elif noc_cmd_wr_inline:
                self.handle_inline_write_transfer(
                    noc_cmd_wr_be, noc_cmd_wr_inline, noc_cmd_resp_marked
                )
            else:
                self.handle_none_inline_write(
                    noc_cmd_wr_be, noc_cmd_wr_inline, noc_cmd_resp_marked
                )

        def handle_multicast_write_transfer(
            self, noc_cmd_wr_be, noc_cmd_wr_inline, noc_cmd_resp_marked
        ):
            """Multicast write — fan one packet out to every tile in the
            destination rectangle.

            For multicast packets, ``ret_addr_mid`` encodes a rectangle
            instead of a single coord:

            * bits [4:10]   = x_end
            * bits [10:16]  = y_end
            * bits [16:22]  = x_start
            * bits [22:28]  = y_start

            (This is the bit packing of ``NOC_MULTICAST_ADDR`` in
            ``tt_metal/hw/inc/wormhole/noc/noc_parameters.h``.) On real
            silicon the NoC routes a single packet that the routers split
            along the rectangle; tt-sim transmits one ``WRITE`` request
            per destination. The master's ``REQS_OUTSTANDING`` counter
            (and the per-trid FIFO) is bumped by ``num_dests`` so the
            kernel's ``noc_async_write_barrier`` waits for all N ACKs.
            """
            noc_packet_transaction_id = extract_bits(self.packet_tag, 4, 10)

            (
                x_start,
                y_start,
                x_end,
                y_end,
            ) = self.nui.noc_coord_strategy.broadcast_coords(self)

            destinations = [
                (x, y)
                for x in range(x_start, x_end + 1)
                for y in range(y_start, y_end + 1)
            ]
            num_dests = len(destinations)

            if noc_cmd_resp_marked:
                self.nui.nui_counters.increment(
                    NUI.NUICounters.CounterNames.NIU_MST_REQS_OUTSTANDING_ID_0
                    + noc_packet_transaction_id,
                    num_dests,
                )
                self.nui.nui_counters.increment(
                    [
                        NUI.NUICounters.CounterNames.NIU_MST_NONPOSTED_WR_REQ_STARTED,
                        NUI.NUICounters.CounterNames.NIU_MST_NONPOSTED_WR_REQ_SENT,
                    ]
                )
                self.nui.nui_counters.increment(
                    NUI.NUICounters.CounterNames.NIU_MST_NONPOSTED_WR_DATA_WORD_SENT,
                    self.at_len_be / 4,
                )
            else:
                self.nui.nui_counters.increment(
                    [
                        NUI.NUICounters.CounterNames.NIU_MST_POSTED_WR_REQ_STARTED,
                        NUI.NUICounters.CounterNames.NIU_MST_POSTED_WR_REQ_SENT,
                    ]
                )
                self.nui.nui_counters.increment(
                    NUI.NUICounters.CounterNames.NIU_MST_POSTED_WR_DATA_WORD_SENT,
                    self.at_len_be / 4,
                )

            self.nui.nui_counters.increment(
                NUI.NUICounters.CounterNames.NIU_MST_CMD_ACCEPTED
            )
            self.cmd_ctrl = 0

            # A multicast write is still an L1 -> L1 write per destination, so
            # the same C16 rule applies (and the single source/destination
            # address pair is shared by every destination in the rectangle).
            self._check_alignment(
                L1_CONGRUENCE,
                self.target_addr_low,
                self.ret_addr_low,
                path="L1 -> L1 multicast write",
            )

            data = self.nui.attached_memory.read(self.target_addr_low, self.at_len_be)

            # One packet leaves this NIU however wide the rectangle is -- the
            # routers do the splitting -- so the injection port is claimed once
            # for the whole fan-out and every copy shares the wait. Charging
            # each modelled unicast its own injection time would invent
            # serialisation the hardware does not have.
            payload_bytes = len(data) if data else 0
            queued = self.nui.claim_injection_port(payload_bytes)

            # ...and the same argument, one level out, for every link on the
            # way. A multicast crosses a *tree*: the union of the
            # dimension-ordered routes to each destination, which is exactly
            # what those routes' links de-duplicate to. Claiming a link once
            # per destination instead would serialise the fan-out against
            # itself on the launch-message path every tt-metal program uses --
            # the same over-charge ``claim_injection_port`` exists to avoid,
            # one resource further along. First-appearance order is kept so the
            # tree is walked outwards from this NIU, as the packet does.
            endpoints = [self.nui.resolve_destination(c) for c in destinations]
            self.nui.report_multicast_gaps(
                (x_start, y_start, x_end, y_end), destinations, endpoints
            )
            tree = dict.fromkeys(
                link
                for destination in endpoints
                for link in self.nui.route_links_to(destination)
            )
            link_wait = self.nui.claim_route_links(tuple(tree), payload_bytes, queued)

            for destination in endpoints:
                seq = self.nui.next_request_seq()
                write_req = NUI.NoCDataRequest(
                    self.ret_addr_low,
                    NUI.NoCDataRequest.DataRequestAction.WRITE,
                    self.at_len_be,
                    self.nui.id_pair,
                    noc_packet_transaction_id,
                    data,
                    bool(noc_cmd_resp_marked),
                    reply_to=self.nui,
                    seq=seq,
                )
                # Each destination's ACK clears its own entry. The rectangle is
                # the one place in the tree where one trid's requests genuinely
                # go to many tiles at once, so it is also where their ACKs are
                # most obviously free to come back in any order.
                self.nui.add_outstanding_noc_request(
                    noc_packet_transaction_id,
                    (noc_cmd_wr_inline, noc_cmd_resp_marked),
                    seq,
                )
                self.nui.send_to(
                    destination, write_req, queued=queued, link_wait=link_wait
                )

            if self.nui.snoop:
                print(
                    f"[NoC{self.nui.noc_number} {self.nui.id_pair}]: Issue "
                    f"multicast write id {noc_packet_transaction_id} to "
                    f"rectangle ({x_start}, {y_start})..({x_end}, {y_end}) "
                    f"= {num_dests} dests, writing from "
                    f"{hex(self.target_addr_low)} to {hex(self.ret_addr_low)} "
                    f"of size {hex(self.at_len_be)}"
                )

        def handle_atomic_transfer(self):
            """Atomic add to a remote 32-bit word — what ``noc_semaphore_inc``
            issues. Per the ISA docs, the operation kind is encoded in
            ``at_len_be`` (ATINC / ATCAS / ATSWAP / ATINCGET); we model
            atomic-add only here because that's what semaphore signalling
            needs. Bigger ops are §D ThCon scope.
            """
            noc_packet_transaction_id = extract_bits(self.packet_tag, 4, 10)
            noc_cmd_resp_marked = extract_bits(self.ctrl, 1, 4)

            if noc_cmd_resp_marked:
                self.nui.nui_counters.increment(
                    [
                        NUI.NUICounters.CounterNames.NIU_MST_REQS_OUTSTANDING_ID_0
                        + noc_packet_transaction_id,
                        NUI.NUICounters.CounterNames.NIU_MST_NONPOSTED_ATOMIC_STARTED,
                        NUI.NUICounters.CounterNames.NIU_MST_NONPOSTED_ATOMIC_SENT,
                        NUI.NUICounters.CounterNames.NIU_MST_CMD_ACCEPTED,
                    ]
                )
            else:
                self.nui.nui_counters.increment(
                    [
                        NUI.NUICounters.CounterNames.NIU_MST_POSTED_ATOMIC_SENT,
                        NUI.NUICounters.CounterNames.NIU_MST_CMD_ACCEPTED,
                    ]
                )

            self.cmd_ctrl = 0

            target_tile_x, target_tile_y = self.nui.noc_coord_strategy.target_coord(
                self
            )
            destination = self.nui.resolve_destination((target_tile_x, target_tile_y))

            seq = self.nui.next_request_seq()
            atomic_req = NUI.NoCDataRequest(
                self.target_addr_low,
                NUI.NoCDataRequest.DataRequestAction.ATOMIC,
                4,
                self.nui.id_pair,
                noc_packet_transaction_id,
                noc_cmd_resp_marked=bool(noc_cmd_resp_marked),
                at_data=self.at_data,
                reply_to=self.nui,
                seq=seq,
            )
            if noc_cmd_resp_marked:
                self.nui.add_outstanding_noc_request(
                    noc_packet_transaction_id, None, seq
                )
            self.nui.send_to(destination, atomic_req)

            if self.nui.snoop:
                print(
                    f"[NoC{self.nui.noc_number} {self.nui.id_pair}]: Issue atomic-add id {atomic_req.request_id} "
                    f"to NUI {(target_tile_x, target_tile_y)} at "
                    f"{hex(atomic_req.tgt_address)} += {self.at_data}"
                )

        def initiate(self):
            if self.cmd_ctrl == 1:
                # Following the protocol at
                # https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/NoC/Counters.md
                # however this is different as completing immediately.
                # NOC_PACKET_TRANSACTION_ID is extracted from packet_tag in
                # handle_read_transfer / handle_*_write below; per-trid
                # outstanding requests are tracked as FIFO queues in
                # ``NUI.outstanding_noc_requests``.

                """
                If reading, then target has remote memory and ret has the local memory. If writing
                then this is the other way around (target has local, ret has remote). Therefore
                ret is always the thing being written into. Currently we assume one end is local
                here. But this can be improved by making the target an NUI too
                """

                mode = extract_bits(self.ctrl, 2, 0)
                if mode == 0:
                    self.handle_read_transfer()
                elif mode == 1:
                    self.handle_atomic_transfer()
                elif mode == 2:
                    self.handle_write_transfer()

    class NUICounters:
        class CounterNames(IntEnum):
            NIU_MST_ATOMIC_RESP_RECEIVED = 0
            NIU_MST_WR_ACK_RECEIVED = 1
            NIU_MST_RD_RESP_RECEIVED = 2
            NIU_MST_RD_DATA_WORD_RECEIVED = 3
            NIU_MST_CMD_ACCEPTED = 4
            NIU_MST_RD_REQ_SENT = 5
            NIU_MST_NONPOSTED_ATOMIC_SENT = 6
            NIU_MST_POSTED_ATOMIC_SENT = 7
            NIU_MST_NONPOSTED_WR_DATA_WORD_SENT = 8
            NIU_MST_POSTED_WR_DATA_WORD_SENT = 9
            NIU_MST_NONPOSTED_WR_REQ_SENT = 10
            NIU_MST_POSTED_WR_REQ_SENT = 11
            NIU_MST_NONPOSTED_WR_REQ_STARTED = 12
            NIU_MST_POSTED_WR_REQ_STARTED = 13
            NIU_MST_RD_REQ_STARTED = 14
            NIU_MST_NONPOSTED_ATOMIC_STARTED = 15
            NIU_MST_REQS_OUTSTANDING_ID_0 = 16
            NIU_MST_REQS_OUTSTANDING_ID_1 = 17
            NIU_MST_REQS_OUTSTANDING_ID_2 = 18
            NIU_MST_REQS_OUTSTANDING_ID_3 = 19
            NIU_MST_REQS_OUTSTANDING_ID_4 = 20
            NIU_MST_REQS_OUTSTANDING_ID_5 = 21
            NIU_MST_REQS_OUTSTANDING_ID_6 = 22
            NIU_MST_REQS_OUTSTANDING_ID_7 = 23
            NIU_MST_REQS_OUTSTANDING_ID_8 = 24
            NIU_MST_REQS_OUTSTANDING_ID_9 = 25
            NIU_MST_REQS_OUTSTANDING_ID_10 = 26
            NIU_MST_REQS_OUTSTANDING_ID_11 = 27
            NIU_MST_REQS_OUTSTANDING_ID_12 = 28
            NIU_MST_REQS_OUTSTANDING_ID_13 = 29
            NIU_MST_REQS_OUTSTANDING_ID_14 = 30
            NIU_MST_REQS_OUTSTANDING_ID_15 = 31
            NIU_MST_WRITE_REQS_OUTGOING_ID_0 = 32
            NIU_MST_WRITE_REQS_OUTGOING_ID_1 = 33
            NIU_MST_WRITE_REQS_OUTGOING_ID_2 = 34
            NIU_MST_WRITE_REQS_OUTGOING_ID_3 = 35
            NIU_MST_WRITE_REQS_OUTGOING_ID_4 = 36
            NIU_MST_WRITE_REQS_OUTGOING_ID_5 = 37
            NIU_MST_WRITE_REQS_OUTGOING_ID_6 = 38
            NIU_MST_WRITE_REQS_OUTGOING_ID_7 = 39
            NIU_MST_WRITE_REQS_OUTGOING_ID_8 = 40
            NIU_MST_WRITE_REQS_OUTGOING_ID_9 = 41
            NIU_MST_WRITE_REQS_OUTGOING_ID_10 = 42
            NIU_MST_WRITE_REQS_OUTGOING_ID_11 = 43
            NIU_MST_WRITE_REQS_OUTGOING_ID_12 = 44
            NIU_MST_WRITE_REQS_OUTGOING_ID_13 = 45
            NIU_MST_WRITE_REQS_OUTGOING_ID_14 = 46
            NIU_MST_WRITE_REQS_OUTGOING_ID_15 = 47
            NIU_SLV_ATOMIC_RESP_SENT = 48
            NIU_SLV_WR_ACK_SENT = 49
            NIU_SLV_RD_RESP_SENT = 50
            NIU_SLV_RD_DATA_WORD_SENT = 51
            NIU_SLV_REQ_ACCEPTED = 52
            NIU_SLV_RD_REQ_RECEIVED = 53
            NIU_SLV_NONPOSTED_ATOMIC_RECEIVED = 54
            NIU_SLV_POSTED_ATOMIC_RECEIVED = 55
            NIU_SLV_NONPOSTED_WR_DATA_WORD_RECEIVED = 56
            NIU_SLV_POSTED_WR_DATA_WORD_RECEIVED = 57
            NIU_SLV_NONPOSTED_WR_REQ_RECEIVED = 58
            NIU_SLV_POSTED_WR_REQ_RECEIVED = 59
            NIU_SLV_NONPOSTED_WR_REQ_STARTED = 60
            NIU_SLV_POSTED_WR_REQ_STARTED = 61

        def __init__(self):
            # One slot per CounterNames member: the enum runs 0..61, so 62 slots.
            # A posted (non-response-marked) NoC write increments index 61, which
            # a 61-long list cannot hold -- that path only fires once the device
            # profiler's firmware is running, which is why it stayed latent.
            self.counters = [0] * len(NUI.NUICounters.CounterNames)

        def __getitem__(self, idx):
            return self.counters[idx]

        def __setitem__(self, idx, value):
            self.counters[idx] = value

        def increment(self, idx_to_increment, val=1):
            if isinstance(idx_to_increment, list):
                for idx in idx_to_increment:
                    self.counters[idx] += val
            else:
                self.counters[idx_to_increment] += val

        def decrement(self, idx_to_decrement, val=1):
            if isinstance(idx_to_decrement, list):
                for idx in idx_to_decrement:
                    self.counters[idx] -= val
            else:
                self.counters[idx_to_decrement] -= val

        def __delitem__(self, idx):
            del self.counters[idx]

    # Default (Wormhole) NoC grid dimensions — used to mirror the canonical
    # NoC 0 physical coord onto NoC 1's coord space. NoC 1's origin is the
    # bottom-right tile, so its coords are (NOC_GRID_X-1 - x, NOC_GRID_Y-1 - y).
    # A device passes its architecture's dims (e.g. 17 x 12 for Blackhole) via
    # the constructor; these class constants are the fallback for direct
    # construction that does not supply them.
    NOC_GRID_X = 10
    NOC_GRID_Y = 12

    def __init__(
        self,
        noc_number,
        x_coord,
        y_coord,
        attached_memory,
        snoop=False,
        noc_grid_x=None,
        noc_grid_y=None,
        noc_max_burst_size=None,
        noc_coord_strategy=None,
        noc_blackhole_cmd_buf_layout=False,
        noc_dram_read_congruence=32,
        noc_id_logical_cfg_index=0xE,
        noc_id_logical_mirrored_on_noc1=True,
        tile_kind="T",
        arch=None,
    ):
        """``x_coord`` / ``y_coord`` are the tile's canonical SoC-physical
        NoC 0 coord. The NUI's ``id_pair`` is that canonical coord on BOTH
        NoCs — that's what tt-metal kernels supply when
        ``translation_id_enabled`` is set in the SoC descriptor. Per-NoC
        physical coords for the kernel-visible ``NOC_NODE_ID`` register reads
        are derived by mirroring through the grid dimensions for NoC 1.

        ``id_pair`` is therefore a NoC 0 coord even on NoC 1, and must never
        be used to look a tile up in NoC 1's directory (which is also keyed by
        mirrored coords, in which the same tuple names a different tile). It
        is a directory key on NoC 0 and an identity/diagnostic label
        everywhere else; responses route by endpoint object instead — see
        :meth:`send_response`.

        ``noc_grid_x`` / ``noc_grid_y`` / ``noc_max_burst_size`` come from the
        architecture profile; each falls back to the Wormhole default when not
        supplied. See ``docs/plans/blackhole-support.md``.

        ``arch`` is the profile's name, and its only use is to look up the
        per-hop latency table. Left ``None`` by a directly-constructed NUI
        (unit tests, ``driver/simple``), which therefore never opts into the
        cost model however the environment is set.
        """
        assert noc_number == 0 or noc_number == 1
        self.noc_number = noc_number
        self.noc_grid_x = NUI.NOC_GRID_X if noc_grid_x is None else noc_grid_x
        self.noc_grid_y = NUI.NOC_GRID_Y if noc_grid_y is None else noc_grid_y
        self.noc_max_burst_size = (
            NOC_MAX_BURST_SIZE if noc_max_burst_size is None else noc_max_burst_size
        )
        # Strategy for reading a request's destination coord out of the NIU
        # command registers; defaults to Wormhole's coord-in-MID layout.
        self.noc_coord_strategy = (
            WormholeNocCoords() if noc_coord_strategy is None else noc_coord_strategy
        )
        self.blackhole_cmd_buf_layout = noc_blackhole_cmd_buf_layout
        # Congruence modulus for DRAM-sourced reads (32 Wormhole / 64 Blackhole)
        # and this endpoint's memory kind ('D' DRAM, 'T' Tensix L1, 'E' Eth L1),
        # both used by the alignment checks in ``RequestInitiator``.
        self.noc_dram_read_congruence = noc_dram_read_congruence
        # NIU offset of NOC_ID_LOGICAL within the NOC_CFG(cnt) block, and
        # whether that register mirrors on NoC 1. Both are per-architecture;
        # see ``ArchProfile.noc_id_logical_cfg_index`` /
        # ``ArchProfile.noc_id_logical_mirrored_on_noc1``.
        self.noc_id_logical_offset = 0x100 + 4 * noc_id_logical_cfg_index
        self.noc_id_logical_mirrored_on_noc1 = noc_id_logical_mirrored_on_noc1
        self.tile_kind = tile_kind
        if noc_number == 0:
            self.x_coord = x_coord
            self.y_coord = y_coord
        else:
            self.x_coord = self.noc_grid_x - 1 - x_coord
            self.y_coord = self.noc_grid_y - 1 - y_coord
        self.id_pair = (x_coord, y_coord)
        self.generate_NIU_and_NoC_config()
        self.generate_NoC_node_id()
        self.request_initiators = [
            NUI.RequestInitiator(self),
            NUI.RequestInitiator(self),
            NUI.RequestInitiator(self),
            NUI.RequestInitiator(self),
        ]
        self.nui_counters = NUI.NUICounters()
        self.noc_directory = None
        #: ``(noc_number, coord) -> None`` hook consulted **only** when a
        #: request's destination coord is absent from :attr:`noc_directory`.
        #: The wire bridge installs one so a worker a peer addresses before the
        #: host has launched on it is materialised there and then, instead of
        #: the packet being null-routed into oblivion. ``None`` (the default,
        #: and every non-bridge caller) keeps the historical behaviour and
        #: costs the resolve path nothing: it is reached only on a miss.
        self.directory_miss_hook = None
        #: Multicast rectangles already reported as spanning cells with no
        #: modelled tile — one line per distinct rectangle, not per packet.
        self._reported_multicast_gaps = set()
        self.attached_memory = attached_memory
        #: ``{trid: {seq: state}}`` — what this NIU saved when it issued each
        #: request that is still awaiting a response. See
        #: :meth:`take_outstanding_noc_request` for why it is keyed by ``seq``
        #: rather than being a FIFO.
        self.outstanding_noc_requests = {}
        self._request_seq = 0
        #: Responses that came back before an older one under the same trid.
        #: Zero on every in-tree workload so far; counted rather than assumed,
        #: because the whole point of keying by ``seq`` is that the number is
        #: allowed to be non-zero.
        self.out_of_order_responses = 0
        # Separate these out to ensure we have atleast one clock cycle
        # between a request and it being handled (can increase)
        self.noc_requests_to_handle = []
        self.noc_new_requests_to_handle = []
        # Per-hop flight time, or None (the default, and the only state with
        # TT_SIM_COST_MODEL unset) meaning "deliver on the next cycle" — the
        # two-list swap below. See ``send_to`` and ``docs/plans/cost-model.md``.
        self.noc_latency = noc_cost_model(arch)
        #: ``{arrival_cycle: [packet, ...]}`` for packets still in flight.
        #: Always empty without a latency model.
        self.delayed_arrivals = {}
        #: Earliest key of :attr:`delayed_arrivals`, cached so the pump's
        #: per-tile wake probe is one attribute read rather than a ``min``.
        self.next_arrival = None
        #: Cycle this NIU's outbound link finishes injecting the last packet
        #: handed to :meth:`send_to`. The bandwidth model's serialisation
        #: point; never read without a latency model, so it stays 0 for every
        #: run with the cost model off.
        self._tx_free_cycle = 0
        #: The device's :class:`NocLinkRegistry` for this NoC, or ``None``. Set
        #: by ``TT_Device._register_tile_internals``; ``None`` for a directly
        #: constructed NUI (unit tests, ``driver/simple``), which therefore
        #: charges nothing for a router-to-router link — the same shape as
        #: :attr:`noc_latency` being ``None``.
        self.noc_link_registry = None
        # Guards cross-thread appends to noc_new_requests_to_handle from
        # source tiles' transmit() calls and the owning tile's per-cycle
        # swap in clock_tick(). The destination then drains
        # noc_requests_to_handle without locking — only the owning thread
        # touches it after the swap.
        self._inbox_lock = threading.Lock()
        self.snoop = snoop
        self.unit_id: tuple | None = None
        # The owning tile's TileClock, set by TTDeviceTile._bind_clock. An
        # inbound transmit() is one of the stimuli that can wake a dormant
        # tile, so it must be able to reach the clock. None when the NUI is
        # constructed standalone (unit tests, driver/simple examples).
        self.clock_owner = None

    def get_id_pair(self):
        # Return the ID in this NoC coordinate system
        return self.id_pair

    def next_request_seq(self):
        """A fresh issue number for a request this NIU is about to send."""
        self._request_seq += 1
        return self._request_seq

    def add_outstanding_noc_request(self, request_id, tgt_addr, seq):
        # Per-trid, because tt-metal kernels (e.g. DRAM-sharded reads) issue
        # multiple requests with the same transaction ID before any barrier, so
        # a single slot per trid loses all but the last. Keyed within the trid
        # by the request's own issue number, so which response is which does
        # not depend on the order they come back in.
        self.outstanding_noc_requests.setdefault(request_id, {})[seq] = tgt_addr

    def take_outstanding_noc_request(self, response):
        """The state saved when the request ``response`` answers was issued.

        The alternative this replaces was a per-trid FIFO popped from the
        front, which is correct exactly while responses return in issue order.
        Nothing guarantees that. Hops are directional and per-destination, and
        with the bandwidth model a large transfer to one tile occupies its
        link while a small one to another does not — so two requests sharing a
        trid but not a destination can be answered in either order, and a
        multicast write sends one trid's requests to a whole rectangle at
        once. A FIFO fed out of order does not fail: it hands a read response
        the *other* request's L1 address and writes the right bytes to the
        wrong place, silently. Matching on the issue number removes the
        assumption rather than detecting its violation.

        Instrumented across the in-tree guards, out-of-order arrival is still
        rare-to-absent — :attr:`out_of_order_responses` counts it — so this is
        a hazard closed rather than a bug fixed. What *is* raised is a response
        no outstanding request accounts for, because there is nothing sensible
        to return for one.
        """
        pending = self.outstanding_noc_requests.get(response.request_id)
        if not pending or response.seq not in pending:
            raise NoCResponseError(
                f"NoC{self.noc_number} {self.id_pair}: response "
                f"{response.action.name} for trid {response.request_id} "
                f"(issue #{response.seq}) from {response.source_coord} matches "
                f"no outstanding request; awaiting "
                f"{sorted(pending) if pending else 'nothing'} on that trid"
            )
        if response.seq != next(iter(pending)):
            self.out_of_order_responses += 1
        return pending.pop(response.seq)

    def is_clock_idle(self):
        """No requests in flight, so ``clock_tick`` would only swap two empty
        lists. See ``docs/plans/event-driven-pump.md``; the only way this
        becomes False again is ``transmit()``, which wakes the owning tile."""
        return not (
            self.noc_requests_to_handle
            or self.noc_new_requests_to_handle
            or self.delayed_arrivals
        )

    def next_wake_cycle(self, cycle_num):
        """When this NIU next has something to do.

        The default derivation (``busy_until`` / :meth:`is_clock_idle`) would
        answer "next cycle" for a NIU whose only work is a packet that is still
        forty cycles from arriving, which keeps the whole tile awake for the
        flight. Naming the arrival cycle instead lets a DRAM tile sleep through
        it, and is what makes the latency model a *stride* rather than a spin.
        Identical to the default whenever nothing is in flight, which is every
        run with the cost model off.
        """
        if self.noc_requests_to_handle or self.noc_new_requests_to_handle:
            return cycle_num + 1
        arrival = self.next_arrival
        if arrival is None:
            return None
        return arrival if arrival > cycle_num else cycle_num + 1

    def clock_tick(self, cycle_num):
        arrival = self.next_arrival
        if arrival is not None and arrival <= cycle_num:
            self._land_arrived_packets(cycle_num)
        if not (self.noc_requests_to_handle or self.noc_new_requests_to_handle):
            # Nothing queued and nothing arriving: the drain loop below has no
            # work and the swap would exchange one empty list for another.
            # Early-out so an idle NIU costs neither the lock nor the
            # allocation (ROADMAP §L target 4). Kept alongside the TileClock
            # gate because a tile stays awake for its busy NIU while its other
            # NIU, and both NIUs of a tile whose cores are running, are idle.
            return
        for noc_request in self.noc_requests_to_handle:
            assert isinstance(noc_request, NUI.NoCDataRequest)
            if noc_request.action == NUI.NoCDataRequest.DataRequestAction.READ:
                if self.snoop:
                    print(
                        f"[NoC{self.noc_number} {self.id_pair}]: Read request id {noc_request.request_id} from NUI "
                        f"{noc_request.source_coord} at {hex(noc_request.tgt_address)} of size "
                        f"{hex(noc_request.data_length_bytes)}"
                    )
                self._publish_noc_event(
                    cycle_num,
                    phase="request",
                    txn_type="read",
                    src=noc_request.source_coord,
                    dst=self.id_pair,
                    size_bytes=noc_request.data_length_bytes,
                    txn_id=noc_request.request_id,
                    issue_cycle=noc_request.issue_cycle,
                )
                self.nui_counters.increment(
                    [
                        NUI.NUICounters.CounterNames.NIU_SLV_REQ_ACCEPTED,
                        NUI.NUICounters.CounterNames.NIU_SLV_RD_REQ_RECEIVED,
                    ]
                )

                data = self.attached_memory.read(
                    noc_request.tgt_address, noc_request.data_length_bytes
                )

                self.nui_counters.increment(
                    NUI.NUICounters.CounterNames.NIU_SLV_RD_RESP_SENT
                )

                self.nui_counters.increment(
                    NUI.NUICounters.CounterNames.NIU_SLV_RD_DATA_WORD_SENT,
                    noc_request.data_length_bytes / 4,
                )

                response = NUI.NoCDataRequest(
                    None,
                    NUI.NoCDataRequest.DataRequestAction.RESPONSE_READ,
                    noc_request.data_length_bytes,
                    self.id_pair,
                    noc_request.request_id,
                    data,
                    seq=noc_request.seq,
                )
                self.send_response(noc_request, response)
            elif noc_request.action == NUI.NoCDataRequest.DataRequestAction.WRITE:
                # When handle multiple 8192 size messages then will need to chunk this and slightly different
                # as NIU_SLV_NONPOSTED_WR_REQ_RECEIVED is incremented only for the last flit
                if self.snoop:
                    print(
                        f"[NoC{self.noc_number} {self.id_pair}]: Write request id {noc_request.request_id} from NUI "
                        f"{noc_request.source_coord} to {hex(noc_request.tgt_address)} of size "
                        f"{hex(noc_request.data_length_bytes)}"
                    )
                self._publish_noc_event(
                    cycle_num,
                    phase="request",
                    txn_type="write",
                    src=noc_request.source_coord,
                    dst=self.id_pair,
                    size_bytes=noc_request.data_length_bytes,
                    txn_id=noc_request.request_id,
                    issue_cycle=noc_request.issue_cycle,
                )
                if noc_request.noc_cmd_resp_marked:
                    self.nui_counters.increment(
                        [
                            NUI.NUICounters.CounterNames.NIU_SLV_NONPOSTED_WR_REQ_STARTED,
                            NUI.NUICounters.CounterNames.NIU_SLV_NONPOSTED_WR_DATA_WORD_RECEIVED,
                            NUI.NUICounters.CounterNames.NIU_SLV_NONPOSTED_WR_REQ_RECEIVED,
                        ]
                    )
                else:
                    self.nui_counters.increment(
                        [
                            NUI.NUICounters.CounterNames.NIU_SLV_POSTED_WR_REQ_STARTED,
                            NUI.NUICounters.CounterNames.NIU_SLV_POSTED_WR_DATA_WORD_RECEIVED,
                            NUI.NUICounters.CounterNames.NIU_SLV_POSTED_WR_REQ_RECEIVED,
                        ]
                    )
                self.attached_memory.write(noc_request.tgt_address, noc_request.data)

                if noc_request.noc_cmd_resp_marked:
                    self.nui_counters.increment(
                        NUI.NUICounters.CounterNames.NIU_SLV_WR_ACK_SENT
                    )

                response = NUI.NoCDataRequest(
                    None,
                    NUI.NoCDataRequest.DataRequestAction.ACK,
                    noc_request.data_length_bytes,
                    self.id_pair,
                    noc_request.request_id,
                    seq=noc_request.seq,
                )
                self.send_response(noc_request, response)
            elif (
                noc_request.action == NUI.NoCDataRequest.DataRequestAction.RESPONSE_READ
            ):
                tgt_addr = self.take_outstanding_noc_request(noc_request)
                self.attached_memory.write(tgt_addr, noc_request.data)

                if self.snoop:
                    print(
                        f"[NoC{self.noc_number} {self.id_pair}]: Read response id {noc_request.request_id} from NUI "
                        f"{noc_request.source_coord}, stored in to {hex(tgt_addr)} of size "
                        f"{hex(noc_request.data_length_bytes)}"
                    )
                self._publish_noc_event(
                    cycle_num,
                    phase="response",
                    txn_type="read",
                    src=noc_request.source_coord,
                    dst=self.id_pair,
                    size_bytes=noc_request.data_length_bytes,
                    txn_id=noc_request.request_id,
                    issue_cycle=noc_request.issue_cycle,
                )

                self.nui_counters.increment(
                    NUI.NUICounters.CounterNames.NIU_MST_RD_RESP_RECEIVED
                )
                # Each flit is 32 bytes, increment by this number
                self.nui_counters.increment(
                    NUI.NUICounters.CounterNames.NIU_MST_RD_DATA_WORD_RECEIVED,
                    noc_request.data_length_bytes / 4,
                )

                self.nui_counters.decrement(
                    NUI.NUICounters.CounterNames.NIU_MST_REQS_OUTSTANDING_ID_0
                    + noc_request.request_id
                )
                # NB: do NOT del self.outstanding_noc_requests[trid]. The FIFO
                # may still hold writes / atomics queued under the same trid;
                # callers also collide on a single trid when multiple cores
                # share a NUI (e.g. ROADMAP §A multi-Tensix nine/, with reader
                # and sender both on NoC 0). Leaving an empty list around is
                # safe — add_outstanding_noc_request just appends.
            elif noc_request.action == NUI.NoCDataRequest.DataRequestAction.ATOMIC:
                if self.snoop:
                    print(
                        f"[NoC{self.noc_number} {self.id_pair}]: Atomic-add id "
                        f"{noc_request.request_id} from NUI {noc_request.source_coord} "
                        f"at {hex(noc_request.tgt_address)} += {noc_request.at_data}"
                    )
                self._publish_noc_event(
                    cycle_num,
                    phase="request",
                    txn_type="atomic",
                    src=noc_request.source_coord,
                    dst=self.id_pair,
                    size_bytes=noc_request.data_length_bytes,
                    txn_id=noc_request.request_id,
                    issue_cycle=noc_request.issue_cycle,
                )
                if noc_request.noc_cmd_resp_marked:
                    self.nui_counters.increment(
                        [
                            NUI.NUICounters.CounterNames.NIU_SLV_REQ_ACCEPTED,
                            NUI.NUICounters.CounterNames.NIU_SLV_NONPOSTED_ATOMIC_RECEIVED,
                        ]
                    )
                else:
                    self.nui_counters.increment(
                        [
                            NUI.NUICounters.CounterNames.NIU_SLV_REQ_ACCEPTED,
                            NUI.NUICounters.CounterNames.NIU_SLV_POSTED_ATOMIC_RECEIVED,
                        ]
                    )

                old_bytes = self.attached_memory.read(noc_request.tgt_address, 4)
                old_val = conv_to_uint32(old_bytes)
                new_val = (old_val + noc_request.at_data) & 0xFFFFFFFF
                self.attached_memory.write(
                    noc_request.tgt_address, conv_to_bytes(new_val)
                )

                if noc_request.noc_cmd_resp_marked:
                    self.nui_counters.increment(
                        NUI.NUICounters.CounterNames.NIU_SLV_ATOMIC_RESP_SENT
                    )
                    response = NUI.NoCDataRequest(
                        None,
                        NUI.NoCDataRequest.DataRequestAction.RESPONSE_ATOMIC,
                        noc_request.data_length_bytes,
                        self.id_pair,
                        noc_request.request_id,
                        data=conv_to_bytes(old_val),
                        seq=noc_request.seq,
                    )
                    self.send_response(noc_request, response)
            elif (
                noc_request.action
                == NUI.NoCDataRequest.DataRequestAction.RESPONSE_ATOMIC
            ):
                if self.snoop:
                    print(
                        f"[NoC{self.noc_number} {self.id_pair}]: Atomic response id "
                        f"{noc_request.request_id} from NUI {noc_request.source_coord}"
                    )
                self._publish_noc_event(
                    cycle_num,
                    phase="response",
                    txn_type="atomic",
                    src=noc_request.source_coord,
                    dst=self.id_pair,
                    size_bytes=noc_request.data_length_bytes,
                    txn_id=noc_request.request_id,
                    issue_cycle=noc_request.issue_cycle,
                )
                self.nui_counters.increment(
                    NUI.NUICounters.CounterNames.NIU_MST_ATOMIC_RESP_RECEIVED
                )
                self.nui_counters.decrement(
                    NUI.NUICounters.CounterNames.NIU_MST_REQS_OUTSTANDING_ID_0
                    + noc_request.request_id
                )
                self.take_outstanding_noc_request(noc_request)
            elif noc_request.action == NUI.NoCDataRequest.DataRequestAction.ACK:
                if self.snoop:
                    print(
                        f"[NoC{self.noc_number} {self.id_pair}]: Write acknowledge to response id "
                        f"{noc_request.request_id} from NUI {noc_request.source_coord}"
                    )
                self._publish_noc_event(
                    cycle_num,
                    phase="response",
                    txn_type="write",
                    src=noc_request.source_coord,
                    dst=self.id_pair,
                    size_bytes=noc_request.data_length_bytes,
                    txn_id=noc_request.request_id,
                    issue_cycle=noc_request.issue_cycle,
                )

                self.nui_counters.decrement(
                    NUI.NUICounters.CounterNames.NIU_MST_WRITE_REQS_OUTGOING_ID_0
                    + noc_request.request_id
                )
                _noc_cmd_wr_inline, noc_cmd_resp_marked = (
                    self.take_outstanding_noc_request(noc_request)
                )
                if noc_cmd_resp_marked:
                    self.nui_counters.increment(
                        NUI.NUICounters.CounterNames.NIU_MST_WR_ACK_RECEIVED
                    )
                    self.nui_counters.decrement(
                        NUI.NUICounters.CounterNames.NIU_MST_REQS_OUTSTANDING_ID_0
                        + noc_request.request_id
                    )

        # Now copy over the new requests to the requests to handle
        with self._inbox_lock:
            self.noc_requests_to_handle = self.noc_new_requests_to_handle
            self.noc_new_requests_to_handle = []

    def _publish_noc_event(
        self, cycle_num, phase, txn_type, src, dst, size_bytes, txn_id, issue_cycle=-1
    ):
        if self.unit_id is None:
            return
        bus = get_bus()
        if not bus.is_enabled(EventCategory.NOC):
            return
        bus.publish(
            NoCEvent(
                cycle=cycle_num,
                unit_id=self.unit_id,
                phase=phase,
                txn_type=txn_type,
                src=tuple(src) if not isinstance(src, tuple) else src,
                dst=tuple(dst) if not isinstance(dst, tuple) else dst,
                size_bytes=int(size_bytes),
                txn_id=int(txn_id),
                # ``cycle_num`` is the arrival — this NIU servicing the packet
                # — so pairing the two gives the flight time. See
                # ``NUI.transmit``, which stamps it.
                issue_cycle=int(issue_cycle),
            )
        )

    def transmit(self, data_request, delay=None):
        """Accept a packet addressed to this NIU, arriving ``delay`` cycles hence.

        ``delay=None`` (and any delay of one cycle or less) keeps the original
        behaviour exactly: the packet lands in the arrivals list, the two-list
        swap in :meth:`clock_tick` hands it to the drain loop, and it is
        serviced on the next cycle. A longer delay parks it in
        :attr:`delayed_arrivals` until its cycle comes round.

        The delay is computed by the *sender* (:meth:`send_to`), because the
        sender is the endpoint that knows both its own per-NoC coord and the
        latency model. A NIU built without an architecture has no model and
        every caller passes ``None``, so the whole path collapses back to the
        original two lines.
        """
        cycle = self._current_cycle()
        if cycle is not None:
            # Trace-only: what makes NoCEvent.issue_cycle a measurement rather
            # than a placeholder, in both regimes. Hoisted out of the delayed
            # branch below so an un-modelled (next-cycle) flight is stamped too.
            data_request.issue_cycle = cycle
        if delay is not None and delay > 1:
            if cycle is not None:
                arrival = cycle + delay
                with self._inbox_lock:
                    self.delayed_arrivals.setdefault(arrival, []).append(data_request)
                    if self.next_arrival is None or arrival < self.next_arrival:
                        self.next_arrival = arrival
                owner = self.clock_owner
                if owner is not None:
                    owner.wake()
                return
        with self._inbox_lock:
            self.noc_new_requests_to_handle.append(data_request)
        owner = self.clock_owner
        if owner is not None:
            owner.wake()

    def _current_cycle(self):
        """The cycle being simulated, or ``None`` for an unclocked NIU.

        A packet's arrival cycle has to be absolute, and the only component
        that knows the absolute cycle at ``transmit`` time — which happens
        inside some *other* tile's tick — is the pump. ``None`` (a NIU with no
        owning tile clock: the unit tests and ``driver/simple``) means the
        flight cannot be timed, so the packet is delivered on the next cycle
        as it always was.
        """
        owner = self.clock_owner
        return None if owner is None else owner.current_cycle

    def _land_arrived_packets(self, cycle_num):
        """Move every packet whose flight ended by ``cycle_num`` into the drain.

        Called from the top of :meth:`clock_tick`, so an arriving packet is
        serviced on its arrival cycle rather than the one after it — which is
        what makes a one-cycle delay identical to the un-modelled path and a
        delay of N cost exactly N.
        """
        with self._inbox_lock:
            due = [c for c in self.delayed_arrivals if c <= cycle_num]
            for c in sorted(due):
                self.noc_requests_to_handle.extend(self.delayed_arrivals.pop(c))
            self.next_arrival = min(self.delayed_arrivals, default=None)

    def send_to(
        self, destination, packet, queued=None, link_wait=None, *, sent_from=None
    ):
        """Send ``packet`` from this NIU to ``destination``, timing the flight.

        The one place a NoC packet's latency is decided, for requests and
        responses alike. Both endpoints' coords are read in *this NoC's own
        space* (:attr:`x_coord` / :attr:`y_coord`, mirrored on NoC 1), never
        from the packet, so this cannot reintroduce the cross-NoC coordinate
        confusion :meth:`send_response` exists to prevent.

        The size of the packet is charged here too, and it is a different shape
        from the distance: see :meth:`_bandwidth_delay`. ``queued`` and
        ``link_wait`` are pre-claimed occupancies, for the multicast fan-out —
        see :meth:`claim_injection_port` and :meth:`claim_route_links`.

        ``sent_from`` overrides *this* end of the journey, for the one tile
        kind that answers to more than one NoC cell: a DRAM controller replies
        from the interface the request arrived at, not from its primary one.
        See :class:`AliasedEndpoint` and :meth:`send_response`.
        """
        model = self.noc_latency
        if model is None:
            destination.transmit(packet)
            return
        payload = _payload_bytes(packet)
        if queued is None:
            queued = self.claim_injection_port(payload)
        if link_wait is None:
            link_wait = self.claim_route_links(
                self.route_links_to(destination, sent_from=sent_from), payload, queued
            )
        destination.transmit(
            packet,
            self._bandwidth_delay(
                packet,
                self.flight_cycles_to(destination, sent_from=sent_from),
                queued,
                link_wait,
            ),
        )

    def route_links_to(self, destination, *, sent_from=None):
        """The router-to-router links a packet from here to ``destination``
        crosses, in order — or ``()`` when nothing shares links.

        Both coords are read in this NoC's own space, exactly as
        :meth:`flight_cycles_to` reads them, so the route and the hop count it
        is charged for are the same journey by construction.
        """
        if self.noc_link_registry is None or self.noc_latency is None:
            return ()
        try:
            return noc_route_links(
                (self.x_coord, self.y_coord) if sent_from is None else sent_from,
                _endpoint_noc_coord(destination),
                self.noc_grid_x,
                self.noc_grid_y,
            )
        except NoCCoordinateError as exc:
            raise self._off_grid(exc, destination) from None

    def claim_route_links(self, links, payload_bytes, queued=0):
        """Hold each of ``links`` long enough to carry ``payload_bytes``.

        The router-to-router half of what :meth:`claim_injection_port` does for
        this NIU's own port, and the same number spent: one flit per cycle per
        axis (``noc.hops.router_to_router.throughput_flits_per_cycle``,
        ``isa_doc``), which the model already charges once at the injecting NIU
        and now charges on each link the packet crosses.

        ``queued`` is how long the packet waited for the injection port, which
        is when its head actually reaches the first router — a packet held at
        the port arrives at the link later and is correspondingly less likely
        to find it busy. Returns the cycles it then waited for a busy link.

        **This is inert for a single flow**, which is the property that keeps
        it from being a second charge for the same thing: a flow's own packets
        leave its port one occupancy apart, so they reach each link exactly one
        occupancy apart and never queue behind themselves. Only *another
        tile's* traffic on the same link costs anything, which is precisely the
        resource the silicon measurement resolves.
        """
        registry = self.noc_link_registry
        if registry is None or not links:
            return 0
        now = self._current_cycle()
        if now is None:
            return 0
        occupancy = self.noc_latency.serialisation_cycles(payload_bytes)
        return registry.claim(links, now + queued, occupancy)

    def claim_injection_port(self, payload_bytes):
        """Hold this NIU's outbound link long enough to inject ``payload_bytes``.

        Returns how long the packet has to wait before it can start — which is
        how much of the *previous* packet is still going out. Separate from
        :meth:`_bandwidth_delay` for one caller: a multicast write is a single
        packet that the routers fan out, so it is injected **once** however
        many tiles are in the rectangle. tt-sim models the fan-out as N
        unicasts, and charging each of them the full injection time would
        invent serialisation the hardware does not have — the over-charging
        direction this project's cost policy asks callers to avoid. So the
        multicast path claims the port once and passes the answer to every
        :meth:`send_to` in the group.
        """
        model = self.noc_latency
        if model is None:
            return 0
        occupancy = model.serialisation_cycles(payload_bytes)
        now = self._current_cycle()
        if occupancy is None or now is None:
            return 0
        queued = self._tx_free_cycle - now
        if queued < 0:
            queued = 0
        self._tx_free_cycle = now + queued + occupancy
        return queued

    def _bandwidth_delay(self, packet, flight, queued=None, link_wait=None):
        """Add the bandwidth terms to a packet's ``flight`` time.

        Bandwidth is not a per-packet latency, it is an **occupancy of a
        link**: the NoC carries one flit per cycle, so a packet of N flits
        spends N cycles being pushed onto the wire. That single number is
        spent twice, in two different ways:

        * **The tail.** The last flit arrives ``N - 1`` cycles after the first,
          so the packet's arrival moves out by that much. Once, not per hop,
          because the NoC is wormhole-routed and the tail follows the head
          rather than being re-assembled at each router.
        * **The port.** This NIU's injection link is held for the whole N
          cycles, so the *next* packet this NIU sends departs no earlier than
          that. This is the serialisation half, and it is where an 8 KiB tile
          read actually differs from a semaphore poke: not by arriving 255
          cycles later, but by keeping everything behind it waiting.

        The port occupancy is also what keeps the model honest about ordering.
        Without it a one-flit packet issued right after a 256-flit one to the
        same destination would *overtake* it, which no NoC does and which the
        outstanding-request bookkeeping would have to cope with; with it,
        departures are monotonic and arrivals to a given destination are too.
        (Two *different* destinations can still reorder — that is real, and
        :meth:`take_outstanding_noc_request` is what makes it safe.)

        Since 2026-08-05 there is a **third** place the same occupancy is
        spent, and it is not a third number: each router-to-router link the
        packet crosses is held for the same N cycles, by
        :meth:`claim_route_links`, and ``link_wait`` is how long this packet
        waited for one of them to come free. That charge is zero for a single
        flow — a flow's own packets are already one occupancy apart when they
        leave the port — and non-zero only where another tile's traffic crosses
        the same link, which is the resource the Blackhole measurement in
        ``docs/bh_arch.md`` §4.2 resolves.
        """
        occupancy = self.noc_latency.serialisation_cycles(_payload_bytes(packet))
        if occupancy is None:
            return flight
        if queued is None:
            # An NIU with no clock (unit tests, ``driver/simple``) cannot hold
            # a port for N cycles because it does not know when N cycles are
            # up, and :meth:`claim_injection_port` answers 0 for it. The tail
            # is still charged; the queue is not.
            queued = self.claim_injection_port(_payload_bytes(packet))
        if link_wait is None:
            link_wait = 0
        return (0 if flight is None else flight) + queued + link_wait + occupancy - 1

    def tail_cycles_for(self, packet):
        """Cycles ``packet``'s last flit arrives after its first, or 0.

        The half of :meth:`_bandwidth_delay` that an endpoint with no clock of
        its own can still charge — see :meth:`NullEndpoint._respond`.
        """
        model = self.noc_latency
        if model is None:
            return 0
        tail = model.tail_cycles(_payload_bytes(packet))
        return 0 if tail is None else tail

    def flight_cycles_to(self, destination, *, sent_from=None):
        """Modelled cycles for a packet from here to ``destination``, or ``None``.

        ``sent_from`` replaces this NIU's own coord — see :meth:`send_to`.
        """
        model = self.noc_latency
        if model is None:
            return None
        try:
            hops = noc_hop_count(
                (self.x_coord, self.y_coord) if sent_from is None else sent_from,
                _endpoint_noc_coord(destination),
                self.noc_grid_x,
                self.noc_grid_y,
            )
        except NoCCoordinateError as exc:
            raise self._off_grid(exc, destination) from None
        return model.flight_cycles(hops)

    def _off_grid(self, exc, destination):
        """``exc`` again, saying which two endpoints the journey was between.

        Built here rather than passed into the distance functions because
        those are called once per packet and this is called once per bug.
        """
        return NoCCoordinateError(
            f"{exc} — NoC{self.noc_number} {self.id_pair} to "
            f"{type(destination).__name__} "
            f"{_endpoint_noc_coord(destination)}"
        )

    def flight_cycles_from(self, coord):
        """Modelled cycles for a packet from ``coord`` to here, or ``None``.

        The mirror image of :meth:`flight_cycles_to`, and not the same number:
        hops are directional (see :func:`noc_hop_count`). Used by
        :class:`NullEndpoint`, which has no latency model of its own.
        """
        model = self.noc_latency
        if model is None:
            return None
        return model.flight_cycles(
            noc_hop_count(
                coord,
                (self.x_coord, self.y_coord),
                self.noc_grid_x,
                self.noc_grid_y,
                src_role="source NullEndpoint",
            )
        )

    def send_response(self, noc_request, response):
        """Return a response to whoever issued ``noc_request``.

        The single choke point for the response direction — and deliberately
        the *only* routing decision on the NoC that is not a coordinate
        lookup. A response goes to ``noc_request.reply_to``, the endpoint
        object that issued the request, so it cannot be misdelivered no matter
        how the directories are keyed.

        This matters because the two directories are keyed in two different
        coordinate spaces that are both plain ``(x, y)`` tuples: NoC 0 by the
        canonical SoC-physical coord, NoC 1 additionally by the mirrored
        ``(GRID_X-1-x, GRID_Y-1-y)`` coord that tt-metal's bank-to-noc table
        emits (see ``TT_Device._register_tile_internals``). Those two spaces
        overlap — on Wormhole the DRAM column ``x=5`` mirrors onto the worker
        column ``x=4`` — so routing a response by the requester's coord
        resolved a *different* tile's NUI, which then popped an empty
        outstanding-request FIFO and killed the run. Requests keep resolving
        by coord because a request's destination coord genuinely arrives from
        the kernel in that NoC's own space; only the response direction had a
        space to get wrong, and it no longer carries one.

        The per-hop latency model landed here exactly as that suggests: the
        return flight is timed by :meth:`send_to`, which delays the
        ``transmit`` and reads both coords off the endpoint objects. No lookup
        was reintroduced — ``arrived_at`` is stamped on the request by the
        endpoint it arrived at (:class:`AliasedEndpoint`), not looked up, and
        is ``None`` for every tile that answers to a single cell.
        """
        assert noc_request.reply_to is not None, (
            f"NoC{self.noc_number} request {noc_request.request_id} from "
            f"{noc_request.source_coord} needs a response but carries no "
            f"reply_to endpoint"
        )
        self.send_to(noc_request.reply_to, response, sent_from=noc_request.arrived_at)

    def set_noc_directory(self, noc_directory):
        self.noc_directory = noc_directory

    def report_multicast_gaps(self, rectangle, destinations, endpoints):
        """Warn when a multicast rectangle spans cells with no modelled tile.

        The kernel tells its own software counter how many ACKs to expect
        (``num_dests``, the argument to ``noc_async_write_multicast``); the NIU
        counts the ACKs that actually arrive. tt-sim never sees ``num_dests`` —
        it is not written to any command register — so the mismatch cannot be
        diagnosed directly. What *can* be: a rectangle that covers cells no tile
        answers for, which is how the mismatch arises in practice. Blackhole's
        worker columns are ``1..7`` and ``10..16``, so a rectangle written as
        ``(2,2)..(10,2)`` is 9 cells wide while the caller counted 7 workers.
        Every cell is ACKed, the sender's expected count is 7, and
        ``noc_async_write_barrier`` spins on an equality that never holds.

        Naming the offending cells turns that unbounded hang into one line
        pointing at the caller. Always on, and deduplicated per rectangle: it
        fires only on a rectangle that is already outside what the caller can
        have counted, which is always a bug worth reporting.
        """
        if rectangle in self._reported_multicast_gaps:
            return
        unknown = [
            coord
            for coord, endpoint in zip(destinations, endpoints)
            if isinstance(endpoint, NullEndpoint)
        ]
        if not unknown:
            return
        self._reported_multicast_gaps.add(rectangle)
        x_start, y_start, x_end, y_end = rectangle
        print(
            f"[NoC{self.noc_number} {self.id_pair}]: multicast rectangle "
            f"({x_start}, {y_start})..({x_end}, {y_end}) covers "
            f"{len(destinations)} cells, {len(unknown)} of which have no "
            f"modelled tile: {unknown}. Every cell is ACKed, so if the kernel's "
            f"num_dests counted only the real destinations its "
            f"noc_async_write_barrier will never see its ACK count match.",
            file=sys.stderr,
        )

    def resolve_destination(self, coord):
        """Look up the endpoint a *request* is addressed to.

        The only coordinate lookup left on the NoC, and the only one that is
        well defined: ``coord`` comes straight out of the initiator's command
        registers, so it is already expressed in this NoC's own coordinate
        space. Unknown coords get a :class:`NullEndpoint` so the requester
        still completes. Responses do not come through here — see
        :meth:`send_response`.

        A miss first offers the coord to :attr:`directory_miss_hook`, if one is
        installed. That is what closes the *early-packet race*: a worker that
        left reset a hundred cycles before its peers can address one of them
        before the host has launched a kernel there, and null-routing that
        packet is a silently wrong answer rather than a hang. The hook may
        materialise the tile — which registers it in this very directory — so
        the lookup is retried before any null route is installed, and the
        packet proceeds to a real L1. Only a coord nothing can materialise
        (off-grid, or a tile kind the simulator does not model) reaches the
        NullEndpoint, and that answer is then cached as before.
        """
        dest = self.noc_directory.get(coord)
        if dest is None:
            if self.directory_miss_hook is not None:
                self.directory_miss_hook(self.noc_number, coord)
                dest = self.noc_directory.get(coord)
            if dest is None:
                dest = NullEndpoint(coord)
                self.noc_directory[coord] = dest
                if self.snoop:
                    print(
                        f"[NoC{self.noc_number} {self.id_pair}]: null-route installed for unknown destination {coord}"
                    )
        return dest

    def generate_NIU_and_NoC_config(self):
        # https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/NoC/MemoryMap.md#niu-and-noc-router-configuration

        self.niu_cfg_0 = clear_bit(0, 12)  # tile clock disable, 1=disable and 0=enable
        self.niu_cfg_0 = clear_bit(
            self.niu_cfg_0, 13
        )  # double store disable, 1=disable and 0=enable
        self.niu_cfg_0 = clear_bit(
            self.niu_cfg_0, 14
        )  # coordinate translation enable, 1=enable and 0=disable

        self.router_cfg_0 = 0
        self.router_cfg_1 = 0
        self.router_cfg_2 = 0
        self.router_cfg_3 = 0
        self.router_cfg_4 = 0

        # The coordinate a *core* reports as its own: tt-metal's firmware fills
        # ``my_x[noc]`` / ``my_y[noc]`` from this register and then both
        # compares it against that NoC's bank table (``is_local_bank``) and
        # emits it as a destination (single-argument ``get_noc_addr``). So it
        # follows the same per-arch convention as NoC 1's directory keys:
        # mirrored on Wormhole, canonical on Blackhole. ``NOC_NODE_ID`` below
        # is the physical node ID and stays per-NoC on both.
        if self.noc_id_logical_mirrored_on_noc1:
            logical_x, logical_y = self.x_coord, self.y_coord
        else:
            logical_x, logical_y = self.id_pair
        self.noc_id_logical = replace_bits(0, logical_x, 0, 6)
        self.noc_id_logical = replace_bits(self.noc_id_logical, logical_y, 6, 6)

        # Backing store for the NOC_CFG(cnt) register block (0x100 + cnt*4)
        # that isn't modelled with dedicated semantics — e.g. the NoC ID
        # translation tables / masks the firmware programs then reads back.
        # Behaves like plain config registers (read-what-you-wrote, default 0),
        # which is enough for the init sequences that touch them.
        self.noc_config_regs = {}

    def generate_NoC_node_id(self):
        self.noc_node_id = replace_bits(0, self.x_coord, 0, 6)
        self.noc_node_id = replace_bits(self.noc_node_id, self.y_coord, 6, 6)
        self.noc_node_id = replace_bits(self.noc_node_id, 10, 12, 7)
        self.noc_node_id = replace_bits(self.noc_node_id, 12, 19, 7)
        self.noc_node_id = clear_bit(self.noc_node_id, 26)
        self.noc_node_id = clear_bit(self.noc_node_id, 27)
        self.noc_node_id = clear_bit(self.noc_node_id, 28)

    def generate_NoC_endpoint_id(self):
        self.noc_endpoint_id = replace_bits(0, 0, 8, 0)
        self.noc_endpoint_id = replace_bits(self.noc_endpoint_id, 0, 8, 8)
        self.noc_endpoint_id = replace_bits(self.noc_endpoint_id, 1, 16, 8)
        self.noc_endpoint_id = replace_bits(
            self.noc_endpoint_id, self.noc_number, 24, 8
        )

    # Blackhole command-buffer register offset (within a 0x800-stride buffer) ->
    # the Wormhole-canonical offset (within a 0x400-stride buffer) that read()/
    # write() below are written against. Blackhole inserts AT_LEN_BE_1 at 0x24,
    # which shifts AT_DATA to 0x28 and CMD_CTRL/NODE_ID/ENDPOINT_ID up to
    # 0x40/0x44/0x48 (see blackhole/noc_parameters.h). AT_LEN_BE_1 (0x24) has no
    # Wormhole equivalent and is not modelled, so it maps to ``None`` (ignored).
    _BH_CMD_REG_TO_WH = {
        0x0: 0x0,
        0x4: 0x4,
        0x8: 0x8,
        0xC: 0xC,
        0x10: 0x10,
        0x14: 0x14,
        0x18: 0x18,
        0x1C: 0x1C,
        0x20: 0x20,
        0x24: None,
        0x28: 0x24,
        0x40: 0x28,
        0x44: 0x2C,
        0x48: 0x30,
    }

    def _to_canonical_cmd_addr(self, addr):
        """Map a Blackhole NIU address to the Wormhole-canonical layout.

        Returns the canonical address, or ``None`` for the unmodelled
        AT_LEN_BE_1 register. A no-op for Wormhole.
        """
        if not self.blackhole_cmd_buf_layout:
            return addr
        buf, reg = divmod(addr, 0x800)
        if 0 <= buf < 4 and reg in NUI._BH_CMD_REG_TO_WH:
            wh_reg = NUI._BH_CMD_REG_TO_WH[reg]
            return None if wh_reg is None else buf * 0x400 + wh_reg
        return addr

    def read(self, addr, size):
        canonical = self._to_canonical_cmd_addr(addr)
        if canonical is None:
            return conv_to_bytes(0)  # AT_LEN_BE_1: not modelled, reads as 0
        addr = canonical
        if self.snoop:
            print(f"NoC read {hex(addr)}")
        if addr == self.noc_id_logical_offset:
            return conv_to_bytes(self.noc_id_logical)
        elif addr == 0x100:
            return conv_to_bytes(self.niu_cfg_0)
        elif addr == 0x104:
            return conv_to_bytes(self.router_cfg_0)
        elif addr == 0x108:
            return conv_to_bytes(self.router_cfg_1)
        elif addr == 0x10C:
            return conv_to_bytes(self.router_cfg_2)
        elif addr == 0x110:
            return conv_to_bytes(self.router_cfg_3)
        elif addr == 0x114:
            return conv_to_bytes(self.router_cfg_4)
        elif addr == 0x002C or addr == 0x042C or addr == 0x82C or addr == 0xC2C:
            return conv_to_bytes(self.noc_node_id)
        elif addr == 0x0030 or addr == 0x0430 or addr == 0x830 or addr == 0xC30:
            return conv_to_bytes(self.noc_endpoint_id)
        elif addr == 0x64 or addr == 0x68:
            # CMD_BUF_AVAIL (0x64) and CMD_BUF_OVFL (0x68), Blackhole only;
            # Wormhole's noc_parameters.h has no analogue for either. 0x64 is
            # four 5-bit per-command-buffer occupancy fields whose depth the ISA
            # docs decline to state, and 0x68 is the overflow register beside
            # it -- the only reading that can turn a measured peak occupancy
            # into a depth rather than a lower bound. Reading both on silicon is
            # the whole point of perfbench/nocreadbench. This NUI models no
            # command buffer at all, so there is no honest value to return and
            # a fabricated small integer would be indistinguishable from the
            # measurement. Return the all-ones sentinel nocreadbench already
            # uses for "this part does not expose the register", so a simulator
            # run reads as *absent* rather than as a number.
            return conv_to_bytes(0xFFFFFFFF)
        elif addr == 0x0:
            return conv_to_bytes(self.request_initiators[0].target_addr_low)
        elif addr == 0x4:
            return conv_to_bytes(self.request_initiators[0].target_addr_mid)
        # NOC_TARG_ADDR_HI / NOC_RET_ADDR_HI. The write path has always decoded
        # these; the read path did not, so a register the kernel had itself
        # written back raised NotImplementedError. tt-metal's device profiler
        # reads 0x14 while starting up, which is what took the whole profiler
        # path down under tt-sim and left the paper's end-to-end cycle table
        # with no simulator column.
        elif addr == 0x8:
            return conv_to_bytes(self.request_initiators[0].target_addr_hi)
        elif addr == 0xC:
            return conv_to_bytes(self.request_initiators[0].ret_addr_low)
        elif addr == 0x10:
            return conv_to_bytes(self.request_initiators[0].ret_addr_mid)
        elif addr == 0x14:
            return conv_to_bytes(self.request_initiators[0].ret_addr_hi)
        elif addr == 0x18:
            return conv_to_bytes(self.request_initiators[0].packet_tag)
        elif addr == 0x1C:
            return conv_to_bytes(self.request_initiators[0].ctrl)
        elif addr == 0x20:
            return conv_to_bytes(self.request_initiators[0].at_len_be)
        elif addr == 0x24:
            return conv_to_bytes(self.request_initiators[0].at_data)
        elif addr == 0x28:
            return conv_to_bytes(self.request_initiators[0].cmd_ctrl)
        elif addr == 0x400:
            return conv_to_bytes(self.request_initiators[1].target_addr_low)
        elif addr == 0x404:
            return conv_to_bytes(self.request_initiators[1].target_addr_mid)
        elif addr == 0x408:
            return conv_to_bytes(self.request_initiators[1].target_addr_hi)
        elif addr == 0x40C:
            return conv_to_bytes(self.request_initiators[1].ret_addr_low)
        elif addr == 0x410:
            return conv_to_bytes(self.request_initiators[1].ret_addr_mid)
        elif addr == 0x414:
            return conv_to_bytes(self.request_initiators[1].ret_addr_hi)
        elif addr == 0x418:
            return conv_to_bytes(self.request_initiators[1].packet_tag)
        elif addr == 0x41C:
            return conv_to_bytes(self.request_initiators[1].ctrl)
        elif addr == 0x420:
            return conv_to_bytes(self.request_initiators[1].at_len_be)
        elif addr == 0x424:
            return conv_to_bytes(self.request_initiators[1].at_data)
        elif addr == 0x428:
            return conv_to_bytes(self.request_initiators[1].cmd_ctrl)
        elif addr == 0x800:
            return conv_to_bytes(self.request_initiators[2].target_addr_low)
        elif addr == 0x804:
            return conv_to_bytes(self.request_initiators[2].target_addr_mid)
        elif addr == 0x808:
            return conv_to_bytes(self.request_initiators[2].target_addr_hi)
        elif addr == 0x80C:
            return conv_to_bytes(self.request_initiators[2].ret_addr_low)
        elif addr == 0x810:
            return conv_to_bytes(self.request_initiators[2].ret_addr_mid)
        elif addr == 0x814:
            return conv_to_bytes(self.request_initiators[2].ret_addr_hi)
        elif addr == 0x818:
            return conv_to_bytes(self.request_initiators[2].packet_tag)
        elif addr == 0x81C:
            return conv_to_bytes(self.request_initiators[2].ctrl)
        elif addr == 0x820:
            return conv_to_bytes(self.request_initiators[2].at_len_be)
        elif addr == 0x824:
            return conv_to_bytes(self.request_initiators[2].at_data)
        elif addr == 0x828:
            return conv_to_bytes(self.request_initiators[2].cmd_ctrl)
        elif addr == 0xC00:
            return conv_to_bytes(self.request_initiators[3].target_addr_low)
        elif addr == 0xC04:
            return conv_to_bytes(self.request_initiators[3].target_addr_mid)
        elif addr == 0xC08:
            return conv_to_bytes(self.request_initiators[3].target_addr_hi)
        elif addr == 0xC0C:
            return conv_to_bytes(self.request_initiators[3].ret_addr_low)
        elif addr == 0xC10:
            return conv_to_bytes(self.request_initiators[3].ret_addr_mid)
        elif addr == 0xC14:
            return conv_to_bytes(self.request_initiators[3].ret_addr_hi)
        elif addr == 0xC18:
            return conv_to_bytes(self.request_initiators[3].packet_tag)
        elif addr == 0xC1C:
            return conv_to_bytes(self.request_initiators[3].ctrl)
        elif addr == 0xC20:
            return conv_to_bytes(self.request_initiators[3].at_len_be)
        elif addr == 0xC24:
            return conv_to_bytes(self.request_initiators[3].at_data)
        elif addr == 0xC28:
            return conv_to_bytes(self.request_initiators[3].cmd_ctrl)
        elif addr >= 0x200 and addr <= 0x2F4:
            counter_idx = int((addr - 0x200) / 4)
            return conv_to_bytes(self.nui_counters[counter_idx])
        elif 0x100 <= addr <= 0x1FC and (addr & 0x3) == 0:
            # NOC_CFG(cnt) register block, generic read-what-you-wrote.
            return conv_to_bytes(self.noc_config_regs.get(addr, 0))
        else:
            raise NotImplementedError(
                f"Reading from address {hex(addr)} not yet supported by NoC"
            )

    def write(self, addr, value, size=None):
        canonical = self._to_canonical_cmd_addr(addr)
        if canonical is None:
            return  # AT_LEN_BE_1: not modelled, writes ignored
        addr = canonical
        if self.snoop:
            print(f"NoC write {hex(addr)}")
        if addr == self.noc_id_logical_offset:
            self.noc_id_logical = conv_to_uint32(value)
        elif addr == 0x100:
            self.niu_cfg_0 = conv_to_uint32(value)
        elif addr == 0x104:
            self.router_cfg_0 = conv_to_uint32(value)
        elif addr == 0x108:
            self.router_cfg_1 = conv_to_uint32(value)
        elif addr == 0x10C:
            self.router_cfg_2 = conv_to_uint32(value)
        elif addr == 0x110:
            self.router_cfg_3 = conv_to_uint32(value)
        elif addr == 0x114:
            self.router_cfg_4 = conv_to_uint32(value)
        elif addr == 0x0:
            self.request_initiators[0].target_addr_low = conv_to_uint32(value)
        elif addr == 0x4:
            self.request_initiators[0].target_addr_mid = conv_to_uint32(value)
        elif addr == 0x8:
            self.request_initiators[0].target_addr_hi = conv_to_uint32(value)
        elif addr == 0xC:
            self.request_initiators[0].ret_addr_low = conv_to_uint32(value)
        elif addr == 0x10:
            self.request_initiators[0].ret_addr_mid = conv_to_uint32(value)
        elif addr == 0x14:
            self.request_initiators[0].ret_addr_hi = conv_to_uint32(value)
        elif addr == 0x18:
            self.request_initiators[0].packet_tag = conv_to_uint32(value)
        elif addr == 0x1C:
            self.request_initiators[0].ctrl = conv_to_uint32(value)
        elif addr == 0x20:
            self.request_initiators[0].at_len_be = conv_to_uint32(value)
        elif addr == 0x24:
            self.request_initiators[0].at_data = conv_to_uint32(value)
        elif addr == 0x28:
            self.request_initiators[0].cmd_ctrl = conv_to_uint32(value)
            self.request_initiators[0].initiate()
        elif addr == 0x400:
            self.request_initiators[1].target_addr_low = conv_to_uint32(value)
        elif addr == 0x404:
            self.request_initiators[1].target_addr_mid = conv_to_uint32(value)
        elif addr == 0x408:
            self.request_initiators[1].target_addr_hi = conv_to_uint32(value)
        elif addr == 0x40C:
            self.request_initiators[1].ret_addr_low = conv_to_uint32(value)
        elif addr == 0x410:
            self.request_initiators[1].ret_addr_mid = conv_to_uint32(value)
        elif addr == 0x414:
            self.request_initiators[1].ret_addr_hi = conv_to_uint32(value)
        elif addr == 0x418:
            self.request_initiators[1].packet_tag = conv_to_uint32(value)
        elif addr == 0x41C:
            self.request_initiators[1].ctrl = conv_to_uint32(value)
        elif addr == 0x420:
            self.request_initiators[1].at_len_be = conv_to_uint32(value)
        elif addr == 0x424:
            self.request_initiators[1].at_data = conv_to_uint32(value)
        elif addr == 0x428:
            self.request_initiators[1].cmd_ctrl = conv_to_uint32(value)
            self.request_initiators[1].initiate()
        elif addr == 0x800:
            self.request_initiators[2].target_addr_low = conv_to_uint32(value)
        elif addr == 0x804:
            self.request_initiators[2].target_addr_mid = conv_to_uint32(value)
        elif addr == 0x808:
            self.request_initiators[2].target_addr_hi = conv_to_uint32(value)
        elif addr == 0x80C:
            self.request_initiators[2].ret_addr_low = conv_to_uint32(value)
        elif addr == 0x810:
            self.request_initiators[2].ret_addr_mid = conv_to_uint32(value)
        elif addr == 0x814:
            self.request_initiators[2].ret_addr_hi = conv_to_uint32(value)
        elif addr == 0x818:
            self.request_initiators[2].packet_tag = conv_to_uint32(value)
        elif addr == 0x81C:
            self.request_initiators[2].ctrl = conv_to_uint32(value)
        elif addr == 0x820:
            self.request_initiators[2].at_len_be = conv_to_uint32(value)
        elif addr == 0x824:
            self.request_initiators[2].at_data = conv_to_uint32(value)
        elif addr == 0x828:
            self.request_initiators[2].cmd_ctrl = conv_to_uint32(value)
            self.request_initiators[2].initiate()
        elif addr == 0xC00:
            self.request_initiators[3].target_addr_low = conv_to_uint32(value)
        elif addr == 0xC04:
            self.request_initiators[3].target_addr_mid = conv_to_uint32(value)
        elif addr == 0xC08:
            self.request_initiators[3].target_addr_hi = conv_to_uint32(value)
        elif addr == 0xC0C:
            self.request_initiators[3].ret_addr_low = conv_to_uint32(value)
        elif addr == 0xC10:
            self.request_initiators[3].ret_addr_mid = conv_to_uint32(value)
        elif addr == 0xC14:
            self.request_initiators[3].ret_addr_hi = conv_to_uint32(value)
        elif addr == 0xC18:
            self.request_initiators[3].packet_tag = conv_to_uint32(value)
        elif addr == 0xC1C:
            self.request_initiators[3].ctrl = conv_to_uint32(value)
        elif addr == 0xC20:
            self.request_initiators[3].at_len_be = conv_to_uint32(value)
        elif addr == 0xC24:
            self.request_initiators[3].at_data = conv_to_uint32(value)
        elif addr == 0xC28:
            self.request_initiators[3].cmd_ctrl = conv_to_uint32(value)
            self.request_initiators[3].initiate()
        elif 0x100 <= addr <= 0x1FC and (addr & 0x3) == 0:
            # NOC_CFG(cnt) register block, generic backing store.
            self.noc_config_regs[addr] = conv_to_uint32(value)
        else:
            raise NotImplementedError(
                f"Writing to address {hex(addr)} not yet supported by NoC"
            )

    def getSize(self):
        return 0xFFFF
