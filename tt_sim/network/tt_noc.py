from enum import IntEnum

from tt_sim.device.clock import Clockable
from tt_sim.memory.mem_mapable import MemMapable
from tt_sim.util.conversion import (
    clear_bit,
    conv_to_bytes,
    conv_to_uint32,
    extract_bits,
    replace_bits,
)


class NoCOverlay(MemMapable):
    def __init__(self):
        pass

    def read(self, addr, size):
        return 0

    def write(self, addr, value, size=None):
        pass

    def getSize(self):
        return 0x3FFFF


class NUI(MemMapable, Clockable):
    class NoCDataRequest:
        class DataRequestAction(IntEnum):
            READ = 0
            WRITE = 1
            RESPONSE_READ = 2
            ACK = 3

        def __init__(
            self,
            tgt_address,
            action,
            data_length_bytes,
            source,
            request_id,
            data=None,
            noc_cmd_resp_marked=True,
        ):
            self.tgt_address = tgt_address
            self.action = action
            self.request_id = request_id
            self.data_length_bytes = data_length_bytes
            self.source = source
            self.data = data
            self.noc_cmd_resp_marked = noc_cmd_resp_marked

    class RequestInitiator:
        def __init__(self, nui):
            self.target_addr_low = 0
            self.target_addr_mid = 0
            self.ret_addr_low = 0
            self.ret_addr_mid = 0
            self.packet_tag = 0
            self.ctrl = 0
            self.at_len_be = 0
            self.at_data = 0
            self.cmd_ctrl = 0
            self.nui = nui

        def handle_read_transfer(self):
            noc_packet_transaction_id = extract_bits(self.packet_tag, 10, 4)

            self.nui.nui_counters.increment(
                [
                    NUI.NUICounters.CounterNames.NIU_MST_REQS_OUTSTANDING_ID_0
                    + noc_packet_transaction_id,
                    NUI.NUICounters.CounterNames.NIU_MST_CMD_ACCEPTED,
                    NUI.NUICounters.CounterNames.NIU_MST_RD_REQ_STARTED,
                ]
            )

            # TODO: handle request splitting when message > 8192
            self.cmd_ctrl = 0

            target_tile_x = extract_bits(self.target_addr_mid, 6, 4)
            target_tile_y = extract_bits(self.target_addr_mid, 12, 4)
            assert (target_tile_x, target_tile_y) in self.nui.noc_directory

            read_req = NUI.NoCDataRequest(
                self.target_addr_low,
                NUI.NoCDataRequest.DataRequestAction.READ,
                self.at_len_be,
                self.nui.id_pair,
                noc_packet_transaction_id,
            )
            self.nui.add_outstanding_noc_request(
                noc_packet_transaction_id, self.ret_addr_low
            )

            self.nui.noc_directory[(target_tile_x, target_tile_y)].transmit(read_req)
            self.nui.nui_counters.increment(
                NUI.NUICounters.CounterNames.NIU_MST_RD_REQ_SENT
            )

            if self.nui.snoop:
                print(
                    f"[NoC {self.nui.id_pair}]: Issue read request id {read_req.request_id} to NUI "
                    f"{(target_tile_x, target_tile_y)}, reading at {hex(read_req.tgt_address)} of size "
                    f"{hex(read_req.data_length_bytes)}"
                )

        def handle_inline_write_transfer(
            self, noc_cmd_wr_be, noc_cmd_wr_inline, noc_cmd_resp_marked
        ):
            noc_packet_transaction_id = extract_bits(self.packet_tag, 10, 4)

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
            ret_tile_x = extract_bits(self.ret_addr_mid, 6, 4)
            ret_tile_y = extract_bits(self.ret_addr_mid, 12, 4)
            assert (ret_tile_x, ret_tile_y) in self.nui.noc_directory

            data = self.nui.attached_memory.read(self.target_addr_low, self.at_len_be)

            write_req = NUI.NoCDataRequest(
                self.ret_addr_low,
                NUI.NoCDataRequest.DataRequestAction.WRITE,
                self.at_len_be,
                self.nui.id_pair,
                noc_packet_transaction_id,
                data,
                noc_cmd_resp_marked,
            )
            self.nui.add_outstanding_noc_request(
                noc_packet_transaction_id, (noc_cmd_wr_inline, noc_cmd_resp_marked)
            )
            self.nui.noc_directory[(ret_tile_x, ret_tile_y)].transmit(write_req)

        def handle_none_inline_write(
            self, noc_cmd_wr_be, noc_cmd_wr_inline, noc_cmd_resp_marked
        ):
            noc_packet_transaction_id = extract_bits(self.packet_tag, 10, 4)

            if noc_cmd_resp_marked:
                self.nui.nui_counters.increment(
                    NUI.NUICounters.CounterNames.NIU_MST_REQS_OUTSTANDING_ID_0
                    + noc_packet_transaction_id,
                    noc_cmd_wr_be,
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

            # Send write request
            ret_tile_x = extract_bits(self.ret_addr_mid, 6, 4)
            ret_tile_y = extract_bits(self.ret_addr_mid, 12, 4)
            assert (ret_tile_x, ret_tile_y) in self.nui.noc_directory

            data = self.nui.attached_memory.read(self.target_addr_low, self.at_len_be)

            write_req = NUI.NoCDataRequest(
                self.ret_addr_low,
                NUI.NoCDataRequest.DataRequestAction.WRITE,
                self.at_len_be,
                self.nui.id_pair,
                noc_packet_transaction_id,
                data,
                noc_cmd_resp_marked,
            )
            self.nui.add_outstanding_noc_request(
                noc_packet_transaction_id, (noc_cmd_wr_inline, noc_cmd_resp_marked)
            )
            self.nui.noc_directory[(ret_tile_x, ret_tile_y)].transmit(write_req)

            if noc_cmd_resp_marked:
                self.nui.nui_counters.increment(
                    NUI.NUICounters.CounterNames.NIU_MST_NONPOSTED_WR_REQ_SENT
                )
                self.nui.nui_counters.increment(
                    NUI.NUICounters.CounterNames.NIU_MST_NONPOSTED_WR_DATA_WORD_SENT,
                    self.at_len_be / 4,
                )
            else:
                self.nui.nui_counters.increment(
                    NUI.NUICounters.CounterNames.NIU_MST_POSTED_WR_REQ_SENT
                )
                self.nui.nui_counters.increment(
                    NUI.NUICounters.CounterNames.NIU_MST_POSTED_WR_DATA_WORD_SENT,
                    self.at_len_be / 4,
                )
            self.nui.nui_counters.decrement(
                NUI.NUICounters.CounterNames.NIU_MST_WRITE_REQS_OUTGOING_ID_0
            )

            if self.nui.snoop:
                print(
                    f"[NoC {self.nui.id_pair}]: Issue write request id {write_req.request_id} to NUI "
                    f"{(ret_tile_x, ret_tile_y)}, writing at {hex(write_req.tgt_address)} of size "
                    f"{hex(write_req.data_length_bytes)}"
                )

        def handle_write_transfer(self):
            noc_cmd_wr_be = extract_bits(self.ctrl, 1, 2)
            noc_cmd_wr_inline = extract_bits(self.ctrl, 1, 3)
            noc_cmd_resp_marked = extract_bits(self.ctrl, 1, 4)

            if noc_cmd_wr_inline:
                self.handle_inline_write_transfer(
                    noc_cmd_wr_be, noc_cmd_wr_inline, noc_cmd_resp_marked
                )
            else:
                self.handle_none_inline_write(
                    noc_cmd_wr_be, noc_cmd_wr_inline, noc_cmd_resp_marked
                )

        def initiate(self):
            if self.cmd_ctrl == 1:
                # Following the protocol at
                # https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/NoC/Counters.md
                # however this is different as completing immediately
                # TODO need to hook up with NOC_PACKET_TRANSACTION_ID

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
                    raise NotImplementedError(
                        "NoC does not support atomic data transfers"
                    )
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
            self.counters = [0] * 61

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

    def __init__(self, noc_number, x_coord, y_coord, attached_memory, snoop=False):
        assert noc_number == 0 or noc_number == 1
        self.noc_number = noc_number
        self.x_coord = x_coord
        self.y_coord = y_coord
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
        self.attached_memory = attached_memory
        self.outstanding_noc_requests = {}
        # Separate these out to ensure we have atleast one clock cycle
        # between a request and it being handled (can increase)
        self.noc_requests_to_handle = []
        self.noc_new_requests_to_handle = []
        self.snoop = snoop

    def get_id_pair(self):
        return self.id_pair

    def add_outstanding_noc_request(self, request_id, tgt_addr):
        self.outstanding_noc_requests[request_id] = tgt_addr

    def clock_tick(self, cycle_num):
        for noc_request in self.noc_requests_to_handle:
            assert isinstance(noc_request, NUI.NoCDataRequest)
            if noc_request.action == NUI.NoCDataRequest.DataRequestAction.READ:
                if self.snoop:
                    print(
                        f"[NoC {self.id_pair}]: Read request id {noc_request.request_id} from NUI "
                        f"{noc_request.source} at {hex(noc_request.tgt_address)} of size "
                        f"{hex(noc_request.data_length_bytes)}"
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
                )
                self.noc_directory[noc_request.source].transmit(response)
            elif noc_request.action == NUI.NoCDataRequest.DataRequestAction.WRITE:
                # When handle multiple 8192 size messages then will need to chunk this and slightly different
                # as NIU_SLV_NONPOSTED_WR_REQ_RECEIVED is incremented only for the last flit
                if self.snoop:
                    print(
                        f"[NoC {self.id_pair}]: Write request id {noc_request.request_id} from NUI "
                        f"{noc_request.source} to {hex(noc_request.tgt_address)} of size "
                        f"{hex(noc_request.data_length_bytes)}"
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
                )
                self.noc_directory[noc_request.source].transmit(response)
            elif (
                noc_request.action == NUI.NoCDataRequest.DataRequestAction.RESPONSE_READ
            ):
                tgt_addr = self.outstanding_noc_requests[noc_request.request_id]
                self.attached_memory.write(tgt_addr, noc_request.data)

                if self.snoop:
                    print(
                        f"[NoC {self.id_pair}]: Read response id {noc_request.request_id} from NUI "
                        f"{noc_request.source}, stored in to {hex(tgt_addr)} of size "
                        f"{hex(noc_request.data_length_bytes)}"
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
                del self.outstanding_noc_requests[noc_request.request_id]
            elif noc_request.action == NUI.NoCDataRequest.DataRequestAction.ACK:
                if self.snoop:
                    print(
                        f"[NoC {self.id_pair}]: Write acknowledge to response id "
                        f"{noc_request.request_id} from NUI {noc_request.source}"
                    )

                self.nui_counters.decrement(
                    NUI.NUICounters.CounterNames.NIU_MST_WRITE_REQS_OUTGOING_ID_0
                    + noc_request.request_id
                )
                noc_cmd_resp_marked = self.outstanding_noc_requests[
                    noc_request.request_id
                ][1]
                if noc_cmd_resp_marked:
                    self.nui_counters.increment(
                        NUI.NUICounters.CounterNames.NIU_MST_WR_ACK_RECEIVED
                    )
                    self.nui_counters.decrement(
                        NUI.NUICounters.CounterNames.NIU_MST_REQS_OUTSTANDING_ID_0
                        + noc_request.request_id
                    )
                del self.outstanding_noc_requests[noc_request.request_id]

        # Now copy over the new requests to the requests to handle
        self.noc_requests_to_handle = self.noc_new_requests_to_handle
        self.noc_new_requests_to_handle = []

    def transmit(self, data_request):
        self.noc_new_requests_to_handle.append(data_request)

    def set_noc_directory(self, noc_directory):
        self.noc_directory = noc_directory

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

        self.noc_id_logical = replace_bits(0, self.x_coord, 0, 6)
        self.noc_id_logical = replace_bits(self.noc_id_logical, self.y_coord, 6, 6)

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

    def read(self, addr, size):
        if self.snoop:
            print(f"NoC read {hex(addr)}")
        if addr == 0x0138:
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
        elif addr == 0x0:
            return conv_to_bytes(self.request_initiators[0].target_addr_low)
        elif addr == 0x4:
            return conv_to_bytes(self.request_initiators[0].target_addr_mid)
        elif addr == 0xC:
            return conv_to_bytes(self.request_initiators[0].ret_addr_low)
        elif addr == 0x10:
            return conv_to_bytes(self.request_initiators[0].ret_addr_mid)
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
        elif addr == 0x40C:
            return conv_to_bytes(self.request_initiators[1].ret_addr_low)
        elif addr == 0x410:
            return conv_to_bytes(self.request_initiators[1].ret_addr_mid)
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
        elif addr == 0x80C:
            return conv_to_bytes(self.request_initiators[2].ret_addr_low)
        elif addr == 0x810:
            return conv_to_bytes(self.request_initiators[2].ret_addr_mid)
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
        elif addr == 0xC0C:
            return conv_to_bytes(self.request_initiators[3].ret_addr_low)
        elif addr == 0xC10:
            return conv_to_bytes(self.request_initiators[3].ret_addr_mid)
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
        else:
            raise NotImplementedError(
                f"Reading from address {hex(addr)} not yet supported by NoC"
            )

    def write(self, addr, value, size=None):
        if self.snoop:
            print(f"NoC write {hex(addr)}")
        if addr == 0x0138:
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
        elif addr == 0xC:
            self.request_initiators[0].ret_addr_low = conv_to_uint32(value)
        elif addr == 0x10:
            self.request_initiators[0].ret_addr_mid = conv_to_uint32(value)
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
        elif addr == 0x40C:
            self.request_initiators[1].ret_addr_low = conv_to_uint32(value)
        elif addr == 0x410:
            self.request_initiators[1].ret_addr_mid = conv_to_uint32(value)
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
        elif addr == 0x80C:
            self.request_initiators[2].ret_addr_low = conv_to_uint32(value)
        elif addr == 0x810:
            self.request_initiators[2].ret_addr_mid = conv_to_uint32(value)
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
        elif addr == 0xC0C:
            self.request_initiators[3].ret_addr_low = conv_to_uint32(value)
        elif addr == 0xC10:
            self.request_initiators[3].ret_addr_mid = conv_to_uint32(value)
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
        else:
            raise NotImplementedError(
                f"Writing to address {hex(addr)} not yet supported by NoC"
            )

    def getSize(self):
        return 0xFFFF
