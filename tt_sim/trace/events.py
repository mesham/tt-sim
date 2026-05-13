"""Typed events published on the simulator's event bus.

Every event subclass fixes its CATEGORY as a ClassVar so the bus can
route purely by ``type(event)``, no per-instance discriminator field.
SCHEMA_VERSION is bumped on any breaking change to event shape; writers
should refuse to emit unknown versions.
"""

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar


class EventCategory(Enum):
    INSTR = "instr"
    MEM = "mem"
    NOC = "noc"
    COMPUTE = "compute"
    SYNC = "sync"
    DISPATCH = "dispatch"
    LIFECYCLE = "lifecycle"


class Unit(Enum):
    BRISC = "BRISC"
    NCRISC = "NCRISC"
    TRISC0 = "TRISC0"
    TRISC1 = "TRISC1"
    TRISC2 = "TRISC2"
    FPU = "FPU"
    SFPU = "SFPU"
    PACKER = "PACKER"
    UNPACKER = "UNPACKER"
    MATRIX = "MATRIX"
    MOVER = "MOVER"
    THCON = "THCON"
    SYNC = "SYNC"
    TDMA = "TDMA"
    CFG = "CFG"
    MISC = "MISC"
    MAILBOX = "MAILBOX"
    TTSYNC = "TTSYNC"
    NOC0 = "NOC0"
    NOC1 = "NOC1"
    HOST = "HOST"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class Event:
    SCHEMA_VERSION: ClassVar[int] = 2
    CATEGORY: ClassVar[EventCategory | None] = None
    cycle: int
    unit_id: tuple


@dataclass(frozen=True, slots=True)
class InstrEvent(Event):
    CATEGORY: ClassVar[EventCategory] = EventCategory.INSTR
    pc: int
    instruction: int
    stalled: bool = False
    # Destination register write captured at retirement.
    # reg_write_idx == -1 means no architectural register was written
    # (stores, branches, jumps without link). x0 writes are not
    # recorded (Spike's commitlog format excludes them).
    reg_write_idx: int = -1
    reg_write_value: int = 0


@dataclass(frozen=True, slots=True)
class DispatchEvent(Event):
    CATEGORY: ClassVar[EventCategory] = EventCategory.DISPATCH
    opcode: str
    target_unit: str
    thread_id: int


@dataclass(frozen=True, slots=True)
class NoCEvent(Event):
    CATEGORY: ClassVar[EventCategory] = EventCategory.NOC
    phase: str
    txn_type: str
    src: tuple
    dst: tuple
    size_bytes: int = 0
    txn_id: int = 0


@dataclass(frozen=True, slots=True)
class LifecycleEvent(Event):
    CATEGORY: ClassVar[EventCategory] = EventCategory.LIFECYCLE
    kind: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class MemEvent(Event):
    CATEGORY: ClassVar[EventCategory] = EventCategory.MEM
    op: str  # "read" or "write"
    address: int
    size: int
    region: str = ""  # "L1", "DRAM", "MMIO", etc. — coarse classifier


@dataclass(frozen=True, slots=True)
class ComputeEvent(Event):
    CATEGORY: ClassVar[EventCategory] = EventCategory.COMPUTE
    op: str  # opcode name (ELWADD, SFPCONFIG, PACK, UNPACK, ...)
    target_unit: str  # FPU / SFPU / PACKER / UNPACKER / MATRIX / MOVER / THCON
    thread_id: int = -1  # -1 when not thread-attributed
    detail: str = ""


@dataclass(frozen=True, slots=True)
class SyncEvent(Event):
    CATEGORY: ClassVar[EventCategory] = EventCategory.SYNC
    kind: str  # "mailbox_send" / "mailbox_recv" / "ttsync_signal" / "ttsync_wait"
    detail: str = ""
