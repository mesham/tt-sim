"""``ebreak`` on a baby RISC-V is a device-side assertion, not a no-op.

The five Tensix baby cores have no debugger attached and no trap handler
installed, so an ``ebreak`` is terminal: the core stops there. That is the
mechanism the tt-metal device-side debug tooling is built on --

* ``ASSERT()`` in a kernel (``api/debug/assert.h``) compiles to a waypoint
  write followed by a halt;
* the **LLK sanitizer** (``TT_METAL_LLK_SANITIZER=1``) reports a violated LLK
  contract through ``LLK_ASSERT``, and in the assert-only build
  (``TT_METAL_LLK_ASSERTS=1`` without the watcher) ``LLK_ASSERT`` is literally
  ``asm volatile("ebreak")`` -- see ``tt-llk/common/inc/llk_assert.h``;
* ``__builtin_trap()`` from a compiler-inserted unreachable path.

tt-sim used to decode ``ebreak`` and return "handled, nothing to do", so every
one of those assertions fired into the void and the kernel carried on running
past a point its own authors had declared impossible. A simulator that runs
*through* the assertion is worse than one that never had it: the program
reports success while the same binary stops dead on silicon. So the decode
raises, naming the core and the PC, and pointing at the tooling that carries
the actual message (the text of an ``LLK_ASSERT`` only exists in the DPRINT or
watcher build -- the assert-only build compiles the string away entirely).

``ecall`` stays ignored. Nothing in the tt-metal firmware or kernel path issues
one, and unlike ``ebreak`` it is not the encoding a failed assertion lowers to.

Set :data:`IGNORE_ENV_VAR` truthy to restore the old skip-and-continue
behaviour, e.g. to see how far past a known assertion a program would get.
"""

import os

#: Truthy to make ``ebreak`` a no-op again.
IGNORE_ENV_VAR = "TT_SIM_IGNORE_EBREAK"

_TRUTHY = {"1", "true", "yes", "on"}


class RiscvBreakpoint(RuntimeError):
    """Raised when a baby RISC-V core executes ``ebreak``."""

    def __init__(self, memory_space=None):
        # ``RV32.clock_tick`` stamps ``(unit_id, core_label, pc)`` on the
        # memory space every cycle for the trace/diagnostic path, so the core
        # identity is already to hand here without threading it through every
        # ISA handler signature.
        ctx = getattr(memory_space, "caller_context", None) or ()
        self.unit_id = ctx[0] if len(ctx) > 0 else None
        self.core_label = ctx[1] if len(ctx) > 1 else "?"
        self.pc = ctx[2] if len(ctx) > 2 else None
        where = f"{self.core_label}"
        # unit_id is the trace registry's (chip, core_y, core_x, unit) tuple;
        # the unit is already ``core_label``, so report just the tile.
        if isinstance(self.unit_id, tuple) and len(self.unit_id) >= 3:
            where += f" on tile (x={self.unit_id[2]}, y={self.unit_id[1]})"
        elif self.unit_id is not None:
            where += f" ({self.unit_id})"
        at = "" if self.pc is None else f" at pc={hex(self.pc)}"
        super().__init__(
            f"{where} executed `ebreak`{at}: a device-side assertion fired.\n"
            "  On silicon this halts the core. It is what a failed tt-metal "
            "`ASSERT()` and\n"
            "  a tripped LLK sanitizer check (TT_METAL_LLK_SANITIZER=1) lower "
            "to, so the\n"
            "  kernel is reporting that it is wrong -- the simulator has not "
            "gone wrong.\n"
            "  The assertion's message is compiled out of the assert-only "
            "build; rebuild\n"
            "  with the watcher or DPRINT enabled to read it.\n"
            f"  ({IGNORE_ENV_VAR}=1 to execute through it instead.)"
        )


def _env_ignored() -> bool:
    return os.environ.get(IGNORE_ENV_VAR, "").strip().lower() in _TRUTHY


#: Cached at import so a hot decode path never touches ``os.environ``.
#: Call :func:`refresh_from_env` after mutating the environment.
_ignored = _env_ignored()


def refresh_from_env() -> bool:
    """Re-read :data:`IGNORE_ENV_VAR`. Returns whether ``ebreak`` now traps."""
    global _ignored
    _ignored = _env_ignored()
    return not _ignored


def trapping_enabled() -> bool:
    """Whether ``ebreak`` raises rather than being skipped."""
    return not _ignored
