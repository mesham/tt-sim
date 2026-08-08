"""An optional Numba acceleration of the exact FPU datapath's inner kernel.

``MatrixUnit.perform_mvmul_exact`` evaluates one MVMUL/GAPOOL/DOTPV as two
numpy passes -- :meth:`~MatrixUnit._fpu_group_sums_batch` and
:meth:`~MatrixUnit._fpu_accumulate_batch`. Those are already batched, and they
are still the single most expensive thing in a matmul workload, because the
arrays are *small*: one instruction is a 16-lane reduction over at most 16 SrcB
rows by 16 Dst columns, i.e. 2048 elements, and roughly fifty numpy operations
each allocate and traverse a fresh 16 KB temporary to do 2048 elements' worth
of integer arithmetic. The arithmetic is not the cost; the temporaries are.

``mvmul_fused`` below is the same arithmetic written as one scalar loop nest,
so every intermediate stays in a register and Dst is written once. Fusing the
two passes also removes the ``(2, rows, columns)`` group-sum arrays entirely.
Measured on this machine: **534 us -> 46 us per MVMUL, 13.3x**, bit-exact.

Three things keep this honest:

- **It is optional.** Numba is not a dependency of the simulator; CLAUDE.md's
  "no build step -- pure Python" is the project's reason to exist, and the
  roadmap rejected Cython precisely because a build step breaks hackability.
  If ``numba`` will not import, :func:`get_fused_mvmul` returns ``None`` for
  ever and the numpy pair runs exactly as before. Nothing else changes.
- **It is bit-exact, and tested as such.** ``fpu_accumulate_test.py`` already
  pins the numpy pair against the scalar port of ttsim's C; it now fuzzes this
  kernel against the numpy pair too, so a divergence is a test failure rather
  than a wrong number in a matmul.
- **It only engages once it can pay for itself.** Getting to a callable kernel
  costs ~3.4 s cold and **~800 ms** per process warm, so it is pure loss on a
  workload with a handful of MVMULs. The import is deferred until
  ``TT_SIM_NUMBA_THRESHOLD`` of them have run (default 512).
  ``TT_SIM_NUMBA=0`` disables it outright; ``TT_SIM_NUMBA=1`` compiles on the
  first MVMUL.

**Where the ~800 ms goes, and why it is a floor.** It was once recorded as an
"on-disk cache load", which is the one thing it mostly is not:

===============================================  ========  ======================
stage                                            cost      what it actually is
===============================================  ========  ======================
``import numba``                                  ~450 ms  numba's package import
first call, numba's lazy ``refresh()``            ~350 ms  importing
                                                           ``numba.cpython.*`` /
                                                           ``numba.np.*`` and
                                                           installing ~250 typing
                                                           and lowering registries
first call, reading the ``.nbc``                   ~30 ms  this kernel's object
                                                           code
===============================================  ========  ======================

Only the last row is ours. The middle row is numba initialising its target
context, which it defers until something is compiled *or loaded from cache*; a
one-line ``njit`` kernel measures the same, so shrinking or re-signaturing
``mvmul_fused`` buys nothing. Nor is dropping ``cache=True`` an option --
compiling cold costs ~3.4 s against the ~380 ms a cache load costs -- and no
narrower import (``numba.core.decorators``, ``numba.core.registry``) is
measurably cheaper than ``import numba``. The only ways further down are an
ahead-of-time build or a second dependency, which are the two things the
project refuses. **Treat the ~800 ms as fixed.**

**Do not try to hide it on a background thread.** Measured, nine interleaved
rounds: warming up in a daemon thread while the numpy path keeps serving
MVMULs is **+42 %** against just blocking, and +38 % against never engaging at
all. The main thread's numpy calls release and re-acquire the GIL thousands of
times a second, and against a compile thread that holds it for whole switch
intervals they convoy -- the simulator ran ~6x slower for the duration, which
is far more than the stall the thread was hiding.

**Why 512 and not less.** Engaging costs ~800 ms and saves ~507 us per later
MVMUL, so it repays only after ~1580 further calls: a workload of M MVMULs
wins iff ``M > threshold + 1580``. Measured on the in-tree guards (medians,
Blackhole, cost model off):

=============  =======  =======  =======  =======  =======
guard          MVMULs   off      t=128    t=512    t=1024
=============  =======  =======  =======  =======  =======
``matmulidx``      384  2540 ms  3482 ms  2693 ms  2587 ms
``matmulblock``   1536  4178 ms  4182 ms  4302 ms  4563 ms
``six``           4096  7115 ms  5841 ms  5987 ms  6165 ms
=============  =======  =======  =======  =======  =======

``six`` is what the accelerator is for -- **1128 ms, -16 %** against numpy.
But dropping the threshold to 128 makes ``matmulidx`` engage for only 256
further calls and lose **789 ms (+29 %, the slowest arm in 12 of 12 rounds)**,
which costs more than the 146 ms it buys ``six``; and raising it to 1024 makes
``matmulblock`` pay the same 800 ms over fewer calls, +260 ms. The three
non-engaging arms on ``matmulidx`` (off / 512 / 1024) agree within noise, which
is what says the harness is measuring the switch and not the weather. 512 is
the compromise; it is not a placeholder.

``nogil=True`` is deliberate: the kernel holds no Python objects, so it drops
the GIL for its whole run. That is the only route to the roadmap's "threading
revival" that does not need a free-threaded interpreter -- note that the
convoy above is an argument about *compiling* on a thread, not about running
the compiled kernel on one, which is exactly what ``nogil`` makes safe.
"""

import os

# Resolved on first use and then frozen: the compiled kernel, or None if numba
# is unavailable, disabled, or failed to compile.
_fused = None
_resolved = False
_calls = 0

_DEFAULT_THRESHOLD = 512


def _threshold():
    """``None`` to stay on numpy for ever, else the call count to switch after.

    Same truthy vocabulary as ``TT_SIM_THREADED`` and the ``TT_SIM_DIAG_*``
    flags: ``1``/``true``/``yes``/``on``, case-insensitive.
    """
    raw = os.environ.get("TT_SIM_NUMBA")
    if raw is not None:
        return 0 if raw.strip().lower() in ("1", "true", "yes", "on") else None
    try:
        return max(0, int(os.environ.get("TT_SIM_NUMBA_THRESHOLD", _DEFAULT_THRESHOLD)))
    except ValueError:
        return _DEFAULT_THRESHOLD


def get_fused_mvmul():
    """The compiled fused MVMUL kernel, or ``None`` to use the numpy pair.

    Cheap on every call after the first decision: two module-global reads.
    """
    global _fused, _resolved, _calls
    if _resolved:
        return _fused
    threshold = _threshold()
    if threshold is None:
        _resolved = True
        return None
    _calls += 1
    if _calls <= threshold:
        return None
    _resolved = True
    try:
        from tt_sim.pe.tensix.backends.fpu_jit_kernel import mvmul_fused
    except Exception:
        # No numba, an incompatible numba, or a compile failure. The numpy pair
        # is a complete implementation, so this is not an error.
        _fused = None
        return None
    _fused = mvmul_fused
    return _fused


def reset_for_test():
    """Forget the resolved decision. For tests that toggle the environment."""
    global _fused, _resolved, _calls
    _fused = None
    _resolved = False
    _calls = 0
