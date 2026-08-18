# DRAFT — not filed. Read §0 before deciding whether to file.

**Target:** `tenstorrent/tt-umd` (vendored as `tt_metal/third_party/umd`, VERSION `0.9.7`,
inside tt-metal `0.74.0` / `c49bb7625e`).
**Status:** draft only. Nothing has been filed, pushed or sent.

---

## §0. For our reviewer — the blocker claim did not survive verification

The premise this ticket was drafted against was *"a wire-protocol simulation chip cannot
declare NoC translation support"*. **That is false.** UMD already exposes the field, it is
documented for `ChipType::SIMULATION`, tt-metal already plumbs an env var to it, and it
works end to end against our wire-protocol simulator. Details and the A/B measurement are
in §4.

So the strong ticket is unnecessary and should not be filed. What remains is a smaller,
genuine wart — a capability decision made by testing a filename extension, plus two
inconsistencies that fall out of it. §1–§3 are written as a fileable issue for that
smaller thing; file it, or don't, but don't file the blocker.

---

## §1. Ask

`Cluster` decides whether a simulated chip supports NoC coordinate translation by testing
the *file extension of the simulator path*. Please make that decision expressible
directly — e.g. an optional `ClusterOptions` field — and leave the extension test to do
only what it can actually answer: which backend transports the requests.

```cpp
// device/cluster.cpp:382-387 (UMD 0.9.7)
// Noc translation is enabled for mock chips and for ttsim simulation, but disabled for versim/vcs
// simulation.
bool is_ttsim_simulation =
    (options.chip_type == ChipType::SIMULATION && options.simulator_directory.extension() == ".so");
bool noc_translation_enabled = options.chip_type == ChipType::MOCK ||
                               options.chip_type == ChipType::SWEMULE || is_ttsim_simulation;
```

The comment states a **capability** ("software simulators can translate, RTL simulators
cannot"). The code tests a **transport**. Those are orthogonal, and the same expression is
load-bearing for backend selection elsewhere, so they cannot be separated by changing this
line alone:

- `device/tt_device/simulation_device_factory.cpp:17` and `:29` — `.so` ⇒ `TTSimTTDevice`
  (in-process `dlopen`), otherwise `RtlSimulationTTDevice` (nng wire protocol).
- `device/simulation/simulation_connector.cpp:35` — same test again, host vs client.
- `device/simulation/simulation_chip.cpp:34-36` — `.so` relocates the expected
  `soc_descriptor.yaml` to the parent directory.
- `device/simulation/tt_sim_communicator.cpp:88-126` — the `.so` path is `dlopen`ed and
  must export `libttsim_init` / `libttsim_tile_rd_bytes` / `libttsim_clock` /
  `libttsim_pci_*`.

A software simulator that speaks the nng wire protocol (rather than exporting the
`libttsim_*` ABI) is therefore classified with versim/vcs — against the comment's own
intent — and there is no way to say otherwise short of handing UMD a complete
`ClusterDescriptor` (§4).

## §2. Why it matters — the failure mode when translation is off

With translation disabled, tt-metal's device firmware init emits **two different
coordinate conventions on NoC 1 in the same run**, and on Wormhole they collide:

- Worker coordinates are not virtualised, and the kernel-side address helpers do not
  mirror them per NoC: `NOC_0_X(noc_index, noc_size_x, x)` is the identity
  (`tt_metal/hw/inc/internal/tt-1xx/wormhole/noc_nonblocking_api.h:24-25`), so
  `get_noc_addr` / `get_noc_multicast_addr` place the **unmirrored** coordinate in the
  command registers on both NoCs.
- The DRAM bank table is built per NoC from `hal_.noc_coordinate(...)`, i.e. **mirrored**
  on NoC 1, because `dram_is_virtualized` is false
  (`tt_metal/impl/device/firmware/risc_firmware_initializer.cpp:549-567`; Wormhole's
  virtualised core types are `{TENSIX, ETH}` only —
  `tt_metal/llrt/hal/tt-1xx/wormhole/wh_hal.cpp:377`).

A device model must therefore resolve NoC 1 destination coordinates in two conventions
whose numeric ranges overlap on a 10×12 (Wormhole) / 17×12 (Blackhole) grid. Where a
mirrored DRAM (or worker) coordinate lands on a live worker's canonical coordinate, one of
the two is unreachable, and the wrong tile **accepts and ACKs** the write: the sender's
`noc_async_write_barrier()` returns normally and neither end can tell. As an indication of
how much of the grid is affected in our model: 56 of Wormhole's 80 worker coordinates and
102 of Blackhole's 140 resolve to a foreign tile on NoC 1 (NoC 0 is clean).

With translation **on**, the collision cannot arise: workers and eth move to the translated
band (Wormhole `x ∈ 18..25`, worker `y ∈ 18..27`, eth `y ∈ 16..17`) while Wormhole DRAM
keeps physical coordinates (`x ∈ {0,5}`, `y < 12`), so the two spaces are numerically
disjoint. We measured exactly that on the wire (§4).

That is the whole of the request: the model *can* implement ID translation; UMD is what
decides it isn't allowed to say so.

## §3. Proposed change

Smallest change that separates capability from transport, and is a no-op for every
existing caller:

```diff
--- a/device/api/umd/device/cluster.hpp
+++ b/device/api/umd/device/cluster.hpp
@@ struct ClusterOptions {
      std::filesystem::path simulator_directory = "";
+
+    /**
+     * Only used for SIMULATION chip type. Declares whether the simulated chip implements
+     * NoC coordinate translation. If unset, UMD infers it from the simulator backend
+     * (ttsim: enabled; RTL simulation: disabled).
+     */
+    std::optional<bool> noc_translation_enabled = std::nullopt;

--- a/device/cluster.cpp
+++ b/device/cluster.cpp
@@ -382,9 +382,11 @@
-                // Noc translation is enabled for mock chips and for ttsim simulation, but disabled for versim/vcs
-                // simulation.
+                // Mock and SW-emulated chips translate. For simulation, the caller may declare it; otherwise
+                // infer from the backend, which the simulator path's extension selects (.so => in-process ttsim).
                 bool is_ttsim_simulation =
                     (options.chip_type == ChipType::SIMULATION && options.simulator_directory.extension() == ".so");
-                bool noc_translation_enabled = options.chip_type == ChipType::MOCK ||
-                                               options.chip_type == ChipType::SWEMULE || is_ttsim_simulation;
+                bool noc_translation_enabled = options.noc_translation_enabled.value_or(
+                    options.chip_type == ChipType::MOCK || options.chip_type == ChipType::SWEMULE ||
+                    is_ttsim_simulation);
```

Backward compatibility: unset is the default, so versim/vcs keep translation disabled and
`.so` ttsim users keep it enabled. Nothing about backend selection moves.

If you take this, tt-metal would want a matching pass-through (one `std::optional<bool>`
from `RunTimeOptions` into the `ClusterOptions{}` built at
`tt_metal/llrt/tt_cluster.cpp:411-416`) — we are happy to write that patch too.

### Alternatives we considered and rejected

- **Honour the SoC descriptor's `features.noc.translation_id_enabled`.** Most natural home
  on the face of it, and the key already exists — but it is currently *write-only* in UMD:
  it is emitted by `SocDescriptor::serialize` (`device/soc_descriptor.cpp:388`) and parsed
  nowhere (`device/soc_arch_descriptor.cpp` reads no such key). Starting to honour it would
  silently change behaviour for any existing descriptor that carries it, and it is
  arguably the wrong layer: `SocDescriptor` takes `noc_translation_enabled` from
  `ChipInfo` (`device/soc_descriptor.cpp:121`), i.e. per-chip runtime state owned by the
  cluster descriptor, not from the arch description. Worth deciding separately whether to
  parse the key or drop it — a field that is serialized and never read invites exactly the
  misreading we made.
- **A new `ChipType`** (`SIMULATION_TTSIM` / `SIMULATION_RTL`). Still encodes the backend
  rather than the capability, and churns a public enum plus every switch over it.
- **A marker file in the simulator directory.** Undiscoverable magic; no API to document.
- **Do nothing; require a full `ClusterDescriptor`** — the route that works today (§4). It
  costs every wire-protocol simulator a hand-written cluster descriptor plus, from
  tt-metal, an env var named `..._MOCK_...`, and leaves the misleading comment in place.

### Two smaller inconsistencies noticed on the way

Reported for completeness; both are latent today.

1. `TTSimTTDevice::get_noc_translation_enabled()` returns **false**, with the comment
   "TTSim operates on logical/virtual coordinates end-to-end; NOC translation is never
   applied" (`device/tt_device/tt_sim_tt_device.cpp:411-414`) — while `cluster.cpp:384-387`
   declares translation **enabled** for exactly that device. Only the cluster-descriptor
   value reaches `SocDescriptor` for simulation chips
   (`device/cluster.cpp:285-292`), so the `TTDevice` accessor is unused on this path and
   the disagreement is invisible; but one of the two is wrong.
2. `translation_id_enabled` in the SoC descriptor YAML is serialized and never parsed (see
   above).

---

## §4. The route that already works (verified end to end)

`ClusterOptions::cluster_descriptor` is documented as *"Only used for SIMULATION and MOCK
chip types"* (`device/api/umd/device/cluster.hpp:104-107`), and when it is non-null,
`cluster.cpp:369` takes a branch that never evaluates the extension test: the supplied
descriptor's own per-chip `noc_translation` flag is used
(`ClusterDescriptor::create_constrained_cluster_descriptor`, `cluster_descriptor.cpp:394`;
the YAML key is parsed at `cluster_descriptor.cpp:870`). From tt-metal, a wire-protocol
simulation run reaches that field by setting `TT_METAL_MOCK_CLUSTER_DESC_PATH` alongside
`TT_METAL_SIMULATOR` — a deliberately supported combination
(`tt_metal/llrt/rtoptions.cpp:443-459`, "If both MOCK and SIMULATOR are set, SIMULATOR
takes precedence"; the branch that consumes it is `tt_metal/llrt/tt_cluster.cpp:398-409`).
`Cluster::is_mock_or_emulated()` is `Mock || Emule` only (`tt_cluster.hpp:369-371`), so a
Simulator target with a mock cluster descriptor still takes the real firmware-init path
and the descriptor's translation flag is honoured there
(`risc_firmware_initializer.cpp:549-550`).

Measured, same host binary, one env var apart, against our wire-protocol simulator
(tt-metal 0.74.0, slow dispatch, single Tensix example):

| | worker addressing on the wire | result |
| --- | --- | --- |
| baseline | `(1,1)` — physical NoC0 | passes |
| `TT_METAL_MOCK_CLUSTER_DESC_PATH=<yaml with noc_translation: true>` | `(18,18) … (25,27)` workers, `(18,16)/(18,17)…` eth, DRAM still `(0,11)`, `(0,5)` … | fails in *our* model, which does not yet accept translated coordinates |

The descriptor used was the shipped
`tests/cluster_descriptor_examples/wormhole_N150.yaml` with `harvest_mask: 0`. (With
`board_type: n150` and a zero mask, `cluster_descriptor.cpp:1288-1297` logs a harvesting
consistency warning — cosmetic, but a reason a simulator-shaped board type might be worth
having.)

So: **translation over the wire protocol works.** The failure in the second row is ours to
fix, not UMD's, and is exactly the work §5 describes.

## §5. What we do on our side once this lands (or with the workaround today)

We maintain a Python functional simulator that speaks the UMD wire protocol. With
translation enabled we would:

1. Model the NIU ID-translation tables instead of leaving `NIU_CFG_0` bit 14 clear, and
   accept translated coordinates on the wire (workers `x ∈ 18..25`, `y ∈ 18..27` on
   Wormhole).
2. Stop keying NoC 1 in two conventions. (Done, and the shape is the opposite of
   what this section first guessed: it is the *canonical* — unmirrored — key that
   goes, because a NoC 1 coordinate is the grid mirror and translated kernels emit
   neither. Wormhole keeps its mirrors, whose space stays unambiguous because its
   translated bands are off the physical grid; Blackhole drops them, its translated
   coords being physical coords.)
3. Keep the non-translated path working for as long as UMD's default keeps it, so both
   configurations remain testable.

None of that is work we are asking UMD to carry. The ask in §1 is one optional field.

## §6. Reproducing

**Without a simulator.** The classification is a pure function of the path extension and
can be read at `device/cluster.cpp:384-387`; a unit test over
`ClusterDescriptor::create_mock_cluster` shows the two outcomes directly. There is no
`ClusterOptions` field, env var, cluster-descriptor default or SoC-descriptor key that
changes it — we checked all four (§3, §4), and the only lever is supplying a whole
`ClusterDescriptor`.

**With a simulator.** A runtime repro needs something that speaks the nng wire protocol
(`device/simulation/simulation_device.fbs`) — an RTL simulator, or any stub that answers
`WRITE`/`READ`/`RESET_*`. Shortest path we know:

```bash
# any directory-style TT_METAL_SIMULATOR (i.e. not a .so) — worker coords arrive physical
TT_METAL_SIMULATOR=/path/to/sim  TT_METAL_SLOW_DISPATCH_MODE=1  ./any_metal_program
# => first writes address (1,1); cluster desc reports noc_translation = false

# same run, translation declared through the cluster descriptor
TT_METAL_MOCK_CLUSTER_DESC_PATH=/path/to/wormhole_N150.yaml  # with noc_translation: true
# => first writes address (18,18); i.e. the capability is real, only the declaration was missing
```

---

## §7. Verified vs inferred

**Verified by reading the named code and/or measuring:**

- Every file:line quoted above (UMD 0.9.7 in tt-metal 0.74.0 / `c49bb7625e`).
- The extension test drives both translation and backend selection, and relocates the
  SoC-descriptor path.
- `translation_id_enabled` is emitted by UMD and parsed by nothing.
- `TTSimTTDevice::get_noc_translation_enabled()` returns false while `cluster.cpp`
  declares true for the same device.
- The `cluster_descriptor` route works: measured A/B on one host binary, translated worker
  coordinates on the wire (§4).
- The 56/80 and 102/140 shadowing counts are computed and asserted in our own test suite
  (`tt_sim/network/noc_routing_test.py`, 17 tests, passing).

**Inferred, not proven:**

- That the maintainers intend the comment's reading ("software simulators translate") to
  cover wire-protocol software simulators. That is our reading of the comment, not a
  statement anyone made.
- That no in-flight UMD work already replaces this heuristic. We checked only the version
  vendored in tt-metal 0.74.0; we did not read `tt-umd` `main`.
- That the ergonomics matter to anyone beyond us. If wire-protocol simulators are expected
  to carry a cluster descriptor anyway, the right outcome here may be a comment fix and a
  line of documentation rather than a new field.
