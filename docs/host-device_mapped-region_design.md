# HostDeviceMappedRegion Design

## Purpose

`HostDeviceMappedRegion` is a low-level L3 parent-to-chip-child/NPU
communication primitive in Simpler. It exposes a child-owned memory region
through a narrow host-side API:

- device-visible data and signal addresses that can be passed to kernels,
- a raw data area addressed by byte offset,
- cache-line-sized signal slots,
- explicit host-side datacopy into and out of the region, and
- explicit host-side notify/wait operations on signal slots.

The primitive is intentionally lower-level than task tensor payload handling,
shared-buffer protocols, and send/recv protocols. It does not construct
`ContinuousTensor` descriptors, infer dependencies, manage queue metadata, or
define a message format. Higher-level protocols can build those policies on
top of the mapped data area and signal slots.

This design extracts only the first-layer `datacopy + notify/wait` primitive
from the lessons of PR803. Higher-level shared-buffer and send/recv protocols
are left to later designs.

## Runtime Ownership

`HostDeviceMappedRegion` is owned by the process that owns `DeviceContext`.

In L3 PROCESS mode, that owner is the chip child process containing
`ChipWorker`, `DeviceRunner`, and the loaded host runtime. The L3 parent owns
only an opaque region wrapper and reaches the child through the existing
parent-child mailbox RPC path.

The NPU does not own the allocation lifetime. It participates by reading and
writing the device-visible addresses returned by `mapped_region_info`.

Host mappings are owner-process implementation details:

- public Python `mapped_region_info()` always reports `host_data_ptr == 0`,
  and `host_signal_ptr == 0`;
- host-side public access uses `mapped_region_datacopy_h2region()` and
  `mapped_region_datacopy_region2h()` in all modes;
- L3 parent callers may receive `device_data_ptr` and `device_signal_ptr`,
  because those values are task/kernel arguments rather than parent
  dereferenceable addresses.

The parent-child mailbox is a host-side control and proxy transport. It is not
the CPU-NPU mapped-region primitive itself.

### Lifetime And Handle Ownership

Each opened region is registered in the owning `DeviceContext`. The public
handle is the child-side pointer value, carried as an opaque token by the L3
parent. The parent must never dereference that value.

The owner context registry is still required. All `info`, datacopy, notify,
wait, and close operations validate that the handle exists in the supplied
context. A handle from another context, a stale handle, or a double-close is
invalid.

`close_mapped_region` releases the owner-side host mapping and device
allocation, but it is not a device synchronization operation. The caller must
ensure no in-flight kernel, AICPU code, or other device participant still uses
`device_data_ptr` or `device_signal_ptr` before closing the region. Usually
that means waiting for task completion or for the protocol's completion signal
before close.

`close_mapped_region` does not inspect task state, signal state, or implicitly
wait for device work. In-flight device access at close time is a caller
protocol error.

`finalize_device` / `destroy_device_context` releases any still-registered
mapped regions as a resource cleanup fallback. That cleanup does not make close
safe while device code is still accessing the region.

## Public ABI

The runtime C ABI defines an opaque handle plus config and info structures in
`src/common/worker/pto_runtime_c_api.h`:

```cpp
typedef void *HostDeviceMappedRegionHandle;

typedef struct {
    uint64_t data_bytes;
    uint32_t signal_count;
    uint32_t flags;
} HostDeviceMappedRegionConfig;

typedef struct {
    uint64_t host_data_ptr;
    uint64_t device_data_ptr;
    uint64_t data_bytes;
    uint64_t host_signal_ptr;
    uint64_t device_signal_ptr;
    uint32_t signal_count;
    uint64_t total_bytes;
    uint32_t flags;
} HostDeviceMappedRegionInfo;
```

`flags` is reserved and must be `0` in this design. Non-zero flags are invalid.

The C ABI entry points are:

```cpp
int open_host_device_mapped_region_ctx(
    DeviceContextHandle ctx,
    const HostDeviceMappedRegionConfig *cfg,
    HostDeviceMappedRegionHandle *out_region
);
int close_host_device_mapped_region_ctx(
    DeviceContextHandle ctx,
    HostDeviceMappedRegionHandle region
);
int host_device_mapped_region_info_ctx(
    DeviceContextHandle ctx,
    HostDeviceMappedRegionHandle region,
    HostDeviceMappedRegionInfo *info
);
int host_device_mapped_region_datacopy_h2region_ctx(
    DeviceContextHandle ctx,
    HostDeviceMappedRegionHandle region,
    uint64_t offset,
    const void *src,
    size_t nbytes
);
int host_device_mapped_region_datacopy_region2h_ctx(
    DeviceContextHandle ctx,
    HostDeviceMappedRegionHandle region,
    uint64_t offset,
    void *dst,
    size_t nbytes
);
int host_device_mapped_region_notify_ctx(
    DeviceContextHandle ctx,
    HostDeviceMappedRegionHandle region,
    uint32_t signal_id,
    uint32_t value
);
int host_device_mapped_region_wait_ctx(
    DeviceContextHandle ctx,
    HostDeviceMappedRegionHandle region,
    uint32_t signal_id,
    uint32_t target,
    uint32_t timeout_us
);
```

`open_host_device_mapped_region_ctx` returns `0` on success and writes the
handle to `out_region`. On failure it writes `NULL` to `out_region` and returns
a negative errno-style code.

`close_host_device_mapped_region_ctx` takes the handle by value. It does not
clear caller-side storage. Reusing the same handle after close is invalid.

### Error Model

The C ABI uses negative errno-style return codes:

- `0`: success.
- `-EINVAL`: invalid context, handle, config, range, signal id, value, or null
  pointer.
- `-EAGAIN` / `-EWOULDBLOCK`: non-blocking wait miss or bounded wait timeout.
- `-ENOMEM`: allocation or wrapper-allocation failure.
- `-EIO`: backend mapping, datacopy, or signal failure.
- `-ENOTSUP`: unsupported platform or unsupported backend feature.

Python bindings map invalid user input to `ValueError`, wait miss or timeout
to `TimeoutError`, and backend, allocation, or unsupported-platform failures
to `RuntimeError`.

## Python API

The Python API exposes matching expert operations through the existing
`ChipWorker`, `Worker`, and `Orchestrator` control chain:

```python
region = worker.open_mapped_region(
    data_bytes,
    signal_count=2,
    flags=0,
    worker_id=0,
)
info = worker.mapped_region_info(region)

worker.mapped_region_datacopy_h2region(region, offset, data)
out = worker.mapped_region_datacopy_region2h(region, offset, nbytes)

worker.mapped_region_notify(region, signal_id, value)
worker.mapped_region_wait(region, signal_id, target, timeout_us)
worker.close_mapped_region(region)
```

`region` is a lightweight Python wrapper, not a naked integer. It records at
least:

- the opaque child-side handle value,
- the owning `worker_id`,
- whether the region is closed, and
- the opened `data_bytes`, `signal_count`, and `flags`.

Follow-up operations default to `region.worker_id`. If the caller explicitly
passes a different `worker_id`, Python raises `ValueError` before sending RPC.
Operations on a closed region also raise `ValueError`.

The first implementation uses manual close only. Context-manager syntax such as
`with worker.open_mapped_region(...)` is out of scope.

`mapped_region_info(region)` returns a structured `MappedRegionInfo` object,
not a dictionary. The fields mirror `HostDeviceMappedRegionInfo`, but public
Python host pointers are always masked:

```python
MappedRegionInfo(
    host_data_ptr=0,
    device_data_ptr=...,
    data_bytes=...,
    host_signal_ptr=0,
    device_signal_ptr=...,
    signal_count=...,
    total_bytes=...,
    flags=...,
)
```

The primitive returns bare device pointers only. It does not construct
`ContinuousTensor`, `TaskArgs`, queue metadata, or message descriptors. If a
kernel expects tensor metadata, a higher layer may wrap
`info.device_data_ptr + offset` in `ContinuousTensor(..., child_memory=True)`.

`mapped_region_datacopy_h2region(region, offset, data)` accepts readable,
C-contiguous bytes-like objects through the Python buffer protocol. `str` is
rejected; callers must encode explicitly. Non-contiguous buffers are invalid.

`mapped_region_datacopy_region2h(region, offset, nbytes)` returns a new
`bytes` object. The first implementation does not support writing into a
caller-provided mutable buffer.

## Region Layout

The region has a fixed internal layout:

```text
HostDeviceMappedRegionHeader
HostDeviceMappedRegionSignalSlot[signal_count]
data[data_bytes]
padding to 64B
```

Header:

```cpp
static constexpr uint32_t HDMR_MAGIC = 0x48444D52U;  // "HDMR"
static constexpr uint32_t HDMR_VERSION = 1;

struct alignas(64) HostDeviceMappedRegionHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t flags;
    uint32_t signal_count;
    uint64_t signal_offset;
    uint64_t data_offset;
    uint64_t data_bytes;
    uint64_t total_bytes;
    uint64_t reserved[2];
};
```

Signal slot:

```cpp
struct alignas(64) HostDeviceMappedRegionSignalSlot {
    volatile uint32_t value;
    uint32_t reserved0;
    uint64_t reserved[7];
};
```

Required static layout checks:

```cpp
static_assert(sizeof(HostDeviceMappedRegionHeader) == 64);
static_assert(alignof(HostDeviceMappedRegionHeader) == 64);
static_assert(sizeof(HostDeviceMappedRegionSignalSlot) == 64);
static_assert(alignof(HostDeviceMappedRegionSignalSlot) == 64);
```

Each signal slot occupies one 64B cache line. This keeps independent signal
words from sharing a line and gives device code a conservative documented
layout. Signal values are `uint32_t`.

`HostDeviceMappedRegionInfo` is fixed at 64B:

```cpp
static_assert(offsetof(HostDeviceMappedRegionInfo, host_data_ptr) == 0);
static_assert(offsetof(HostDeviceMappedRegionInfo, device_data_ptr) == 8);
static_assert(offsetof(HostDeviceMappedRegionInfo, data_bytes) == 16);
static_assert(offsetof(HostDeviceMappedRegionInfo, host_signal_ptr) == 24);
static_assert(offsetof(HostDeviceMappedRegionInfo, device_signal_ptr) == 32);
static_assert(offsetof(HostDeviceMappedRegionInfo, signal_count) == 40);
static_assert(offsetof(HostDeviceMappedRegionInfo, total_bytes) == 48);
static_assert(offsetof(HostDeviceMappedRegionInfo, flags) == 56);
static_assert(sizeof(HostDeviceMappedRegionInfo) == 64);
```

Sizing:

```text
signal_offset = sizeof(HostDeviceMappedRegionHeader)
data_offset   = align64(signal_offset + signal_count * sizeof(SignalSlot))
total_bytes   = align64(data_offset + data_bytes)
```

Validation rules:

- `data_bytes > 0`
- `signal_count > 0`
- `flags == 0`
- `offset <= data_bytes`
- `nbytes <= data_bytes - offset`
- `signal_id < signal_count`
- all arithmetic is checked for overflow

This makes zero-length datacopy at `offset == data_bytes` valid while
rejecting non-zero copies at the end of the data area. Implementations should
use the subtraction form above rather than computing `offset + nbytes`, so
overflow cannot turn an out-of-range request into an in-range value.

A newly opened region is zero-initialized. Header reserved fields, signal
values, data bytes, and padding are all zero before the region is returned.

## L3 Control Transport

L3 parent calls proxy through the existing mailbox control path. Small control
operations use mailbox arguments. Structured replies and payload-bearing
operations use POSIX `SharedMemory` side-band.

The mailbox control argument layout is extended to support four input values:

```text
CTRL_OFF_ARG0   = 16
CTRL_OFF_ARG1   = 24
CTRL_OFF_ARG2   = 32
CTRL_OFF_ARG3   = 40
CTRL_OFF_RESULT = 48
```

This moves `CTRL_OFF_RESULT` from offset 40 to 48. Existing control commands
must be updated on both C++ and Python sides.

Mailbox-argument operations:

- `open_mapped_region`
- `close_mapped_region`
- `mapped_region_notify`
- `mapped_region_wait`

Side-band operations:

- `mapped_region_info`
- `mapped_region_datacopy_h2region`
- `mapped_region_datacopy_region2h`

`open_mapped_region` uses:

```text
ARG0 = data_bytes
ARG1 = signal_count
ARG2 = flags
RESULT = region_handle
```

`close_mapped_region` uses:

```text
ARG0 = region_handle
```

`mapped_region_notify` uses:

```text
ARG0 = region_handle
ARG1 = signal_id
ARG2 = value
```

`mapped_region_wait` uses:

```text
ARG0 = region_handle
ARG1 = signal_id
ARG2 = target
ARG3 = timeout_us
```

### Side-Band Header

The side-band request/reply memory has a fixed little-endian 64B header. The
mailbox carries only the control command and the NUL-terminated shared-memory
name.

```text
magic      u32  "HMRD"
version    u16  1
op         u16  1=info, 2=h2region, 3=region2h
region     u64  opaque handle value
offset     u64
nbytes     u64
status     i32  child writes 0 or negative errno
reserved   u32
reserved2       zero padding to 64B
payload         starts at offset 64
```

For `mapped_region_info`, the child writes a 64B
`HostDeviceMappedRegionInfo` payload after the header and masks
`host_data_ptr` and `host_signal_ptr` to zero before publishing
`CONTROL_DONE`.

For `mapped_region_datacopy_h2region`, the parent writes the payload before
sending the mailbox request. The child validates the request and copies from
the side-band payload to the owner-process mapped region.

For `mapped_region_datacopy_region2h`, the parent creates a side-band segment
large enough for the header plus `nbytes`. The child validates the request,
copies from the mapped region into the side-band payload, writes `status`, and
then publishes `CONTROL_DONE`. The parent reads the payload only after
`CONTROL_DONE`.

The parent closes and unlinks the `SharedMemory` segment after the control
round trip. The child closes its mapping before publishing `CONTROL_DONE`.

## Datacopy Semantics

The datacopy APIs move raw bytes between a caller-provided host buffer and the
mapped region data area:

```text
datacopy_h2region:
    host buffer -> region data[offset:offset+nbytes]

datacopy_region2h:
    region data[offset:offset+nbytes] -> host buffer
```

In L3, host buffer transfer has two host-side steps:

```text
h2region:
    parent buffer -> SharedMemory payload -> child host mapping

region2h:
    child host mapping -> SharedMemory payload -> parent bytes
```

The side-band `SharedMemory` segment is only the L3 parent-to-child payload
transport. It does not provide CPU-NPU visibility semantics.

Datacopy does not wait, notify, check protocol phase, update ring metadata, or
construct tensor descriptors. Protocols compose the primitives explicitly. For
example:

```text
producer write = datacopy_h2region + notify
consumer read  = wait + datacopy_region2h
```

For public Python callers, host writes become publishable only through a
successful `mapped_region_datacopy_h2region` call. `mapped_region_notify` is a
release publication point for those completed datacopy writes. Direct host
pointer writes are not part of the public Python contract.

## Signal Semantics

Signal values are `uint32_t`. A signal slot is a lightweight doorbell or phase
word for bounded protocol epochs, not a long-lived channel sequence counter.
Higher-level protocols such as SPSC channels that need long-running head,
tail, or sequence values should define their own `uint64_t` metadata in the
mapped data area, including their own wrap-around rules.

```text
notify(signal_id, value)
    publish value to signal[signal_id]

wait(signal_id, target, timeout_us)
    complete when observed signal[signal_id] >= target
    otherwise return would-block after timeout policy
```

Signal values are monotonic within one protocol epoch. Host-side notify
rejects a value lower than the current signal value with `-EINVAL`. Device-side
signal publication must also be monotonic; violating this is a protocol error
and wait behavior is no longer guaranteed.

Wrap-around handling is out of scope.

All signal values are initialized to zero. Therefore
`wait(signal_id, target=0, timeout_us=0)` is a legal non-blocking probe that
immediately succeeds.

`wait` has only two modes:

- `timeout_us == 0`: non-blocking probe.
- `timeout_us > 0`: bounded wait.

There is no infinite wait mode.

`timeout_us` is a best-effort upper bound measured with a monotonic host clock.
The implementation may return later than requested because of host scheduler
latency. It must not report timeout before the deadline, except when
`timeout_us == 0`, which performs no waiting.

`device_signal_ptr` points to the device-visible signal slot array. Device code
may use that documented layout directly. There is no required device-side
helper API for the first implementation.

### Memory Ordering

`notify` is a release publication point for completed host datacopy writes.
`wait` is an acquire observation point for reads sequenced after it.

For CPU produces / NPU consumes:

```text
host datacopy_h2region(...)
host notify(signal_id, seq)
device acquire-poll signal_id >= seq
device reads data
```

If device code observes `signal_id >= seq`, device reads after that
observation must see host datacopy writes completed before the matching
`notify`.

For NPU produces / CPU consumes:

```text
device writes data
device release-stores signal_id = seq
host wait(signal_id, seq, timeout_us)
host datacopy_region2h(...)
```

If host wait succeeds, host datacopy reads after wait must see device writes
completed before the matching device signal publication.

The first a2a3 implementation uses atomic release/acquire operations on mapped
signal slots. If the real-NPU round-trip validation is flaky or fails, the
backend must not expose weaker best-effort semantics. It must either add the
required Ascend cache-maintenance primitives and document them, or return
`-ENOTSUP` for a2a3 onboard.

## Relationship To Task Tensor Payloads

Simpler's normal task tensor payload path is built around `TaskArgs` and
`ContinuousTensor`. It is task-scoped and tensor-oriented:

1. The user adds a `ContinuousTensor` to `TaskArgs`.
2. The task is dispatched to a chip child.
3. `init_runtime_impl` prepares device-side task arguments.
4. For ordinary tensors, the runtime allocates device memory and calls
   `copy_to_device()` from the host pointer in `ContinuousTensor.data`.
5. The runtime replaces the tensor's data pointer with the device pointer
   before launching device orchestration and kernels.
6. During validation/copy-back, recorded tensor pairs can be copied from
   device memory back to the original host pointer.

That path is convenient for normal kernel inputs and outputs. The runtime owns
the per-task tensor staging details, and the user describes tensors rather than
explicit data movement phases.

`child_memory=True` is an opt-out from that automatic staging path. When a
`ContinuousTensor` is marked as child memory, `init_runtime_impl` treats
`ContinuousTensor.data` as an existing child-managed device pointer. It passes
the tensor through without allocating new device memory and without
`copy_to_device()`. The caller is responsible for allocating and populating the
device buffer.

`HostDeviceMappedRegion` is different from both:

- Ordinary task tensor: `ContinuousTensor` host pointer, implicit
  `copy_to_device()` and copy-back around a task, task/runtime-managed
  lifetime, TensorMap and task dependencies for sync.
- `child_memory=True` tensor: existing device pointer, caller-managed H2D/D2H
  copies, caller/child-managed lifetime, TensorMap can still see the tensor
  argument.
- Mapped region: data offsets plus signal slots, explicit
  `datacopy_h2region` / `datacopy_region2h`, explicit open/close on a
  child-owned region, explicit notify/wait.

Mapped regions produce device addresses that can be passed through existing
task arguments. They do not require changing `TaskArgs`, `ContinuousTensor`,
TensorMap, or the normal `copy_to_device()` path.

If a mapped-region-backed tensor is submitted through `TaskArgs`, callers
should choose the appropriate tensor tag for scheduler behavior, usually
`NO_DEP` when synchronization is handled by the mapped region's signal
protocol.

## Platform Support

Supported platforms:

- `a2a3` onboard
- `a2a3sim`
- `a5sim`

Unsupported stub:

- `a5` onboard

Common implementation files:

- `src/common/worker/host_device_mapped_region.h`
- `src/common/worker/host_device_mapped_region.cpp`

Platform allocation and mapping live in existing runtime C ABI implementation
files:

- `src/a2a3/platform/onboard/host/pto_runtime_c_api.cpp`
- `src/a2a3/platform/sim/host/pto_runtime_c_api.cpp`
- `src/a5/platform/sim/host/pto_runtime_c_api.cpp`
- `src/a5/platform/onboard/host/pto_runtime_c_api.cpp`

The common implementation owns layout validation, bounds checks, datacopy,
host-side notify, and host-side wait. Platform code owns allocation, host
mapping, unmapping, and unsupported stubs.

### a2a3 Onboard Backend

The a2a3 onboard backend reuses the existing repo transport pattern:

```text
dev_ptr = DeviceRunner::allocate_tensor(total_bytes)
halHostRegister(dev_ptr, total_bytes, DEV_SVM_MAP_HOST, device_id, &host_ptr)
common_init(host_ptr, total_bytes, cfg)
```

`DeviceRunner::allocate_tensor` uses the platform `MemoryAllocator`, backed by
device memory allocation. `halHostRegister` is resolved from
`libascend_hal.so`, matching the existing profiling, tensor-dump, PMU, and
dep-gen paths.

`common_init` writes the header and zero-fills signal, data, reserved, and
padding bytes through the owner-process host mapping.

On close or cleanup, the backend unregisters the host mapping before freeing
device memory:

```text
halHostUnregister(host_ptr, device_id)
DeviceRunner::free_tensor(dev_ptr)
```

If `libascend_hal.so` cannot be loaded, the HAL symbols are missing, or
`halHostRegister` fails, open returns `-EIO`.

### Sim Backends

`a2a3sim` and `a5sim` may use ordinary host memory as both the host mapping and
device-visible pointer in the simulation address space. They must still follow
the same layout, validation, and signal semantics as onboard.

### a5 Onboard

`a5` onboard returns `-ENOTSUP` for open. It must export the ABI symbols as
explicit unsupported stubs rather than omitting them.

## Thin NPU Example

Add a minimal real-NPU round-trip example for `a2a3` onboard:

```text
host:
    data_bytes = 2 * N * sizeof(float)
    signal_count = 2
    open mapped region
    repeat seq in 1..10:
        datacopy input pattern(seq) into data[0:N]
        notify signal[0] = seq
        submit or let AIV observe device_data_ptr/device_signal_ptr

AIV:
    acquire-poll signal[0] until >= seq
    read input pattern(seq)
    output[i] = input[i] + add_const + seq_adjustment
    release-store signal[1] = seq

host:
    wait signal[1] >= seq with bounded timeout
    datacopy output from data[N:2N]
    verify exact expected bytes for seq
```

This proves:

- host datacopy into mapped region is visible to NPU,
- host notify is visible to NPU,
- NPU writes to mapped region are visible to host,
- NPU signal publication is visible to host wait, and
- device pointers returned by `mapped_region_info` can be passed to kernels.

The example reuses the same mapped region for all 10 iterations. Signal values
must increase monotonically. Input and output patterns must depend on `seq` so
stale reads cannot pass accidentally.

If this example is flaky or fails, `a2a3` onboard must not be marked supported
until the missing ordering or cache-maintenance mechanism is implemented and
documented.

## Tests

Common C++ unit tests:

- layout size, alignment, and `offsetof` assertions,
- invalid config handling,
- overflow and bounds checks,
- zero-initialization,
- datacopy h2region and region2h,
- host notify/wait,
- decreasing host notify rejection, and
- stale handle, double-close, and cross-context handle rejection.

Python sim tests:

- `a2a3sim` and `a5sim` smoke coverage,
- direct and L3 proxy paths,
- public Python info returns `host_*_ptr == 0`,
- info returns valid `device_*_ptr`,
- region wrapper rejects mismatched `worker_id`,
- closed region operations raise `ValueError`,
- h2region rejects `str` and non-contiguous buffers,
- region2h returns `bytes`,
- L3 datacopy payloads larger than mailbox capacity, and
- datacopy plus host notify/wait behavior.

Mailbox and side-band tests:

- mapped-region `wait` uses `ARG3`,
- existing mailbox controls still read result at `CTRL_OFF_RESULT == 48`,
- malformed side-band magic/version/op is rejected,
- side-band `status` propagates negative errno-style failures, and
- `info` side-band replies mask host pointers in the child.

Onboard example:

- run the 10-iteration `a2a3` real-NPU round trip described above,
- verify the NPU code demonstrates acquire poll and release store against the
  documented signal slot layout.

Error and lifetime tests:

- invalid config, stale handle, double-close, and cross-context handle use,
- non-blocking wait miss and bounded wait timeout map to timeout errors in
  Python,
- unsupported `a5` onboard stubs fail explicitly,
- `finalize_device` / `destroy_device_context` release regions that were not
  explicitly closed, and
- runtime shared objects export all mapped-region ABI symbols.

## Out Of Scope

`HostDeviceMappedRegion` does not define:

- migrating PR803 `HostDeviceMemory` or `HostDeviceChannel`,
- send/recv message protocols,
- ring/channel protocols,
- queue metadata,
- TensorMap dependency publication,
- automatic tensor descriptor construction,
- formal device-side helper APIs,
- multi-chip communication protocols,
- performance benchmarking contracts,
- `a5` onboard allocation/mapping behavior,
- context-manager support for the Python wrapper,
- infinite wait semantics,
- non-zero flag semantics, or
- signal wrap-around handling.
