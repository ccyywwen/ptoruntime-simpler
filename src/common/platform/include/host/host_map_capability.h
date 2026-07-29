/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
#ifndef SIMPLER_COMMON_PLATFORM_INCLUDE_HOST_HOST_MAP_CAPABILITY_H_
#define SIMPLER_COMMON_PLATFORM_INCLUDE_HOST_HOST_MAP_CAPABILITY_H_

#include <dlfcn.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>

namespace simpler::host_map {

static constexpr unsigned int DEV_SVM_MAP_HOST_FLAG = 2;

using HalHostRegisterFn = int (*)(void *dev_ptr, std::size_t size, unsigned int flags, int device_id, void **host_ptr);
using HalHostUnregisterFn = int (*)(void *dev_ptr, int device_id);

enum class CapabilityStatus : uint32_t {
    Supported = 0,
    Unsupported = 1,
    ProbeError = 2,
};

struct HostMapCapability {
    CapabilityStatus status{CapabilityStatus::ProbeError};
    bool hal_loaded{false};
    bool register_symbol_found{false};
    bool unregister_symbol_found{false};
    int register_rc{0};
    int unregister_rc{0};
    uintptr_t device_va{0};
    uintptr_t host_va{0};
    bool identity_va{false};
    bool cleanup_ok{true};
    std::string stage;
    std::string reason;
};

inline const char *status_name(CapabilityStatus status) {
    switch (status) {
        case CapabilityStatus::Supported:
            return "supported";
        case CapabilityStatus::Unsupported:
            return "unsupported";
        case CapabilityStatus::ProbeError:
            return "probe_error";
    }
    return "probe_error";
}

inline bool is_known_unsupported_rc(int rc) {
    switch (rc) {
        case 8:
        case 22:
        case 87:
            return true;
        default:
            return false;
    }
}

inline HostMapCapability classify_host_map_primitive_probe(
    bool hal_loaded, HalHostRegisterFn register_fn, HalHostUnregisterFn unregister_fn, void *dev_ptr, std::size_t bytes,
    int device_id
) {
    HostMapCapability result{};
    result.hal_loaded = hal_loaded;
    result.register_symbol_found = register_fn != nullptr;
    result.unregister_symbol_found = unregister_fn != nullptr;
    result.device_va = reinterpret_cast<uintptr_t>(dev_ptr);
    if (!hal_loaded) {
        result.status = CapabilityStatus::Unsupported;
        result.stage = "hal_load";
        result.reason = "libascend_hal load failed";
        return result;
    }
    if (register_fn == nullptr) {
        result.status = CapabilityStatus::Unsupported;
        result.stage = "symbol_lookup";
        result.reason = "halHostRegister symbol missing";
        return result;
    }
    if (unregister_fn == nullptr) {
        result.status = CapabilityStatus::Unsupported;
        result.stage = "symbol_lookup";
        result.reason = "halHostUnregister symbol missing";
        return result;
    }
    if (dev_ptr == nullptr || bytes == 0) {
        result.status = CapabilityStatus::ProbeError;
        result.stage = "allocation";
        result.reason = "probe allocation is invalid";
        return result;
    }

    void *host_ptr = nullptr;
    result.register_rc = register_fn(dev_ptr, bytes, DEV_SVM_MAP_HOST_FLAG, device_id, &host_ptr);
    result.host_va = reinterpret_cast<uintptr_t>(host_ptr);
    result.identity_va = host_ptr == dev_ptr && host_ptr != nullptr;
    if (result.register_rc != 0) {
        result.status = is_known_unsupported_rc(result.register_rc) ? CapabilityStatus::Unsupported
                                                                    : CapabilityStatus::ProbeError;
        result.stage = "register";
        result.reason = "halHostRegister rc=" + std::to_string(result.register_rc);
        return result;
    }
    if (host_ptr == nullptr) {
        result.status = CapabilityStatus::Unsupported;
        result.stage = "register";
        result.reason = "halHostRegister returned null host_va";
        return result;
    }

    result.unregister_rc = unregister_fn(dev_ptr, device_id);
    if (result.unregister_rc != 0) {
        result.status = CapabilityStatus::ProbeError;
        result.cleanup_ok = false;
        result.stage = "unregister";
        result.reason = "halHostUnregister rc=" + std::to_string(result.unregister_rc);
        return result;
    }

    result.status = CapabilityStatus::Supported;
    result.stage = "complete";
    result.reason = "host-map primitive supported";
    return result;
}

namespace detail {

static constexpr std::size_t kPrimitiveProbeBytes = 4096;
static constexpr int kAclSuccess = 0;
static constexpr int kAclErrorRepeatInitialize = 100002;
static constexpr int kAclMemMallocHugeFirst = 0;

using AclInitFn = int (*)(const char *);
using AclrtSetDeviceFn = int (*)(int);
using AclrtMallocFn = int (*)(void **, std::size_t, int);
using AclrtFreeFn = int (*)(void *);

struct HalRuntimeSymbols {
    bool loaded{false};
    HalHostRegisterFn register_fn{nullptr};
    HalHostUnregisterFn unregister_fn{nullptr};
    std::string error;
};

struct AclRuntimeSymbols {
    bool loaded{false};
    AclInitFn aclInit{nullptr};
    AclrtSetDeviceFn aclrtSetDevice{nullptr};
    AclrtMallocFn aclrtMalloc{nullptr};
    AclrtFreeFn aclrtFree{nullptr};
    std::string error;
};

inline void *resolve_symbol(void *lib, const char *name) {
    void *sym = dlsym(RTLD_DEFAULT, name);
    if (sym == nullptr && lib != nullptr) {
        sym = dlsym(lib, name);
    }
    return sym;
}

inline HalRuntimeSymbols &hal_runtime_symbols() {
    static HalRuntimeSymbols symbols = []() {
        HalRuntimeSymbols result{};
        void *lib = dlopen("libascend_hal.so", RTLD_NOW | RTLD_LOCAL);
        if (lib == nullptr) {
            const char *err = dlerror();
            result.error = err != nullptr ? err : "dlopen failed";
            return result;
        }
        result.loaded = true;
        result.register_fn = reinterpret_cast<HalHostRegisterFn>(resolve_symbol(lib, "halHostRegister"));
        result.unregister_fn = reinterpret_cast<HalHostUnregisterFn>(resolve_symbol(lib, "halHostUnregister"));
        return result;
    }();
    return symbols;
}

inline AclRuntimeSymbols &acl_runtime_symbols() {
    static AclRuntimeSymbols symbols = []() {
        AclRuntimeSymbols result{};
        void *lib = dlopen("libascendcl.so", RTLD_NOW | RTLD_LOCAL);
        if (lib == nullptr) {
            lib = dlopen("libascendcl.so.1", RTLD_NOW | RTLD_LOCAL);
        }
        if (lib == nullptr && dlsym(RTLD_DEFAULT, "aclrtMalloc") == nullptr) {
            const char *err = dlerror();
            result.error = err != nullptr ? err : "dlopen failed";
            return result;
        }
        result.loaded = true;
        result.aclInit = reinterpret_cast<AclInitFn>(resolve_symbol(lib, "aclInit"));
        result.aclrtSetDevice = reinterpret_cast<AclrtSetDeviceFn>(resolve_symbol(lib, "aclrtSetDevice"));
        result.aclrtMalloc = reinterpret_cast<AclrtMallocFn>(resolve_symbol(lib, "aclrtMalloc"));
        result.aclrtFree = reinterpret_cast<AclrtFreeFn>(resolve_symbol(lib, "aclrtFree"));
        return result;
    }();
    return symbols;
}

inline void require_acl_runtime(AclRuntimeSymbols &symbols) {
    if (!symbols.loaded) {
        throw std::runtime_error("failed to load libascendcl.so: " + symbols.error);
    }
    if (symbols.aclInit == nullptr) {
        throw std::runtime_error("aclInit symbol missing");
    }
    if (symbols.aclrtSetDevice == nullptr) {
        throw std::runtime_error("aclrtSetDevice symbol missing");
    }
    if (symbols.aclrtMalloc == nullptr) {
        throw std::runtime_error("aclrtMalloc symbol missing");
    }
    if (symbols.aclrtFree == nullptr) {
        throw std::runtime_error("aclrtFree symbol missing");
    }

    static std::once_flag init_once;
    static int init_rc = kAclSuccess;
    std::call_once(init_once, [&]() { init_rc = symbols.aclInit(nullptr); });
    if (init_rc != kAclSuccess && init_rc != kAclErrorRepeatInitialize) {
        throw std::runtime_error("aclInit failed with code " + std::to_string(init_rc));
    }
}

}  // namespace detail

inline HostMapCapability query_host_map_capability(int device_id) {
    detail::HalRuntimeSymbols &hal = detail::hal_runtime_symbols();
    void *dev_ptr = nullptr;
    bool allocated = false;
    HostMapCapability cap{};
    try {
        if (hal.loaded && hal.register_fn != nullptr && hal.unregister_fn != nullptr) {
            detail::AclRuntimeSymbols &acl = detail::acl_runtime_symbols();
            detail::require_acl_runtime(acl);
            int rc = acl.aclrtSetDevice(device_id);
            if (rc != detail::kAclSuccess) {
                throw std::runtime_error("aclrtSetDevice failed with code " + std::to_string(rc));
            }
            rc = acl.aclrtMalloc(&dev_ptr, detail::kPrimitiveProbeBytes, detail::kAclMemMallocHugeFirst);
            if (rc != detail::kAclSuccess) {
                throw std::runtime_error("aclrtMalloc failed with code " + std::to_string(rc));
            }
            if (dev_ptr == nullptr) {
                throw std::runtime_error("aclrtMalloc returned null");
            }
            allocated = true;
        }
        cap = classify_host_map_primitive_probe(
            hal.loaded, hal.register_fn, hal.unregister_fn, dev_ptr, detail::kPrimitiveProbeBytes, device_id
        );
    } catch (const std::exception &exc) {
        cap.status = CapabilityStatus::ProbeError;
        cap.hal_loaded = hal.loaded;
        cap.register_symbol_found = hal.register_fn != nullptr;
        cap.unregister_symbol_found = hal.unregister_fn != nullptr;
        cap.device_va = reinterpret_cast<uintptr_t>(dev_ptr);
        cap.stage = allocated ? "probe" : "allocation";
        cap.reason = exc.what();
    }
    if (allocated) {
        int free_rc = detail::acl_runtime_symbols().aclrtFree(dev_ptr);
        if (free_rc != detail::kAclSuccess && cap.status == CapabilityStatus::Supported) {
            cap.status = CapabilityStatus::ProbeError;
            cap.cleanup_ok = false;
            cap.stage = "free";
            cap.reason = "aclrtFree rc=" + std::to_string(free_rc);
        }
    }
    return cap;
}

}  // namespace simpler::host_map

#endif  // SIMPLER_COMMON_PLATFORM_INCLUDE_HOST_HOST_MAP_CAPABILITY_H_
