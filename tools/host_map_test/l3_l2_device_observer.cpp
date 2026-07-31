/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <stdint.h>

#include "aicpu/device_time.h"
#include "aicpu/l3_l2_orch_endpoint.h"
#include "common/memory_barrier.h"
#include "common/platform_config.h"
#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

namespace {

constexpr int kExpectedArgCount = 12;
constexpr uint64_t kTimeoutTicks = PLATFORM_PROF_SYS_CNT_FREQ * 5ULL;

enum class ObserverMode : uint64_t {
    RawDeviceAddress = 0,
    RegionDescriptor = 1,
};

void report_endpoint_error(const L3L2OrchEndpoint &endpoint) {
    const L3L2EndpointError &err = endpoint.error();
    rt_report_fatal(
        PTO2_ERROR_EXPLICIT_ORCH_FATAL,
        "L3-L2 host-map observer endpoint error op=%s kind=%u region=%llu counter_addr=%llu counter_operand=%d "
        "observed_counter=%d msg=%s",
        l3_l2_endpoint_op_to_string(err.op), static_cast<unsigned>(err.kind),
        static_cast<unsigned long long>(err.region_id), static_cast<unsigned long long>(err.counter_addr),
        err.counter_operand, err.observed_counter, err.message
    );
}

uint32_t wait_tail_or_die(volatile uint32_t *tail, uint32_t expected_tail) {
    const uint64_t start = get_sys_cnt_aicpu();
    uint32_t observed = 0;
    do {
        observed = *tail;
        if (observed >= expected_tail) {
            rmb();
            return observed;
        }
    } while (get_sys_cnt_aicpu() - start < kTimeoutTicks);

    rt_report_fatal(
        PTO2_ERROR_EXPLICIT_ORCH_FATAL,
        "L3-L2 host-map observer tail timeout expected=%u observed=%u tail_addr=%llu",
        expected_tail, observed, static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(tail))
    );
    return observed;
}

void observe_payload_tail(
    uint64_t payload_addr, uint64_t tail_addr, uint64_t completion_addr, uint64_t expected_payload,
    uint32_t expected_tail
) {
    volatile uint32_t *tail = reinterpret_cast<volatile uint32_t *>(static_cast<uintptr_t>(tail_addr));
    (void)wait_tail_or_die(tail, expected_tail);

    volatile uint64_t *payload = reinterpret_cast<volatile uint64_t *>(static_cast<uintptr_t>(payload_addr));
    const uint64_t got = *payload;
    if (got != expected_payload) {
        rt_report_fatal(
            PTO2_ERROR_EXPLICIT_ORCH_FATAL,
            "L3-L2 host-map observer payload mismatch expected=%llu got=%llu payload_addr=%llu",
            static_cast<unsigned long long>(expected_payload), static_cast<unsigned long long>(got),
            static_cast<unsigned long long>(payload_addr)
        );
        return;
    }

    volatile uint32_t *completion = reinterpret_cast<volatile uint32_t *>(static_cast<uintptr_t>(completion_addr));
    *completion = expected_tail;
    wmb();
}

}  // namespace

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{.expected_arg_count = kExpectedArgCount};
}

__attribute__((visibility("default"))) void host_map_device_observer(const L2TaskArgs &orch_args) {
    const auto mode = static_cast<ObserverMode>(orch_args.scalar(0));
    const uint64_t payload_offset = orch_args.scalar(7);
    const uint64_t expected_payload = orch_args.scalar(8);
    const uint64_t tail_offset = orch_args.scalar(9);
    const uint64_t completion_offset = orch_args.scalar(10);
    const uint32_t expected_tail = static_cast<uint32_t>(orch_args.scalar(11));

    if (mode == ObserverMode::RawDeviceAddress) {
        const uint64_t base = orch_args.scalar(1);
        observe_payload_tail(base + payload_offset, base + tail_offset, base + completion_offset, expected_payload, expected_tail);
        return;
    }

    if (mode == ObserverMode::RegionDescriptor) {
        uint64_t desc_scalars[L3L2_ORCH_REGION_DESC_SCALAR_COUNT] = {
            orch_args.scalar(1), orch_args.scalar(2), orch_args.scalar(3),
            orch_args.scalar(4), orch_args.scalar(5), orch_args.scalar(6),
        };
        L3L2OrchEndpoint endpoint(desc_scalars, L3L2_ORCH_REGION_DESC_SCALAR_COUNT);
        if (endpoint.error().kind != L3L2EndpointErrorKind::NONE) {
            report_endpoint_error(endpoint);
            return;
        }

        L3L2OrchPayloadView payload{};
        if (!endpoint.payload_read(payload_offset, sizeof(uint64_t), payload)) {
            report_endpoint_error(endpoint);
            return;
        }
        uint64_t tail_addr = 0;
        uint64_t completion_addr = 0;
        if (!endpoint.counter_addr(tail_offset, tail_addr) || !endpoint.counter_addr(completion_offset, completion_addr)) {
            report_endpoint_error(endpoint);
            return;
        }
        observe_payload_tail(payload.gm_addr, tail_addr, completion_addr, expected_payload, expected_tail);
        return;
    }

    rt_report_fatal(
        PTO2_ERROR_EXPLICIT_ORCH_FATAL,
        "L3-L2 host-map observer invalid mode=%llu",
        static_cast<unsigned long long>(orch_args.scalar(0))
    );
}

}  // extern "C"
