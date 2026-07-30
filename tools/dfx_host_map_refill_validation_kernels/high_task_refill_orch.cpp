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

#include <cstdint>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

namespace {

constexpr int32_t kFuncNoop = 0;
constexpr uint64_t kMaxTasks = 150000;

}  // namespace

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{.expected_arg_count = 2};
}

__attribute__((visibility("default"))) void dfx_host_map_refill_high_task_orch(const L2TaskArgs &orch_args) {
    const Tensor &input = orch_args.tensor(0).ref();
    uint64_t task_count = orch_args.scalar(0);
    if (task_count < 1 || task_count > kMaxTasks) {
        rt_report_fatal(
            PTO2_ERROR_INVALID_ARGS, "dfx_host_map_refill_high_task_orch: invalid task_count=%llu",
            static_cast<unsigned long long>(task_count)
        );
        return;
    }

    for (uint64_t i = 0; i < task_count; ++i) {
        L0TaskArgs args;
        args.add_input(input);
        rt_submit_aic_task(kFuncNoop, args);
    }
}

}  // extern "C"
