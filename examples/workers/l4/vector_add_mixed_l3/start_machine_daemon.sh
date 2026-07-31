#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

: "${SIMPLER_VECTOR_ADD_MIXED_L3_ROLE:=machine}"
: "${SIMPLER_VECTOR_ADD_MIXED_L3_HOST:=0.0.0.0}"
: "${SIMPLER_VECTOR_ADD_MIXED_L3_PORT:=19073}"

cd "${ROOT_DIR}"
source .venv/bin/activate

echo "[vector-add-mixed-l3] role=${SIMPLER_VECTOR_ADD_MIXED_L3_ROLE}"
echo "[vector-add-mixed-l3] daemon=${SIMPLER_VECTOR_ADD_MIXED_L3_HOST}:${SIMPLER_VECTOR_ADD_MIXED_L3_PORT}"

exec python -m simpler.remote_l3_worker \
  --host "${SIMPLER_VECTOR_ADD_MIXED_L3_HOST}" \
  --port "${SIMPLER_VECTOR_ADD_MIXED_L3_PORT}"
