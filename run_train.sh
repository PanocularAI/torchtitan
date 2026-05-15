#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# use envs as local overwrites for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./run_train.sh
#
# COMM_MODE options for debugging:
#
# 1. "fake_backend" - Dry-run mode for config validation without GPU execution
#    - Uses fake process groups (no actual communication)
#    - Runs on a single GPU without torchrun or NCCL initialization
#    - Useful for validating configuration and model setup
#    Example: NGPU=32 COMM_MODE="fake_backend" ./run_train.sh
#    Set RANK to simulate a nonzero global rank, for example RANK=16.
#
# 2. "local_tensor" - Single-GPU debugging mode with simulated multi-GPU behavior
#    - All communication and computation execute on a single shared GPU
#    - Simulates the full training workflow without actual distributed communication
#    - Useful for debugging distributed training logic locally
#    Example: NGPU=32 COMM_MODE="local_tensor" ./run_train.sh

NGPU=${NGPU:-"8"}
export LOG_RANK=${LOG_RANK:-0}
MODULE=${MODULE:-"llama3"}
CONFIG=${CONFIG:-"llama3_debugmodel"}
COMM_MODE=${COMM_MODE:-""}
FT_ENABLE=${FT_ENABLE:-"false"}

TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}

if [ -n "$COMM_MODE" ]; then
    # Communication mode specified: validate configuration or run in debug mode
    echo "Running with comm_mode=${COMM_MODE}"
    NGPU="${NGPU}" LOCAL_RANK=0 python3 -m torchtitan.train --module ${MODULE} --config ${CONFIG} "$@" --comm.mode=${COMM_MODE} --training.steps 1
else
    # Normal training with torchrun
    if [ "${FT_ENABLE,,}" == "true" ]; then
        
        : "${FT_REPLICA_ID:?FT_REPLICA_ID must be set when FT_ENABLE=true}"
        : "${FT_GROUP_SIZE:?FT_GROUP_SIZE must be set when FT_ENABLE=true}"
        : "${FT_SYNC_STEPS:?FT_SYNC_STEPS must be set when FT_ENABLE=true}"

        MASTER_ADDR=${MASTER_ADDR:-"localhost"}
        MASTER_PORT=${MASTER_PORT:-"0"}
        LOCAL_ADDR=${LOCAL_ADDR:-"localhost"}
        NNODES=${NNODES:-"1"}
        ISHOST=${ISHOST:-"true"}
        FT_NUM_FRAGMENTS=${FT_NUM_FRAGMENTS:-"1"}
        FT_PROCESS_GROUP=${FT_PROCESS_GROUP:-"gloo"}
        FT_PROCESS_GROUP_TIMEOUT_MS=${FT_PROCESS_GROUP_TIMEOUT_MS:-"10000"}
        FT_RANK_0_SYNC=${FT_RANK_0_SYNC:-"false"}

        FT_RANK_0_SYNC_FLAG=""
        if [ "${FT_RANK_0_SYNC,,}" == "true" ]; then
            FT_RANK_0_SYNC_FLAG="--fault_tolerance.rank0_synchronization_only"
        fi
        
        PYTORCH_ALLOC_CONF="expandable_segments:True" \
        TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
        torchrun --nproc_per_node=${NGPU} --nnodes=${NNODES} --rdzv_id 101 --rdzv_backend c10d \
        --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" --local_addr=${LOCAL_ADDR} \
        --rdzv-conf is_host=${ISHOST} --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
        -m torchtitan.train --module ${MODULE} --config ${CONFIG} \
        --fault_tolerance.enable \
        --fault_tolerance.replica_id=${FT_REPLICA_ID} \
        --fault_tolerance.group_size=${FT_GROUP_SIZE} \
        --fault_tolerance.sync_steps=${FT_SYNC_STEPS} \
        --fault_tolerance.num_fragments=${FT_NUM_FRAGMENTS} \
        --fault_tolerance.process_group=${FT_PROCESS_GROUP} \
        --fault_tolerance.process_group_timeout_ms=${FT_PROCESS_GROUP_TIMEOUT_MS} \
        ${FT_RANK_0_SYNC_FLAG} "$@"
    else
        PYTORCH_ALLOC_CONF="expandable_segments:True" \
        TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
        torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
        --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
        -m torchtitan.train --module ${MODULE} --config ${CONFIG} "$@"
    fi
fi
