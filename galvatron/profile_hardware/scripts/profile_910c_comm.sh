#!/bin/bash
# Run on the Atlas 800I A3 after CANN env is loaded.
# One process per die. Do not count the MCU.
#
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   bash profile_910c_comm.sh
#
# Copy the four JSON files under hardware_configs/ back to the dev machine.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR/.."

NPROC=${NPROC:-16}
MASTER_PORT=${MASTER_PORT:-29500}

export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1800}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-1800}

if command -v torchrun >/dev/null 2>&1; then
  LAUNCH=(torchrun)
else
  LAUNCH=(python3 -m torch.distributed.run)
fi

"${LAUNCH[@]}" \
  --nproc_per_node="$NPROC" \
  --master_addr=127.0.0.1 \
  --master_port="$MASTER_PORT" \
  profile_910c_comm.py \
  --output-dir hardware_configs \
  --num-nodes 1 \
  --message-mb 256
