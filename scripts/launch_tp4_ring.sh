#!/usr/bin/env bash
# Launch Tencent Hy4 (770B NVFP4 + DFly2) across 4x DGX Spark (GB10) in physical 200G RoCE ring order:
#   rank 0 <-> rank 1 <-> rank 2 <-> rank 3 <-> rank 0
set -euo pipefail

# Customize hostnames, bootstrap IP (rank 0), and management interface for your cluster:
HOSTS=("spark1" "spark2" "asus1" "asus2")
RANK0_IP="${RANK0_IP:-192.168.68.119}"
SOCKET_IF="${SOCKET_IF:-enP7s7}"
PORT="${PORT:-8895}"
CONTEXT="${CONTEXT:-45056}"
BIN="${BIN:-/home/spnichol/hy4-770b/bin/tensorfold-native}"
MODEL_BASE="${MODEL_BASE:-/home/spnichol/hy4-770b/Hy4-NVFP4-rank}"

for r in 0 1 2 3; do
  h="${HOSTS[$r]}"
  echo "Launching rank $r on $h..."
  ssh -o BatchMode=yes "$h" "nohup env \
    LD_LIBRARY_PATH=\$HOME/nccl-2.30.7:/usr/local/cuda/lib64 \
    NCCL_COMM_ID=${RANK0_IP}:29551 \
    NCCL_SOCKET_IFNAME=${SOCKET_IF} \
    NCCL_ALGO=Ring \
    NCCL_CROSS_NIC=1 \
    NCCL_IB_SUBNET_AWARE_ROUTING=1 \
    NCCL_P2P_DISABLE=1 \
    NCCL_SHM_DISABLE=1 \
    NCCL_MIN_NCHANNELS=4 \
    NCCL_MAX_NCHANNELS=4 \
    NCCL_DEBUG=WARN \
    TF_HY4_FP8=1 \
    TENSORFOLD_MEMORY_RESERVE_GIB=4 \
    ${BIN} serve --backend cuda --tp 4 --rank ${r} --host 0.0.0.0 --port ${PORT} --context ${CONTEXT} --name hy4 ${MODEL_BASE}${r} \
    > \$HOME/hy4-770b/serve-rank${r}.log 2>&1 < /dev/null &" &
done
wait
echo "All 4 ranks launched. Cold load takes ~6.5 minutes; check http://${RANK0_IP}:${PORT}/v1/models"
