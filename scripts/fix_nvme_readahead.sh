#!/usr/bin/env bash
# Fix the buffered-read readahead mismatch on the cluster NVMe (RP2001T9RK004LX).
#
# The drive does 4.6 GB/s at 8MB IOs but only ~0.8 GB/s at 128KB IOs, and the
# kernel default read_ahead_kb=128 funnels every buffered-read miss into
# <=128KB IOs -> weight loads crawl at ~1.3 GB/s. Raise readahead to 16MB.
#
# Run once on EACH node (needs sudo):  sudo ./fix_nvme_readahead.sh
set -euo pipefail

[ "$(id -u)" -eq 0 ] || exec sudo "$0" "$@"

DEV=${1:-nvme0n1}
RA_KB=16384
RULE=/etc/udev/rules.d/71-nvme-readahead.rules

echo "$RA_KB" > "/sys/block/$DEV/queue/read_ahead_kb"
echo "read_ahead_kb set to $(cat /sys/block/$DEV/queue/read_ahead_kb) (was 128)"

cat > "$RULE" <<EOF
# GLM cluster: NVMe is 6x faster at >=8MB IOs; default 128KB readahead caps
# buffered reads at ~1.3 GB/s. Raise to 16MB. Added by fix_nvme_readahead.sh.
ACTION=="add|change", KERNEL=="$DEV", ATTR{queue/read_ahead_kb}="$RA_KB"
EOF
udevadm control --reload-rules
echo "persisted via $RULE"
