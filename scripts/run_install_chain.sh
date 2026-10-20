#!/bin/bash
# wait for stage 1 to finish, then run stage 2
while pgrep -f 'Rscript scripts/install_r_baselines.R' > /dev/null; do sleep 20; done
echo "=== stage 1 finished, stage 1 tail: ===" 
tail -16 /home/holiday01/2026_cm/logs/install_r.log
echo "=== starting stage 2 ==="
cd /home/holiday01/2026_cm && Rscript scripts/install_r_stage2.R
