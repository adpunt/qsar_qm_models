#!/bin/bash
# Check status of validation rerun jobs 11448793-11448879

echo "=== Job Status Summary ==="
RUNNING=0
PENDING=0
COMPLETED=0
FAILED=0
TIMEOUT=0
OTHER=0

for JOBID in $(seq 11448793 11448879); do
    STATE=$(sacct -j $JOBID --format=State --noheader -P | head -1)
    case "$STATE" in
        RUNNING)   RUNNING=$((RUNNING + 1)) ;;
        PENDING)   PENDING=$((PENDING + 1)) ;;
        COMPLETED) COMPLETED=$((COMPLETED + 1)) ;;
        FAILED)    FAILED=$((FAILED + 1)) ;;
        TIMEOUT)   TIMEOUT=$((TIMEOUT + 1)) ;;
        *)         OTHER=$((OTHER + 1)) ;;
    esac
done

TOTAL=$((RUNNING + PENDING + COMPLETED + FAILED + TIMEOUT + OTHER))
echo "Total:     $TOTAL / 87"
echo "Completed: $COMPLETED"
echo "Running:   $RUNNING"
echo "Pending:   $PENDING"
echo "Failed:    $FAILED"
echo "Timeout:   $TIMEOUT"
echo "Other:     $OTHER"

# Show any failed/timed out jobs
if [ $FAILED -gt 0 ] || [ $TIMEOUT -gt 0 ]; then
    echo ""
    echo "=== Problem Jobs ==="
    for JOBID in $(seq 11448793 11448879); do
        STATE=$(sacct -j $JOBID --format=State --noheader -P | head -1)
        if [ "$STATE" = "FAILED" ] || [ "$STATE" = "TIMEOUT" ]; then
            NAME=$(sacct -j $JOBID --format=JobName --noheader -P | head -1)
            echo "  $JOBID  $STATE  $NAME"
        fi
    done
fi

if [ $COMPLETED -eq 87 ]; then
    echo ""
    echo "=== ALL 87 JOBS COMPLETED ==="
    echo "Ready to run: cd /data/stat-cadd/scat9264/KIRBy/tests && python /data/stat-cadd/scat9264/qsar_qm_models/slurm_scripts_validation_rerun/merge_results.py --dry-run"
fi
