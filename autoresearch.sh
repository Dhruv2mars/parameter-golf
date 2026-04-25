#!/bin/bash
# Parameter Golf Autoresearch - 30-min training runs on Kaggle
# Each run = push kernel → wait → extract val_bpb

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$REPO/experiments/kaggle_work"
OUT="$REPO/experiments/kaggle_results"
STATE="$REPO/experiments/kaggle_state.json"

# =============================================================================
# FAST PRE-CHECK (syntax, <1s)
# =============================================================================

python3 -m py_compile "$REPO/train_kaggle.py" 2>/dev/null || {
    echo "SYNTAX ERROR train_kaggle.py"
    exit 1
}

# =============================================================================
# RUN ID
# =============================================================================

RUN_ID="${RUN_ID:-pg-ar-$(date +%Y%m%d-%H%M%S)-$$}"
TIMEOUT="${TIMEOUT:-2400}"  # 40 min Kaggle timeout (gives buffer)

# =============================================================================
# LAUNCH KAGGLE KERNEL
# =============================================================================

cd "$REPO"

# Build kernel with current env vars baked in
"$REPO/experiments/pg_autoresearch.py" launch \
    --name "ar-$RUN_ID" \
    --env "RUN_ID=$RUN_ID" \
    --env "ITERATIONS=7000" \
    --env "MAX_WALLCLOCK_SECONDS=1800" \
    2>&1 | tee /tmp/ar_launch_$$.json

KERNEL_REF=$(python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('ref',''))" < /tmp/ar_launch_$$.json)

if [ -z "$KERNEL_REF" ]; then
    echo "ERROR: Failed to launch kernel"
    exit 1
fi

echo "[AR] Kernel launched: $KERNEL_REF"

# =============================================================================
# WAIT FOR COMPLETION
# =============================================================================

"$REPO/experiments/pg_autoresearch.py" status --ref "$KERNEL_REF" || true
echo "[AR] Waiting for completion (timeout=${TIMEOUT}s)..."

python3 -c "
import subprocess, json, time, sys
ref = '$KERNEL_REF'
timeout = $TIMEOUT
poll = 30
deadline = time.time() + timeout
while time.time() < deadline:
    cp = subprocess.run(['$REPO/experiments/pg_autoresearch.py', 'status', '--ref', ref], capture_output=True, text=True)
    status = cp.stdout.strip()
    print(f'[AR] status={status}', flush=True)
    if 'COMPLETE' in status or 'ERROR' in status:
        print(f'[AR] Done: {status}', flush=True)
        sys.exit(0)
    time.sleep(poll)
print('[AR] TIMEOUT', flush=True)
sys.exit(1)
" || {
    echo "[AR] Wait failed or timeout"
    exit 1
}

# =============================================================================
# COLLECT RESULTS
# =============================================================================

"$REPO/experiments/pg_autoresearch.py" collect \
    --ref "$KERNEL_REF" \
    --name "$RUN_ID" \
    2>&1 | tee /tmp/ar_result_$$.json

# =============================================================================
# EXTRACT METRICS FOR AUTORESEARCH
# =============================================================================

METRICS=$(python3 -c "
import json, sys
try:
    d = json.load(open('/tmp/ar_result_$$.json'))
    val_bpb = d.get('metric', d.get('val_bpb', d.get('best_val_bpb', 999.0)))
    best_val_bpb = d.get('best_val_bpb', val_bpb)
    ttt_val_bpb = d.get('ttt_val_bpb')
    steps = d.get('steps', 0)
    compressed = d.get('compressed_bytes')
    runtime = d.get('runtime_seconds')
    status = d.get('status', 'UNKNOWN')
    error = d.get('error', '')
    non_finite = d.get('non_finite', False)
    
    print(f'val_bpb={val_bpb}')
    print(f'best_val_bpb={best_val_bpb}')
    if ttt_val_bpb:
        print(f'ttt_val_bpb={ttt_val_bpb}')
    print(f'steps={steps}')
    if compressed:
        print(f'compressed_bytes={compressed}')
    if runtime:
        print(f'runtime_seconds={runtime}')
    print(f'status={status}')
    
    if error:
        print(f'ERROR: {error[:200]}', file=sys.stderr)
    if non_finite or status == 'ERROR':
        print('[AR] NON-FINITE or ERROR detected', file=sys.stderr)
        sys.exit(1)
except Exception as e:
    print(f'PARSE ERROR: {e}', file=sys.stderr)
    sys.exit(1)
" 2>&1)

METRIC_LINE=$(echo "$METRICS" | grep "^val_bpb=" || echo "val_bpb=999.0")
echo "[AR] METRIC $METRIC_LINE"
echo "$METRICS"

# Exit with success if we got a valid result
exit 0
