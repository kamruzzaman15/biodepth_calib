#!/bin/bash
# Run all post-generation pipeline steps.
# Called automatically once generate_responses.py finishes.
signal -s SIGHUP ignore 2>/dev/null || true

set -e
cd /home/kamruzzaman1/biosecurity/biodepth-calib

MODEL_SHORT="qwen2.5-3b"
JUDGE_SHORT="judge-3b"
JUDGE_MODEL="Qwen/Qwen2.5-3B-Instruct"

RAW="data/outputs_raw/responses_${MODEL_SHORT}.jsonl"
REFUSAL="data/outputs_raw/responses_${MODEL_SHORT}_refusal.jsonl"
BASELINE="data/outputs_raw/responses_${MODEL_SHORT}_baseline.jsonl"
SCORED="data/outputs_scored/responses_${MODEL_SHORT}_${JUDGE_SHORT}.jsonl"

echo "=== Step 1: Refusal detection ==="
python scripts/detect_refusals.py --input "$RAW" --output "$REFUSAL"

echo "=== Step 2: Public-baseline retrieval ==="
python scripts/retrieve_public_baseline.py --input "$REFUSAL" --output "$BASELINE"

echo "=== Step 3: Judge scoring ==="
python scripts/judge_responses.py \
    --input "$BASELINE" --output "$SCORED" \
    --judge_model "$JUDGE_MODEL" --judge_short "$JUDGE_SHORT" \
    --resume

echo "=== Step 4: Framing sensitivity analysis ==="
python scripts/analyze_framing.py \
    --input "$SCORED" \
    --output "results/tables/framing_${MODEL_SHORT}.csv" \
    --judge_short "$JUDGE_SHORT"

echo "=== Step 5: Stochastic stability analysis ==="
python scripts/analyze_stability.py \
    --input "$SCORED" \
    --output "results/tables/stability_${MODEL_SHORT}.csv" \
    --judge_short "$JUDGE_SHORT"

echo ""
echo "=== ALL STEPS COMPLETE ==="
echo "Results in: results/tables/"
