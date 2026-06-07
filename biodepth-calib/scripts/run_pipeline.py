#!/usr/bin/env python3
"""
BioDepth-Calib Part I pipeline orchestrator.

Runs the full pipeline for one generation model + one judge model:
  1. generate_responses       → data/outputs_raw/responses_<model>.jsonl
  2. detect_refusals          → data/outputs_raw/responses_<model>_refusal.jsonl
  3. retrieve_public_baseline → data/outputs_raw/responses_<model>_baseline.jsonl
  4. judge_responses          → data/outputs_scored/responses_<model>_<judge>.jsonl
  5. analyze_framing          → results/tables/framing_<model>.csv
  6. analyze_stability        → results/tables/stability_<model>.csv

For two-judge disagreement analysis, run both judges then:
  python scripts/run_pipeline.py --merge_judges \
      --model_short qwen2.5-3b \
      --judge1 judge-3b --judge2 judge-7b

Usage examples:
  # Run generation + scoring for qwen2.5-3b with judge-3b
  python scripts/run_pipeline.py \
      --model Qwen/Qwen2.5-3B-Instruct --model_short qwen2.5-3b \
      --judge_model Qwen/Qwen2.5-3B-Instruct --judge_short judge-3b

  # Skip generation (already done), only judge + analyze
  python scripts/run_pipeline.py \
      --model_short qwen2.5-3b \
      --judge_model Qwen/Qwen2.5-7B-Instruct --judge_short judge-7b \
      --skip_generate

  # Merge two judge outputs and run disagreement analysis
  python scripts/run_pipeline.py \
      --model_short qwen2.5-3b \
      --merge_judges --judge1 judge-3b --judge2 judge-7b
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_step(cmd: list, step_name: str):
    print(f"\n{'='*60}")
    print(f"STEP: {step_name}")
    print(f"CMD:  {' '.join(cmd)}")
    print("=" * 60)
    # start_new_session=True puts the child in its own process group so it
    # is not killed by SIGHUP when the parent terminal closes
    result = subprocess.run(cmd, check=True, start_new_session=True)
    return result


def merge_judge_outputs(model_short: str, judge1: str, judge2: str):
    """
    Merge two judge output files into a single file with both sets of scores.
    Matches records by (prompt_id, generation_id).
    """
    scored_dir = ROOT / "data" / "outputs_scored"
    f1 = scored_dir / f"responses_{model_short}_{judge1}.jsonl"
    f2 = scored_dir / f"responses_{model_short}_{judge2}.jsonl"

    if not f1.exists():
        print(f"ERROR: {f1} not found")
        sys.exit(1)
    if not f2.exists():
        print(f"ERROR: {f2} not found")
        sys.exit(1)

    # Load judge2 scores indexed by (prompt_id, gen_id)
    j2_index = {}
    with open(f2) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                key = (rec["prompt_id"], rec["generation_id"])
                j2_index[key] = rec

    # Merge judge2 fields into judge1 records
    merged = []
    j2_fields_prefix = f"_{judge2}"
    with open(f1) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (rec["prompt_id"], rec["generation_id"])
            j2_rec = j2_index.get(key, {})
            for k, v in j2_rec.items():
                if k.endswith(f"_{judge2}"):
                    rec[k] = v
            merged.append(rec)

    out_path = scored_dir / f"responses_{model_short}_both_judges.jsonl"
    with open(out_path, "w") as f:
        for rec in merged:
            f.write(json.dumps(rec) + "\n")

    print(f"Merged {len(merged)} records → {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="BioDepth-Calib pipeline orchestrator")

    # Generation model
    parser.add_argument("--model", default=None,
                        help="HuggingFace model name for generation")
    parser.add_argument("--model_short", required=True,
                        help="Short label for generation model (e.g. qwen2.5-3b)")

    # Judge model
    parser.add_argument("--judge_model", default=None,
                        help="HuggingFace model name for judge")
    parser.add_argument("--judge_short", default=None,
                        help="Short label for judge (e.g. judge-3b)")

    # Skip flags
    parser.add_argument("--skip_generate", action="store_true",
                        help="Skip generation step (use existing outputs_raw file)")
    parser.add_argument("--skip_judge", action="store_true",
                        help="Skip judge step (use existing scored file)")
    parser.add_argument("--skip_analysis", action="store_true",
                        help="Skip analysis steps")
    parser.add_argument("--resume", action="store_true",
                        help="Pass --resume to generation and judge steps")

    # Merge mode
    parser.add_argument("--merge_judges", action="store_true",
                        help="Merge two judge outputs and run disagreement analysis")
    parser.add_argument("--judge1", default=None)
    parser.add_argument("--judge2", default=None)

    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--n_samples", type=int, default=None)
    args = parser.parse_args()

    scripts_dir = ROOT / "scripts"
    raw_dir = ROOT / "data" / "outputs_raw"
    scored_dir = ROOT / "data" / "outputs_scored"
    results_dir = ROOT / "results" / "tables"

    raw_dir.mkdir(parents=True, exist_ok=True)
    scored_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Merge mode ────────────────────────────────────────────────────────────
    if args.merge_judges:
        if not args.judge1 or not args.judge2:
            print("ERROR: --merge_judges requires --judge1 and --judge2")
            sys.exit(1)
        merged_path = merge_judge_outputs(args.model_short, args.judge1, args.judge2)

        run_step(
            [sys.executable, str(scripts_dir / "analyze_judge_disagreement.py"),
             "--input", str(merged_path),
             "--output", str(results_dir / f"judge_disagreement_{args.model_short}.csv"),
             "--judge1", args.judge1,
             "--judge2", args.judge2],
            "Disagreement analysis",
        )
        print("\nDisagreement analysis complete.")
        return

    # ── Normal pipeline ───────────────────────────────────────────────────────
    raw_file           = raw_dir     / f"responses_{args.model_short}.jsonl"
    refusal_file       = raw_dir     / f"responses_{args.model_short}_refusal.jsonl"
    baseline_file      = raw_dir     / f"responses_{args.model_short}_baseline.jsonl"

    # Step 1: Generate
    if not args.skip_generate:
        if not args.model:
            print("ERROR: --model required unless --skip_generate is set")
            sys.exit(1)
        gen_cmd = [
            sys.executable, str(scripts_dir / "generate_responses.py"),
            "--model", args.model,
            "--model_short", args.model_short,
            "--config", args.config,
        ]
        if args.n_samples:
            gen_cmd += ["--n_samples", str(args.n_samples)]
        if args.resume:
            gen_cmd.append("--resume")
        run_step(gen_cmd, "Generate responses")
    else:
        print(f"\nSkipping generation — using existing: {raw_file}")

    # Step 2: Refusal detection
    run_step(
        [sys.executable, str(scripts_dir / "detect_refusals.py"),
         "--input", str(raw_file),
         "--output", str(refusal_file)],
        "Refusal detection",
    )

    # Step 3: Public-baseline retrieval
    run_step(
        [sys.executable, str(scripts_dir / "retrieve_public_baseline.py"),
         "--input", str(refusal_file),
         "--output", str(baseline_file),
         "--config", args.config],
        "Public-baseline retrieval",
    )

    # Step 4: Judge scoring
    if not args.skip_judge:
        if not args.judge_model or not args.judge_short:
            print("ERROR: --judge_model and --judge_short required unless --skip_judge is set")
            sys.exit(1)
        judge_file = scored_dir / f"responses_{args.model_short}_{args.judge_short}.jsonl"
        judge_cmd = [
            sys.executable, str(scripts_dir / "judge_responses.py"),
            "--input", str(baseline_file),
            "--output", str(judge_file),
            "--judge_model", args.judge_model,
            "--judge_short", args.judge_short,
            "--config", args.config,
        ]
        if args.resume:
            judge_cmd.append("--resume")
        run_step(judge_cmd, "LLM judge scoring")
    else:
        judge_file = scored_dir / f"responses_{args.model_short}_{args.judge_short}.jsonl"
        print(f"\nSkipping judge — using existing: {judge_file}")

    # Step 5: Analysis
    if not args.skip_analysis and args.judge_short:
        run_step(
            [sys.executable, str(scripts_dir / "analyze_framing.py"),
             "--input", str(judge_file),
             "--output", str(results_dir / f"framing_{args.model_short}.csv"),
             "--judge_short", args.judge_short],
            "Framing sensitivity analysis",
        )

        run_step(
            [sys.executable, str(scripts_dir / "analyze_stability.py"),
             "--input", str(judge_file),
             "--output", str(results_dir / f"stability_{args.model_short}.csv"),
             "--judge_short", args.judge_short],
            "Stochastic stability analysis",
        )

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"  Raw responses:      {raw_file}")
    print(f"  With refusal tags:  {refusal_file}")
    print(f"  With baseline tags: {baseline_file}")
    if not args.skip_judge and args.judge_short:
        print(f"  Scored:             {scored_dir}/responses_{args.model_short}_{args.judge_short}.jsonl")
    print(f"  Results:            {results_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
