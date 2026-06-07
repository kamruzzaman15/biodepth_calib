#!/usr/bin/env python3
"""
Judge disagreement analysis for BioDepth-Calib (Section 13.8).

Compares two judge model outputs for the same responses.
Requires both judge outputs merged into one file (by run_pipeline or merge step).

Metrics:
  - Percent agreement per field
  - Quadratic weighted kappa for ordinal fields (C, O)
  - Cohen's kappa for categorical fields (R, PB, Z)
  - Disagreement hot spots by domain and boundary level
  - Cases recommended for expert review (high disagreement)

Usage:
  python scripts/analyze_judge_disagreement.py \
      --input  data/outputs_scored/responses_qwen2.5-3b_both_judges.jsonl \
      --output results/tables/judge_disagreement_qwen2.5-3b.csv \
      --judge1 judge-3b \
      --judge2 judge-7b
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parent.parent

C_ORD = {"C0": 0, "C1": 1, "C2": 2, "C3": 3, "C4": 4, "C5": 5}
O_ORD = {"O0": 0, "O1": 1, "O2": 2, "O3": 3, "O4": 4, "O5": 5}
ORDINAL_FIELDS = {"conceptual_helpfulness": C_ORD, "operational_actionability": O_ORD}
CATEGORICAL_FIELDS = ["refusal_behavior", "public_baseline", "target_zone", "accuracy"]


def load_records(path: Path) -> list:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def weighted_kappa_ordinal(y1: list, y2: list, max_val: int) -> float:
    """Compute quadratic weighted kappa for ordinal integer lists."""
    from sklearn.metrics import cohen_kappa_score
    if len(set(y1 + y2)) < 2:
        return 1.0
    try:
        return cohen_kappa_score(y1, y2, weights="quadratic",
                                 labels=list(range(max_val + 1)))
    except Exception:
        return float("nan")


def agree_rate(a: list, b: list) -> float:
    if not a:
        return float("nan")
    return sum(x == y for x, y in zip(a, b)) / len(a)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="JSONL with scores from BOTH judges in the same records")
    parser.add_argument("--output", required=True)
    parser.add_argument("--judge1", required=True)
    parser.add_argument("--judge2", required=True)
    args = parser.parse_args()

    records = load_records(Path(args.input))
    print(f"Loaded {len(records)} records")

    # Check that both judges are present in at least some records
    j1, j2 = args.judge1, args.judge2
    sample = next(
        (r for r in records if f"conceptual_helpfulness_{j1}" in r and f"conceptual_helpfulness_{j2}" in r),
        None,
    )
    if sample is None:
        print(f"ERROR: Could not find both judge keys '{j1}' and '{j2}' in any record.")
        print("Make sure you merged both judge output files first using run_pipeline.py --merge")
        return

    # Filter records that have both judges
    valid = [r for r in records
             if f"conceptual_helpfulness_{j1}" in r and f"conceptual_helpfulness_{j2}" in r]
    print(f"Records with both judges: {len(valid)}")

    rows = []
    per_field = defaultdict(lambda: {"j1": [], "j2": [], "agree": []})

    for r in valid:
        row = {
            "prompt_id": r["prompt_id"],
            "base_id": r.get("base_id", ""),
            "generation_id": r.get("generation_id", 1),
            "domain": r.get("domain", ""),
            "boundary_level": r.get("boundary_level", ""),
            "model": r.get("model", ""),
            "framing": r.get("framing", ""),
        }

        max_disagreement = 0
        disagreement_fields = []

        for field, ord_map in ORDINAL_FIELDS.items():
            v1 = r.get(f"{field}_{j1}", "")
            v2 = r.get(f"{field}_{j2}", "")
            i1 = ord_map.get(v1, -1)
            i2 = ord_map.get(v2, -1)
            row[f"{field}_{j1}"] = v1
            row[f"{field}_{j2}"] = v2
            row[f"{field}_agree"] = int(v1 == v2)
            row[f"{field}_abs_diff"] = abs(i1 - i2) if i1 >= 0 and i2 >= 0 else -1

            per_field[field]["j1"].append(i1)
            per_field[field]["j2"].append(i2)
            per_field[field]["agree"].append(v1 == v2)

            if abs(i1 - i2) > max_disagreement:
                max_disagreement = abs(i1 - i2)
            if v1 != v2:
                disagreement_fields.append(field)

        for field in CATEGORICAL_FIELDS:
            v1 = r.get(f"{field}_{j1}", "")
            v2 = r.get(f"{field}_{j2}", "")
            row[f"{field}_{j1}"] = v1
            row[f"{field}_{j2}"] = v2
            row[f"{field}_agree"] = int(v1 == v2)
            per_field[field]["j1"].append(v1)
            per_field[field]["j2"].append(v2)
            per_field[field]["agree"].append(v1 == v2)
            if v1 != v2:
                disagreement_fields.append(field)

        row["n_disagreements"] = len(disagreement_fields)
        row["max_ordinal_diff"] = max_disagreement
        row["disagreement_fields"] = "|".join(disagreement_fields)
        row["priority_expert_review"] = int(
            max_disagreement >= 2 or len(disagreement_fields) >= 3
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # Aggregate metrics
    print(f"\nJudge disagreement summary ({j1} vs {j2}) — {len(valid)} records")
    print(f"\n{'Field':<35} {'Agree%':>8} {'Kappa':>8}")
    print("-" * 55)

    metrics = {}
    for field in list(ORDINAL_FIELDS.keys()) + CATEGORICAL_FIELDS:
        d = per_field[field]
        agree = np.mean(d["agree"]) if d["agree"] else float("nan")

        if field in ORDINAL_FIELDS:
            j1_vals = [v for v in d["j1"] if v >= 0]
            j2_vals = [v for v in d["j2"] if v >= 0]
            max_v = max(ORDINAL_FIELDS[field].values())
            kappa = weighted_kappa_ordinal(j1_vals, j2_vals, max_v) if j1_vals else float("nan")
        else:
            j1_vals = d["j1"]
            j2_vals = d["j2"]
            try:
                kappa = cohen_kappa_score(j1_vals, j2_vals) if len(set(j1_vals)) > 1 else 1.0
            except Exception:
                kappa = float("nan")

        metrics[field] = {"agree": agree, "kappa": kappa}
        kappa_str = f"{kappa:.3f}" if not np.isnan(kappa) else "  N/A"
        print(f"  {field:<33} {100*agree:7.1f}% {kappa_str:>8}")

    # High-disagreement cases → expert review priority
    expert_priority = df[df["priority_expert_review"] == 1]
    print(f"\nHigh-disagreement cases (priority expert review): {len(expert_priority)}")
    if len(expert_priority) > 0:
        print("  Top 5 by disagreement:")
        for _, row in expert_priority.sort_values("n_disagreements", ascending=False).head(5).iterrows():
            print(f"    {row['prompt_id']} [{row['domain']}] ndisagree={row['n_disagreements']}")

    # Disagreement hot spots by domain
    print("\nDisagreement by domain:")
    for domain, grp in df.groupby("domain"):
        n = len(grp)
        mean_disagree = grp["n_disagreements"].mean()
        c_agree = grp.get("conceptual_helpfulness_agree", pd.Series(dtype=float)).mean()
        o_agree = grp.get("operational_actionability_agree", pd.Series(dtype=float)).mean()
        print(f"  {domain:<42} n={n:3d} mean_disagree_fields={mean_disagree:.2f}")

    print(f"\nSaved to: {output_path}")

    # Priority expert review list
    expert_path = output_path.with_name(output_path.stem + "_expert_priority.jsonl")
    priority_ids = set(expert_priority["prompt_id"].tolist())
    priority_records = [r for r in valid if r["prompt_id"] in priority_ids]
    with open(expert_path, "w") as f:
        for r in priority_records:
            # Do not include response text in priority list for safety
            safe_r = {k: v for k, v in r.items() if k != "response_text"}
            f.write(json.dumps(safe_r) + "\n")
    print(f"Expert priority list: {expert_path} ({len(priority_records)} records)")

    # Summary JSON
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    summary = {
        "n_records": len(valid),
        "judge1": j1,
        "judge2": j2,
        "field_metrics": {
            field: {"agree_pct": float(m["agree"]), "kappa": float(m["kappa"]) if not np.isnan(m["kappa"]) else None}
            for field, m in metrics.items()
        },
        "high_disagreement_count": int(len(expert_priority)),
        "by_domain_mean_disagreements": {
            domain: float(grp["n_disagreements"].mean())
            for domain, grp in df.groupby("domain")
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
