# BioDepth-Calib: Evaluating LLM Alignment on Biosecurity Questions

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Status](https://img.shields.io/badge/status-Part%20I%20Complete-green.svg)

Can language models be both **helpful** and **safe** when answering biosecurity questions? 

**BioDepth-Calib** is an empirical framework to measure this tradeoff. We quantify how well LLMs handle defensive biosecurity Q&A without escalating to operational procedures.

## Quick Results

We evaluated **Qwen2.5-3B** and **Gemma-3-4B** on 100 prompts across 5 biosecurity domains:

| Metric | Result |
|--------|--------|
| Operational escalation flags (O4/O5) | **0 / 600 responses** ✓ |
| Public baseline alignment | **100%** ✓ |
| Educational helpfulness (C2+) | **99%+** ✓ |
| Refusal consistency | **0% flips across framings** ✓ |

**Verdict**: Both models are safe for deployment in educational and defensive contexts.

Read the full [Results Summary](RESULTS_SUMMARY.md) or [Technical Documentation](README.md).

---

## What This Project Does

**The Problem**: Biosecurity educators need LLMs that are both:
- **Helpful**: Provide accurate, nuanced explanations for defensive purposes
- **Safe**: Don't escalate to step-by-step procedures or novel implementation details

We had no rigorous way to measure this tradeoff—until now.

**The Solution**: A 6-dimensional evaluation rubric + automated pipeline that classifies LLM responses on:

1. **Accuracy** (A0–A3): Is it factually correct?
2. **Conceptual Helpfulness** (C0–C5): Educational value?
3. **Operational Actionability** (O0–O5): Risk beyond public baseline? ← *The key safety metric*
4. **Refusal Behavior** (R0–R5): Does it cooperate or over-refuse?
5. **Public Baseline** (PB-0–PB-4): Aligned with CDC/WHO/institutional guidance?
6. **Target Zone** (Z-A–Z-E): What's the recommended response type?

---

## Installation

### Requirements
- Python 3.9+
- CUDA-capable GPU (RTX 3090 or equivalent, 12GB+ VRAM recommended)
- 100GB+ free disk for model weights

### Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/biodepth-calib.git
cd biodepth-calib

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Models

Models are auto-downloaded from Hugging Face on first run:
- Qwen2.5-3B-Instruct: ~3.3 GB
- Gemma-3-4B-it: ~4.2 GB

---

## Quick Start

### Generate Responses

```bash
python scripts/generate_responses.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --model_short qwen2.5-3b
```

Output: `data/outputs_raw/responses_qwen2.5-3b.jsonl` (300 responses)

### Run Full Pipeline (Generation → Scoring → Analysis)

```bash
python scripts/run_pipeline.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --model_short qwen2.5-3b \
    --judge_model Qwen/Qwen2.5-3B-Instruct \
    --judge_short judge-3b
```

Outputs:
- `data/outputs_raw/responses_qwen2.5-3b.jsonl` — Raw model outputs
- `data/outputs_scored/responses_qwen2.5-3b_judge-3b.jsonl` — Scored responses
- `results/tables/framing_qwen2.5-3b.csv` — Framing sensitivity analysis
- `results/tables/stability_qwen2.5-3b.csv` — Stochastic stability analysis

### Compare Two Models

```bash
python scripts/run_pipeline.py \
    --model_short qwen2.5-3b \
    --merge_judges \
    --judge1 judge-3b \
    --judge2 judge-gemma
```

---

## Directory Structure

```
biodepth-calib/
├── README.md                           # This file
├── RESULTS_SUMMARY.md                  # Key findings
├── config.yaml                         # Pipeline configuration
├── requirements.txt                    # Dependencies
│
├── data/
│   ├── prompts_pilot.jsonl            # 100 base prompts (2 framings each)
│   ├── references/                     # Public baseline corpus (CDC, WHO, etc)
│   │   ├── biosafety_education.txt
│   │   ├── lab_incident_response.txt
│   │   ├── misinformation_response.txt
│   │   ├── public_health_communication.txt
│   │   └── surveillance_triage.txt
│   ├── outputs_raw/                    # Raw model outputs
│   │   ├── responses_qwen2.5-3b.jsonl
│   │   ├── responses_gemma3-4b.jsonl
│   │   └── *_baseline.jsonl, *_refusal.jsonl
│   └── outputs_scored/                 # Judge-scored responses
│       ├── responses_qwen2.5-3b_judge-3b.jsonl
│       └── responses_gemma3-4b_judge-gemma.jsonl
│
├── results/
│   ├── tables/
│   │   ├── framing_qwen2.5-3b.csv
│   │   ├── framing_gemma3-4b.csv
│   │   ├── stability_qwen2.5-3b.csv
│   │   └── stability_gemma3-4b.csv
│   └── logs/
│
└── scripts/
    ├── generate_responses.py            # Model inference
    ├── detect_refusals.py              # Pattern-based R0–R5 classification
    ├── retrieve_public_baseline.py     # Semantic similarity to references
    ├── judge_responses.py              # LLM-as-judge scoring
    ├── analyze_framing.py              # Framing sensitivity analysis
    ├── analyze_stability.py            # Stochastic stability analysis
    └── run_pipeline.py                 # Orchestrator
```

---

## Key Findings

### Safety Profile ✓

Both models achieved:
- **0 operational escalation flags** (no O4/O5 detected)
- **100% public baseline alignment** (all responses match CDC/WHO/institutional guidance)
- **0% refusal inconsistency** (behavior locked across different framings)

### Helpfulness ✓

- **99%+ educational value** (C2–C4 responses)
- **56–67% direct answers** (no unnecessary hedging)
- **0% over-refusal** (models cooperate on legitimate defensive questions)

### Stability ✓

- **Deterministic safety**: Refusal behavior identical across 3 independent generations
- **Operationally predictable**: O-scores vary <0.042 σ
- **Framing-robust**: Safety properties unaffected by prompt rewording

### Model Comparison

| Aspect | Qwen2.5-3B | Gemma-3-4B |
|--------|-----------|-----------|
| **Helpfulness** | 55.3% C4 (richer) | 44.3% C4 (conservative) |
| **Stability** | 0.042 O-variance | **0.036 O-variance** |
| **Framing sensitivity** | 0.396 C-delta | **0.326 C-delta** |
| **Best for** | Education | High-stakes |

→ [Full results](RESULTS_SUMMARY.md)

---

## Usage Examples

### Evaluate a Custom Model

```bash
python scripts/generate_responses.py \
    --model meta-llama/Llama-3-8B-Instruct \
    --model_short llama3-8b

python scripts/run_pipeline.py \
    --model_short llama3-8b \
    --skip_generate \
    --judge_model Qwen/Qwen2.5-3B-Instruct \
    --judge_short judge-3b
```

### Resume Interrupted Run

```bash
python scripts/run_pipeline.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --model_short qwen2.5-3b \
    --judge_model Qwen/Qwen2.5-3B-Instruct \
    --judge_short judge-3b \
    --resume
```

### Analyze Results

```python
import json
import pandas as pd

# Load scored responses
with open('data/outputs_scored/responses_qwen2.5-3b_judge-3b.jsonl') as f:
    responses = [json.loads(line) for line in f]

# Convert to DataFrame
df = pd.DataFrame(responses)

# Analyze by domain
domain_stats = df.groupby('domain')[['conceptual_helpfulness_judge-3b', 
                                      'operational_actionability_judge-3b']].mean()
print(domain_stats)

# Find high-actionability responses
high_action = df[df['operational_actionability_judge-3b'] == 'O3']
print(f"Found {len(high_action)} O3 responses")
```

---

## Citation

If you use BioDepth-Calib in your research, please cite:

```bibtex
@software{biodepth_calib_2026,
  title = {BioDepth-Calib: Empirical Evaluation Framework for LLM Biosecurity Alignment},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/biodepth-calib},
  note = {Part I: Response Classification}
}
```

---

## Roadmap: Part II

- [ ] Multi-judge disagreement analysis (Qwen vs Gemma vs Llama as judges)
- [ ] Extended domain evaluation (synthetic biology, pathogen surveillance)
- [ ] Adversarial red-teaming (jailbreak prompts, roleplay)
- [ ] Larger model comparison (7B, 13B, 70B models)
- [ ] Fine-tuning for improved C-scores with maintained O-floor

---

## Contributing

Contributions welcome! Areas of interest:

- Additional biosecurity domains
- New models to evaluate
- Improvements to refusal detection
- Enhanced visualization of results
- Dataset expansion

Please open an issue or submit a PR.

---

## License

MIT License — see [LICENSE](LICENSE) file for details.

---

## Contact & Citation

- **Questions?** Open an issue or email [your email]
- **Want to collaborate?** Email [your email]
- **Find a bug?** GitHub Issues

---

## Acknowledgments

- Reference corpus: CDC, WHO, institutional biosafety guidance
- Models: Qwen/Alibaba, Google (Gemma), Meta (Llama)
- Framework inspired by: AI safety evaluation practices, responsible disclosure principles

---

## Disclaimer

This project is for **defensive biosecurity research only**. It is not intended to enable harm. All models are evaluated to ensure they **do not** provide operational escalation for dangerous activities.

If you observe a model providing O4/O5 content (specific procedures beyond public baseline), please report it confidentially to [your contact].

---

**Last Updated**: June 2, 2026  
**Status**: Part I Complete | Part II in Planning
