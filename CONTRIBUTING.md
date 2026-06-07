# Contributing to BioDepth-Calib

Thanks for your interest in contributing! This document outlines how to get involved.

## Areas We're Looking For

- **New evaluation domains**: Synthetic biology, pathogen surveillance, lab safety protocols
- **Model evaluations**: Test additional models (Llama, Mistral, Claude, GPT, etc.)
- **Methodology improvements**: Better refusal detection, refined rubric scoring
- **Analysis tools**: Visualization, statistical tests, dataset utilities
- **Documentation**: Improved README, tutorials, examples

## Before You Start

1. Check [Issues](../../issues) — someone might already be working on it
2. Read the [technical documentation](README.md)
3. Understand the evaluation rubric (see Appendix in README.md)

## Development Workflow

### 1. Fork & Clone

```bash
git clone https://github.com/yourusername/biodepth-calib.git
cd biodepth-calib
```

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

Branch naming:
- `feature/` — new capability (e.g., `feature/llama-eval`)
- `fix/` — bug fix (e.g., `fix/json-parsing`)
- `docs/` — documentation (e.g., `docs/api-guide`)
- `analysis/` — new analysis script (e.g., `analysis/disagreement`)

### 3. Make Changes

- Add your code to the appropriate script or new script
- Update docstrings and comments
- Add unit tests if adding new functions

### 4. Test Locally

```bash
# Quick test on a small subset
python scripts/run_pipeline.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --model_short test-run \
    --judge_model Qwen/Qwen2.5-3B-Instruct \
    --judge_short judge-test \
    --n_samples 1
```

### 5. Submit PR

Include in your PR description:
- **What**: Brief description of the change
- **Why**: Motivation or problem it solves
- **How**: Approach taken
- **Results**: Any output or metrics (if applicable)

Example:

```
## What
Added support for evaluating Llama-3 models

## Why
Expanding model coverage beyond Qwen/Gemma; testing scaling laws

## How
- Modified generate_responses.py to handle llama-3 chat template
- Tested on 10 prompts with resume flag

## Results
✓ Generation: 10/10 successful, avg 2.3s/prompt
✓ Scoring: 0 parse failures
✓ Output: data/outputs_scored/responses_llama3-70b_judge-3b.jsonl (10 responses)
```

## Code Style

- **Python**: Follow PEP 8
- **Comments**: Only when WHY is non-obvious, not WHAT
- **Functions**: Include docstrings with inputs/outputs
- **Error handling**: Fail loudly (useful error messages)

Example:

```python
def assign_pb_tag(max_sim: float, response_text: str) -> tuple:
    """
    Assign public baseline tag based on similarity score.
    
    Args:
        max_sim: Highest cosine similarity to reference corpus
        response_text: Model response to classify
    
    Returns:
        (pb_tag, rationale): e.g., ("PB-2", "High similarity to public reference")
    """
    # Implementation...
```

## Testing Your Changes

If you're adding a new script or modifying the pipeline:

```bash
# Test on small subset (1 prompt, 1 generation, 1 framing)
python scripts/run_pipeline.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --model_short test \
    --judge_model Qwen/Qwen2.5-3B-Instruct \
    --judge_short judge-test \
    --n_samples 1
```

Verify:
- ✓ No errors/crashes
- ✓ Output files created
- ✓ JSON is valid
- ✓ Expected fields present

## Reporting Issues

Found a bug? Open an issue with:

1. **Description**: What's broken?
2. **Reproduction**: Exact command to reproduce
3. **Expected vs Actual**: What should happen vs what did
4. **Environment**: Model name, GPU, Python version

Example:

```
Title: Judge scoring produces NaN accuracy scores for Qwen output

Description:
When scoring Qwen2.5-3B responses, judge model sometimes outputs invalid 
JSON with NaN fields instead of accuracy scores.

Reproduction:
python scripts/judge_responses.py --input data/outputs_raw/responses_qwen2.5-3b_baseline.jsonl ...

Expected:
All 300 responses scored with valid A0–A3 / A-UNK labels

Actual:
149/300 scored; remaining have "accuracy": NaN in JSON output

Environment:
- GPU: RTX 3090
- Python: 3.9.16
- torch: 2.2.0
- transformers: 4.40.0
```

## Questions?

- Check [Issues](../../issues) for similar questions
- Review [technical docs](README.md) for methodology questions
- Email [your contact] for broader discussion

## Code of Conduct

- Be respectful and inclusive
- Assume good intent
- Focus on ideas, not personalities
- Support other contributors

Thanks for contributing! 🎉
