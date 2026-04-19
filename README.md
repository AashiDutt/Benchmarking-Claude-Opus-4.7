# Claude Opus 4.7 Memory Performance Benchmark

Tests how well models maintain context and constraints across a long
multi-step reasoning chain within a single session.

## The question this answers

Does the model remember what it established in step 1 when it gets to step 5?

## What it measures

Memory-focused metrics:

- **constraint_citations** — how often the model references step 1 constraints in steps 4–5  
- **step_coherence** — steps completed without contradicting earlier steps  
- **self_corrections** — explicit corrections (thinking blocks when available; otherwise response text on OpenRouter)  
- **memory_failure** — LLM-judge: did the model violate the step 1 constraint when it mattered?

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env
# Set OPENROUTER_API_KEY and/or ANTHROPIC_API_KEY (see .env.example)

# Question: does Opus 4.7 maintain context better than 4.6? (single task, cheap validation)
python benchmark.py --task T1 --axis model

# Question: does effort level change memory coherence? (hard tasks only)
python benchmark.py --axis effort

# Full study — all axes, all tasks
python benchmark.py
```

**OpenRouter:** Thinking blocks are not returned — `self_corrections` is scored from the visible answer. Axis 3 (task budget) is skipped unless you use the direct Anthropic API.

## Cost estimate

| Axis | Runs | Approx cost |
|------|------|-------------|
| Model (2 models × 5 tasks) | 10 | ~$1.0–1.2 |
| Effort (3 levels × 4 hard tasks) | 12 | varies |
| Budget (direct API only) | 6 | — |
| Full (`--axis all` on OpenRouter) | 22 | ~$2.5+ (effort + model; no budget axis) |

## Observations (OpenRouter runs, Apr 2026)

Runs used **OpenRouter** with Anthropic model slugs (`anthropic/claude-opus-4-7` vs `anthropic/claude-opus-4-6`), **high** effort on axis 1, and **standard / high / xhigh** on axis 2. Full runs: `20260419_054228`; standalone model sweep: `20260419_075724`.

### Axis 1 — Model comparison (Opus 4.7 vs 4.6, high effort)

| Run | Model | Pass@1 | mem_fail rate | Avg citations | Avg coherence | Avg self-corr | Avg latency | Avg cost |
|-----|--------|--------|---------------|---------------|---------------|---------------|-------------|----------|
| Full + model-only | **opus-4-7** | 100% | **40%** | 1.2–2.0 | 5.0 | 0.2–0.8 | ~45–50 s | ~$0.105–0.111 |
| Full + model-only | **opus-4-6** | 100% | **100%** | 1.8–2.2 | 5.0 | 1.0–2.4 | ~105–127 s | ~$0.147–0.172 |

**Takeaways:** On this harness, **4.7** showed a much lower **memory_failure** rate than **4.6** (judge flagged failures on every 4.6 row in the aggregate; 4.7 sat around 40%). **4.6** also ran **~2× slower** and **~40–50% higher cost** on average for the same tasks. Coherence sat at **5/5** for both — the stricter signal here is **mem_fail** and **citations**, not the coarse coherence score.

### Axis 2 — Effort calibration (Opus 4.7 only, hard/expert tasks)

Aggregates from run `20260419_054228`:

| Effort | Pass@1 | mem_fail rate | Avg citations | Avg self-corr | Avg latency | Avg cost |
|--------|--------|---------------|---------------|---------------|-------------|----------|
| standard | 100% | 60% | 1.4 | 0.2 | ~35.7 s | ~$0.099 |
| high | 100% | 80% | 1.4 | 0.4 | ~44.8 s | ~$0.119 |
| xhigh | 100% | 60% | **2.0** | 0.6 | ~47.9 s | ~$0.132 |

**Takeaways:** **xhigh** produced the **highest average constraint citations** (2.0) and more self-corrections than standard, with **mem_fail** matching standard (60%) while **high** looked worse (80%) on this sample — worth repeating with direct API + thinking for a cleaner effort signal. Cost and latency increase with effort as expected.

### Axis 3 — Task budget

Not exercised on OpenRouter (provider does not forward the task-budget beta). Use **direct Anthropic API** to benchmark that axis.

---

*Metrics depend on judge and heuristics; use these as directional, not ground truth.*

## Repository

[Benchmarking-Claude-Opus-4.7](https://github.com/AashiDutt/Benchmarking-Claude-Opus-4.7)
