"""
benchmark.py — Claude Opus 4.7 Memory Performance Benchmark
=============================================================

Tests how well models maintain context and constraints across a long
multi-step reasoning chain within a single session.

Three comparison axes:
  Axis 1: Opus 4.7 vs Opus 4.6 — same effort, does 4.7 maintain context better?
  Axis 2: xhigh vs high vs standard effort on Opus 4.7 — does effort affect memory?
  Axis 3: task_budget vs no budget — does constraining tokens hurt coherence?
         (Axis 3 requires direct Anthropic API — skipped on OpenRouter)

Usage:
  python benchmark.py                        # full benchmark, all axes
  python benchmark.py --axis model           # axis 1 only
  python benchmark.py --axis effort          # axis 2 only
  python benchmark.py --task T1 --axis model # single task, single axis

OpenRouter:
  Set OPENROUTER_API_KEY in .env — script auto-detects.
  Note: OpenRouter strips thinking blocks — self_corrections falls back
  to response text scanning. Axis 3 (task_budget) is skipped.
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Optional

import anthropic
from dotenv import load_dotenv
from rich.console import Console
from tabulate import tabulate

sys.path.insert(0, os.path.dirname(__file__))
from benchmark_tasks import TASKS

load_dotenv()
console = Console()

# ── Constants ──────────────────────────────────────────────────────────────────

MODELS = {
    "opus_4_7": "claude-opus-4-7",
    "opus_4_6": "claude-opus-4-6",
}

OPENROUTER_MODELS = {
    "opus_4_7": "anthropic/claude-opus-4-7",
    "opus_4_6": "anthropic/claude-opus-4-6",
}

JUDGE_MODEL            = "claude-opus-4-6"
OPENROUTER_JUDGE_MODEL = "anthropic/claude-opus-4-6"

PRICING = {
    "claude-opus-4-7":           {"input": 5.00, "output": 25.00},
    "claude-opus-4-6":           {"input": 5.00, "output": 25.00},
    "anthropic/claude-opus-4-7": {"input": 5.00, "output": 25.00},
    "anthropic/claude-opus-4-6": {"input": 5.00, "output": 25.00},
}

EFFORT_LEVELS       = ["standard", "high", "xhigh"]
RESULTS_DIR         = os.getenv("RESULTS_DIR", "./results")
OPENROUTER_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://openrouter.ai/api")
HTTP_TIMEOUT_S      = float(os.getenv("BENCHMARK_HTTP_TIMEOUT_S", "3600"))

os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Client factory ─────────────────────────────────────────────────────────────

def make_client():
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    anthropic_key  = os.getenv("ANTHROPIC_API_KEY")

    if openrouter_key:
        console.print("[cyan]Provider: OpenRouter[/cyan]")
        console.print(
            "[yellow]Note: thinking blocks stripped by OpenRouter — "
            "self_corrections scored from response text. "
            "Axis 3 (task_budget) will be skipped.[/yellow]\n"
        )
        saved = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            client = anthropic.Anthropic(
                api_key=None,
                auth_token=openrouter_key,
                base_url=OPENROUTER_BASE_URL,
            )
        finally:
            if saved is not None:
                os.environ["ANTHROPIC_API_KEY"] = saved
        return client, OPENROUTER_MODELS, OPENROUTER_JUDGE_MODEL, True

    if anthropic_key:
        console.print("[cyan]Provider: Anthropic (direct)[/cyan]\n")
        client = anthropic.Anthropic(api_key=anthropic_key)
        return client, MODELS, JUDGE_MODEL, False

    console.print("[red]No API key found. Set OPENROUTER_API_KEY or ANTHROPIC_API_KEY in .env[/red]")
    sys.exit(1)


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class TaskResult:
    task_id: str
    task_title: str
    model: str
    effort: str
    task_budget: Optional[int]
    constraint_citations: int
    step_coherence_score: int
    self_corrections: int
    memory_failure: bool
    pass_at_1: bool
    latency_s: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    run_id: str
    timestamp: str
    response_excerpt: str


# ── Scoring ────────────────────────────────────────────────────────────────────

def judge_memory_failure(client, judge_model, task, response_text):
    prompt = f"""You are evaluating a model's response to a multi-step engineering problem.

The task established this constraint in Step 1:
"{task['constraint_to_track']}"

The known memory failure pattern for this task is:
"{task['memory_failure_signal']}"

Here is the model's full response:
---
{response_text[:6000]}
---

Did the model violate the Step 1 constraint in its Step 4 or Step 5 answer,
OR fail to reference it when directly relevant?

Answer with exactly one word: YES or NO.
"""
    try:
        resp = client.messages.create(
            model=judge_model,
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
            timeout=min(120.0, HTTP_TIMEOUT_S),
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        return text.strip().upper().startswith("YES")
    except Exception as e:
        console.print(f"    [yellow]Judge call failed ({e}), defaulting NO[/yellow]")
        return False


def score_response(task, response_text, thinking_text, client, judge_model):
    response_lower = response_text.lower()
    thinking_lower = (thinking_text or "").lower()
    hints          = task["verification_hints"]
    constraint     = task["constraint_to_track"].lower()

    constraint_keywords  = [w for w in constraint.split() if len(w) > 4]
    constraint_citations = sum(1 for kw in constraint_keywords if response_lower.count(kw) > 0)

    memory_failure = judge_memory_failure(client, judge_model, task, response_text)

    hints_found = sum(
        1 for hint in hints
        if any(kw in response_lower for kw in hint.lower().split()[:4])
    )
    step_coherence_score = min(hints_found + 1, task["expected_steps"])

    correction_phrases = [
        "wait,", "actually,", "let me reconsider", "i made an error",
        "that's wrong", "i need to correct", "on second thought",
        "let me re-examine", "i was wrong", "correction:", "mistake:",
        "i realize", "i should clarify", "to correct myself",
    ]
    thinking_available = bool(thinking_lower.strip())
    search_text        = thinking_lower if thinking_available else response_lower
    self_corrections   = sum(1 for p in correction_phrases if p in search_text)

    return {
        "constraint_citations":  constraint_citations,
        "step_coherence_score":  step_coherence_score,
        "self_corrections":      self_corrections,
        "memory_failure":        memory_failure,
        "pass_at_1":             step_coherence_score >= task["expected_steps"] - 1,
        "thinking_available":    thinking_available,
    }


def calculate_cost(model, input_tokens, output_tokens):
    p = PRICING.get(model, {"input": 5.00, "output": 25.00})
    return round((input_tokens / 1e6) * p["input"] + (output_tokens / 1e6) * p["output"], 6)


_EFFORT_SYSTEM_NOTES = {
    "standard": "Apply moderate reasoning depth — be concise but still complete every step.",
    "high":     "Apply strong reasoning depth — check each step carefully before moving on.",
    "xhigh":    "Apply maximum reasoning depth — be exhaustive; self-check and correct when needed.",
    "max":      "Apply maximum reasoning depth — be exhaustive; self-check and correct when needed.",
}


def _system_with_effort(base, effort):
    note = _EFFORT_SYSTEM_NOTES.get(effort, _EFFORT_SYSTEM_NOTES["high"])
    return f"{base}\n\nEffort setting for this run: {effort}. {note}"


def _api_effort(effort):
    return {"standard": "medium", "high": "high", "xhigh": "xhigh", "max": "max"}.get(effort, "high")


# ── Core runner ────────────────────────────────────────────────────────────────

def run_task(client, task, model, effort, judge_model, using_openrouter,
             task_budget=None, run_id=""):

    system_base = """You are a senior software engineer performing an adversarial
code review. You will work through a multi-step problem.

Critical instructions:
- Complete every step in sequence. Do not skip steps.
- Each step may reference findings from earlier steps. Do so explicitly.
- If a step asks you to evaluate something against earlier constraints,
  name those constraints explicitly.
- If you catch an error in your own reasoning, correct it and say so clearly.
- Be precise. Vague answers will miss the bugs.
"""
    system_prompt = _system_with_effort(system_base, effort)
    max_tokens    = 64000 if effort in ("xhigh", "max") else 32000
    use_beta      = task_budget is not None and not using_openrouter
    start_time    = time.time()

    try:
        base_kwargs = dict(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": task["description"]}],
        )

        # ── KEY FIX: pass output_config and thinking as top-level kwargs.
        # extra_body is NOT forwarded by messages.stream() to the API request
        # body — only top-level kwargs reach the wire. This was the root cause
        # of thinking blocks being empty and effort having no effect.
        if not using_openrouter:
            base_kwargs["thinking"]      = {"type": "adaptive"}
            base_kwargs["output_config"] = {"effort": _api_effort(effort)}
            if use_beta:
                base_kwargs["output_config"]["task_budget"] = {
                    "type": "tokens",
                    "total": task_budget,
                }

        if use_beta:
            stream_ctx = client.beta.messages.stream(
                **base_kwargs,
                betas=["task-budgets-2026-03-13"],
                timeout=HTTP_TIMEOUT_S,
            )
        else:
            if task_budget and using_openrouter:
                console.print("    [yellow]task_budget skipped — not supported on OpenRouter[/yellow]")
            stream_ctx = client.messages.stream(**base_kwargs, timeout=HTTP_TIMEOUT_S)

        response_text = ""
        thinking_text = ""

        with stream_ctx as stream:
            final = stream.get_final_message()

        latency = time.time() - start_time

        for block in final.content:
            if block.type == "thinking":
                thinking_text += getattr(block, "thinking", "") or ""
            elif block.type == "text":
                response_text += block.text

        input_tokens  = final.usage.input_tokens
        output_tokens = final.usage.output_tokens

        if not thinking_text.strip():
            console.print(
                "    [dim]thinking blocks empty — "
                "self_corrections scored from response text[/dim]"
            )
        else:
            console.print(
                f"    [green]thinking blocks present — "
                f"{len(thinking_text)} chars[/green]"
            )

        cost   = calculate_cost(model, input_tokens, output_tokens)
        scores = score_response(task, response_text, thinking_text, client, judge_model)

        return TaskResult(
            task_id=task["id"], task_title=task["title"],
            model=model, effort=effort, task_budget=task_budget,
            constraint_citations=scores["constraint_citations"],
            step_coherence_score=scores["step_coherence_score"],
            self_corrections=scores["self_corrections"],
            memory_failure=scores["memory_failure"],
            pass_at_1=scores["pass_at_1"],
            latency_s=round(latency, 2),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            run_id=run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            response_excerpt=response_text[:300].replace("\n", " "),
        )

    except Exception as e:
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
        console.print(f"[red]Error on {task['id']} ({model}/{effort}): {e}[/red]")
        return TaskResult(
            task_id=task["id"], task_title=task["title"],
            model=model, effort=effort, task_budget=task_budget,
            constraint_citations=0, step_coherence_score=0,
            self_corrections=0, memory_failure=True, pass_at_1=False,
            latency_s=round(time.time() - start_time, 2),
            input_tokens=0, output_tokens=0, cost_usd=0.0,
            run_id=run_id, timestamp=datetime.now(timezone.utc).isoformat(),
            response_excerpt=f"ERROR: {str(e)[:200]}",
        )


# ── Benchmark axes ─────────────────────────────────────────────────────────────

def run_axis_model(client, model_map, judge_model, using_openrouter, tasks, run_id):
    console.print("\n[bold cyan]Axis 1: Model Comparison — Opus 4.7 vs Opus 4.6 (high effort)[/bold cyan]")
    results = []
    for task in tasks:
        for key, model_id in model_map.items():
            console.print(f"  [{key}] {task['id']}: {task['title'][:55]}...")
            r = run_task(client, task, model_id, "high", judge_model, using_openrouter, run_id=run_id)
            results.append(r)
            console.print(
                f"    citations={r.constraint_citations} | "
                f"coherence={r.step_coherence_score}/{task['expected_steps']} | "
                f"corrections={r.self_corrections} | "
                f"mem_fail={'✗' if r.memory_failure else '✓'} | "
                f"latency={r.latency_s}s | cost=${r.cost_usd:.4f}"
            )
    return results


def run_axis_effort(client, model_map, judge_model, using_openrouter, tasks, run_id):
    console.print("\n[bold cyan]Axis 2: Effort Calibration — xhigh vs high vs standard (Opus 4.7)[/bold cyan]")
    results    = []
    hard_tasks = [t for t in tasks if t["difficulty"] in ("hard", "expert")]
    opus_47    = model_map["opus_4_7"]
    for task in hard_tasks:
        for effort in EFFORT_LEVELS:
            console.print(f"  [effort={effort}] {task['id']}: {task['title'][:55]}...")
            r = run_task(client, task, opus_47, effort, judge_model, using_openrouter, run_id=run_id)
            results.append(r)
            console.print(
                f"    citations={r.constraint_citations} | "
                f"coherence={r.step_coherence_score}/{task['expected_steps']} | "
                f"corrections={r.self_corrections} | "
                f"latency={r.latency_s}s | cost=${r.cost_usd:.4f}"
            )
    return results


def run_axis_budget(client, model_map, judge_model, using_openrouter, tasks, run_id):
    if using_openrouter:
        console.print(
            "\n[yellow]Axis 3 skipped — task_budget beta headers not forwarded "
            "by OpenRouter. Use direct Anthropic API to test this axis.[/yellow]"
        )
        return []

    console.print(
        "\n[bold cyan]Axis 3: Task Budget — "
        "does constraining tokens hurt memory coherence? (Opus 4.7 xhigh)[/bold cyan]"
    )
    results    = []
    opus_47    = model_map["opus_4_7"]
    test_tasks = [t for t in tasks if t["id"] in ("T1", "T3")]
    for task in test_tasks:
        for budget in [None, 64000, 32000]:
            label = f"budget={budget}" if budget else "no_budget"
            console.print(f"  [{label}] {task['id']}: {task['title'][:55]}...")
            r = run_task(
                client, task, opus_47, "xhigh", judge_model, using_openrouter,
                task_budget=budget, run_id=run_id,
            )
            results.append(r)
            console.print(
                f"    citations={r.constraint_citations} | "
                f"coherence={r.step_coherence_score}/{task['expected_steps']} | "
                f"mem_fail={'✗' if r.memory_failure else '✓'} | "
                f"latency={r.latency_s}s | cost=${r.cost_usd:.4f}"
            )
    return results


# ── Reporting ──────────────────────────────────────────────────────────────────

def short_model(model_id):
    return model_id.replace("anthropic/claude-", "").replace("claude-", "")


def print_table(results, title, tasks):
    if not results:
        return
    console.print(f"\n[bold]{title}[/bold]\n")
    task_steps = {t["id"]: t["expected_steps"] for t in tasks}
    rows = [[
        r.task_id, short_model(r.model), r.effort,
        r.task_budget or "—", r.constraint_citations,
        f"{r.step_coherence_score}/{task_steps.get(r.task_id, 5)}",
        r.self_corrections,
        "✗ FAIL" if r.memory_failure else "✓ pass",
        "✓" if r.pass_at_1 else "✗",
        f"{r.latency_s}s", f"${r.cost_usd:.4f}",
    ] for r in results]
    print(tabulate(rows, headers=[
        "Task", "Model", "Effort", "Budget", "Citations", "Coherence",
        "Self-Corr", "Mem Status", "Pass@1", "Latency", "Cost",
    ], tablefmt="rounded_outline"))


def print_insights(results, axis, model_map):
    if not results:
        return
    console.print("\n[bold dim]Axis insights:[/bold dim]")

    def stats(rs):
        n = len(rs)
        return (
            f"pass={sum(1 for r in rs if r.pass_at_1)/n:.0%} | "
            f"mem_fail={sum(1 for r in rs if r.memory_failure)/n:.0%} | "
            f"citations={sum(r.constraint_citations for r in rs)/n:.1f} | "
            f"coherence={sum(r.step_coherence_score for r in rs)/n:.1f} | "
            f"corrections={sum(r.self_corrections for r in rs)/n:.1f} | "
            f"latency={sum(r.latency_s for r in rs)/n:.1f}s | "
            f"cost=${sum(r.cost_usd for r in rs)/n:.4f}"
        )

    if axis == "model":
        for model_id in model_map.values():
            rs = [r for r in results if r.model == model_id]
            if rs:
                console.print(f"  [bold]{short_model(model_id)}[/bold]: {stats(rs)}")
    elif axis == "effort":
        for effort in EFFORT_LEVELS:
            rs = [r for r in results if r.effort == effort]
            if rs:
                console.print(f"  [bold]{effort}[/bold]: {stats(rs)}")
    elif axis == "budget":
        for budget in [None, 64000, 32000]:
            rs = [r for r in results if r.task_budget == budget]
            label = f"budget={budget}" if budget else "no_budget"
            if rs:
                console.print(f"  [bold]{label}[/bold]: {stats(rs)}")


def save_results(results, run_id, axis):
    if not results:
        return
    path = os.path.join(RESULTS_DIR, f"run_{run_id}_{axis}.json")
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    console.print(f"[dim]Saved: {path}[/dim]")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Claude Opus 4.7 Memory Performance Benchmark")
    parser.add_argument("--axis", choices=["model", "effort", "budget", "all"], default="all")
    parser.add_argument("--task", help="Single task ID, e.g. T1")
    args = parser.parse_args()

    client, model_map, judge_model, using_openrouter = make_client()

    tasks = TASKS
    if args.task:
        tasks = [t for t in TASKS if t["id"] == args.task]
        if not tasks:
            console.print(f"[red]Task {args.task} not found. Available: {[t['id'] for t in TASKS]}[/red]")
            sys.exit(1)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    console.print(f"\n[bold white]{'='*65}[/bold white]")
    console.print(f"[bold white]Opus 4.7 Memory Performance Benchmark — {run_id}[/bold white]")
    console.print(f"[bold white]{'='*65}[/bold white]")
    console.print(f"Tasks: {[t['id'] for t in tasks]} | Axis: {args.axis}")

    if args.axis in ("model", "all"):
        r = run_axis_model(client, model_map, judge_model, using_openrouter, tasks, run_id)
        print_table(r, "Axis 1: Model Comparison", tasks)
        print_insights(r, "model", model_map)
        save_results(r, run_id, "model")

    if args.axis in ("effort", "all"):
        r = run_axis_effort(client, model_map, judge_model, using_openrouter, tasks, run_id)
        print_table(r, "Axis 2: Effort Calibration", tasks)
        print_insights(r, "effort", model_map)
        save_results(r, run_id, "effort")

    if args.axis in ("budget", "all"):
        r = run_axis_budget(client, model_map, judge_model, using_openrouter, tasks, run_id)
        print_table(r, "Axis 3: Task Budget", tasks)
        print_insights(r, "budget", model_map)
        save_results(r, run_id, "budget")

    console.print("\n[bold green]Benchmark complete.[/bold green]")
    console.print(f"Results: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()