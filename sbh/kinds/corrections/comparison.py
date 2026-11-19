"""The corrections kind's closure comparisons -- EVALUATOR-ONLY (sbh/harbor_pack.py EVALUATOR_ONLY): which one a task
uses (CorrectionsTask.closure_comparison: "z_peak", peak.py; "balance", balance.py) and what records and pages name
it by.

Every comparison is a module with RULE (stamped on the post stage's record and closure.json's baseline), LABEL (the
score breakdown's term), NOUN (messages), PENDING (the scorer's pending note), candidates(dataset_dir, spec, flavour,
cache_dir), closure(spec, cand, cset, result, request, flavour, seed, fallback) and fallback_loss(spec, cand, request,
flavour, seed, cache_dir), on the common core of peak.py; the task describes it per flavour (Task.peak_spec).  The
post stage runs it, the scorer takes it as Delta_peak whatever it compares (docs/METRICS.md "Function outputs").
"""
from __future__ import annotations

import importlib

COMPARISONS = {"z_peak": "peak", "balance": "balance"}


def module(task):
    """The comparison module of a task (CorrectionsTask.closure_comparison)."""
    name = task.closure_comparison
    if name not in COMPARISONS:
        raise SystemExit(f"{getattr(task, 'name', task)}: unknown closure_comparison {name!r} "
                         f"({', '.join(sorted(COMPARISONS))})")
    return importlib.import_module(f".{COMPARISONS[name]}", __package__)


def rule(task) -> str:
    return module(task).RULE


def label(task) -> str:
    return module(task).LABEL


def noun(task) -> str:
    return module(task).NOUN


def pending(task) -> str:
    return module(task).PENDING
