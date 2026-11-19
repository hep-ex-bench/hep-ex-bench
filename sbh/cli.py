"""Command line: python3 -m sbh <command> ..."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import ROOT, __version__
from .runner import Limits, ensure_agent_env, run_one
from .snapshot import REFUSED_FILE, freeze
from .task import load_task
# >>> evaluator only
import time

from .runner import run_suite
from .suite import CLOSURE_LABEL, load_manifest, seed_dirs, suite_task
# <<< evaluator only
# >>> dev-harness only
import sys

from .suite import build_suite
from .task import write_participant_task_stub
# <<< dev-harness only

# >>> dev-harness only
# <<< dev-harness only
USAGE = """
Participant side
  check      run your submission on the public dataset, validate the contract, see the pulls
"""
# >>> evaluator only
USAGE += """
Evaluator side
  closure    run the reference pipeline on a suite; require it to recover the truths; set sigma_ref
  eval       freeze + run on sealed seeds + score one submission
  stress     evaluate the reference and every attacker / alternate reference (sbh stress --list)
"""
# <<< evaluator only
# >>> dev-harness only
USAGE += """
Development harness
  build-suite         draw sealed truths and generate the datasets
  scoreboard          merge the public summaries of several runs
"""
USAGE += """  post-score          the output kind's steps after scoring
"""
USAGE += """  pack                the participant bundle
"""
# <<< dev-harness only



# >>> evaluator only
def format_tables(*args, **kw):
    from .score import format_tables as f
    return f(*args, **kw)


def score_run(*args, **kw):
    from .score import score_run as f
    return f(*args, **kw)
# <<< evaluator only


def _limits(a, task=None, label: Optional[str] = None) -> Limits:
    own = task.own_budget(label) if label and hasattr(task, "own_budget") else {}
    t = a.timeout if a.timeout is not None else own.get("run_timeout_s", getattr(task, "run_timeout_s", 600))
    m = a.memory_mb if a.memory_mb is not None else own.get("run_memory_mb", getattr(task, "run_memory_mb", 6144))
    c = getattr(a, "cpus", None)
    c = c if c is not None else own.get("run_cpus", getattr(task, "run_cpus", 0))
    return Limits(timeout_s=t, memory_mb=m, cpus=c, network=a.allow_network,
                  worker_memory_mb=int(own.get("worker_memory_mb", getattr(task, "worker_memory_mb", 0)) or 0))


def _scaled_limits(a, task, label: Optional[str], work: Path) -> Limits:
    """_limits, with the task's time limit scaled to this evaluation's hardware (timeout_scale) when the task states
    its anchor and the evaluation has a yardstick; an explicit --timeout is kept."""
    lim = _limits(a, task, label)
    if a.timeout is not None or not getattr(task, "run_timeout_reference_wall_s", None):
        return lim
    from .score import run_yardstick
    from .yardstick import timeout_scale
    scale = timeout_scale(task, run_yardstick(work, task))
    if scale > 1.0:
        lim = dataclasses.replace(lim, timeout_s=int(-(-lim.timeout_s * scale // 1)))
        print(f"[{work.name}] run time limit {lim.timeout_s} s: the stated {_limits(a, task, label).timeout_s} s x "
              f"{scale:.3f} (this evaluation's yardstick over the task's {task.run_timeout_reference_wall_s} s)")
    return lim


def memory_verdict(peak_mb, cap_mb: int):
    """(level, message): peak memory vs the per-dataset cap.  FAIL above the cap; WARN above half, since evaluation
    datasets carry about twice the public dataset's luminosity."""
    if peak_mb is None:
        return "OK", "peak memory not measured"
    if cap_mb <= 0:
        return "OK", f"peak {peak_mb:.0f} MB (this task states no memory cap)"
    if peak_mb > cap_mb:
        return "FAIL", (f"peak {peak_mb:.0f} MB exceeds the evaluation's {cap_mb} MB cap: the evaluation would "
                        "kill this run")
    if peak_mb > 0.5 * cap_mb:
        return "WARN", (f"peak {peak_mb:.0f} MB is above half the evaluation's {cap_mb} MB cap, and the evaluation "
                        "datasets carry about twice this dataset's luminosity: if your memory grows with the "
                        "data, the evaluation can kill the run")
    return "OK", f"peak {peak_mb:.0f} MB of the evaluation's {cap_mb} MB cap"


def _add_run_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--timeout", type=int, default=None, help="per-seed wall clock [s] (default: the task's budget)")
    ap.add_argument("--memory-mb", type=int, default=None, help="RLIMIT_AS per run; 0 = none (default: the task's budget)")
    ap.add_argument("--cpus", type=int, default=None,
                    help="CPU affinity per run; 0 = host allocation (default: the task's worker spec)")
    ap.add_argument("--allow-network", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--seeds", default=None, help="subset like 1-4 or 2,5")
    ap.add_argument("--work", default="work")
    ap.add_argument("--allow-violations", action="store_true", help="freeze despite hygiene violations")


# >>> evaluator only
def _parse_seeds(spec: Optional[str], dirs: List[Path]) -> List[Path]:
    if not spec:
        return dirs
    idx = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        idx.update(range(int(lo), int(hi or lo) + 1))
    return [d for d in dirs if int(d.name.split("_")[1]) in idx]


def _work(a, suite: Path, label: str) -> Path:
    return Path(a.work) / suite.name / label


def _sigma_ref(suite: Path):
    p = suite / "closure.json"
    if not p.exists():
        return None
    cl = json.loads(p.read_text())
    if not cl.get("passed"):
        print(f"WARNING: {p} records a FAILED closure — ignoring its sigma_ref "
              "(scores will be self-referenced diagnostics)")
        return None
    return cl["sigma_ref"]


def _sigma(v: dict) -> float:
    return v["total"] if v.get("total") else (v.get("stat", 0.0) ** 2 + v.get("syst", 0.0) ** 2) ** 0.5


def _scored_rows(priv: dict, poi: str):
    """(truth, value, sigma) for every completed seed on which ``poi`` is scored."""
    rows = []
    for s in priv["per_seed"]:
        v, t = s["pois"].get(poi), s.get("scoring_truth", {}).get(poi)
        if s.get("completed") and v is not None and t is not None and _sigma(v) > 0:
            rows.append((float(t), float(v["value"]), _sigma(v)))
    return rows


def reference_bias(priv: dict) -> Dict[str, float]:
    """Mean (reported - injected) of the reference per POI over its scored seeds (the offset `--calibrate-bias`
    removes from submissions before the gates)."""
    out = {}
    for p in {k for s in priv["per_seed"] for k in s["pois"]}:
        rows = _scored_rows(priv, p)
        if rows:
            out[p] = float(sum(v - t for t, v, _ in rows) / len(rows))
    return out


def _calibrated_gates(priv: dict, pois, offsets: Dict[str, float], max_pull: float):
    """Gates on reference-calibrated results (value - offset[poi]): bias against the naive 3/sqrt(n) band, slope
    error from the quoted uncertainties (WLS) rather than the few residuals."""
    from .score import _wls_slope
    import numpy as np
    crit, pulls_all = {}, []
    for p in pois:
        rows = _scored_rows(priv, p)
        n = len(rows)
        if n < 3:
            crit[f"{p}: >= 3 scored datasets"] = False
            continue
        t, v, sg = (np.array(c, dtype=float) for c in zip(*rows))
        v = v - offsets.get(p, 0.0)
        pulls = (v - t) / sg
        pulls_all += list(np.abs(pulls))
        if np.ptp(t) > 0:
            b, se = _wls_slope(t, v, sg)
            crit[f"{p} sensitivity |slope-1| <= max(0.15, 3 err)"] = bool(abs(b - 1) <= max(0.15, 3 * se))
        crit[f"{p} bias |mean pull| <= max(3/sqrt(n), 0.05)"] = bool(abs(pulls.mean()) <= max(3 / n ** 0.5, 0.05))
        crit[f"{p} pull width |w-1| <= max(0.25, 3/sqrt(2(n-1)))"] = bool(
            abs(pulls.std(ddof=1) - 1) <= max(0.25, 3 / (2 * (n - 1)) ** 0.5))
    mp = float(max(pulls_all)) if pulls_all else float("inf")
    crit[f"max |pull| <= {max_pull}"] = bool(mp <= max_pull)
    return crit, mp


def _bias_ok(pub: dict, pois) -> bool:
    """The closure's bias gate of ``pois``: the cluster-robust bias where the closure has it (a POI without an entry
    is not gated), else every POI's per-seed bias_pass."""
    cb = pub.get("cluster_bias") or {}
    return all(cb[p]["pass"] for p in pois if p in cb) if cb else all(pub["pois"][p]["bias_pass"] for p in pois)


def _pass_criteria(pub: dict, priv: dict, max_pull: float, min_precision: Optional[float] = None,
                   no_false_discovery: bool = False, offsets: Optional[Dict[str, float]] = None,
                   max_leakage: Optional[float] = None, leakage_nsigma: float = 2.0,
                   leakage_se_cap: Optional[float] = None,
                   scored_calibration: Optional[List[str]] = None,
                   calibration_slope_tol: float = 0.25):
    """The closure criteria (with ``min_precision`` / ``no_false_discovery``: the binary verdict's rule), named
    booleans that must all hold; ``offsets`` runs the POI gates on calibrated results.
    Returns (criteria, max |pull|)."""
    if offsets is not None:
        gcrit, mp = _calibrated_gates(priv, list(pub["pois"]), offsets, max_pull)
        crit = {"recovery == 1": pub["recovery"] == 1.0, **gcrit}
    else:
        pulls = [abs(v) for s in priv["per_seed"] for v in s["pulls"].values() if v is not None]
        mp = max(pulls) if pulls else float("inf")
        gates = all(m["sensitivity_pass"] and m["calibration_pass"] for m in pub["pois"].values()) and \
            _bias_ok(pub, pub["pois"])
        crit = {"recovery == 1": pub["recovery"] == 1.0, "all POI gates pass": gates,
                f"max |pull| <= {max_pull}": mp <= max_pull}
    if min_precision is not None:
        for p, m in pub["pois"].items():
            v = m.get("precision")
            crit[f"{p} precision >= {min_precision}"] = v is not None and v == v and v >= min_precision
    if max_leakage is not None:
        for key, m in ((pub.get("calibration_response") or {}).get("leakage") or {}).items():
            if leakage_se_cap is not None and m["se"] > leakage_se_cap:
                continue
            crit[f"{key} leakage |k| <= {max_leakage} + {leakage_nsigma:g} sigma"] = \
                bool(abs(m["k"]) <= max_leakage + leakage_nsigma * m["se"])
    if scored_calibration:
        cal = pub.get("calibration") or pub.get("nuisances") or {}
        for key in scored_calibration:
            m = cal.get(key)
            if m is None:
                crit[f"{key} registered on every dataset"] = False
                continue
            exp = m.get("n_expected")
            full = m.get("n") == exp if exp is not None else True
            if not m.get("truth_varies"):
                crit[f"{key} registered on every dataset"] = bool(full)
                continue
            b, se = m.get("slope"), m.get("slope_err")
            ok = full and b is not None and b == b and abs(b - 1.0) <= max(calibration_slope_tol, 3.0 * se)
            crit[f"{key} measured on every dataset: |slope-1| <= max({calibration_slope_tol}, 3 err)"] = bool(ok)
    if no_false_discovery and pub.get("fdr") is not None:
        crit["no discovery claimed on a null (mu = 0) dataset"] = pub["fdr"] == 0.0
    return crit, mp


POI_MAX_PULL = 3.5
# <<< evaluator only


def _forbidden_outputs(task):
    """The kind's further output files a submission may not ship (refused at freeze like RESULT.json)."""
    from .kinds import get_kind
    return get_kind(task).forbidden_outputs


# >>> evaluator only


def _write_json_atomic(path: Path, obj) -> None:
    """Write JSON whole: a writer killed mid-write leaves the previous file, never half of one."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)




_BUNDLE_CHECKED = set()


def _check_suite_bundle(task, suite: Path) -> None:
    """Refuse a suite whose datasets' mc/ is not the task's current MC bundle (sbh/bundle.py), once per process."""
    key = (task.name, str(Path(suite).resolve()))
    if key in _BUNDLE_CHECKED:
        return
    from .bundle import dataset_problems, require
    require(dataset_problems(task, [*seed_dirs(suite, "public"), *seed_dirs(suite, "sealed")]))
    _BUNDLE_CHECKED.add(key)


def _do_eval(a, submission: Path, suite: Path, label: str, which: str = "sealed", sigma_ref=None,
             making_baseline: bool = False, yardstick: bool = True, keep_scored: bool = False) -> Optional[dict]:
    """phase (eval / closure / stress --phase): all = freeze, yardstick, run, post, score.  Split form for one seed
    per batch job: prepare = freeze + build env; run = the --seeds subset on that snapshot; yardstick = time the
    reference on the public dataset (Task.compute_yardstick_k); post = Kind.post_dataset on --seeds; score = score the
    runs on disk.  All but score return None.  yardstick=False: skip the yardstick (a caller that times the reference
    itself).  keep_scored (`sbh eval`): prepare refuses to replace a label that
    already holds records or a score unless --replace.  The evaluator's identity (sbh.provenance) is recorded at
    prepare and at the score; the split run, yardstick, post and score phases refuse to run under another (a score
    after refused runs would charge their missing records to the submission)."""
    from . import provenance
    phase = getattr(a, "phase", "all")
    task = suite_task(suite, getattr(a, "task", None))
    work = _work(a, suite, label)
    if phase in ("run", "yardstick", "post", "score"):
        provenance.check_unchanged(work, phase)

    def scored():
        if not making_baseline:
            from .suite import closure_gate
            why = closure_gate(suite, task, label)
            if why:
                raise SystemExit(f"[{label}] not scored: {why}")
        _settle_post(task, work, label)
        pub = score_run(work, suite, task, label, sigma_ref=sigma_ref, making_baseline=making_baseline)
        provenance.record(work, "score")
        return pub
    if phase == "score":
        return scored()
    from .kinds import get_kind
    post = phase in ("all", "post") and get_kind(task).post_shape(task) is not None
    if phase == "post":
        if post:
            _post_datasets(a, task, work, suite, which)
        return None
    timed = yardstick and getattr(task, "compute_yardstick_k", None) is not None
    if phase == "yardstick":
        if timed:
            if not (work / "snapshot" / "snapshot.tar.gz").exists():
                raise SystemExit(f"{work / 'snapshot' / 'snapshot.tar.gz'} missing: run --phase prepare first")
            _yardstick(a, task, work, suite)
        return None
    if phase in ("all", "prepare"):
        _check_suite_bundle(task, suite)
        if work.exists():
            if keep_scored and not getattr(a, "replace", False) and (
                    (work / "public" / "summary.json").exists() or any((work / "runs").glob("seed_*/record.json"))):
                raise SystemExit(f"{work} already holds this label's records or score: pass --replace to discard "
                                 "them, or evaluate under another --label")
            shutil.rmtree(work)
        try:
            info = freeze(submission, work / "snapshot", a.allow_violations,
                          forbidden_names=_forbidden_outputs(task), vendor_dir=task.vendor_dir)
        except SystemExit as e:
            _refused(work, "freeze", str(e))
            raise
        from .runner import sandbox_mode
        if sandbox_mode() == "setpriv":
            (work / "snapshot").chmod(0o700)
        provenance.record(work, "prepare")
        print(f"[{label}] frozen {info['n_files']} files, sha256 {info['sha256'][:12]}"
              + (f", git {info['git']['head'][:8]}{' (dirty)' if info['git']['dirty'] else ''}" if info["git"] else ""))
        tarball = Path(info["tarball"])
    else:
        tarball = work / "snapshot" / "snapshot.tar.gz"
        if not tarball.exists():
            raise SystemExit(f"{tarball} missing: run --phase prepare first")
    if phase == "prepare":
        _prefix, err = ensure_agent_env(tarball, work / "agent_env", task)
        if err:
            _refused(work, "environment", f"environment: {err}")
            raise SystemExit(f"[{label}] environment: {err}")
        print(f"[{label}] environment ready")
        return None
    if phase == "all" and timed:
        _yardstick(a, task, work, suite)
        if (a.jobs or 1) > 1:
            print(f"[{label}] note: the datasets run {a.jobs} at a time against a yardstick timed alone, so the "
                  "compute term charges their contention (the split chain runs one dataset per task; --jobs 1 matches "
                  "it)")
    dirs = _parse_seeds(a.seeds, seed_dirs(suite, which))
    shield = [suite / "truth", suite / "manifest.json"] if which == "sealed" else []
    recs = run_suite(tarball, dirs, work / "runs", _scaled_limits(a, task, label, work), task, jobs=a.jobs,
                     shield=shield)
    for r in recs:
        msg = "ok" if r.ok else ("TIMEOUT" if r.timed_out else f"FAIL exit={r.exit_code}")
        extra = "" if r.ok else f"  {r.result_errors or r.stderr_tail[-160:].strip()}"
        print(f"  {r.seed}: {msg} ({r.wall_s:.1f}s){extra}")
    if phase == "run":
        return None
    if post:
        _post_datasets(a, task, work, suite, which)
    return scored()


def _refused(work: Path, stage: str, why: str) -> None:
    """Record a submission the prepare step refused (freeze violations or an env that does not build)."""
    lines = [x.strip() for x in why.splitlines() if x.strip()]
    head, items = lines[0], lines[1:]
    shown = items[:5] + ([f"and {len(items) - 5} more"] if len(items) > 5 else [])
    text = head + (" " + "; ".join(shown) if shown else "")
    work.mkdir(parents=True, exist_ok=True)
    (work / REFUSED_FILE).write_text(json.dumps({"ok": False, "stage": stage, "why": text, "detail": items},
                                                indent=1))


def _yardstick(a, task, work: Path, suite: Path) -> None:
    """The compute yardstick (Task.compute_yardstick_k, opt-in): the reference run once on the public dataset with
    this evaluation's flags, recorded in <run>/yardstick.json for the scorer.  A failing reference: no compute term.
    With Task.compute_yardstick_seeds: one replica per sealed seed of --seeds that is a yardstick seed (all of them
    without --seeds), each in <run>/yardstick/<seed>.json; the scorer takes their median (run_yardstick)."""
    from .yardstick import RECORD_FILE, REPLICA_DIR, measure, reference_wall
    seeds = getattr(task, "compute_yardstick_seeds", None)
    if seeds:
        want = {d.name for d in _parse_seeds(a.seeds, seed_dirs(suite, "sealed"))} if a.seeds else set(seeds)
        (work / REPLICA_DIR).mkdir(exist_ok=True)
        os.chmod(work / REPLICA_DIR, 0o700)
        for name in [s for s in seeds if s in want]:
            y = argparse.Namespace(**vars(a))
            y.reference, y.dataset, y.suite, y.task = (str(task.reference_submission), str(suite / "sealed" / name),
                                                       str(suite), task.name)
            y.k, y.jobs = float(task.compute_yardstick_k), 1
            y.out, y.work = str(work / REPLICA_DIR / f"{name}.json"), str(work / f"yardstick_work_{name}")
            rec = measure(y)
            print(f"[{work.name}] yardstick replica {name}: {rec['status']}"
                  + (f"; {rec['wall_s']:.1f} s on {rec.get('host')}" if rec.get("ok") else ""))
        return
    public = seed_dirs(suite, "public")
    if not public:
        (work / RECORD_FILE).write_text(json.dumps({"ok": False, "status": f"no public dataset in {suite}",
                                                    "k": float(task.compute_yardstick_k)}))
        print(f"[{work.name}] yardstick: no public dataset in {suite}; no compute term")
        return
    y = argparse.Namespace(**vars(a))
    y.reference, y.dataset, y.suite, y.task = str(task.reference_submission), str(public[0]), str(suite), task.name
    y.k, y.jobs = float(task.compute_yardstick_k), 1
    y.out, y.work = str(work / RECORD_FILE), str(work / "yardstick_work")
    rec = measure(y)
    ref = reference_wall(rec)
    print(f"[{work.name}] yardstick: {rec['status']}"
          + (f"; Y = {rec['wall_s']:.1f} s x k {rec['k']:g} = {ref:.1f} s" if ref is not None else "; no compute term"))


def _kind_post(task, name: str):
    """The function ``name`` of the task's kind's post-stage module (sbh/kinds/<kind>/post.py), or None where the kind
    has no such module or function."""
    from importlib import import_module
    from .kinds import KINDS
    try:
        mod = import_module(KINDS[getattr(task, "kind", "pois")] + ".post")
    except (ImportError, KeyError):
        return None
    return getattr(mod, name, None)


def _post_datasets(a, task, work: Path, suite: Path, which: str) -> None:
    """Kind.post_dataset on the --seeds subset, between the runs and the score, as one batch where the kind's post stage
    grades several datasets together (its post module's batch: every dataset once, then the retries, so whether a fault
    is systemic is decided over all of their first attempts, whatever their order)."""
    import contextlib
    from .kinds import get_kind
    kind = get_kind(task)
    batch = _kind_post(task, "batch")
    with (batch(work, suite, task) if batch is not None else contextlib.nullcontext()):
        for d in _parse_seeds(a.seeds, seed_dirs(suite, which)):
            kind.post_dataset(work, suite, task, d.name)


def _settle_post(task, work: Path, label: str) -> None:
    """Before the score: where the kind's post stage grades its datasets one per job (scan_likelihood, corrections), its
    records' systemic-fault marks settled over the run's final records (the post module's settle_systemic), so they
    never depend on the order the jobs ran in; no dataset is graded."""
    settle = _kind_post(task, "settle_systemic")
    if settle is None or not (work / "runs").is_dir():
        return
    marked = settle(work, task)
    if marked:
        print(f"[{label}] systemic evaluator fault: {len(marked)} grader record(s) marked over the run's final records")


# <<< evaluator only

# >>> dev-harness only
def cmd_build_suite(a) -> int:
    task = load_task(a.task)
    if not a.manifest_only:
        from .bundle import bundle_problems, require
        require(bundle_problems(task))
    design, design_file = None, None
    if a.design:
        design = json.loads(Path(a.design).read_text())["seeds"]
        design_file = str(a.design)
    only = None
    if a.seeds is not None:
        only = set()
        for part in str(a.seeds).split(","):
            lo, _, hi = part.partition("-")
            only.update(range(int(lo), int(hi or lo) + 1))
    out = build_suite(task, a.name, a.n_seeds, a.master_seed, Path(a.out),
                      priors_override=json.loads(a.priors) if a.priors else None,
                      design=design, response=not a.no_response, force=a.force, design_file=design_file,
                      only=only, manifest_only=a.manifest_only)
    if only is not None and not a.manifest_only:
        print(f"suite {a.name}: generated seeds {sorted(only)} -> {out} (no manifest; "
              f"finish with --manifest-only)")
        return 0
    m = load_manifest(out)
    print(f"suite {m['suite']} ({task.name}): {m['n_seeds']} seeds, seed_000 public -> {out}")
    for s in m["seeds"]:
        shown = {**{k: s["truth"][k] for k in task.pois if k in s["truth"]}, **task.calibration_truth(s["truth"])}
        vals = ", ".join(f"{k}={v:.4g}" for k, v in shown.items())
        print(f"  {s['seed']} {'PUBLIC' if s['public'] else 'sealed'}  {vals}")
    return 0


def cmd_bundle_fingerprint(a) -> int:
    """The fingerprint of an MC bundle or a dataset's mc/ (sbh/bundle.py); with --task, compared with the task's."""
    from .bundle import fingerprint
    task = load_task(a.task) if a.task else None
    try:
        d = Path(a.directory) if a.directory else (task.mc_bundle() if task else None)
    except RuntimeError as e:
        raise SystemExit(f"bundle-fingerprint: {e}")
    if d is None or not Path(d).is_dir():
        raise SystemExit(f"bundle-fingerprint: no directory {d} (give one, or --task with an MC bundle)")
    fp = fingerprint(d, task.mc_bundle_evaluator_files if task else ())
    print(f"{d}: {fp}")
    if task is None or task.mc_bundle_fingerprint is None:
        return 0
    same = fp == task.mc_bundle_fingerprint
    print(f"{task.name}: " + ("the task's MC bundle" if same else
                              f"NOT the task's MC bundle (the task records {task.mc_bundle_fingerprint})"))
    return 0 if same else 1
# <<< dev-harness only


# >>> evaluator only
def cmd_eval(a) -> int:
    suite = Path(a.suite)
    label = a.label or Path(a.submission).name
    pub = _do_eval(a, Path(a.submission), suite, label, sigma_ref=_sigma_ref(suite), keep_scored=True)
    if pub is None:
        return 0
    print("\n" + format_tables([pub], with_attackers=a.with_attackers))
    print(f"public summary: {_work(a, suite, label) / 'public' / 'summary.json'}")
    return 0


def _closure_rescore(a, suite: Path, task, kind_fields: dict) -> dict:
    """The closure's reference scored again, against the grader's baseline the kind just re-derived from its
    verification on disk (a gated task: `sbh closure --phase score` after the reference's post-score, or after a
    `verify --stage grid --prior-curve` re-scan).  The first score read whatever closure.json held then (an older
    closure's mu rule, or none), so its mu cases, and with them the closure's summary, criteria and reference_score
    (every run's relative score), would not be the ones every later run is compared with.  The baseline goes into
    closure.json first (sbh.suite.update_closure, stale mu entries and baseline keys dropped); a failure restores the
    file as it was."""
    import contextlib
    import io
    from . import provenance
    from .score import score_run
    from .suite import baseline_keys, baseline_pois, mu_rule, update_closure
    cj = suite / "closure.json"
    before = cj.read_bytes() if cj.exists() else None
    base = {"mu_rule": mu_rule(), **kind_fields}
    print(f"{'/'.join(baseline_pois(task))} baseline: the grader's, from {CLOSURE_LABEL}'s verification on disk; "
          f"scoring {CLOSURE_LABEL} again against it")
    try:
        if before is None:
            cj.write_text(json.dumps(base, indent=1))
        else:
            update_closure(suite, base, replace_pois=baseline_pois(task), replace_keys=baseline_keys(task))
        work = _work(a, suite, CLOSURE_LABEL)
        with contextlib.redirect_stdout(io.StringIO()):
            pub = score_run(work, suite, task, CLOSURE_LABEL, sigma_ref=None, making_baseline=True)
        provenance.record(work, "score")
        return pub
    except BaseException:
        if before is None:
            cj.unlink(missing_ok=True)
        else:
            cj.write_bytes(before)
        raise


def cmd_closure(a) -> int:
    suite = Path(a.suite)
    task = suite_task(suite, getattr(a, "task", None))
    pub = _do_eval(a, task.reference_submission, suite, CLOSURE_LABEL, making_baseline=True)
    if pub is None:
        return 0
    from .kinds import get_kind
    from .suite import BASELINE_GRADER, baseline_pois
    kind_fields = get_kind(task).closure_fields(_work(a, suite, CLOSURE_LABEL), suite, task)
    if baseline_pois(task) and kind_fields.get("baseline_source") == BASELINE_GRADER:
        pub = _closure_rescore(a, suite, task, kind_fields)
    priv = json.loads((_work(a, suite, CLOSURE_LABEL) / "private" / "scores.json").read_text())
    offsets = reference_bias(priv) if a.calibrate_bias else None
    sc = (sorted(task.calibration_scored) if a.scored_calibration == "auto"
          else a.scored_calibration.split(",") if a.scored_calibration else None)
    own = get_kind(task).closure_criteria(pub, priv, task, a, offsets=offsets, scored_calibration=sc)
    fields = {}
    if own is not None:
        crit, max_pull, fields = own
    else:
        a.max_pull = a.max_pull if a.max_pull is not None else POI_MAX_PULL
        crit, max_pull = _pass_criteria(pub, priv, a.max_pull, offsets=offsets, max_leakage=a.max_leakage,
                                        leakage_nsigma=a.leakage_nsigma if a.leakage_nsigma is not None else 2.0,
                                        leakage_se_cap=a.leakage_se_cap, scored_calibration=sc,
                                        calibration_slope_tol=a.calibration_slope_tol if a.calibration_slope_tol is not None else 0.25)
    fields = {**fields, **kind_fields}
    if offsets is not None:
        diag, _ = _pass_criteria(pub, priv, a.max_pull)
        print(f"reference bias (removed from every submission): { {k: round(v, 4) for k, v in offsets.items()} }")
    passed = all(crit.values())
    sigma_ref = pub["sigma_ref"]
    from .suite import BASELINE_PENDING, BASELINE_GRADER, baseline_pois, mu_rule
    grader_pois = baseline_pois(task)
    if not isinstance(priv.get("reference_pois"), dict):
        raise SystemExit(f"{CLOSURE_LABEL}'s private/scores.json records no reference_pois (the per-dataset baselines its "
                         f"score is graded on): score the run with this evaluator first")
    ref_pois = {s: dict(e) for s, e in priv["reference_pois"].items()}
    for s, e in (fields.pop("reference_pois", None) or {}).items():
        ref_pois[s] = {**ref_pois.get(s, {}), **(e or {})}
    versioned = {"mu_rule": mu_rule(), "baseline_source": BASELINE_PENDING} if grader_pois else {}
    from .score import closure_multiworld, reference_record
    mw = closure_multiworld(pub.get("v2"))
    (suite / "closure.json").write_text(json.dumps(
        {"passed": passed, "sigma_ref": sigma_ref, "max_abs_pull": max_pull, "criteria": crit,
         **({"reference_bias": offsets, "criteria_uncalibrated": diag} if offsets is not None else {}),
         **versioned, **fields,
         "reference_pois": ref_pois, "multiworld": mw, "summary": pub,
         "reference_score": reference_record(pub, priv)}, indent=1))
    print("\n" + format_tables([pub]))
    shown = ({k: float(f"{v:.4g}") for k, v in sigma_ref.items()} if own is not None
             else {k: round(v, 4) for k, v in sigma_ref.items()})
    print(f"closure {'PASSED' if passed else 'FAILED'}: recovery={pub['recovery']:.2f} max|pull|={max_pull:.2f} "
          f"sigma_ref={shown}")
    shown_mw = {p: m for p, m in mw["per_poi"].items() if m["bias"] is not None}
    if shown_mw:
        print("bias term (diagnostic): the reference's combination per POI, z* / B_p / its part of the POI's "
              "charge: " + ", ".join(f"{p} {m['zstar']:+.2f} / {m['bias']:+.4f} / {m['delta_multiworld']:+.4f}"
                                      for p, m in shown_mw.items()))
    if (pub.get("v3") or {}).get("failed"):
        print(f"the reference FAILED on {pub['v3']['failed']['n_datasets']} datasets (S = 0): "
              + "; ".join(f"{d['seed']}: {', '.join(d['reasons'])[:160]}" for d in pub["v3"]["failed"]["datasets"][:5]))
    if grader_pois:
        src = {**versioned, **fields}.get("baseline_source")
        mb = fields.get("mu_baseline") or {}
        print(f"{'/'.join(grader_pois)} baseline ({versioned['mu_rule']}): "
              + (f"the grader's (scan {fields.get('scan_version')}) on {mb.get('n_datasets')} of "
                 f"{mb.get('n_expected')} datasets" if src == BASELINE_GRADER else
                 f"PENDING the grader -- sbh post-score --label {CLOSURE_LABEL} writes it from the reference's "
                 "model verification; no other run is scored before"
                 + (f" ({str(fields.get('baseline_note'))[:300]})" if fields.get("baseline_note") else "")))
        for s, why in sorted((mb.get("excluded") or {}).items()):
            print(f"  {s}: no {grader_pois[0]} baseline ({str(why)[:160]})")
    print(f"wrote {suite / 'closure.json'} (sigma_ref now used by eval/stress on this suite)")
    return 0 if passed else 1


def stress_submissions(task, only: Optional[str] = None) -> List[Path]:
    """The entries `sbh stress` evaluates, each under its directory's name (the task's reference under
    Task.reference_label): --only, else task.stress_submissions, else the reference and every entry of this task or
    shared, but not a further reference."""
    from .task import submission_entries
    entries = {task.reference_label: (Path(task.reference_submission), task.name), **submission_entries()}
    names = sorted(entries)
    if only:
        return [entries[k][0] for k in names if k in only.split(",")]
    if getattr(task, "stress_submissions", None) is not None:
        missing = sorted(set(task.stress_submissions) - set(entries))
        if missing:
            raise SystemExit(f"stress: {task.name}'s stress submissions {missing} are not submissions")
        return [entries[k][0] for k in names if k in task.stress_submissions]
    further = set(getattr(task, "further_references", None) or {})
    return [entries[k][0] for k in names if entries[k][1] in (None, task.name) and k not in further]


def _label(p: Path, task) -> str:
    """A stress entry's run label: its directory's name, Task.reference_label for the task's reference."""
    return task.reference_label if p == Path(task.reference_submission) else p.name


def cmd_stress(a) -> int:
    suite = Path(a.suite)
    task = suite_task(suite, getattr(a, "task", None))
    subs = stress_submissions(task, a.only)
    if a.list:
        print("\n".join(_label(s, task) for s in subs))
        return 0
    if getattr(a, "recheck", False):
        files = {_label(s, task): _work(a, suite, _label(s, task)) / "public" / "summary.json" for s in subs}
        missing = [k for k, f in files.items() if not f.exists()]
        pubs = [json.loads(f.read_text()) for f in files.values() if f.exists()]
        provisional = [x["label"] for x in pubs if ((x.get("v3") or {}).get("compute") or {}).get("gate_provisional")]
        sides = [w for w in ("reference", "attacker") if not any(x["label"].startswith(w) for x in pubs)]
        why = [w for w in (f"no scored summary for {missing}" if missing else "",
                           f"provisional scores {provisional}" if provisional else "",
                           f"no {' or '.join(sides)} among the runs" if sides else "") if w]
        if why:
            print("STRESS RECHECK FAILED: " + "; ".join(why))
            return 1
        return _stress_invariant(pubs)
    sigma_ref = _sigma_ref(suite)
    pubs = [_do_eval(a, s, suite, _label(s, task), sigma_ref=sigma_ref) for s in subs]
    if getattr(a, "phase", "all") in ("prepare", "run", "yardstick", "post"):
        return 0
    table = format_tables(pubs, with_attackers=True)
    out = Path(a.work) / suite.name / "STRESS_SCOREBOARD.md"
    out.write_text(f"# Stress test on suite `{suite.name}`\n\nsigma_ref source: "
                   f"{'reference closure' if sigma_ref else 'self-referenced'}\n\n" + table)
    print("\n" + table)
    print(f"wrote {out}")
    return _stress_invariant(pubs)


def _stress_invariant(pubs) -> int:
    """The stress invariant on scored summaries: every attacker strictly below the reference, on S_phys where both
    have it and on S (v3).  0 when it holds, 1 when it fails."""
    refs = [x for x in pubs if x["label"].startswith("reference")]
    atks = [x for x in pubs if x["label"].startswith("attacker")]
    if refs and atks:
        phys = [x["s_phys"] for x in refs if x["s_phys"] is not None]
        ref_s = max(phys) if phys else None
        bad = [x["label"] for x in atks if ref_s is not None and x["s_phys"] is not None and x["s_phys"] >= ref_s]
        ref_v3 = max((x["v3"]["s_v3"] for x in refs if x.get("v3")), default=None)
        if ref_v3 is not None:
            bad += [f"{x['label']} (v3)" for x in atks
                    if x.get("v3") and x["v3"]["s_v3"] >= ref_v3]
        ref_txt = f"{ref_s:.3f}" if ref_s is not None else "(S_phys not computed)"
        if bad:
            print(f"STRESS FAILED: attacker(s) not below the reference ({ref_txt}): {bad}")
            return 1
        v3s = f", v3 < {ref_v3:.3f}" if ref_v3 is not None else ""
        print(f"stress invariant holds: all {len(atks)} attackers < reference {ref_txt}{v3s}")
    return 0
# <<< evaluator only


# >>> dev-harness only


def cmd_post_score(a) -> int:
    """The kind's steps after a run is scored (Kind.post_score; binned_likelihood: model verify, rescore), then the
    closure.json fields on the reference run.  Exit 3 when there are no runs, 1 when a required step failed (after the
    rest have run)."""
    suite = Path(a.suite).absolute()
    run_dir = _work(a, suite, a.label).absolute()
    if not (run_dir / "runs").is_dir():
        print(f"no runs under {run_dir} — nothing to post-process")
        return 3
    task = suite_task(suite, a.task)
    from .kinds import get_kind
    failed = get_kind(task).post_score(run_dir, suite, task, a.label)
    if a.label == CLOSURE_LABEL:
        from .score import closure_multiworld, reference_record
        from .suite import (BASELINE_GRADER, BASELINE_KEYS, baseline_keys, baseline_pois, closure_gate,
                            update_closure)
        rec = None
        rescore_failed = any(str(f).startswith("rescore") for f in (failed or ()))
        if baseline_pois(task) and rescore_failed:
            print(f"{a.label}: the rescore failed, reference_score not updated (closure_gate admits no run until "
                  f"sbh post-score --label {CLOSURE_LABEL} rescores the reference against the grader baseline)")
        elif (run_dir / "public" / "summary.json").exists():
            summary = json.loads((run_dir / "public" / "summary.json").read_text())
            rec = reference_record(summary, json.loads((run_dir / "private" / "scores.json").read_text()))
        stale = closure_gate(suite, task, CLOSURE_LABEL) if baseline_pois(task) else None
        fields, keys = get_kind(task).closure_fields(run_dir, suite, task), baseline_keys(task)
        if stale:
            print(f"{a.label}: closure.json not updated ({stale})")
            fields, rec, keys = {}, None, ()
        elif keys:
            try:
                src = json.loads((suite / "closure.json").read_text()).get("baseline_source")
            except (OSError, ValueError):
                src = None
            if src != BASELINE_GRADER:
                fields = {k: v for k, v in fields.items()
                          if k not in BASELINE_KEYS and k not in ("reference_pois", "sigma_ref")}
                keys = ()
        if fields or rec:
            update_closure(suite, {**fields, **({"reference_score": rec, "multiworld": closure_multiworld(
                               summary.get("v2"))} if rec else {})},
                           replace_pois=baseline_pois(task) if not stale else (), replace_keys=keys)
    if failed:
        print(f"{a.label}: post-score FAILED: {'; '.join(failed)}", file=sys.stderr)
        return 1
    return 0


def cmd_scoreboard(a) -> int:
    pubs = [json.loads((Path(d) / "public" / "summary.json").read_text()) for d in a.runs]
    print(format_tables(pubs, with_attackers=a.with_attackers))
    return 0


# <<< dev-harness only


def check_work_root(work: str, sub: Path) -> Tuple[Path, Optional[str]]:
    """(scratch root, note): --work, or a fresh temp dir when it lies inside the submission (would fail hygiene at
    freeze) or is not writable.  The note names --work as given."""
    import tempfile
    root = Path(work).resolve()
    if root == sub or sub in root.parents:
        why = "lies inside the submission"
    else:
        try:
            root.mkdir(parents=True, exist_ok=True)
            writable = os.access(root, os.W_OK)
        except OSError:
            writable = False
        if writable:
            return root, None
        why = "cannot be written to"
    tmp = Path(tempfile.mkdtemp(prefix="sbh_check_"))
    return tmp, f"note: --work {work} {why}; using {tmp} instead"


def cmd_check(a) -> int:
    """Participant self-check: hygiene, sandboxed run on one dataset, contract, pulls if truth.json exists."""
    import hashlib
    task = load_task(a.task)
    ds = Path(a.dataset).resolve()
    sub = Path(a.submission).resolve()
    work_root, note = check_work_root(a.work, sub)
    if note:
        print(note)
    if not (ROOT / ".pixi" / "envs" / (getattr(task, "base_env", None) or "agentbase")).exists():
        os.environ["SBH_BASE_ENV_FALLBACK"] = "1"
        if os.environ.get("SBH_BASE_ENV_ONLY"):
            print("note: fixed-runtime evaluation -- extra packages ship INSIDE the "
                  "submission (pip install --target <submission>/pylib; see the "
                  "contract's runtime section); pixi manifests are ignored here")
            from .snapshot import runtime_drift
            base = os.environ.get("SBH_RUNTIME_FREEZE")
            drift = runtime_drift(base)
            if drift:
                print(f"WARNING: this container's Python packages differ from the evaluation runtime ({base}): "
                      + "; ".join(drift[:12]) + (f"; ... {len(drift) - 12} more" if len(drift) > 12 else ""))
                print("  The evaluation runs a fresh copy of the runtime as built; this self-check runs your "
                      "pipeline with THIS interpreter and compares pylib/ against ITS packages, so it can pass "
                      "where the graded run fails (a missing import, a pylib/ file the grader rejects).  Put what "
                      "your pipeline needs into pylib/ (the contract's runtime section).")
        else:
            print("note: the task base env is not installed here, so a submission without its own "
                  "pixi.toml runs with this interpreter; the evaluation's base env has only what "
                  "EVAL_CONTRACT.md promises -- declare anything else in pixi.toml + pixi.lock")
    work = work_root / "check" / f"{sub.name}-{hashlib.sha1(str(sub).encode()).hexdigest()[:8]}"
    if work.exists():
        shutil.rmtree(work)
    info = freeze(Path(a.submission), work / "snapshot", a.allow_violations,
                  forbidden_names=_forbidden_outputs(task), vendor_dir=task.vendor_dir)
    print(f"snapshot: {info['n_files']} files, sha256 {info['sha256'][:12]}, hygiene "
          f"{'OK' if not info['hygiene_violations'] else info['hygiene_violations']}")
    from .runner import ensure_agent_env, _agent_base_prefix, sandbox_mode, private_tmp, network_isolated
    env_prefix, env_err = ensure_agent_env(Path(info["tarball"]), work / "agent_env", task)
    if env_err:
        print(f"environment: FAILED — {env_err}")
        return 1
    print("environment:", "submission pixi.lock (built OK)"
          if env_prefix != _agent_base_prefix(task) else "task base environment")
    lim = _limits(a, task)
    cap = lim.worker_memory_mb
    mode = sandbox_mode(lim.network)
    tmp_counts = private_tmp(mode)
    mem_txt = (f"RLIMIT_AS {lim.memory_mb} MB (virtual)" if lim.memory_mb else
               f"{cap} MB resident{', files in /tmp included' if tmp_counts else ''} (the evaluation kills a run "
               "above it; measured here, not enforced)" if cap else
               "not capped")
    net_txt = ("on (--allow-network)" if lim.network else "off" if network_isolated(mode) else
               f"NOT isolated in this self-check (sandbox '{mode}' keeps this machine's network; "
               "the evaluation has none)")
    print(f"limits (as in the evaluation): wall clock {lim.timeout_s} s, "
          f"{f'{lim.cpus} CPUs' if lim.cpus else 'CPUs as allocated'}, memory {mem_txt}")
    print(f"network: {net_txt}")
    rec = run_one(Path(info["tarball"]), ds, work / "run", lim, task, env_prefix=env_prefix, self_check=True)
    tree_txt = (f"whole process tree + files in /tmp {rec.tree_mem_mb} MB sampled, of which /tmp {rec.tmpfs_mb} MB"
                if tmp_counts else f"whole process tree {rec.tree_mem_mb} MB sampled")
    print(f"run: exit={rec.exit_code} wall={rec.wall_s:.1f}s cpu={rec.cpu_s}s peak memory {rec.mem_peak_mb} MB "
          f"({tree_txt}; largest process {rec.maxrss_mb} MB) isolation={rec.isolation}")
    level, msg = memory_verdict(rec.mem_peak_mb, cap)
    print(f"memory: {level} -- {msg}")
    if not rec.ok:
        print("FAILED:", rec.result_errors or "(nonzero exit / timeout)")
        print(rec.stderr_tail)
        return 1
    print("contract: OK")
    from .kinds import get_kind
    own = get_kind(task).check_report(a, rec, ds, work, info, env_prefix, task, level, msg)
    if own is not None:
        return own
    from .hep.nuisances import working_point_problems
    for w in working_point_problems(rec.result, task):
        print("WARNING:", w)
    for p in rec.result["pois"]:
        print(f"  {p['name']:8s} = {get_kind(task).poi_text(p)}")
    if task.result_type == "search":
        print(f"  significance_obs = {rec.result['significance_obs']:.2f}  discovery = {rec.result['discovery']}")
    from .kinds import get_kind
    from .snapshot import VENDOR_DIR
    rc = get_kind(task).check_outputs(work / "run" / "out", rec.result, task, ds, work / "run" / "repo" / VENDOR_DIR)
    if rc:
        return rc
    _grid_self_check(work / "run" / "out", task, rec.result)
    tf = ds / "truth.json"
    if tf.exists():
        truth = json.loads(tf.read_text())
        print("pulls vs disclosed truth:")
        for p in rec.result["pois"]:
            n = p["name"]
            if n in truth and task.poi_applicable(n, truth):
                sg = get_kind(task).poi_sigma(p, float(truth[n]))
                print(f"  {n:8s} truth={truth[n]:.4g}  pull={(p['value'] - truth[n]) / sg:+.2f}")
        for n, c in (rec.result.get("nuisances") or {}).items():
            if n in truth and c.get("unc", 0) > 0:
                print(f"  {n:8s} truth={truth[n]:.4g}  pull={(c['value'] - truth[n]) / c['unc']:+.2f}  (nuisance)")
    if a.determinism:
        rec2 = run_one(Path(info["tarball"]), ds, work / "run2", _limits(a, task), task, env_prefix=env_prefix,
                       self_check=True)
        same = rec2.ok and all(abs(p1["value"] - p2["value"]) <= 1e-9 * max(1.0, abs(p1["value"]))
                               for p1, p2 in zip(rec.result["pois"], rec2.result["pois"]))
        print(f"determinism (second clean run): {'PASS' if same else 'FAIL'}")
        if not same:
            return 1
    if level == "FAIL":
        print(f"self-check FAILED on memory: {msg}")
        return 1
    return 0


def grid_self_check(out_dir: Path, task, result=None) -> Optional[dict]:
    """The evaluator's MODEL_GRID coverage and content checks from the workspaces' JSON alone (no fit; the verify
    stage runs the same functions on its fitted points): {"n_points", "mass_range", "coverage", "content"} per
    sbh.kinds.binned_likelihood.verify_grid (grid_coverage, grid_content on template_facts of every grid file), the
    band anchored on MODEL.json's own template at the reported mH (``result``'s POI, else the grid's nearest point).
    None without a task mass range, a yield curve (Task.grid_yield_curve) or a grid.  Fit-dependent checks (dead
    points by a mu on its bound, the 2NLL minimum) are left to the evaluator."""
    curve = getattr(task, "grid_yield_curve", None)
    curve = curve() if callable(curve) else None
    rng = getattr(task, "mass_range", None)
    if curve is None or rng is None:
        return None
    from .kinds.binned_likelihood.model import read_point
    from .kinds.binned_likelihood.verify_grid import _load_grid, grid_content, grid_coverage, template_facts
    grid = _load_grid(Path(out_dir))
    if not grid:
        return None
    lo, hi = float(rng[0]), float(rng[1])

    def spec_of(x):
        return read_point(x)
    points = []
    for m in sorted(grid):
        spec = spec_of(grid[m])
        in_range = lo <= m <= hi
        points.append({"mass": float(m), "in_range": in_range, "status": "ok" if in_range else "out_of_range",
                       "mu_hat": None, "twice_nll_abs": None, "sigma_hess": None, "bounds": None,
                       **template_facts(spec, spec.get("_meta"), mass_range=(lo, hi))})
    m0 = s0 = None
    mh = next((p for p in (result or {}).get("pois") or [] if p.get("name") == "mH"), None)
    if mh is not None and isinstance(mh.get("value"), (int, float)):
        m0 = float(mh["value"])
    mp = Path(out_dir) / "MODEL.json"
    if mp.exists():
        from .contract import load_output_json
        s0 = template_facts(load_output_json(mp), None, mass_range=(lo, hi)).get("yield_mu1")
    con = grid_content(points, (lo, hi), curve, model_mass=m0, model_yield=s0)
    cov, _ = grid_coverage(points, (lo, hi), model_mass=m0)
    return {"n_points": len(points), "mass_range": [lo, hi], "coverage": cov, "content": con, "points": points}


def grid_link_self_check(out_dir: Path, task, result=None) -> Optional[dict]:
    """Whether the declared grid holds MODEL.json at the reported mH: the evaluator's first route linking the grid to
    MODEL.json (sbh.kinds.binned_likelihood.model.grid_identity, which the participant bundle ships; pyhf-free, no
    fits: the grid workspace nearest the reported mH within the local spacing, the same workspace to the tolerance
    MODEL.json's own size sets) -> {"mass": the reported mH, "identical", "identical_point", "identical_file",
    "identical_distance", "identical_spacing", "identical_rel", "identical_diff" / "identical_why"}.  None for a task
    without a "grid" POI or whose grader links by the fit alone (Task.grid_link_identical False), or a run without a
    grid or MODEL.json."""
    if "grid" not in (getattr(task, "model_poi_sources", None) or {}).values() \
            or not getattr(task, "grid_link_identical", True):
        return None
    from .contract import load_output_json
    from .kinds.binned_likelihood.model import grid_identity, load_grid
    grid = load_grid(Path(out_dir))
    mp = Path(out_dir) / "MODEL.json"
    if not grid or not mp.is_file():
        return None
    mh = next((p for p in (result or {}).get("pois") or [] if isinstance(p, dict) and p.get("name") == "mH"), None)
    mass = mh.get("value") if mh is not None else None
    return {"mass": mass, **grid_identity(grid, mass, load_output_json(mp))}


def _grid_self_check(out_dir: Path, task, result=None) -> None:
    """`sbh check`'s MODEL_GRID report: the coverage and content of a grid (_grid_coverage_report) and whether it holds
    MODEL.json at the reported mH (_grid_link_report)."""
    _grid_coverage_report(out_dir, task, result)
    _grid_link_report(out_dir, task, result)


def _grid_link_report(out_dir: Path, task, result=None) -> None:
    """The grid link's diagnostic (grid_link_self_check): the grid's workspace nearest the reported mH is MODEL.json's
    (the evaluator links the grid to MODEL.json without comparing fits), or what differs, or that the grid has no
    workspace within its local spacing of that mass (the evaluator then compares its own fit of the grid point nearest
    the reported mH with its fit of MODEL.json).  Nothing where grid_link_self_check gives None."""
    try:
        rep = grid_link_self_check(out_dir, task, result)
    except Exception as e:  # noqa: BLE001
        print(f"MODEL_GRID link: not checked ({type(e).__name__}: {str(e)[:160]})")
        return
    if not rep:
        return
    fit_route = ("the evaluator then links the grid to MODEL.json only if its fit of the grid point nearest the "
                 "reported mH reproduces its fit of MODEL.json (the same mu and likelihood, to numerical precision)")
    if rep.get("identical_why") == "no reported mass":
        print("MODEL_GRID link: RESULT.json reports no mH, so MODEL.json cannot be located on the grid")
        return
    m = f"{float(rep['mass']):g}"
    if rep["identical"]:
        print(f"MODEL_GRID link: {rep['identical_file']}, the grid point nearest the reported mH {m} GeV, is "
              "MODEL.json's own workspace: linked")
    elif rep["identical"] is False:
        print(f"MODEL_GRID link: {rep['identical_file']}, the grid point nearest the reported mH {m} GeV, is not "
              f"MODEL.json's workspace (first difference: {rep.get('identical_diff')}); {fit_route}")
    else:
        print(f"MODEL_GRID link: the grid does not hold MODEL.json at the reported mH {m} GeV "
              f"({rep.get('identical_why')}); {fit_route}.  A grid holding MODEL.json itself at the grid point nearest "
              "the reported mass is linked without that comparison")


def _grid_coverage_report(out_dir: Path, task, result=None) -> None:
    """The coverage and content report for a task that publishes its signal-yield curve: the evaluator's coverage check
    (the grid reaches both ends of the mass range, no wide in-range gap) and content check (each point's template
    against the curve, centroid, smoothness, dead points), grid_self_check above (pyhf-free, no fits).  A grid failing
    either is scored on one mass point.  Nothing for a task without a curve or a run without a grid; a checkout
    without the evaluator's grid functions says so."""
    try:
        rep = grid_self_check(out_dir, task, result)
    except ImportError as e:
        print(f"MODEL_GRID: the coverage / content self-check is not in this checkout ({e})")
        return
    except Exception as e:  # noqa: BLE001
        print(f"MODEL_GRID: coverage / content not checked ({type(e).__name__}: {str(e)[:160]})")
        return
    if not rep:
        return
    cov, con = rep["coverage"], rep["content"]
    lo, hi = rep["mass_range"]

    def gev(x):
        return "none" if x is None else f"{x:.3g}" if abs(x) < 10 else f"{x:.6g}"
    print(f"MODEL_GRID: {rep['n_points']} points, {cov.get('n_in_range')} in {lo:g}-{hi:g} GeV; coverage "
          f"{'OK' if cov.get('ok') else 'FAILS'} (first {gev(cov.get('first_mass'))}, last {gev(cov.get('last_mass'))}, "
          f"largest in-range gap {gev(cov.get('max_gap_in_range'))} GeV)")
    print(f"  content {'OK' if con.get('usable') else 'FAILS'}"
          + (": " + "; ".join(map(str, con.get("reasons") or [])) if con.get("reasons") else ""))
    if not (cov.get("ok") and con.get("usable")):
        print("  the evaluation scores the signal strength of a grid failing coverage or content on one mass point "
              "(the contract's fallback): extend or repair the grid before submitting")


PARTICIPANT_README = """# {task} — participant bundle

You are given ONE development dataset:

    datasets/public/seed_000/      (described in DATASET.md)

Read, in this order: `PROMPT.md` (the task), `DATASET.md` (what the data
is), `EVAL_CONTRACT.md` (what your submission must deliver and how it is
executed — the contract is the single source for that).

Develop however you like — internet included.  The evaluation run itself
is offline and reproducible from your submission alone (EVAL_CONTRACT.md).

## Self-check before submitting

    python3 -m sbh check --task {task} --submission <your_dir> \\
        --dataset datasets/public/seed_000 --determinism

`examples/stub_submission/` shows the minimal interface (it never reads the
data and scores zero -- do not copy its physics).
"""


# >>> dev-harness only
HOST_FILES = ("README.md", "LICENSE", ".gitattributes", ".cache")


def copy_public_dataset(task, public: Path, dest: Path) -> None:
    """The public dataset as participants receive it, copied from `public` (a suite's public/seed_000, or a released
    dataset) into `dest`: the dataset's own files only (not the slimmer's .tbs_slim* records, nor the host's files,
    HOST_FILES), without its truth, brought to the task's shipping rules (Task.ship_fixup), and holding the task's
    current MC bundle (refused otherwise)."""
    def ignore(d, names):
        return [n for n in names if n.startswith(".tbs_slim") or (Path(d) == Path(public) and n in HOST_FILES)]
    shutil.copytree(public, dest, ignore=ignore)
    (dest / "truth.json").unlink(missing_ok=True)
    task.ship_fixup(dest)
    from .bundle import dataset_problems, require
    require(dataset_problems(task, [dest]))


def cmd_pack(a) -> int:
    """The participant bundle from a suite's public dataset, or from a released public dataset (--public-dataset, the
    dataset as published) with --task."""
    public_dataset, task_name = getattr(a, "public_dataset", None), getattr(a, "task", None)
    if public_dataset:
        if not task_name:
            raise SystemExit("pack: --public-dataset needs --task")
        task, name, public = load_task(task_name), "public", Path(public_dataset)
        if not public.is_dir():
            raise SystemExit(f"pack: no public dataset directory {public}")
    else:
        if task_name:
            raise SystemExit("pack: --task goes with --public-dataset (a suite names its task)")
        m = load_manifest(Path(a.suite))
        task, name, public = load_task(m["task"]), m["suite"], Path(a.suite) / "public" / "seed_000"
    from .task import submission_dirs
    example = submission_dirs().get(getattr(task, "example_submission", "attacker_constant"), Path("/nonexistent"))
    if not (example / "run.sh").is_file():
        raise SystemExit(f"pack: {task.name}'s example submission {example} does not exist")
    out = Path(a.out) / f"{task.name}_{name}_participant"
    if out.exists():
        shutil.rmtree(out)
    try:
        copy_public_dataset(task, public, out / "datasets" / "public" / "seed_000")
    except SystemExit:
        shutil.rmtree(out)
        raise
    for d in task.docs():
        shutil.copy(d, out / d.name)
    if getattr(a, "prompt", None):
        shutil.copy(a.prompt, out / "PROMPT.md")
    from .harbor_pack import finish_participant_bundle, participant_ignore
    ign = participant_ignore()
    shutil.copytree(ROOT / "sbh", out / "sbh", ignore=participant_ignore())
    (out / "tasks").mkdir()
    (out / "tasks" / "__init__.py").write_text("")
    write_participant_task_stub(task, out / "tasks" / task.name)
    finish_participant_bundle(out, task)
    shutil.copytree(example, out / "examples" / "stub_submission", ignore=ign)
    (out / "README.md").write_text(PARTICIPANT_README.format(task=task.name, suite=name))
    tarball = shutil.make_archive(str(out), "gztar", root_dir=out.parent, base_dir=out.name)
    print(f"participant bundle: {out}\n  tarball: {tarball}\n  (sealed seeds and truths are NOT included)")
    return 0
# <<< dev-harness only


# >>> evaluator only
# <<< evaluator only


# >>> dev-harness only


# <<< dev-harness only

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sbh", description=__doc__ + USAGE,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"sbh {__version__}")
    sp = ap.add_subparsers(dest="cmd", required=True)

    # >>> dev-harness only
    p = sp.add_parser("build-suite", help="draw sealed truths and generate datasets")
    p.add_argument("--seeds", default=None,
                   help="generate only these seed indices (e.g. 0 or 2,5-7): one Slurm array task per "
                        "seed builds the same bytes as a serial run; no manifest is written")
    p.add_argument("--manifest-only", action="store_true",
                   help="assemble manifest.json from the per-seed truth files after an array build")
    p.add_argument("--task", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--n-seeds", type=int, default=11, help="including the public seed (ignored with --design)")
    p.add_argument("--master-seed", type=int, required=True)
    p.add_argument("--design", help="JSON file: {\"seeds\": [{\"mH\": 120, \"mu\": 2}, ...]}")
    p.add_argument("--priors", help="JSON object overriding task priors")
    p.add_argument("--no-response", action="store_true", help="build with variations off")
    p.add_argument("--out", default="suites")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_build_suite)
    p = sp.add_parser("bundle-fingerprint", help="the fingerprint of an MC bundle or a dataset's mc/ (sbh/bundle.py)")
    p.add_argument("directory", nargs="?", help="default: the task's current bundle (--task)")
    p.add_argument("--task", help="leave out its bundle's evaluator files and compare with its recorded fingerprint")
    p.set_defaults(fn=cmd_bundle_fingerprint)
    # <<< dev-harness only

    # >>> evaluator only
    cmds = [("eval", cmd_eval, "freeze + run on sealed seeds + score one submission"),
            ("closure", cmd_closure, "run the reference pipeline; require it to recover the truths; set sigma_ref"),
            ("stress", cmd_stress, "evaluate the reference and every attacker / alternate reference")]
    for name, fn, hlp in cmds:
        p = sp.add_parser(name, help=hlp)
        p.add_argument("--suite", required=True)
        p.add_argument("--task", default=None, help="score the suite for this task (one listed in the manifest's "
                                                    "\"tasks\"; default: the manifest's task)")
        _add_run_args(p)
        if name == "eval":
            p.add_argument("--submission", required=True)
            p.add_argument("--label")
            p.add_argument("--with-attackers", action="store_true")
            p.add_argument("--replace", action="store_true",
                           help="discard the label's earlier records and score (prepare / phase all refuse otherwise)")
        if name in ("closure", "verify"):
            p.add_argument("--leakage-nsigma", type=float, default=None,
                           help="sigmas of slack on each leakage probe (default 2)")
            p.add_argument("--scored-calibration", default=None,
                           help="'auto' (the task's value-scored set) or a comma list of logged nuisances that "
                                "must be reported; truth-varying ones must track the injected truth (slope gate)")
            p.add_argument("--calibration-slope-tol", type=float, default=None,
                           help="|slope-1| tolerance for scored calibration entries (default 0.25)")
            p.add_argument("--leakage-se-cap", type=float, default=None,
                           help="gate a leakage probe only if se(k) <= this (an underpowered probe cannot "
                                "distinguish calibrated from not); default: gate every probe")
            p.add_argument("--max-leakage", type=float, default=None,
                           help="also require every calibration-leakage probe |k| <= this + 2 sigma")
            p.add_argument("--calibrate-bias", action="store_true",
                           help=(
                               "remove the reference's mean (reported - injected) per POI before the "
                               "gates (shared-background suites); closure stores it"
                           ))
        if name == "closure":
            p.add_argument("--max-pull", type=float, default=None,
                           help=f"largest |pull| (default {POI_MAX_PULL}; an output kind with closure criteria of its "
                                "own reads it as its own bound: the corrections kind's largest |mean pull|, default the "
                                "task's closure_max_pull)")
        if name in ("eval", "closure", "stress"):
            p.add_argument("--phase", choices=("all", "prepare", "run", "yardstick", "post", "score"), default="all",
                           help="split form for one seed per batch job: prepare, run, yardstick "
                                "(the task's reference timed for the compute term, when it asks for it), post (the "
                                "output kind's per-dataset post stage), score; default all (every step in this "
                                "process: the runs --jobs at a time, the post stage one dataset after the other)")
        if name == "stress":
            p.add_argument("--only", help="comma list of submission dir names")
            p.add_argument("--list", action="store_true",
                           help="print the stress submissions' names (the split chains' labels) and exit")
            p.add_argument("--recheck", action="store_true",
                           help="check the invariant on the stressed runs' stored scores (after their post stage, "
                                "which rescores them); nothing is run")
        p.set_defaults(fn=fn)
    # <<< evaluator only

    # >>> dev-harness only
    p = sp.add_parser("scoreboard", help="merge public summaries of several runs")
    p.add_argument("runs", nargs="+")
    p.add_argument("--with-attackers", action="store_true")
    p.set_defaults(fn=cmd_scoreboard)

    # <<< dev-harness only

    p = sp.add_parser("check", help="participant self-check on one dataset")
    p.add_argument("--task", required=True)
    p.add_argument("--submission", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--determinism", action="store_true")
    _add_run_args(p)
    p.set_defaults(fn=cmd_check)

    # >>> dev-harness only

    p = sp.add_parser("post-score", help="the output kind's steps after a run is scored")
    p.add_argument("--suite", required=True)
    p.add_argument("--task", default=None, help="the task the suite is scored for (default: the manifest's)")
    p.add_argument("--label", required=True, help="the run directory under <work>/<suite>/")
    p.add_argument("--work", default="work")
    p.set_defaults(fn=cmd_post_score)

    p = sp.add_parser("pack", help="write the participant bundle (public seed + docs + self-check tool)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--suite", help="a suite: its public dataset and task")
    g.add_argument("--public-dataset", help="a released public dataset (its downloaded directory), with --task")
    p.add_argument("--task", default=None, help="with --public-dataset: the task it is for")
    p.add_argument("--out", default="dist")
    p.add_argument("--prompt", help="prompt tier file to ship as PROMPT.md (default: the task's baseline)")
    p.set_defaults(fn=cmd_pack)

    # <<< dev-harness only

    a = ap.parse_args(argv)
    return a.fn(a)
