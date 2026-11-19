"""MODEL.json checks: contract rules beyond "builds", the grader's pyhf, and the build/refit check shared by the run
gate, `sbh check` and the grader."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ...contract import CONTRACT_PREFIX, _finite, load_output_json
from ...hep import names as NN

GATE_PREFIXES = ("invalid workspace", "MODEL.json unreadable", CONTRACT_PREFIX)

POI_POWERS = (-1.0, -0.5, 0.0, 0.5, 1.0)

ENGINE_MODIFIERS = ("normfactor", "normsys", "histosys", "staterror", "shapesys", "lumi")


def constraint_width_problem(cfg) -> Optional[str]:
    """Why a built pyhf model (its ``cfg``, model.config) has no likelihood a fit can minimise, from its constraint
    terms: the first Gaussian constraint whose width (a lumi parameter's sigmas, a staterror bin's relative
    uncertainty; 1 for normsys and histosys) is not a positive finite number, or the first Poisson constraint
    (shapesys) whose factor (nominal / uncertainty)^2 is not finite, i.e. whose relative width is 0 or not a number,
    as "constraint width not a positive finite number: <name> width <w> (<kind> constraint)"; None when every width is
    one.  pyhf's own fit fails on each of these (a normal density of width <= 0 or not a number has no
    log-probability, an infinite one none that varies), so the grader refuses them before any fit, on both engines and
    both grading kinds.  pyhf fixes a staterror bin of zero uncertainty itself (width 1), so that one is no problem."""
    import math
    for name in cfg.auxdata_order:
        ps = cfg.param_set(name)
        n = int(ps.n_parameters)
        if ps.pdf_type == "normal":
            widths, kind = list(getattr(ps, "sigmas", None) or [1.0] * n), "Gaussian"
            bad = [(j, w) for j, w in enumerate(widths) if not (math.isfinite(float(w)) and float(w) > 0.0)]
        elif ps.pdf_type == "poisson":
            kind = "Poisson"
            bad = [(j, 0.0 if math.isinf(float(f)) else float("nan"))
                   for j, f in enumerate(getattr(ps, "factors", None) or []) if not math.isfinite(float(f))]
        else:
            continue
        if bad:
            j, w = bad[0]
            label = name if n == 1 else f"{name}[{j}]"
            return f"constraint width not a positive finite number: {label} width {float(w):g} ({kind} constraint)"
    return None


def poi_scaling_entries(table, spec) -> dict:
    """{(channel, sample): (c, its monomials [{P: p}, ...], its markers {P with a non-zero power})} of a POI-scaling
    table {channel: {sample: entry}} of the pyhf ``spec`` (the grammar, one parser for every reader): entry
    {"factor": c, "powers": {P: p}} or {"factor": c, "min_of": [{P: p}, ...]}, exactly one of the two; c finite and
    non-zero; every p in POI_POWERS; the sample carries a normfactor named P for exactly each P with a non-zero power.
    ValueError naming the entry on anything else."""
    import math
    by_name = {c["name"]: c for c in spec.get("channels", [])}
    out = {}
    for ch, per in table.items():
        for name, e in per.items():
            sample = next((s for s in by_name[ch]["samples"] if s["name"] == name), None) if ch in by_name else None
            if sample is None:
                raise ValueError(f"poi_scaling {ch}/{name}: no such sample")
            if set(e) not in ({"factor", "powers"}, {"factor", "min_of"}):
                raise ValueError(f"poi_scaling {ch}/{name}: keys {sorted(e)}, want factor and one of powers, min_of")
            c = float(e["factor"])
            monos = [e["powers"]] if "powers" in e else list(e["min_of"])
            if not math.isfinite(c) or c == 0 or not monos or any(p not in POI_POWERS
                                                                   for m in monos for p in m.values()):
                raise ValueError(f"poi_scaling {ch}/{name}: factor {c} (finite, non-zero) and powers in {POI_POWERS}, "
                                 "at least one monomial")
            markers = frozenset(P for m in monos for P, p in m.items() if p != 0)
            nf = {m["name"] for m in sample.get("modifiers", []) if m["type"] == "normfactor"}
            carried = nf & set().union(*monos)
            if carried != markers:
                raise ValueError(f"poi_scaling {ch}/{name}: carries the normfactors {sorted(carried)} of its POIs, not "
                                 f"exactly those with a non-zero power, {sorted(markers)}")
            out[(ch, name)] = (c, monos, markers)
    return out


def shape_problems(spec: dict) -> List[str]:
    """What the gate's rules read of a workspace, in pyhf's shape: channels with a name and samples, each sample with
    a name, numbers as data and modifiers with a name and a type; measurements, where given, with a config whose poi,
    where given, is a string."""
    chans = spec.get("channels")
    if not (isinstance(chans, list) and all(isinstance(c, dict) and isinstance(c.get("name"), str)
                                            and isinstance(c.get("samples"), list) for c in chans)):
        return ["channels must be a list of objects with a name (string) and samples (list)"]
    for c in chans:
        for s in c["samples"]:
            if not (isinstance(s, dict) and isinstance(s.get("name"), str) and isinstance(s.get("data"), list)
                    and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in s["data"])
                    and isinstance(s.get("modifiers"), list)
                    and all(isinstance(m, dict) and isinstance(m.get("name"), str) and isinstance(m.get("type"), str)
                            for m in s["modifiers"])):
                return [f"channel {c['name']}: every sample needs a name (string), data (numbers) and modifiers (a "
                        "list), each with a name and a type (strings)"]
    meas = spec.get("measurements", [])
    if not (isinstance(meas, list) and all(isinstance(m, dict) and isinstance(m.get("config"), dict)
                                           and isinstance(m["config"].get("poi", ""), str) for m in meas)):
        return ["measurements must be a list of objects with a config (object) whose poi is a string"]
    return []


def check_model_contract(spec: dict, result=None, task=None, extra_pois=(), extra_rules=None) -> List[str]:
    """Rules beyond "builds": the workspace in pyhf's shape (shape_problems: alone when it fails, the other rules
    read no other), grammar names on every nuisance modifier (not POI or bin-wise MC-stat), each logged in
    RESULT.json["nuisances"], per-channel `_meta.binning` (observable, edges, data_derived), and a kind's
    ``extra_rules`` (spec -> problem texts).  ``extra_pois``: modifier names that are POIs besides the
    measurement's."""
    errs = [CONTRACT_PREFIX + p for p in shape_problems(spec)]
    if errs:
        return errs
    prefix = getattr(task, "nuisance_prefix", "") if task is not None else ""
    names = NN.model_nuisances(spec, extra_pois)
    bad = NN.unparsed(names, prefix, getattr(task, "processes", ()) if task is not None else ())
    if bad:
        errs.append(CONTRACT_PREFIX + "MODEL.json modifier names outside the naming grammar: " + ", ".join(bad[:8])
                    + (" …" if len(bad) > 8 else ""))
    if isinstance(result, dict):
        log = result.get("nuisances") if isinstance(result.get("nuisances"), dict) else {}
        missing = [n for n in names if n not in log]
        if missing:
            errs.append(CONTRACT_PREFIX + "MODEL.json nuisance parameters missing from RESULT.json nuisances: "
                        + ", ".join(missing[:8]) + (" …" if len(missing) > 8 else ""))
    binning = (spec.get("_meta") or {}).get("binning") if isinstance(spec.get("_meta"), dict) else None
    for ch in spec.get("channels") or []:
        name = (ch or {}).get("name")
        b = (binning or {}).get(name) if isinstance(binning, dict) else None
        nb = len(((ch.get("samples") or [{}])[0] or {}).get("data") or [])
        edges = b.get("edges") if isinstance(b, dict) else None
        ok = (isinstance(b, dict) and isinstance(b.get("observable"), str) and b["observable"]
              and isinstance(b.get("data_derived"), bool) and isinstance(edges, list) and len(edges) == nb + 1
              and all(_finite(x) for x in edges) and all(x < y for x, y in zip(edges, edges[1:])))
        if not ok:
            errs.append(CONTRACT_PREFIX + f"_meta.binning.{name}: needs 'observable' (string), 'edges' ({nb + 1} "
                        "increasing numbers) and 'data_derived' (boolean)")
    if extra_rules is not None:
        errs += [CONTRACT_PREFIX + p for p in extra_rules(spec)]
    return errs


GRADER_PYHF = "0.7.6"
PYHF_MISSING_NOTE = (f"pyhf is not importable here, so the workspace build and refit were NOT VALIDATED HERE "
                     f"(the grader builds MODEL.json with pyhf {GRADER_PYHF} and refits that model; a pyhf "
                     "vendored into your "
                     "submission's pylib/ is picked up by this self-check)")
SELF_CHECK_REFIT_NOTE = "the evaluator refits your model itself; agreement here is necessary, not sufficient"


def import_pyhf(pylib=None):
    """pyhf or None: the interpreter's packages, then SBH_GRADER_LIB (root-only in TB).  Only the self-check passes
    `pylib` (the submission's vendored dir), tried last and APPENDED so numpy/scipy keep precedence; the grader never
    imports from a submission."""
    from ... import append_sys_path, use_grader_lib
    use_grader_lib()
    try:
        import pyhf
        return pyhf
    except ImportError:
        if pylib is None or not append_sys_path(pylib):
            return None
    import sys
    for name in [m for m in sys.modules if m == "pyhf" or m.startswith("pyhf.")]:
        del sys.modules[name]
    try:
        import pyhf
        return pyhf
    except ImportError:
        return None


NEGATIVE_RATE_TOL = -1e-9


def check_model(model_path, reported_mu=None, build_only=False, result=None, task=None,
                pylib=None, fit=None, info=None, extra_pois=(), extra_rules=None) -> List[str]:
    """Validate MODEL.json: parseable, check_model_contract (with ``extra_pois``, ``extra_rules``), pyhf-buildable
    and (unless build_only) MLE-fittable with mu within 0.25·σ + 0.05 of the reported one.  Returns problem strings
    (empty = OK).  The requires_model GATE is build_only (GATE_PREFIXES); fit health is what model_consistency (TB
    criterion 7) counts.

    `fit(spec, model, data) -> (pars in pyhf order | None, fit_info)`: the MLE engine; None falls back to pyhf's
    MLE.  `info` receives fit_info, "engine", "mu_model", and diagnostics "min_expected_rate" / "n_negative_rate_bins"
    (pyhf's main-bin rates at the best fit vs NEGATIVE_RATE_TOL), never problems.

    Truth-free, so it runs in the self-check.  pyhf is a grader dependency, not a runtime promise; without it
    PYHF_MISSING_NOTE is returned, which the evaluator never waives (sbh/runner.py).
    """
    try:
        spec = load_output_json(model_path)
    except Exception as e:
        return [f"MODEL.json unreadable: {e}"]
    if not isinstance(spec, dict):
        return ["invalid workspace: MODEL.json is not a JSON object"]
    errs: List[str] = check_model_contract(spec, result, task, extra_pois, extra_rules)
    pyhf = import_pyhf(pylib)
    if pyhf is None:
        return errs + [PYHF_MISSING_NOTE]
    spec.pop("_meta", None)
    try:
        ws = pyhf.Workspace(spec)
        model = ws.model()
        data = ws.data(model)
    except Exception as e:
        return errs + ["invalid workspace: " + str(e).splitlines()[0]]
    if build_only:
        return errs
    rec = info if isinstance(info, dict) else {}
    try:
        pars, engine = None, "pyhf"
        if fit is not None:
            pars, fit_info = fit(spec, model, data)
            rec.update(fit_info or {})
            if pars is not None:
                engine = rec.get("engine") or "custom"
        if pars is None:
            import time
            t0 = time.perf_counter()
            pars = pyhf.infer.mle.fit(data, model)
            rec["pyhf_fit_s"] = round(time.perf_counter() - t0, 3)
        rec["engine"] = engine
        mu_hat = float(pars[model.config.poi_index])
        rec["mu_model"] = mu_hat
        if reported_mu is not None:
            tol = 0.25 * float(reported_mu.get("total", 0.0)) + 0.05
            if abs(mu_hat - float(reported_mu["value"])) > tol:
                errs.append(f"model MLE mu = {mu_hat:.3f} disagrees with reported "
                            f"{reported_mu['value']:.3f} (tolerance {tol:.3f})")
    except Exception as e:
        errs.append("workspace does not fit: " + str(e).splitlines()[0])
        return errs
    try:
        import numpy as np
        lam = np.asarray(model.expected_actualdata(pars), dtype=float)
        rec["min_expected_rate"] = float(lam.min()) if lam.size else None
        rec["n_negative_rate_bins"] = int(np.sum(~(lam >= NEGATIVE_RATE_TOL)))
    except Exception as e:  # noqa: BLE001
        rec["rate_diagnostics_error"] = (str(e).splitlines() or [type(e).__name__])[0][:200]
    return errs


def gate(mp: Path, result, task, self_check: bool = False, pylib=None, extra_pois=(), extra_rules=None):
    """The run gate on MODEL.json at ``mp``: it must exist and pass check_model build_only (with ``extra_pois``,
    ``extra_rules``) -> (problems, notes).  Missing pyhf is an evaluator error, never a waived gate, except in the
    self-check (pyhf-less agent image): pyhf-free rules still gate, plus a note."""
    if not os.path.lexists(mp):
        return ["MODEL.json not produced (required by the contract)"], []
    probs = check_model(mp, build_only=True, result=result, task=task, pylib=pylib, extra_pois=extra_pois,
                        extra_rules=extra_rules)
    notes = []
    if PYHF_MISSING_NOTE in probs:
        if not self_check:
            raise RuntimeError("task requires MODEL.json validation but pyhf is not importable "
                               "in the evaluator environment — refusing to waive the gate")
        probs = [p for p in probs if p != PYHF_MISSING_NOTE]
        notes.append(PYHF_MISSING_NOTE)
    return probs, notes


def model_consistency(runs: Path, reasons: Optional[Dict[str, List[str]]] = None,
                       detail: Optional[Dict[str, dict]] = None, engine: str = "grader", poi: str = "mu"):
    """(agreeing, total) over completed runs: MODEL.json, built by pyhf and refitted, reproduces the reported ``poi``
    within check_model's tolerance (GRADING.md criterion 7).  Any refit problem except contract-name rules counts as
    not reproducing; `reasons` collects them per seed, `detail` each refit record.  Fit engine: the grader's (GRADING.md
    criterion 7), or pyhf's MLE with `engine="pyhf"` or where the grader's is unavailable.
    The reported mu comes from record.json (runner-written), never out/RESULT.json, which the submission's uid can
    still write; MODEL.json is read via contract.read_output (no symlinks)."""
    fit = None
    if engine != "pyhf":
        try:
            from .refit import grader_refit as fit
        except ImportError:
            fit = None
    ok = n = 0
    for d in sorted(runs.glob("seed_*")):
        m = d / "out" / "MODEL.json"
        if not (os.path.lexists(m) and (d / "record.json").is_file()):
            continue
        try:
            res = json.loads((d / "record.json").read_text())["result"]
            mu = next(x for x in res["pois"] if x["name"] == poi)
        except Exception:
            continue
        n += 1
        info: dict = {}
        t0 = time.perf_counter()
        probs = [p for p in check_model(m, reported_mu=mu, fit=fit, info=info) if not p.startswith(CONTRACT_PREFIX)]
        ok += not probs
        if probs and reasons is not None:
            reasons[d.name] = probs
        if detail is not None:
            detail[d.name] = {**info, "mu_reported": mu.get("value"), "reproduces": not probs,
                              "wall_s": round(time.perf_counter() - t0, 3)}
    return ok, n


MODEL_MAX_BYTES = 256 * 2 ** 20
GRID_MAX_BYTES = 256 * 2 ** 20
GRID_MAX_BYTES_MARGINAL = 512 * 2 ** 20


class GridOverCap(ValueError):
    """The declared grid's files exceed the cap load_grid was given (grid_cap)."""


def grid_cap(task) -> Optional[int]:
    """The cap in bytes on ``task``'s declared grid, its files in all, as its contract states it (module comment):
    GRID_MAX_BYTES_MARGINAL for a task scoring mu on the mass-marginal density, else GRID_MAX_BYTES.  None without a
    task: no cap (the TB grader stage stages the grid within its cap before the verifier reads it)."""
    if task is None:
        return None
    return GRID_MAX_BYTES_MARGINAL if getattr(task, "mu_marginal_over_mass", False) else GRID_MAX_BYTES


def link_twice_nll_tol() -> float:
    """The grid link's 2NLL tolerance, scoring.LINK_TWICE_NLL_TOL (its one home); its value, 0.1, where the scorer is
    not shipped (the participant bundle's self-check; checks/grid_link_check.py keeps the two equal)."""
    try:
        from .scoring import LINK_TWICE_NLL_TOL
        return float(LINK_TWICE_NLL_TOL)
    except Exception:  # noqa: BLE001
        return 0.1


def workspace_events(spec) -> float:
    """sum n + sum nu of a parsed workspace: its observed counts and its samples' nominal yields, every channel (the
    scale of -2 ln L identity_rel divides by); a number that is not finite makes it infinite, a non-number counts 0."""
    import math
    rows = [o.get("data") for o in (spec.get("observations") or []) if isinstance(o, dict)]
    rows += [s.get("data") for c in (spec.get("channels") or []) if isinstance(c, dict)
             for s in (c.get("samples") or []) if isinstance(s, dict)]
    total = 0.0
    for row in rows:
        for x in row if isinstance(row, list) else ():
            if isinstance(x, (int, float)) and not isinstance(x, bool):
                try:
                    v = abs(float(x))
                except OverflowError:
                    return math.inf
                total += v if math.isfinite(v) else math.inf
    return total


def identity_rel(spec, tol: Optional[float] = None) -> float:
    """The same-workspace tolerance of MODEL.json ``spec``: r_max = tol / (4 max(N, 1)), N = workspace_events(spec),
    tol the link's (link_twice_nll_tol) unless given (module comment); 0 (exact equality) when N is not finite."""
    import math
    n = workspace_events(spec) if isinstance(spec, dict) else math.inf
    if not math.isfinite(n):
        return 0.0
    return float(link_twice_nll_tol() if tol is None else tol) / (4.0 * max(n, 1.0))


def nearest_point(masses, m0) -> Tuple[Optional[int], float, float, bool]:
    """The grid link's "grid point at your reported mass" (both routes): (index into ``masses`` of the mass nearest
    ``m0``, its distance, the local spacing = the larger gap to its neighbours in mass order (0 for a single mass),
    near = the distance within that spacing): a grid locates the reported mass to its own resolution there.  Ties go
    to the lower mass.  (None, inf, 0, False) without masses."""
    import math
    ms = [float(m) for m in masses]
    if not ms:
        return None, math.inf, 0.0, False
    order = sorted(range(len(ms)), key=lambda i: ms[i])
    pos = min(range(len(order)), key=lambda p: abs(ms[order[p]] - float(m0)))
    j = order[pos]
    gaps = [abs(ms[order[q]] - ms[j]) for q in (pos - 1, pos + 1) if 0 <= q < len(order)]
    spacing = max(gaps) if gaps else 0.0
    dist = abs(ms[j] - float(m0))
    return j, dist, spacing, bool(dist <= spacing)


def _grid_files(out_dir):
    """The declared grid's files, listed as the contract reads output files (never through a symlink, nothing parsed):
    ("dir", {mH: (path, bytes)}) for MODEL_GRID/ with a point, a regular file named mH_<value>.json with a finite
    value (a symlinked or misnamed entry is skipped, as a stray file is); else ("file", path, bytes) for
    MODEL_GRID.json; None for no grid, also for a MODEL_GRID that is not a directory itself (a symlink, a file)."""
    import math
    import stat
    out_dir = Path(out_dir)
    gd = out_dir / "MODEL_GRID"
    try:
        st = os.lstat(gd)
    except OSError:
        st = None
    if st is not None:
        if not stat.S_ISDIR(st.st_mode):
            return None
        points = {}
        for f in sorted(gd.glob("mH_*.json")):
            try:
                m = float(f.stem[3:])
                fs = os.lstat(f)
            except (ValueError, OSError):
                continue
            if math.isfinite(m) and stat.S_ISREG(fs.st_mode):
                points[m] = (f, int(fs.st_size))
        if points:
            return "dir", points
    gj = out_dir / "MODEL_GRID.json"
    try:
        return "file", gj, int(os.lstat(gj).st_size)
    except OSError:
        return None


def grid_bytes(out_dir) -> Optional[int]:
    """The declared grid's size as load_grid reads it (its point files in all, or MODEL_GRID.json), None without one."""
    files = _grid_files(out_dir)
    if files is None:
        return None
    return sum(n for _, n in files[1].values()) if files[0] == "dir" else files[2]


def load_grid(out_dir, max_bytes: Optional[int] = None) -> dict:
    """{mH: path-or-spec} of the declared grid: MODEL_GRID/mH_<value>.json files, else MODEL_GRID.json
    {"<mH>": spec}, listed by _grid_files (never through a symlink: MODEL_GRID must be a directory itself, each point a
    regular file).  MODEL_GRID.json is read by contract.read_output (a symlink, a hard link or an unreadable file is no
    grid).  Unparsable names/keys and non-finite masses are skipped (a stray file must not fail the seed).
    ``max_bytes`` (grid_cap): a grid whose files exceed it in all raises GridOverCap before anything is read; the
    points' own files are read later, by read_point."""
    import math
    from ...contract import read_output
    files = _grid_files(out_dir)
    if files is None:
        return {}
    size = files[2] if files[0] == "file" else sum(n for _, n in files[1].values())
    if max_bytes is not None and size > max_bytes:
        name = "MODEL_GRID/" if files[0] == "dir" else "MODEL_GRID.json"
        raise GridOverCap(f"{name} is {size / 2 ** 20:.1f} MiB, above the {max_bytes / 2 ** 20:g} MiB cap on the grid")
    if files[0] == "dir":
        return {m: f for m, (f, _) in files[1].items()}
    try:
        blob = json.loads(read_output(files[1]))
    except Exception:  # noqa: BLE001
        return {}
    points = {}
    if isinstance(blob, dict):
        for k, v in blob.items():
            try:
                m = float(k)
            except (TypeError, ValueError):
                continue
            if math.isfinite(m):
                points[m] = v
    return points


def read_point(src):
    """One grid point's workspace as parsed: ``src`` a MODEL_GRID/ file (load_grid; read by contract.read_output, never
    through a symlink) or the spec MODEL_GRID.json held."""
    if isinstance(src, Path):
        from ...contract import read_output
        return json.loads(read_output(src))
    return src


class _Named(dict):
    """A name-keyed list of a workspace as {key: entry} (same_workspace): pyhf orders these lists by name itself."""


def _by_name(xs, key):
    """``xs`` as _Named {key(entry): entry} when every entry is an object with a string name and no key repeats, else
    ``xs`` as it is (compared in order)."""
    if not (isinstance(xs, list) and all(isinstance(x, dict) and isinstance(x.get("name"), str) for x in xs)):
        return xs
    keys = [key(x) for x in xs]
    return _Named(zip(keys, xs)) if len(set(keys)) == len(keys) else xs


def _pyhf_order(ws: dict) -> dict:
    """``ws`` without its top-level `_meta`, its name-keyed lists keyed by name (_by_name): channels, their samples,
    the samples' modifiers by (name, type), observations, and each measurement's parameters (the measurements
    themselves keep their order: Workspace.model() takes the first).  New containers where they change only; ``ws``
    is untouched."""
    def name(x):
        return x["name"]

    def name_type(x):
        return x["name"], str(x.get("type"))
    out = {k: v for k, v in ws.items() if k != "_meta"}
    if isinstance(out.get("channels"), list):
        chans = []
        for c in out["channels"]:
            if isinstance(c, dict) and isinstance(c.get("samples"), list):
                samples = [dict(s, modifiers=_by_name(s["modifiers"], name_type))
                           if isinstance(s, dict) and isinstance(s.get("modifiers"), list) else s
                           for s in c["samples"]]
                c = dict(c, samples=_by_name(samples, name))
            chans.append(c)
        out["channels"] = _by_name(chans, name)
    if isinstance(out.get("observations"), list):
        out["observations"] = _by_name(out["observations"], name)
    if isinstance(out.get("measurements"), list):
        meas = []
        for m in out["measurements"]:
            cfg = m.get("config") if isinstance(m, dict) else None
            if isinstance(cfg, dict) and isinstance(cfg.get("parameters"), list):
                m = dict(m, config=dict(cfg, parameters=_by_name(cfg["parameters"], name)))
            meas.append(m)
        out["measurements"] = meas
    return out


def same_workspace(a, b, rel: Optional[float] = None) -> Optional[str]:
    """None when the parsed workspaces ``a`` (the reference: MODEL.json) and ``b`` are the same workspace: equal as JSON
    with object key order, the top-level `_meta` and the order of the name-keyed lists ignored (_pyhf_order), other
    lists element by element in order, strings, booleans and nulls exactly, numbers to ``rel`` relative (default
    identity_rel(a); an int equals a float of the same value; a boolean is no number; NaN equals nothing; an infinity
    equals only itself).  Otherwise the first difference (object keys in sorted order, list entries in order, a named
    entry by its name), "<path>: <a's> vs <b's>" (e.g. 'channels["ch"].samples["sig"].data[3]: 4.0 vs 4.2').
    Iterative, so any nesting depth compares."""
    import math
    if rel is None:
        rel = identity_rel(a) if isinstance(a, dict) else 0.0

    def is_num(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool)

    def show(x):
        if isinstance(x, _Named):
            return f"a list of {len(x)} named entries"
        if isinstance(x, dict):
            return f"an object of {len(x)} keys"
        if isinstance(x, list):
            return f"a list of {len(x)}"
        return json.dumps(x)[:40]

    def scalar_diff(x, y):
        if is_num(x) and is_num(y):
            return not (x == y or math.isclose(x, y, rel_tol=rel, abs_tol=0.0))
        return type(x) is not type(y) or x != y

    def sub(path, x, k):
        if isinstance(x, _Named):
            label = f"{k[0]} ({k[1]})" if isinstance(k, tuple) else str(k)
            return f"{path}[{json.dumps(label)}]"
        return f"{path}.{k}" if path else str(k)

    if not (isinstance(a, dict) and isinstance(b, dict)):
        return f"(top): {show(a)} vs {show(b)}"
    stack = [("", _pyhf_order(a), _pyhf_order(b))]
    while stack:
        path, x, y = stack.pop()
        if isinstance(x, dict) and isinstance(y, dict) and isinstance(x, _Named) == isinstance(y, _Named):
            for k in sorted(set(x) | set(y), key=str):
                if k not in y or k not in x:
                    return f"{sub(path, x, k)}: {show(x[k]) if k in x else 'absent'} vs " \
                           f"{show(y[k]) if k in y else 'absent'}"
            stack += [(sub(path, x, k), x[k], y[k]) for k in sorted(x, key=str, reverse=True)]
        elif isinstance(x, list) and isinstance(y, list):
            if len(x) != len(y):
                return f"{path}: {show(x)} vs {show(y)}"
            nested = []
            for i, (u, v) in enumerate(zip(x, y)):
                if isinstance(u, (dict, list)) or isinstance(v, (dict, list)):
                    nested.append((f"{path}[{i}]", u, v))
                elif scalar_diff(u, v):
                    return f"{path}[{i}]: {show(u)} vs {show(v)}"
            stack += nested[::-1]
        elif isinstance(x, (dict, list)) or isinstance(y, (dict, list)) or scalar_diff(x, y):
            return f"{path}: {show(x)} vs {show(y)}"
    return None


def grid_identity(points: dict, mass, model_spec, rel: Optional[float] = None) -> dict:
    """Whether the declared grid holds MODEL.json at ``mass`` (the reported mH): ``points`` {mass: path-or-spec}
    (load_grid), ``model_spec`` MODEL.json as parsed (its `_meta` ignored).  The grid workspace compared is the one
    nearest ``mass`` within the local spacing (nearest_point), at any mass the grid declares, to the relative tolerance
    ``rel`` (default identity_rel(model_spec)).  Returns {"identical": True (the same workspace, same_workspace), False
    ("identical_diff" names the first difference) or None (no comparison: "identical_why" says why, e.g. no grid point
    within the local spacing), "identical_point": the grid mass compared, "identical_file": its file (MODEL_GRID/<name>
    or MODEL_GRID.json), "identical_distance" and "identical_spacing" (GeV), "identical_rel": the tolerance}.  Never
    raises."""
    out: dict = {"identical": None}
    try:
        m0 = float(mass)
    except (TypeError, ValueError, OverflowError):
        m0 = float("nan")
    try:
        import math
        if not math.isfinite(m0):
            return {**out, "identical_why": "no reported mass"}
        if not points:
            return {**out, "identical_why": "no grid"}
        masses = sorted(points)
        j, dist, spacing, near = nearest_point(masses, m0)
        at = masses[j]
        out.update(identical_distance=float(dist), identical_spacing=float(spacing))
        if not near:
            return {**out, "identical_why": f"no grid workspace within the local spacing of {m0:g} GeV (nearest "
                                            f"{at:g} GeV, {dist:.3g} GeV away; spacing {spacing:g} GeV)"}
        src = points[at]
        out.update(identical_point=float(at),
                   identical_file=f"MODEL_GRID/{src.name}" if isinstance(src, Path) else "MODEL_GRID.json")
        try:
            spec = read_point(src)
        except Exception as e:  # noqa: BLE001
            return {**out, "identical_why": f"{out['identical_file']} unreadable ({type(e).__name__})"}
        tol = identity_rel(model_spec) if rel is None else float(rel)
        out["identical_rel"] = tol
        diff = same_workspace(model_spec, spec, rel=tol)
        return {**out, "identical": diff is None, **({"identical_diff": diff} if diff else {})}
    except Exception as e:  # noqa: BLE001
        return {**out, "identical": None, "identical_why": f"not compared ({type(e).__name__}: {str(e)[:120]})"}
