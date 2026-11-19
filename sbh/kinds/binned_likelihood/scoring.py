"""Scoring of the binned_likelihood kind (EVALUATOR-ONLY): the score terms read from the evaluator's model
verification (model_verify_toys.json) and the declared MODEL.json -- score-from-fit, nuisance probes / completeness,
GOF, mass-grid profile, truthfulness and claims, calibration gate, yield response, standard-background flag."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ...contract import POI_FIELDS, load_output_json
from ...metrics import PoiResult
from ...hep.calibration import (SPLIT_RULE, _reported, calibration_term_gated, nuisance_log, syst_units,
                                unread_wp_entries)
from ... import score as _core
from ...score import V2_CAP_FLOOR, _cluster_se, _reported_poi, _wls, cluster_of, record_ok

V3_GOF_P_FLOOR = 0.005


GOF_RULE = "ensemble misfit (Fisher, 1 %, z-hinge)"


V3_GOF_ALPHA = 0.01


V3_GOF_DELTA_MIN = 0.0


VERIFY_FILE = "model_verify_toys.json"


def _changes(a, b) -> bool:
    return not np.allclose(np.asarray(a, float), np.asarray(b, float), rtol=1e-9, atol=1e-12)


def _occurrence_delta(t: str, d: dict, nom: np.ndarray, width: Optional[float]):
    """(hi, lo) change of a sample's bins under one constrained modifier at +-1 sigma: normsys nom (hi - 1) /
    nom (lo - 1), histosys hi_data - nom / lo_data - nom, lumi +-nom x relative width; None if unreadable."""
    try:
        if t == "normsys":
            return nom * (float(d.get("hi", 1.0)) - 1.0), nom * (float(d.get("lo", 1.0)) - 1.0)
        if t == "lumi":
            w = float(width or 0.0)
            return nom * w, -nom * w
        if t == "histosys":
            hi, lo = np.asarray(d.get("hi_data"), float), np.asarray(d.get("lo_data"), float)
            if hi.shape != nom.shape or lo.shape != nom.shape:
                return None
            return hi - nom, lo - nom
    except (TypeError, ValueError):
        pass
    return None


def _channel_sizes(parts: Dict[str, Dict[tuple, list]], total: np.ndarray, net: bool = False) -> tuple:
    """(norm, shape) of one effect in one channel of total nominal expectation ``total`` (sized against the
    CHANNEL, so a token sample cannot carry an effect).  ``parts`` = {modifier: {sample: [hi change, lo change]}}.
    Per part, max over hi / lo of
      norm  = sum over samples of sum_b |delta_b|, over sum_b total_b; ``net``: sum_b |sum over samples of delta_b|
              (a change and its negative copy, as in a sample and its negative-coefficient twin, cancel)
      shape = 1/2 sum_b |(total + delta) / sum(total + delta) - total / sum(total)|, delta summed over samples
    and the parts in quadrature."""
    T = float(total.sum())
    if not T > 0:
        return 0.0, 0.0
    unit = total / T
    n2 = s2 = 0.0
    for per_sample in parts.values():
        pn = ps = 0.0
        for k in (0, 1):
            dl = [v[k] for v in per_sample.values()]
            D = sum(x for x in dl if x.shape == total.shape) if any(x.shape == total.shape for x in dl) else None
            if net:
                pn = max(pn, float(np.abs(D).sum()) / T if D is not None else 0.0)
            else:
                pn = max(pn, float(sum(np.abs(x).sum() for x in dl)) / T)
            if D is not None:
                V = total + D
                if float(V.sum()) > 0:
                    ps = max(ps, 0.5 * float(np.abs(V / V.sum() - unit).sum()))
        n2 += pn * pn
        s2 += ps * ps
    return math.sqrt(n2), math.sqrt(s2)


CHANNEL_MINOR_SHARE = 0.02


def _counted_channels(totals: Dict[int, np.ndarray]) -> set:
    """The channels whose sizes count: all but the smallest ones together holding at most CHANNEL_MINOR_SHARE of the
    total nominal expectation (non-positive totals hold none).  No positive total: all channels."""
    tot = {ci: max(float(t.sum()), 0.0) for ci, t in totals.items()}
    T = sum(tot.values())
    if not T > 0:
        return set(tot)
    out, cum = set(), 0.0
    for ci in sorted(tot, key=lambda c: (tot[c], c)):
        if not out and cum + tot[ci] <= CHANNEL_MINOR_SHARE * T:
            cum += tot[ci]
        else:
            out.add(ci)
    return out


def nuisance_probes(spec: dict, names, prefix: str, floors: Optional[Dict[str, float]] = None,
                    sizes: Optional[dict] = None, sample_scale: Optional[Dict[tuple, float]] = None,
                    extra_pois=(), net_size: bool = False, processes=()) -> Dict[str, List[str]]:
    """Behavioural probes of the declared nuisances, per effect (sbh.hep.names.effect) of the modifiers `names`:
    {effect: failures}, empty when it passes.  A name alone covers nothing:
      inert      no occurrence changes a non-empty sample (normsys lo = hi = 1, histosys = nominal, lumi width 0)
      not_shape  a scale / resolution effect with no shape-changing histosys
      not_flat   a luminosity effect declared as other than normsys / lumi
      scope      luminosity missing on an MC-normalized sample (one carrying the POI or an object-efficiency modifier)
      below_floor  (``floors`` = Task.nuisance_floors) the effect's size is below its kind's floor: "shape" for
                 scale / resolution, "norm" otherwise (_channel_sizes), the largest over the counted channels
                 (_counted_channels).  Token modifiers, samples or channels cover nothing.  No floor: not checked.
    ``sizes`` (a dict) receives {effect: {"norm": size, "shape": size}}.  ``sample_scale`` ({(channel, sample):
    factor}): the sample's rate in the sizes and its channel's total is factor x its nominal (a POI-scaled sample's
    rate at the sizing point); whether a modifier moves or shapes it is read off its nominal.  ``extra_pois``: names
    that count as POIs besides the measurement's (scope).  ``net_size``: the norm size of the net per-bin change
    (_channel_sizes net).  ``processes``: the task's own process names (Task.processes), as `prefix` for the grammar."""
    from ...hep.names import CONSTRAINED_TYPES, effect, parse
    names = set(names)
    pois = {(m.get("config") or {}).get("poi") for m in spec.get("measurements") or [] if isinstance(m, dict)}
    pois |= set(extra_pois)
    widths = {p.get("name"): p.get("sigmas") for m in spec.get("measurements") or [] if isinstance(m, dict)
              for p in (m.get("config") or {}).get("parameters") or [] if isinstance(p, dict)}
    aux = {p.get("name"): p.get("auxdata") for m in spec.get("measurements") or [] if isinstance(m, dict)
           for p in (m.get("config") or {}).get("parameters") or [] if isinstance(p, dict)}
    occ: Dict[str, List[dict]] = {}
    mc_samples, where = set(), {}
    delta: Dict[str, Dict[int, Dict[str, Dict[tuple, list]]]] = {}
    totals: Dict[int, np.ndarray] = {}
    for ci, ch in enumerate(spec.get("channels") or []):
        for si, smp in enumerate((ch or {}).get("samples") or []):
            key = (ch.get("name"), smp.get("name"))
            nom = np.asarray(smp.get("data") or [0.0], float)
            live = bool(nom.sum() > 0)
            scale = (sample_scale or {}).get(key, 1.0)
            tot = totals.get(ci)
            if tot is None:
                totals[ci] = nom * scale
            elif tot.shape == nom.shape:
                totals[ci] = tot + nom * scale
            for mod in smp.get("modifiers") or []:
                n, t = mod.get("name"), mod.get("type")
                klass = (parse(n, prefix, processes) or (None,) * 5)[4]
                if n in pois or (klass or "").endswith("_efficiency"):
                    mc_samples.add(key)
                if n not in names or t not in CONSTRAINED_TYPES:
                    continue
                d = mod.get("data") or {}
                moves = shape = False
                if t == "normsys":
                    moves = live and (_changes(d.get("hi", 1.0), 1.0) or _changes(d.get("lo", 1.0), 1.0))
                elif t == "histosys":
                    hi, lo = np.asarray(d.get("hi_data"), float), np.asarray(d.get("lo_data"), float)
                    moves = live and (_changes(hi, nom) or _changes(lo, nom))
                    unit = lambda x: x / x.sum() if x.sum() > 0 else x
                    shape = moves and (_changes(unit(hi), unit(nom)) or _changes(unit(lo), unit(nom)))
                elif t == "lumi":
                    moves = live and any(s > 0 for s in (widths.get(n) or []))
                e = effect(n, prefix, processes)
                occ.setdefault(e, []).append({"type": t, "moves": moves, "shape": shape, "klass": klass})
                where.setdefault(e, set()).add(key)
                if moves:
                    w = None
                    if t == "lumi":
                        a = [x for x in (aux.get(n) or []) if isinstance(x, (int, float)) and x > 0]
                        s = [float(x) for x in (widths.get(n) or []) if isinstance(x, (int, float))]
                        w = max(s, default=0.0) / (float(a[0]) if a else 1.0)
                    dd = _occurrence_delta(t, d, nom, w)
                    if dd is not None:
                        dd = (dd[0] * scale, dd[1] * scale)
                        acc = delta.setdefault(e, {}).setdefault(ci, {}).setdefault(n, {})
                        prev = acc.get((si, key))
                        acc[(si, key)] = [dd[0], dd[1]] if prev is None else [prev[0] + dd[0], prev[1] + dd[1]]
    fl = floors or {}
    counted = _counted_channels(totals)
    out = {}
    for e, os_ in occ.items():
        klass = os_[0]["klass"] or ""
        fails = [] if any(o["moves"] for o in os_) else ["inert"]
        shape_kind = klass.endswith(("_energy_scale", "_resolution"))
        if shape_kind and not any(o["type"] == "histosys" and o["shape"] for o in os_):
            fails.append("not_shape")
        if klass == "luminosity":
            if any(o["type"] not in ("normsys", "lumi") for o in os_):
                fails.append("not_flat")
            if mc_samples - where[e]:
                fails.append("scope")
        per_ch = [_channel_sizes(parts, totals[ci], net_size) for ci, parts in (delta.get(e) or {}).items()
                  if ci in counted]
        size = {"norm": max((x[0] for x in per_ch), default=0.0), "shape": max((x[1] for x in per_ch), default=0.0)}
        kind = "shape" if shape_kind else "norm"
        floor = float(fl.get(kind) or 0.0)
        if floor > 0 and not {"inert", "not_shape"} & set(fails) and size[kind] < floor * (1.0 - 1e-9):
            fails.append("below_floor")
        if sizes is not None:
            sizes[e] = size
        out[e] = fails
    return out


def _fixed_parameters(spec: dict) -> set:
    """Names the model's measurement (the first, pyhf's default) fixes: a fixed modifier carries no uncertainty."""
    ms = spec.get("measurements") if isinstance(spec, dict) else None
    m = ms[0] if isinstance(ms, list) and ms and isinstance(ms[0], dict) else {}
    cfg = m.get("config") if isinstance(m.get("config"), dict) else {}
    return {p.get("name") for p in cfg.get("parameters") or [] if isinstance(p, dict) and p.get("fixed")}


def model_coverage(run_dir: Path, rec: dict, task, ungraded: Optional[set] = None, sample_scale_of=None,
                   extra_pois=(), net_size: bool = False) -> dict:
    """What one dataset's declared model covers, by nuisance_probes over its unfixed constrained modifiers, with the
    task's floors.  {"model": a readable MODEL.json of a completed run (and, under ``ungraded`` -- the strict rule --
    one the grader evaluated), and two probe sets:
      "modelled", "model_probes", "model_sizes"  every such modifier (the calibration term's "modelled")
      "effects", "probes", "sizes"  those also in RESULT.json's nuisance log (unc >= 0) -- Delta_syst's coverage
    effects = passing effects, probes = {effect: failures}, sizes = {effect: {"norm", "shape"}}}.  A modifier shared
    by several object types (scale_l) stands for each type's effect.  A log at unc 0 counts (avoids a cliff).
    ``sample_scale_of`` (MODEL.json -> nuisance_probes' sample_scale), ``extra_pois`` and ``net_size`` go to the
    probes; a model whose scale cannot be read covers nothing."""
    from ...hep.names import constrained_nuisances, covered as stands_for
    out = {"model": False, "effects": [], "probes": {}, "sizes": {}, "modelled": [], "model_probes": {},
           "model_sizes": {}}
    mj = Path(run_dir) / "runs" / rec["seed"] / "out" / "MODEL.json"
    if not (record_ok(rec) and os.path.lexists(mj) and rec["seed"] not in (ungraded or ())):
        return out
    try:
        spec = load_output_json(mj)
        mods = constrained_nuisances(spec) - _fixed_parameters(spec)
        scale = sample_scale_of(spec) if sample_scale_of is not None else None
    except Exception:
        return out
    log = nuisance_log(rec)
    logged = {n for n, e in log.items() if _reported(e) is not None} - unread_wp_entries(task, log)
    prefix, floors = getattr(task, "nuisance_prefix", ""), getattr(task, "nuisance_floors", None)
    procs = getattr(task, "processes", ())
    sizes: Dict[str, dict] = {}
    probes = nuisance_probes(spec, mods & logged, prefix, floors, sizes, scale, extra_pois, net_size, procs)
    msizes: Dict[str, dict] = {}
    mprobes = nuisance_probes(spec, mods, prefix, floors, msizes, scale, extra_pois, net_size, procs)
    out.update(model=True, effects=sorted({c for e, fails in probes.items() if not fails for c in stands_for(e)}),
               probes={e: list(f) for e, f in sorted(probes.items())}, sizes=sizes,
               modelled=sorted({c for e, fails in mprobes.items() if not fails for c in stands_for(e)}),
               model_probes={e: list(f) for e, f in sorted(mprobes.items())}, model_sizes=msizes)
    return out


def completeness_cases(run_dir: Path, records: List[dict], truths: Dict[str, dict], task,
                       ungraded: Optional[set] = None, coverage: Optional[Dict[str, dict]] = None) -> List[dict]:
    """One case per dataset: delta = fraction of its syst_units (under its log's working points) not covered.  A
    unit is covered when one of its effects is in model_coverage "effects"; a failed or modelless dataset (or, under
    ``ungraded``, one the grader could not evaluate) covers nothing.  `rejected`: unit effects declared but failing a
    probe.  ``coverage`` ({seed: model_coverage}) is computed where not given."""
    from ...hep.names import covered as stands_for
    if not syst_units(task) and not any(syst_units(task, nuisance_log(rec)) for rec in records):
        return []
    cases = []
    for rec in records:
        units = syst_units(task, nuisance_log(rec))
        if not units:
            continue
        unit_effects = {n for _, names in units for n in names}
        cov = (coverage or {}).get(rec["seed"]) or model_coverage(run_dir, rec, task, ungraded)
        effects = set(cov.get("effects") or ())
        rejected = {e: fails for e, fails in sorted((cov.get("probes") or {}).items())
                    if fails and set(stands_for(e)) & unit_effects}
        covered = {label for label, names in units if effects & set(names)}
        cases.append({"seed": rec["seed"], "missing": sorted(label for label, _ in units if label not in covered),
                      "rejected": rejected, "delta": 1.0 - len(covered) / len(units),
                      "units": [label for label, _ in units], "floors": dict(getattr(task, "nuisance_floors", None) or {}),
                      "cluster": cluster_of(task, truths[rec["seed"]], rec["seed"])})
    return cases


BG_RULE = "missing standard background (report-only)"


def _bg_name(name) -> str:
    """A sample name as the flag matches it: lower case, letters and digits only ("Z_plus_X" -> "zplusx")."""
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())


def background_processes(spec: dict, task) -> Optional[dict]:
    """Which of Task.standard_backgrounds one declared model carries.  REPORT-ONLY, truth-free (sample names and
    nominal yields).  Lenient on naming:
      signal      a sample carrying the POI modifier, or matching Task.standard_background_signal
      process     other samples match on _bg_name against each process's patterns (re.search), else against
                  Task.standard_background_merged (h4l: a plain "ZZ" counts for qq -> ZZ and gg -> ZZ)
      present     the process's samples hold >= Task.standard_background_min_share of the nominal background yield
                  (h4l 1e-4) -- else "token" (named, negligible) or "absent"
      flagged     a process is token, or absent with every other sample classified
      unresolved  absent processes only and some sample unclassified (a catch-all could hold them): not flagged
    None for a task with no standard backgrounds."""
    std = list(getattr(task, "standard_backgrounds", None) or [])
    if not std:
        return None
    merged = list(getattr(task, "standard_background_merged", None) or [])
    sig_re = [re.compile(p) for p in (getattr(task, "standard_background_signal", None) or ())]
    floor = float(getattr(task, "standard_background_min_share", 0.0) or 0.0)
    pois = {(m.get("config") or {}).get("poi") for m in spec.get("measurements") or [] if isinstance(m, dict)}
    pois.discard(None)
    yields: Dict[str, float] = {}
    signal = set()
    for ch in spec.get("channels") or []:
        for smp in (ch.get("samples") or []) if isinstance(ch, dict) else []:
            if not isinstance(smp, dict) or smp.get("name") is None:
                continue
            n = str(smp["name"])
            try:
                y = float(np.nansum(np.clip(np.asarray(smp.get("data") or [0.0], float), 0.0, None)))
            except (TypeError, ValueError):
                y = 0.0
            yields[n] = yields.get(n, 0.0) + (y if np.isfinite(y) else 0.0)
            if any(isinstance(m, dict) and m.get("name") in pois for m in smp.get("modifiers") or []):
                signal.add(n)
    signal |= {n for n in yields if any(r.search(_bg_name(n)) for r in sig_re)}
    bkg = {n: y for n, y in yields.items() if n not in signal}
    total = float(sum(bkg.values()))
    cls: Dict[str, List[str]] = {}
    for n in bkg:
        k = _bg_name(n)
        labels = [lab for lab, pats in std if any(re.search(p, k) for p in pats)]
        if not labels:
            labels = next((list(labs) for labs, pats in merged if any(re.search(p, k) for p in pats)), [])
        cls[n] = labels
    procs = {}
    for lab, _ in std:
        names = sorted(n for n, ls in cls.items() if lab in ls)
        share = float(sum(bkg[n] for n in names) / total) if total > 0 else 0.0
        procs[lab] = {"samples": names, "share": round(share, 6),
                      "status": "absent" if not names else ("present" if share >= floor else "token")}
    unclassified = sorted(n for n, ls in cls.items() if not ls)
    token = [lab for lab, _ in std if procs[lab]["status"] == "token"]
    absent = [lab for lab, _ in std if procs[lab]["status"] == "absent"]
    flagged = bool(token or (absent and not unclassified))
    missing = token + absent if flagged else []
    return {"flagged": flagged, "missing": [lab for lab, _ in std if lab in missing],
            "unresolved": [] if flagged or not absent else absent, "processes": procs,
            "signal": sorted(signal), "unclassified": unclassified}


def background_flags(run_dir: Path, records: List[dict], task) -> Optional[dict]:
    """The missing-standard-background flag over a run (REPORT-ONLY): background_processes on each completed
    dataset's MODEL.json (never through a symlink).  None for a task with no standard backgrounds."""
    if not getattr(task, "standard_backgrounds", None):
        return None
    per_seed: Dict[str, dict] = {}
    for rec in records:
        mj = Path(run_dir) / "runs" / rec["seed"] / "out" / "MODEL.json"
        if not (record_ok(rec) and os.path.lexists(mj)):
            continue
        try:
            spec = load_output_json(mj)
            r = background_processes(spec, task) if isinstance(spec, dict) else None
        except Exception:
            r = None
        if r is not None:
            per_seed[rec["seed"]] = r
    flagged = {s: r["missing"] for s, r in sorted(per_seed.items()) if r["flagged"]}
    unresolved = {s: {"absent": r["unresolved"], "unclassified": r["unclassified"]}
                  for s, r in sorted(per_seed.items()) if r["unresolved"]}
    labels = [lab for lab, _ in task.standard_backgrounds]
    return {"rule": BG_RULE, "report_only": True, "processes": labels,
            "min_share": float(getattr(task, "standard_background_min_share", 0.0) or 0.0),
            "n_checked": len(per_seed), "n_flagged": len(flagged), "flagged": flagged,
            "missing_counts": {lab: sum(lab in m for m in flagged.values()) for lab in labels},
            "n_unresolved": len(unresolved), "unresolved": unresolved, "per_seed": per_seed}


def _finite_p(v) -> Optional[float]:
    """A usable toy p-value, or None (absent, None, non-numeric or non-finite)."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


GOF_CHARGE_MAX = -math.log(V3_GOF_P_FLOOR) - 1.0


def gof_delta(p: float) -> dict:
    """Report-only per-dataset diagnostics of a toy p: {"delta_cap": max(V3_GOF_DELTA_MIN, raw), "delta_raw": raw},
    raw = -ln max(p, V3_GOF_P_FLOOR) - 1 (mean 0 for a correct model).  The scored charge is gof_ensemble's."""
    raw = -math.log(max(float(p), V3_GOF_P_FLOOR)) - 1.0
    return {"delta_cap": max(V3_GOF_DELTA_MIN, raw), "delta_raw": raw}


_GOF_MAX = {"delta": GOF_CHARGE_MAX, "delta_cap": GOF_CHARGE_MAX, "delta_raw": GOF_CHARGE_MAX}


def _chi2_logsf_even(x: float, k: int) -> float:
    """ln P(chi2_{2k} > x) exactly, as the Poisson cdf e^(-x/2) sum_{j<k} (x/2)^j / j! in log space (no underflow;
    no scipy in a grader runtime)."""
    if x <= 0:
        return 0.0
    h = 0.5 * x
    terms = [j * math.log(h) - math.lgamma(j + 1) for j in range(k)]
    top = max(terms)
    return -h + top + math.log(sum(math.exp(t - top) for t in terms))


def _z_of_log_p(lp: float) -> float:
    """Phi^-1(1 - exp(lp)): the one-sided Gaussian z of a p given as ln p (no underflow; no scipy in a grader runtime).
    Below ln p = -700 the asymptotic z^2 = -2 ln p - ln(-2 ln p) - ln(2 pi) (good to ~1e-3 there)."""
    if lp > -700.0:
        return -statistics.NormalDist().inv_cdf(math.exp(lp)) if lp < 0 else -math.inf
    t = -2.0 * lp
    return math.sqrt(t - math.log(t) - math.log(2.0 * math.pi))


def gof_charge(lp: float) -> float:
    """Delta_F of an ensemble p given as ln p: min(GOF_CHARGE_MAX, max(0, z(p_F) - z(V3_GOF_ALPHA))), the
    z-hinge: zero while p_F >= V3_GOF_ALPHA (a correct model is charged only when its p_F falls below alpha by
    chance, ~1 % of runs, and then little), linear in the Gaussian z beyond, so a borderline ensemble costs little
    (p_F = 0.002: 0.55) and a clear misfit most of the charge (p_F = 3e-8: 3.09)."""
    return min(GOF_CHARGE_MAX, max(0.0, _z_of_log_p(lp) - _z_of_log_p(math.log(V3_GOF_ALPHA))))


def _gof_usable(c: dict) -> bool:
    """A case in the ensemble test: has a p and is not charged the maximum (a wall stop has a p but is charged)."""
    return c.get("p") is not None and not c.get("charged")


def gof_gauss(ps: List[float]) -> Optional[dict]:
    """REPORT-ONLY Gaussian test of the p's: z_i = Phi^-1(clip(p_i, 0.005, 0.995)), mean m, width w (ddof 1),
    t_m = m sqrt(n), t_w = (w - 1) sqrt(2 (n - 1)) (n >= 3; both ~ N(0, 1) under a correct model).  A misfit drives
    m negative; too-good observations narrow w."""
    if not ps:
        return None
    import statistics
    nd = statistics.NormalDist()
    z = [nd.inv_cdf(min(max(float(p), V3_GOF_P_FLOOR), 1.0 - V3_GOF_P_FLOOR)) for p in ps]
    n = len(z)
    m = sum(z) / n
    out = {"n": n, "m": m, "t_m": m * math.sqrt(n), "w": None, "t_w": None, "report_only": True}
    if n >= 3:
        w = statistics.stdev(z)
        out.update(w=w, t_w=(w - 1.0) * math.sqrt(2.0 * (n - 1)))
    return out


def gof_ensemble(cases: List[dict]) -> dict:
    """The GOF term: over the n_ok cases with a usable p, X = -2 sum ln max(p_i, V3_GOF_P_FLOOR),
    p_F = chi2_sf(X, 2 n_ok), Delta_F = gof_charge(ln p_F) = min(GOF_CHARGE_MAX, max(0, z(p_F) - z(V3_GOF_ALPHA)));
    delta = (n_ok Delta_F + n_fail GOF_CHARGE_MAX) / n, n_fail the cases charged the maximum."""
    ok = [c for c in cases if _gof_usable(c)]
    n, n_ok = len(cases), len(ok)
    out = {"rule": GOF_RULE, "alpha": V3_GOF_ALPHA, "n": n, "n_ok": n_ok, "n_fail": n - n_ok,
           "n_wall": sum(1 for c in cases if c.get("wall_stop")), "fisher_x": None, "dof": 2 * n_ok,
           "log_p_f": None, "p_f": None, "delta_f": None, "delta": None}
    if n_ok:
        x = -2.0 * sum(math.log(max(float(c["p"]), V3_GOF_P_FLOOR)) for c in ok)
        lp = _chi2_logsf_even(x, n_ok)
        out.update(fisher_x=x, log_p_f=lp, p_f=math.exp(lp),
                   delta_f=gof_charge(lp))
    if n:
        out["delta"] = ((n_ok * out["delta_f"] if n_ok else 0.0) + (n - n_ok) * GOF_CHARGE_MAX) / n
    return out


def _gof_apply(cases: List[dict]) -> dict:
    """gof_ensemble, setting each case's "delta" (Delta_F or GOF_CHARGE_MAX) so the term is their mean."""
    e = gof_ensemble(cases)
    for c in cases:
        c["delta"] = e["delta_f"] if _gof_usable(c) else GOF_CHARGE_MAX
    return e


def gof_term(cases: List[dict]) -> Optional[dict]:
    """score_v3's GOF block: gof_ensemble, its SE by delete-one-cluster jackknife (a cluster SE of the deltas would
    read 0 for a fully completed run), and report-only means of delta_cap / delta_raw and gof_gauss."""
    if not cases:
        return None
    raw = [c.get("delta_raw", c.get("delta")) for c in cases]
    cap = [c.get("delta_cap", max(V3_GOF_DELTA_MIN, r)) for c, r in zip(cases, raw)]
    e = _gof_apply(cases)
    cl = [str(c.get("cluster", c.get("seed"))) for c in cases]
    groups = sorted(set(cl))
    se = 0.0
    if len(groups) >= 2:
        ts = [gof_ensemble([c for c, k in zip(cases, cl) if k != g])["delta"] for g in groups]
        tb = sum(ts) / len(ts)
        se = math.sqrt((len(ts) - 1) / len(ts) * sum((t - tb) ** 2 for t in ts))
    return {**e, "se": float(se), "delta_cap": float(np.mean(cap)), "delta_raw": float(np.mean(raw)),
            "delta_min": V3_GOF_DELTA_MIN, "n_charged": int(sum(bool(c.get("charged")) for c in cases)),
            "gauss": gof_gauss([float(c["p"]) for c in cases if _gof_usable(c)])}


def _gof_combined_case(r: dict) -> Optional[dict]:
    """The GOF case of one verified dataset: the combined p (global + window, same toys) of the truth-pinned stage,
    else the at-fit stage.  A p the MODEL cannot have (no dof, no bin in the window) is charged the maximum; a stage
    that gave no p for another reason passes to the next stage, and where none gives one None: gof_cases then decides
    (an evaluator fault on a stage is left out, a stage that ran and gave no p is the model's).  No gof_p_comb
    anywhere: None (the caller reads the legacy global p).  A wall stop (stage key tb_wall_rule) is charged the
    maximum, keeping p and p_seq."""
    ft = r.get("freq_toys") if isinstance(r.get("freq_toys"), dict) else {}
    ty = r.get("toys") if isinstance(r.get("toys"), dict) else {}
    stages = [(name, blk) for name, blk in (("truth_pinned", ft), ("at_fit", ty)) if "gof_p_comb" in blk]
    if not stages:
        return None
    for name, blk in stages:
        base = {"stage": name, "p_glob": _finite_p(blk.get("gof_p_glob")), "p_sr": _finite_p(blk.get("gof_p_sr")),
                "n_toys": blk.get("n_toys"), "aux": blk.get("gof_aux"), "n_sr_bins": blk.get("gof_n_sr_bins"),
                "note": blk.get("gof_p_note"), "rule": "combined"}
        pv = _finite_p(blk.get("gof_p_comb"))
        if pv is not None and blk.get("tb_wall_rule"):
            return {**base, "p": pv, "p_seq": _finite_p(blk.get("gof_p_comb_seq")), "charged": True,
                    "wall_stop": str(blk["tb_wall_rule"]), **_GOF_MAX}
        if pv is not None:
            return {**base, "p": pv, "charged": False, **gof_delta(pv)}
        if blk.get("gof_p_missing") in ("dof", "no_window"):
            return {**base, "p": None, "charged": True, "missing": blk["gof_p_missing"], **_GOF_MAX}
    return None


GOF_STAGES = ("freq_toys", "toys")


def gof_cases(run_dir: Path, records: List[dict], combined: bool = False,
              strict: bool = False, rows: Optional[Dict[str, dict]] = None, shipped: bool = False,
              left_out: Optional[Dict[str, str]] = None) -> Optional[List[dict]]:
    """Per-dataset GOF cases from the model verification, or None while it has not run (pending).  Each "delta" is
    the case's charge in gof_ensemble; "delta_cap" / "delta_raw" are gof_delta diagnostics.
    combined (score-from-fit): _gof_combined_case's p; otherwise the global p.  Charged GOF_CHARGE_MAX: a failed
    dataset, one without a usable model, a combine declaration's own failed GOF ("gof_failed") and a missing p that is
    the model's (a toy stage ran and gave no p, no evaluator error on it).  Left out (pending, never charged; in
    ``left_out``, {seed: reason}, when given): a completed dataset without a grader record (not verified yet; under
    strict, its grading lost: sbh.score.NO_GRADER_RECORD), one whose toy stages never ran, and an evaluator fault --
    the grader record (record_fault: its "evaluator_error") or, where no stage gives a p, a toy stage carrying
    "evaluator_error" -- with the reason "evaluator fault: ..." (sbh.score.EVALUATOR_FAULT), which a strict page
    refuses.  A record graded before the grader marked its own errors is read through regrade_marked (an error only
    the evaluator writes: the fault sbh.score.REGRADE).
    strict (Task.grader_strict): an evaluator fault and a missing record are left out all the same (worked out before
    anything strict decides: strict fails only what the contract fails); every other p-less dataset (toys not run) is
    charged.
    ``rows``: {seed: grader record} in place of the run's VERIFY_FILE (a kind whose grader writes one per dataset).
    ``shipped``: the shipped rule's reading (_shipped_gof_cases)."""
    if shipped:
        return _shipped_gof_cases(run_dir, records, combined=combined, strict=strict, rows=rows)
    unverified = f"GOF {_core.NO_GRADER_RECORD}" if strict else "GOF not verified yet"
    if rows is None:
        p = Path(run_dir) / VERIFY_FILE
        if not p.exists():
            if left_out is not None:
                left_out.update({rec["seed"]: unverified for rec in records if record_ok(rec)})
            return None
        rows = {r.get("seed"): r for r in (json.loads(p.read_text()).get("per_seed") or [])}
    rows = {k: regrade_marked(v) for k, v in rows.items()}
    left = left_out if left_out is not None else {}
    cases = []
    for rec in records:
        if record_ok(rec):
            r = rows.get(rec["seed"])
            if r is None:
                left[rec["seed"]] = unverified
                continue
            fault = record_fault(r)
            if fault:
                left[rec["seed"]] = fault
                continue
            if r.get("status") == "ok" and r.get("gof_failed"):
                pass
            elif r.get("status") == "ok":
                if combined:
                    c = _gof_combined_case(r)
                    if c is not None:
                        cases.append({"seed": rec["seed"], **c})
                        continue
                stages = {k: r[k] for k in GOF_STAGES if isinstance(r.get(k), dict)}
                pv = stage = None
                if not (combined and any("gof_p_comb" in b for b in stages.values())):
                    pv, stage = _finite_p((stages.get("freq_toys") or {}).get("gof_p_saturated")), "truth_pinned"
                    if pv is None:
                        pv, stage = _finite_p((stages.get("toys") or {}).get("gof_p_toys")), "at_fit"
                if pv is not None:
                    cases.append({"seed": rec["seed"], "p": float(pv), "stage": stage, "charged": False,
                                  **gof_delta(pv), **({"rule": "legacy_global"} if combined else {})})
                    continue
                stage_fault = next((f for f in (evaluator_fault(b) for b in stages.values()) if f), None)
                if stage_fault:
                    left[rec["seed"]] = stage_fault
                    continue
                if strict:
                    cases.append({"seed": rec["seed"], "p": None, "stage": None, "charged": True,
                                  **_GOF_MAX, "strict": "no GOF p from the grader"})
                    continue
                if not stages:
                    left[rec["seed"]] = "GOF toys not run"
                    continue
                cases.append({"seed": rec["seed"], "p": None, "stage": None, "charged": True, **_GOF_MAX,
                              "missing": next((str(b.get("gof_p_missing")) for b in stages.values()
                                               if b.get("gof_p_missing")), "no p")})
                continue
        cases.append({"seed": rec["seed"], "p": None, "stage": None, "charged": True, **_GOF_MAX})
    if cases:
        _gof_apply(cases)
    return cases or None


def gof_pending_reasons(left_out: Dict[str, str], n_records: int) -> List[str]:
    """The provisional reasons of the GOF datasets left out (gof_cases' ``left_out``), named reasons and counts: a
    stored record graded before the grader marked its own errors (sbh.score.REGRADE: grade it again), any other
    evaluator fault (sbh.score.EVALUATOR_FAULT: fixed and rerun), else not verified or the toys not run."""
    regrade = sum(1 for w in left_out.values() if str(w).startswith(_core.REGRADE_FAULT))
    faults = sum(1 for w in left_out.values() if str(w).startswith(_core.EVALUATOR_FAULT)) - regrade
    rest = len(left_out) - faults - regrade
    out = []
    if regrade:
        out.append(f"GOF left out on {regrade} of {n_records} datasets ({_core.REGRADE}), until they are graded again")
    if faults:
        out.append(f"GOF left out on an {_core.EVALUATOR_FAULT} on {faults} of {n_records} datasets, until it is fixed "
                   "and the evaluation rerun")
    if rest:
        out.append(f"GOF pending on {rest} of {n_records} datasets")
    return out


def _shipped_gof_cases(run_dir: Path, records: List[dict], combined: bool = False,
                       strict: bool = False, rows: Optional[Dict[str, dict]] = None) -> Optional[List[dict]]:
    """gof_cases as the shipped rule (sbh.score.SHIPPED_RULE) reads the grader's records: a dataset without a p from
    any stage, or one the grader raised on, pending (strict: charged)."""
    if rows is None:
        p = Path(run_dir) / VERIFY_FILE
        if not p.exists():
            if not strict:
                return None
            return [{"seed": rec["seed"], "p": None, "stage": None, "charged": True, **_GOF_MAX,
                     "strict": "no grader record"} for rec in records] or None
        rows = {r.get("seed"): r for r in (json.loads(p.read_text()).get("per_seed") or [])}
    per = rows
    cases = []
    for rec in records:
        if record_ok(rec):
            r = per.get(rec["seed"])
            if r is None and strict:
                r = {"status": "not graded"}
            if r is None:
                continue
            if not strict and (r.get("evaluator_error") or (combined and str(r.get("status") or "").startswith(
                    "verify error"))):
                continue
            if r.get("status") == "ok" and r.get("gof_failed"):
                pass
            elif r.get("status") == "ok":
                if combined:
                    c = _gof_combined_case(r)
                    if c is not None:
                        cases.append({"seed": rec["seed"], **c})
                        continue
                    if not strict and any("gof_p_comb" in (r.get(k) or {}) for k in ("freq_toys", "toys")
                                          if isinstance(r.get(k), dict)):
                        continue
                    if strict:
                        cases.append({"seed": rec["seed"], "p": None, "stage": None, "charged": True,
                                      **_GOF_MAX, "strict": "no GOF p from the grader"})
                        continue
                ft = r.get("freq_toys") if isinstance(r.get("freq_toys"), dict) else {}
                ty = r.get("toys") if isinstance(r.get("toys"), dict) else {}
                pv, stage = _finite_p(ft.get("gof_p_saturated")), "truth_pinned"
                if pv is None:
                    pv, stage = _finite_p(ty.get("gof_p_toys")), "at_fit"
                if pv is None and not strict:
                    continue
                if pv is None:
                    cases.append({"seed": rec["seed"], "p": None, "stage": None, "charged": True,
                                  **_GOF_MAX, "strict": "no GOF p from the grader"})
                    continue
                cases.append({"seed": rec["seed"], "p": float(pv), "stage": stage, "charged": False, **gof_delta(pv),
                              **({"rule": "legacy_global"} if combined else {})})
                continue
        cases.append({"seed": rec["seed"], "p": None, "stage": None, "charged": True, **_GOF_MAX})
    if cases:
        _gof_apply(cases)
    return cases or None


def envelope_crossings(mhs, q, root: bool = False):
    """(x_lo, x_hi) of the likelihood ENVELOPE on a grid curve q = 2 DeltaNLL (from the grid minimum): the outermost
    q = 1 crossings bounding every region with q < 1, interpolated linearly in q, or in sqrt(q) with ``root`` (exact
    for a parabola).  A side whose region reaches the grid edge: None."""
    below = [j for j in range(len(q)) if q[j] < 1.0]
    if not below:
        return None, None

    def at(a, b):
        if root:
            ra, rb = math.sqrt(max(q[a], 0.0)), math.sqrt(max(q[b], 0.0))
            f = (1.0 - ra) / (rb - ra) if rb != ra else 0.0
        else:
            f = (1.0 - q[a]) / (q[b] - q[a]) if q[b] != q[a] else 0.0
        return float(mhs[a] + f * (mhs[b] - mhs[a]))
    jl, jr = below[0], below[-1]
    return (at(jl, jl - 1) if jl > 0 else None), (at(jr, jr + 1) if jr < len(q) - 1 else None)


MASS_RANGE_TOL = 1e-3


def grid_profile(mg, mh_range=None) -> dict:
    """The scored mass from a declared grid's record (verify.py mass_grid_check's 2 DeltalnL(mH) curve):
      mh          the parabolic vertex through the grid minimum and its neighbours
      sigma_root  1/2 (s- + s+), s-/s+ the envelope's 2 DeltalnL = 1 crossings (envelope_crossings), interpolated in
                  sqrt(2 DeltalnL) (exact for a parabola; ``sigma_lin``, linear in 2 DeltalnL, underestimates up to 2x)
    usable iff the grid is one model and MODEL.json's (the grader did not mark it unlinked: verify_grid._grid_link),
    >= 3 points were fitted, the minimum is interior and BOTH crossings lie inside the grid with positive half-width (a
    one-sided width would let the submission pick a side).  A grid that is not MODEL.json's is not the run's model:
    no mass measurement, as the contract states for a grid that is not usable.
    ``mh_range`` (task mass_range (lo, hi) GeV): on a side where the grid REACHES the range end (within
    MASS_RANGE_TOL), a missing crossing or one beyond the end becomes the end ("range_bounded"), a vertex beyond it is
    clamped, and a minimum at that grid end is usable with half-width 0 there.  A side that stops short keeps the
    rule above; a profile whose whole region below 1 lies outside the range is unusable.
    Returns {"usable", "reason", ...}."""
    if not isinstance(mg, dict):
        return {"usable": False, "reason": "no grid"}
    curve = mg.get("curve") or []
    if mg.get("error") and not curve:
        return {"usable": False, "reason": f"grid error: {mg['error']}"}
    if not mg.get("grid_uniform", False):
        return {"usable": False, "reason": f"grid is not one model ({mg.get('grid_mismatch') or 'structure differs'})"}
    if mg.get("linked") is False:
        return {"usable": False, "reason": "grid is not MODEL.json's model: "
                                           + str((mg.get("link") or {}).get("why") or "linkage failed")}
    try:
        c = np.asarray(sorted((float(a), float(b)) for a, b, *_ in curve), dtype=float)
    except (TypeError, ValueError):
        return {"usable": False, "reason": "unreadable curve"}
    if len(c) < 3 or not np.all(np.isfinite(c)):
        return {"usable": False, "reason": "fewer than 3 fitted points"}
    mhs, q = c[:, 0], c[:, 1] - c[:, 1].min()
    i = int(np.argmin(q))
    out = {"n_points": int(len(q))}
    reach, r_lo, r_hi = (False, False), None, None
    if mh_range is not None:
        r_lo, r_hi = float(mh_range[0]), float(mh_range[1])
        reach = (bool(mhs[0] <= r_lo + MASS_RANGE_TOL), bool(mhs[-1] >= r_hi - MASS_RANGE_TOL))
        out.update(mh_range=[r_lo, r_hi], reaches_range=list(reach))
    edge = "lower" if i == 0 else ("upper" if i == len(q) - 1 else None)
    if edge and not reach[edge == "upper"]:
        return {**out, "usable": False, "reason": "profile minimum at a grid edge"}
    gaps = np.diff(mhs)
    out["spacing"] = float(max(gaps[max(i - 1, 0)], gaps[min(i, len(gaps) - 1)]))
    if edge:
        mh = r_lo if edge == "lower" else r_hi
        out["min_at_range_end"] = edge
    else:
        mh = _finite_p(mg.get("mh_model"))
        if mh is None:
            mh = float(mhs[i])
            x1, x2 = mhs[i - 1] - mhs[i], mhs[i + 1] - mhs[i]
            y1, y2 = q[i - 1] - q[i], q[i + 1] - q[i]
            den = x1 * x2 * (x1 - x2)
            if den != 0:
                a = (y1 * x2 - y2 * x1) / den
                b = (y2 * x1 * x1 - y1 * x2 * x2) / den
                if a > 1e-12:
                    mh = float(mhs[i] - 0.5 * b / a)
        if reach[0] and mh < r_lo:
            mh = r_lo
        if reach[1] and mh > r_hi:
            mh = r_hi
    out["mh"] = mh

    outside = False
    for tag, root in (("root", True), ("lin", False)):
        x_lo, x_hi = envelope_crossings(mhs, q, root)
        if mh_range is not None and ((x_lo is not None and x_lo >= r_hi) or (x_hi is not None and x_hi <= r_lo)):
            outside = True
        b_lo = bool(reach[0] and (x_lo is None or x_lo < r_lo))
        b_hi = bool(reach[1] and (x_hi is None or x_hi > r_hi))
        if b_lo:
            x_lo = r_lo
        if b_hi:
            x_hi = r_hi
        lo = (mh - x_lo) if x_lo is not None else None
        hi = (x_hi - mh) if x_hi is not None else None
        ok = [s is not None and (s > 0 or (b and s == 0)) for s, b in ((lo, b_lo), (hi, b_hi))]
        sides = [s for s, k in zip((lo, hi), ok) if k]
        out[f"sigma_lo_{tag}"], out[f"sigma_hi_{tag}"] = lo, hi
        out[f"sigma_{tag}"] = float(np.mean(sides)) if sides and max(sides) > 0 else None
        out[f"n_sides_{tag}"] = len(sides)
        if tag == "root":
            out["side_ok"] = ok
            if mh_range is not None:
                out["range_bounded"] = {"lower": b_lo, "upper": b_hi}
    if outside:
        return {**out, "usable": False,
                "reason": f"the profile's interval lies outside the {r_lo:g}-{r_hi:g} GeV mass range"}
    short = (f", and the grid reaches neither end of the {r_lo:g}-{r_hi:g} GeV mass range"
             if mh_range is not None else "")
    if not out["sigma_root"]:
        return {**out, "usable": False, "reason": "no DeltalnL = 0.5 crossing inside the grid" + short}
    if out["n_sides_root"] < 2:
        side = "lower" if not out["side_ok"][0] else "upper"
        end = (f", and the grid stops short of the {r_lo if side == 'lower' else r_hi:g} GeV end of the mass range"
               if mh_range is not None else "")
        return {**out, "usable": False,
                "reason": f"the grid does not cover the mass interval: no DeltalnL = 0.5 crossing on the {side} side"
                          + end}
    out["sigma_min"] = float(min(s for s in (out["sigma_root"], out["sigma_lin"]) if s))
    return {**out, "usable": True, "reason": "usable"}


MU_RULE = "marginal-v1"

SCAN_VERSION = "marginal-v1"

MU_SUPPORT = (0.0, 4.0)

REACH_TOL = 0.5

MAX_GAP_IN_RANGE = 1.0

CELL_HALF_CAP = 1.0

LINK_TWICE_NLL_TOL = 0.1

YIELD_BAND_NEAR, YIELD_BAND_FAR = math.log(1.2), math.log(1.5)
YIELD_BAND_NEAR_GEV = 5.0

CENTROID_CONST_TOL = 1.0
CENTROID_SLOPE = (0.8, 1.2)
CENTROID_MIN_SPAN = 4.0

SLOPE_EXTRA = 0.1

DEAD_YIELD_FRAC, DEAD_MAX_FRAC = 0.05, 0.10

WIDTH_TOL = (0.08, 0.04)
WIDTH_TOL_LOCAL = (0.04, 0.06)
WIDTH_LOCAL_GEV = 1.0
WIDTH_SCALING = (0.0, 0.25, 0.5, 0.75, 1.0)
SHARE_TOL = 0.15

HALF_LN_2PI = 0.5 * math.log(2.0 * math.pi)

MU_AXIS_STEP = 0.0025
LOCAL_HALF_WIDTH = 6.0
LOCAL_STEP = 0.05
BOUND_EDGE = 1e-9

PRIOR_CURVE_MAX_GAP = 2.0
PRIOR_CURVE_MAX_FAILED = 0.10
PRIOR_CURVE_TASK_STEP = 0.5

MU_BASELINE_MIN_FRAC = 0.9


def _stamp_at_least(stamp, floor: str) -> bool:
    """Whether a contract stamp ("2.0", "2.1", ...) is at least ``floor`` (numeric dotted compare; unreadable = no)."""
    def parts(s):
        return tuple(int(p) for p in str(s).strip().split("."))
    try:
        return parts(stamp) >= parts(floor)
    except (TypeError, ValueError):
        return False


def contract_stamp(rec: Optional[dict]) -> str:
    """The contract version a run record was made under (record.json "contract", sbh/runner.py); "2.0" when absent
    (every run before the stamp).  The score keys the coverage regime on it (marginal_mu)."""
    s = (rec or {}).get("contract") if isinstance(rec, dict) else None
    return str(s) if s else "2.0"


def logsumexp(a, axis=None):
    """ln sum exp(a) along ``axis`` without overflow (no scipy in a grader runtime); -inf entries contribute nothing,
    an all -inf slice gives -inf."""
    a = np.asarray(a, dtype=float)
    amax = np.max(a, axis=axis, keepdims=True)
    amax = np.where(np.isfinite(amax), amax, 0.0)
    with np.errstate(divide="ignore"):
        out = np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True)) + amax
    return float(out.reshape(-1)[0]) if axis is None else np.squeeze(out, axis=axis)


def _pchip_edge(h0, h1, m0, m1) -> float:
    """PCHIP's one-sided three-point end slope (Fritsch-Carlson / Fritsch-Butland, as scipy's _edge_case): the
    parabola through the end three points, zeroed when its sign differs from the end secant's, limited to 3x the
    end secant where the two secants differ in sign."""
    d = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    if np.sign(d) != np.sign(m0):
        return 0.0
    if np.sign(m0) != np.sign(m1) and abs(d) > 3.0 * abs(m0):
        return 3.0 * m0
    return float(d)


def pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fritsch-Carlson monotone cubic (PCHIP) derivatives at the knots: the weighted harmonic mean of the two
    neighbouring secants where they agree in sign (w1 = 2 h_k + h_{k-1}, w2 = h_k + 2 h_{k-1}), zero at a local
    extremum, _pchip_edge at the ends.  Exact on linear data; never overshoots monotone data.  No scipy."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    h = np.diff(x)
    mk = np.diff(y) / h
    d = np.zeros(n)
    if n == 2:
        d[:] = mk[0]
        return d
    hl, hr = h[:-1], h[1:]
    ml, mr = mk[:-1], mk[1:]
    w1, w2 = 2.0 * hr + hl, hr + 2.0 * hl
    with np.errstate(divide="ignore", invalid="ignore"):
        whm = (w1 + w2) / (w1 / ml + w2 / mr)
    same = (np.sign(ml) * np.sign(mr) > 0)
    d[1:-1] = np.where(same, whm, 0.0)
    d[0] = _pchip_edge(h[0], h[1], mk[0], mk[1])
    d[-1] = _pchip_edge(h[-1], h[-2], mk[-1], mk[-2])
    return d


def pchip_eval(x: np.ndarray, y: np.ndarray, d: np.ndarray, xq) -> np.ndarray:
    """Cubic Hermite evaluation of the PCHIP (knots x, values y, slopes d from pchip_slopes) at xq; outside the
    knots the end interval's cubic continues (callers continue their own way beyond the scan)."""
    x, y, d = np.asarray(x, float), np.asarray(y, float), np.asarray(d, float)
    xq = np.asarray(xq, float)
    k = np.clip(np.searchsorted(x, xq, side="right") - 1, 0, len(x) - 2)
    h = x[k + 1] - x[k]
    t = (xq - x[k]) / h
    h00 = (1.0 + 2.0 * t) * (1.0 - t) ** 2
    h10 = t * (1.0 - t) ** 2
    h01 = t * t * (3.0 - 2.0 * t)
    h11 = t * t * (t - 1.0)
    return h00 * y[k] + h10 * h * d[k] + h01 * y[k + 1] + h11 * h * d[k + 1]


def _scan_rows(mu_scan) -> Optional[np.ndarray]:
    """A point's mu_scan [[mu, nll_abs], ...] as a sorted (n, 2) array of finite rows with distinct mu (duplicates
    keep the lower nll); None with fewer than 2 rows (the point is dropped and flagged)."""
    rows = []
    for r in mu_scan or []:
        try:
            m, v = float(r[0]), float(r[1])
        except (TypeError, ValueError, IndexError):
            continue
        if np.isfinite(m) and np.isfinite(v):
            rows.append((m, v))
    if len(rows) < 2:
        return None
    rows.sort()
    out = [list(rows[0])]
    for m, v in rows[1:]:
        if abs(m - out[-1][0]) < 1e-12:
            out[-1][1] = min(out[-1][1], v)
        else:
            out.append([m, v])
    return np.asarray(out, float) if len(out) >= 2 else None


def component_nll(point: dict, mu) -> np.ndarray:
    """l_j(mu), the full -ln L of one grid point along ``mu``, from its conditional scan: PCHIP in mu of the signed
    root r = sign(mu - v) sqrt(2 (l - l_v)) (exact for a parabola, shape-preserving on a unimodal profile),
    l = l_v + r^2 / 2, with v the scan's minimum; where that is an end knot and the profile still falls beyond it (a
    deficit at the support edge, a bound), v is the vertex of the parabola through the three end knots, outside the
    scan, so r has no square-root cusp at the edge.  Beyond the scanned range r continues linearly with the end slope
    (a quadratic continuation); +inf outside the workspace's own POI bounds.  NaN where the point has no usable
    scan."""
    mu = np.atleast_1d(np.asarray(mu, float))
    sc = _scan_rows(point.get("mu_scan"))
    if sc is None:
        return np.full(mu.shape, np.nan)
    xs, ys = sc[:, 0], sc[:, 1]
    i0 = int(np.argmin(ys))
    v, lv = float(xs[i0]), float(ys[i0])
    if i0 in (0, len(xs) - 1) and len(xs) >= 3:
        k = np.arange(3) if i0 == 0 else np.arange(len(xs) - 3, len(xs))
        a2, a1, a0 = np.polyfit(xs[k] - xs[i0], ys[k], 2)
        if a2 > 0:
            t = -a1 / (2.0 * a2)
            if (i0 == 0 and t < 0) or (i0 > 0 and t > 0):
                v, lv = float(xs[i0] + t), min(float(a0 - a1 * a1 / (4.0 * a2)), lv)
    r = np.sign(xs - v) * np.sqrt(2.0 * np.maximum(ys - lv, 0.0))
    d = pchip_slopes(xs, r)
    rq = pchip_eval(xs, r, d, np.clip(mu, xs[0], xs[-1]))
    left, right = mu < xs[0], mu > xs[-1]
    if np.any(left):
        rq[left] = r[0] - max(d[0], 0.0) * (xs[0] - mu[left])
    if np.any(right):
        rq[right] = r[-1] + max(d[-1], 0.0) * (mu[right] - xs[-1])
    out = lv + 0.5 * rq * rq
    lo_b, hi_b = _point_bounds(point)
    out[(mu < lo_b - 1e-12) | (mu > hi_b + 1e-12)] = np.inf
    return out


def _point_bounds(point: dict) -> Tuple[float, float]:
    """A grid point's own POI bounds [lo_b, hi_b] (unbounded where it records none or an unreadable pair)."""
    b = point.get("bounds")
    try:
        return (float(b[0]), float(b[1])) if b is not None else (-np.inf, np.inf)
    except (TypeError, ValueError, IndexError):
        return -np.inf, np.inf


def nll_min_on_support(point: dict, support=MU_SUPPORT) -> Optional[float]:
    """min of a point's l_j over the evaluator's support D intersected with its own bounds: the lowest scanned value
    there, or nll_min when the free fit itself lies in it (the scan set holds mu_hat anyway).  The workspace's own
    bounds act only through l = +inf, so a deficit point (mu_hat below D, declared lo_b < 0) is
    ranked at its edge value l(0), never at its free-fit minimum outside D: the single-mass argmin, its tie set and
    Z_G read this, so declaring lo_b << 0 cannot turn the deepest deficit into the "best" point.  None without a scan
    row there."""
    lo_b, hi_b = _point_bounds(point)
    lo, hi = max(float(support[0]), lo_b), min(float(support[1]), hi_b)
    vals = []
    sc = _scan_rows(point.get("mu_scan"))
    if sc is not None:
        on = (sc[:, 0] >= lo - 1e-9) & (sc[:, 0] <= hi + 1e-9)
        if on.any():
            vals.append(float(np.min(sc[on, 1])))
    mh, nm = _finite_p(point.get("mu_hat")), _finite_p(point.get("nll_min"))
    if mh is not None and nm is not None and lo - 1e-9 <= mh <= hi + 1e-9:
        vals.append(nm)
    return min(vals) if vals else None


def grid_cells(masses, mass_range, half_cap: float = CELL_HALF_CAP) -> np.ndarray:
    """Delta m_j of sorted in-range masses: half the distance to each neighbour, the end points' outer halves the
    distance to the range end, every half capped at ``half_cap`` (so a gap wider than 2 caps, or a grid ending more
    than a cap short of a range end, leaves mass uncovered)."""
    m = np.asarray(masses, float)
    lo, hi = float(mass_range[0]), float(mass_range[1])
    if len(m) == 0:
        return m
    gaps = np.diff(m)
    left = np.concatenate([[max(m[0] - lo, 0.0)], 0.5 * gaps])
    right = np.concatenate([0.5 * gaps, [max(hi - m[-1], 0.0)]])
    return np.minimum(left, half_cap) + np.minimum(right, half_cap)


def sigma0_at(curve: Optional[dict], masses) -> Optional[np.ndarray]:
    """sigma0_ref at ``masses`` from a prior curve {"masses", "sigma0"}: log-linear interpolation in mass, the end
    values held beyond the curve's span.  None without a usable curve."""
    if not isinstance(curve, dict):
        return None
    try:
        xm = np.asarray(curve.get("masses"), float)
        ys = np.asarray(curve.get("sigma0"), float)
    except (TypeError, ValueError):
        return None
    ok = np.isfinite(xm) & np.isfinite(ys) & (ys > 0)
    if ok.sum() < 1 or len(xm) != len(ys):
        return None
    xm, ys = xm[ok], ys[ok]
    order = np.argsort(xm)
    xm, ys = xm[order], ys[order]
    return np.exp(np.interp(np.asarray(masses, float), xm, np.log(ys)))


def prior_curve_from_points(points, mass_range) -> Tuple[Optional[dict], Optional[str]]:
    """The per-dataset prior curve {"masses", "sigma0"} from a reference record's in-range ok points carrying
    sigma0_ref (verify --prior-curve), or (None, why): more than PRIOR_CURVE_MAX_FAILED of them without it, or a gap
    wider than PRIOR_CURVE_MAX_GAP between those that have it or between them and a range end (prior_curve_gap; the
    caller then uses the suite-average curve)."""
    lo, hi = float(mass_range[0]), float(mass_range[1])
    rows, n_in = [], 0
    for p in points or []:
        if not isinstance(p, dict) or p.get("status") != "ok":
            continue
        m = _finite_p(p.get("mass"))
        if m is None or not (lo - 1e-9 <= m <= hi + 1e-9):
            continue
        n_in += 1
        s0 = _finite_p(p.get("sigma0_ref"))
        if s0 is not None and s0 > 0:
            rows.append((m, s0))
    if not n_in:
        return None, "no in-range point"
    if not rows:
        return None, "no point carries sigma0_ref (verified without --prior-curve)"
    if (n_in - len(rows)) / n_in > PRIOR_CURVE_MAX_FAILED:
        return None, f"{n_in - len(rows)} of {n_in} in-range points lack sigma0_ref"
    rows.sort()
    curve = {"masses": [float(m) for m, _ in rows], "sigma0": [float(s) for _, s in rows]}
    why = prior_curve_gap(curve, mass_range)
    return (None, why) if why else (curve, None)


def prior_curve_gap(curve: Optional[dict], mass_range, ends_only: bool = False) -> Optional[str]:
    """Why a prior curve {"masses", "sigma0"} cannot weight the whole range, or None: a gap wider than
    PRIOR_CURVE_MAX_GAP between its masses (unless ``ends_only``), OR between a range end and its first / last mass --
    sigma0_at holds the end values flat beyond the span, and sigma0_ref falls tenfold over 110-140 GeV, so a curve
    from a local grid would weight the far end at its edge value (a wider gap goes to the suite-average
    curve)."""
    if not isinstance(curve, dict):
        return "no curve"
    try:
        ms = np.sort(np.asarray([m for m, s in zip(curve.get("masses"), curve.get("sigma0"))
                                 if np.isfinite(float(m)) and np.isfinite(float(s)) and float(s) > 0], float))
    except (TypeError, ValueError):
        return "unreadable curve"
    if not len(ms):
        return "no point with a positive sigma0_ref"
    lo, hi = float(mass_range[0]), float(mass_range[1])
    if ms[0] - lo > PRIOR_CURVE_MAX_GAP + 1e-9 or hi - ms[-1] > PRIOR_CURVE_MAX_GAP + 1e-9:
        return (f"the points with sigma0_ref span {ms[0]:g}-{ms[-1]:g} GeV, short of the {lo:g}-{hi:g} GeV range by "
                f"more than {PRIOR_CURVE_MAX_GAP:g} GeV")
    if not ends_only and len(ms) > 1 and float(np.max(np.diff(ms))) > PRIOR_CURVE_MAX_GAP + 1e-9:
        return f"a {float(np.max(np.diff(ms))):.2f} GeV gap between points with sigma0_ref"
    return None


def prior_curve_average(curves: List[dict], mass_range, step: float = PRIOR_CURVE_TASK_STEP) -> Optional[dict]:
    """The suite-average prior curve: the geometric mean over datasets of sigma0_ref on a common mass grid (the
    range at ``step``), over the curves that span the range (prior_curve_gap: a local curve's flat-held ends would
    bias it).  None without such a curve."""
    curves = [c for c in curves if isinstance(c, dict) and prior_curve_gap(c, mass_range) is None]
    if not curves:
        return None
    lo, hi = float(mass_range[0]), float(mass_range[1])
    grid = np.round(np.arange(lo, hi + 1e-9, step), 9)
    logs = [np.log(s) for s in (sigma0_at(c, grid) for c in curves) if s is not None]
    if not logs:
        return None
    return {"masses": grid.tolist(), "sigma0": np.exp(np.mean(logs, axis=0)).tolist(), "n_datasets": len(logs)}


def _coverage_facts(masses, mass_range) -> dict:
    """Reach and spacing of sorted in-range masses against the task's range: first / last mass, the largest in-range
    gap, whether both ends are reached within REACH_TOL and no gap exceeds MAX_GAP_IN_RANGE ("ok")."""
    m = np.asarray(masses, float)
    lo, hi = float(mass_range[0]), float(mass_range[1])
    if len(m) == 0:
        return {"first_mass": None, "last_mass": None, "n_in_range": 0, "max_gap_in_range": None,
                "reaches_lo": False, "reaches_hi": False, "ok": False}
    gap = float(np.max(np.diff(m))) if len(m) > 1 else 0.0
    r_lo, r_hi = bool(m[0] <= lo + REACH_TOL), bool(m[-1] >= hi - REACH_TOL)
    return {"first_mass": float(m[0]), "last_mass": float(m[-1]), "n_in_range": int(len(m)),
            "max_gap_in_range": gap, "reaches_lo": r_lo, "reaches_hi": r_hi,
            "ok": bool(r_lo and r_hi and gap <= MAX_GAP_IN_RANGE + 1e-9)}


def argmin_point(points: List[dict], tol: float = LINK_TWICE_NLL_TOL,
                 support=MU_SUPPORT) -> Tuple[Optional[int], List[int]]:
    """(index of the scored single-mass point, the tie set): the in-range point with the lowest minimum ON THE
    SUPPORT (nll_min_on_support: a deficit point's free fit below mu = 0 does not count); among the
    points within ``tol`` of it in 2NLL (the tie set) the one with the largest Hessian width -- the submission's
    weakest claim among its statistically equivalent points, independent of restart noise and of where MODEL.json
    sits.  (None, []) without a usable point."""
    vals = [(nll_min_on_support(p, support), i) for i, p in enumerate(points)]
    vals = [(v, i) for v, i in vals if v is not None]
    if not vals:
        return None, []
    best = min(v for v, _ in vals)
    ties = [i for v, i in vals if 2.0 * (v - best) <= tol + 1e-12]
    pick = max(ties, key=lambda i: ((_finite_p(points[i].get("sigma_hess")) or 0.0), -i))
    return pick, ties


def _usable_components(points: List[dict], mass_range) -> Tuple[List[dict], List[str]]:
    """The in-range ok points with a usable scan (>= 2 finite rows and a finite nll_min), sorted by mass, and the
    flags of what was dropped."""
    lo, hi = float(mass_range[0]), float(mass_range[1])
    out, flags = [], []
    n_short = 0
    for p in points or []:
        if not isinstance(p, dict):
            continue
        m = _finite_p(p.get("mass"))
        if m is None:
            continue
        in_range = p.get("in_range")
        if in_range is None:
            in_range = lo - 1e-9 <= m <= hi + 1e-9
        if not in_range or p.get("status") != "ok":
            continue
        if _scan_rows(p.get("mu_scan")) is None or _finite_p(p.get("nll_min")) is None:
            n_short += 1
            continue
        out.append(_band_clipped(dict(p, mass=m)))
    if n_short:
        flags.append(f"{n_short} in-range point(s) without a usable mu scan dropped")
    n_clip = sum(1 for p in out if p.get("mu_stretch_applied"))
    if n_clip:
        flags.append(f"{n_clip} point(s) above the yield band scored at its edge")
    out.sort(key=lambda p: p["mass"])
    return out, flags


def _band_clipped(p: dict) -> dict:
    """A grid point above the yield band (verify_grid.grid_content's "mu_stretch" k = exp(ln r_j - b) > 1) as if its
    POI-scaled template were k times smaller: the signal is mu x template, so l'(mu) = l(mu / k) -- the scan's mu
    axis, the POI bounds, mu_hat and its width all scale by k, the -ln L values do not.  An inflated cell then buys
    no more than one at the band edge, and the grid keeps its marginal."""
    k = _finite_p(p.get("mu_stretch"))
    if k is None or k <= 1.0 + 1e-12:
        return p
    sc = _scan_rows(p.get("mu_scan"))
    out = dict(p, mu_scan=[[float(m) * k, float(v)] for m, v in sc], mu_stretch_applied=float(k))
    lo_b, hi_b = _point_bounds(p)
    if p.get("bounds") is not None:
        out["bounds"] = [lo_b * k, hi_b * k]
    for key in ("mu_hat", "sigma_hess"):
        v = _finite_p(p.get(key))
        if v is not None:
            out[key] = v * k
    return out


def _point_from_model_fit(mf: Optional[dict]) -> Optional[dict]:
    """A single-mass component from a record's mu_fit (MODEL.json's own fixed-mass fit) for the no-grid fallback:
    its mu_scan when the grader stored one, else the parabola of (value, sigma_hess | sigma) sampled as the scan set
    (today's rule, exact only in its Gaussian limit).  None without a finite value and width."""
    if not isinstance(mf, dict) or mf.get("error"):
        return None
    v = _finite_p(mf.get("value"))
    s = _finite_p(mf.get("sigma_hess")) or _finite_p(mf.get("sigma"))
    b = mf.get("bounds")
    out = {"mass": None, "status": "ok", "mu_hat": v, "sigma_hess": s, "bounds": b, "in_range": True}
    if _scan_rows(mf.get("mu_scan")) is not None:
        sc = _scan_rows(mf["mu_scan"])
        return dict(out, mu_scan=sc.tolist(), nll_min=float(np.min(sc[:, 1])))
    if v is None or s is None or s <= 0:
        return None
    ks = np.arange(-6.0, 6.0 + 1e-9, 0.5)
    mus = np.unique(np.round(np.concatenate([[0.0], v + ks * s]), 12))
    return dict(out, nll_min=0.0, mu_scan=[[float(m), 0.5 * ((m - v) / s) ** 2] for m in mus])


def _mu_axis(comps: List[dict], support) -> np.ndarray:
    """The adaptive mu axis: MU_AXIS_STEP over the support united with +-LOCAL_HALF_WIDTH s_j at LOCAL_STEP s_j
    around every component's best fit and the components' bound edges inside the support, each with its neighbour
    just outside (BOUND_EDGE), where l = +inf: the density steps to zero at a bound, so the trapezoid ends at the
    bound instead of spreading half of p(bound) over the next axis step."""
    lo, hi = float(support[0]), float(support[1])
    parts = [np.arange(lo, hi + 0.5 * MU_AXIS_STEP, MU_AXIS_STEP)]
    ks = np.arange(-LOCAL_HALF_WIDTH, LOCAL_HALF_WIDTH + 1e-9, LOCAL_STEP)
    for c in comps:
        v, s = _finite_p(c.get("mu_hat")), _finite_p(c.get("sigma_hess"))
        if v is not None and s is not None and s > 0:
            parts.append(np.clip(v + ks * s, lo, hi))
        lo_b, hi_b = _point_bounds(c)
        edges = [x for x in (lo_b, lo_b - BOUND_EDGE) if lo < lo_b <= hi] + \
                [x for x in (hi_b, hi_b + BOUND_EDGE) if lo <= hi_b < hi]
        parts.append(np.asarray([x for x in edges if lo <= x <= hi], float))
    return np.unique(np.round(np.concatenate(parts), 10))


def _log_trapz(log_f: np.ndarray, x: np.ndarray) -> float:
    """ln of the trapezoid integral of exp(log_f) over x (log space; -inf entries are zeros)."""
    if len(x) < 2:
        return -np.inf
    with np.errstate(divide="ignore"):
        lw = np.log(0.5 * np.diff(x))
    return logsumexp(np.concatenate([log_f[:-1] + lw, log_f[1:] + lw]))


def _trapz(y, x) -> float:
    """The trapezoid integral of y over x (numpy's own is np.trapezoid from 2.0 on; the grader runtime may be 1.26)."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x))) if len(x) > 1 else 0.0


def density_summary(mu_axis: np.ndarray, log_p: np.ndarray) -> dict:
    """Moments and H of a normalised log density on an axis (trapezoid): H = -int p ln p (its mean -ln p), mean,
    sigma (sd), mode, p_lt_0p1 = P(mu < 0.1)."""
    p = np.exp(log_p)
    plnp = np.where(p > 0, p * np.where(p > 0, log_p, 0.0), 0.0)
    mean = float(_trapz(mu_axis * p, mu_axis))
    var = float(_trapz((mu_axis - mean) ** 2 * p, mu_axis))
    m01 = mu_axis <= 0.1 + 1e-12
    return {"entropy": float(-_trapz(plnp, mu_axis)), "mean": mean, "sigma": math.sqrt(max(var, 0.0)),
            "mode": float(mu_axis[int(np.argmax(log_p))]),
            "p_lt_0p1": float(_trapz(p[m01], mu_axis[m01])) if m01.sum() > 1 else 0.0}


def z_global(points: List[dict], mass_range, support=MU_SUPPORT) -> Optional[float]:
    """Z_G = max_j sqrt(max(q0_j, 0)) over the in-range ok points, q0_j = 2 (l_j(0) - min_D l_j) with l_j(0) read off
    the scan at mu = 0 (the verify stage fits mu = 0 under the bounds widened to min(lo_b, 0), so a lo_b > 0 point
    has it too; a point whose mu = 0 fit failed carries none) and the minimum taken on the support D within the
    point's own bounds (nll_min_on_support): one-sided, as mu_fit_block's q0 (0 for mu_hat < 0) -- a deficit is no
    significance, whatever lo_b the workspace declares.  What the delivered grid shows at its best mass, no trials
    constant.  None when no point gives one."""
    lo, hi = float(mass_range[0]), float(mass_range[1])
    best = None
    for p in points or []:
        if not isinstance(p, dict) or p.get("status") != "ok":
            continue
        m = _finite_p(p.get("mass"))
        if m is None or not (lo - 1e-9 <= m <= hi + 1e-9):
            continue
        sc = _scan_rows(p.get("mu_scan"))
        nm = nll_min_on_support(p, support)
        if sc is None or nm is None:
            continue
        at0 = sc[np.abs(sc[:, 0]) < 1e-9]
        if not len(at0):
            continue
        q0 = 2.0 * (float(at0[0, 1]) - nm)
        z = math.sqrt(max(q0, 0.0))
        best = z if best is None else max(best, z)
    return best


def marginal_mu(mass_grid: Optional[dict], mass_range, prior_curve: Optional[dict], task_curve: Optional[dict],
                stamp: str, support=MU_SUPPORT, model_fit: Optional[dict] = None) -> dict:
    """The scored mu density of one dataset from its verify record's mass_grid (module comment above).
    Returns {"status": "ok" | "excluded" | "failed" | "pending", "regime": "marginal" | "single_mass" | None,
    "mu_axis",
    "log_p" (normalised on the support; numpy arrays), "entropy", "mean", "sigma", "mode", "n_eff", "p_lt_0p1",
    "z_g", "coverage_flags", "content_flags", "prior_source": "per_dataset" | "task_average" | "flat" | None,
    "fallback_point": {"mass", "mu_hat", "sigma_hess"} | None, "reason", "stamp", "n_components", "log_z",
    "coverage", "shares" (each component's posterior share)}.
      pending      mass_grid has no scan_version (verified before the scan) or another one: rerun the grader;
      marginal     the sum over the usable in-range components with cells and the prior; stamp >= 2.1 requires the
                   coverage condition (reach within REACH_TOL, gaps <= MAX_GAP_IN_RANGE), stamp 2.0 (stored runs)
                   restricts the prior to the covered cells and flags grid_short / grid_sparse;
      single_mass  the fallback: the exact density of ONE point -- the grid's in-range argmin (argmin_point's
                   tie-break) when the content checks fail, the grid is not one model / not MODEL.json's, or
                   coverage fails at stamp >= 2.1; MODEL.json's own fit (``model_fit``) when there is no grid;
      excluded     the components' declared bounds leave l = +inf on the whole support: an answer that every mu on
                   the support is impossible, scored on the claim (the cap; "sign" the side of the bounds, "bound"
                   the nearest one claimed);
      pending      also on an evaluator fault ("evaluator_fault"): the reference's prior curve leaves no component any
                   measure;
      failed       no usable density (no component, none with a scan on the support, or one that does not
                   normalise).
    n_eff = 1 / (the argmin component's posterior share): the trials factor the grid itself carries (1 for one
    component).  The truth is never used here; the loss reads the density afterwards (log_density_at)."""
    lo_s, hi_s = float(support[0]), float(support[1])
    out = {"status": "failed", "regime": None, "mu_axis": None, "log_p": None, "entropy": None, "mean": None,
           "sigma": None, "mode": None, "n_eff": None, "p_lt_0p1": None, "z_g": None, "coverage_flags": [],
           "content_flags": [], "prior_source": None, "fallback_point": None, "reason": None, "stamp": str(stamp),
           "n_components": 0, "log_z": None, "coverage": None, "shares": None}
    mg = mass_grid if isinstance(mass_grid, dict) else {}
    sv = mg.get("scan_version")
    if sv is None:
        return {**out, "status": "pending",
                "reason": "verified before the mu scan (no scan_version): rerun the grader (verify --stage grid)"}
    if sv != SCAN_VERSION:
        return {**out, "status": "pending",
                "reason": f"scanned under {sv!r}, the score reads {SCAN_VERSION!r}: rerun the grader"}
    points = mg.get("points") if isinstance(mg.get("points"), list) else []
    comps, cflags = _usable_components(points, mass_range)
    out["z_g"] = z_global(points, mass_range, support)
    content = mg.get("content") if isinstance(mg.get("content"), dict) else {}
    content_flags = [str(r) for r in (content.get("reasons") or [])]
    if not content:
        content_flags.append("no content checks recorded")
    content_usable = bool(content.get("usable", True))
    grid_bad = []
    if mg.get("grid_uniform") is False:
        grid_bad.append(f"grid is not one model ({mg.get('grid_mismatch') or 'structure differs'})")
    if mg.get("linked") is False:
        grid_bad.append("grid is not MODEL.json's model: " + str((mg.get("link") or {}).get("why") or "linkage failed"))
    cov = mg.get("coverage") if isinstance(mg.get("coverage"), dict) else None
    facts = _coverage_facts([c["mass"] for c in comps], mass_range)
    has_v = cov is not None and "ok" in cov
    coverage_ok = bool(facts["ok"] and (bool(cov["ok"]) if has_v else True))
    out["coverage"] = {**facts, "ok": coverage_ok, "source": "verify+score" if has_v else "score"}
    cflags = list(cflags)
    if comps and not (facts["reaches_lo"] and facts["reaches_hi"]):
        cflags.append("grid_short")
    if comps and facts["max_gap_in_range"] is not None and facts["max_gap_in_range"] > MAX_GAP_IN_RANGE + 1e-9:
        cflags.append("grid_sparse")
    if cov is not None and "ok" in cov and bool(cov["ok"]) != facts["ok"]:
        cflags.append("verify and score coverage disagree")
    out.update(coverage_flags=cflags, content_flags=content_flags)

    regime, why, fb = "marginal", None, None
    if not comps:
        regime = "single_mass"
        why = "no usable grid point: MODEL.json's own fit" if not points else \
            "no in-range grid point with a usable scan: MODEL.json's own fit"
        pt = _point_from_model_fit(model_fit)
        if pt is None:
            return {**out, "status": "failed", "regime": None,
                    "reason": why + " -- and the model gives no mu fit to fall back on"}
        mf = model_fit or {}
        pt["mass"] = _finite_p(mg.get("mh_model")) or _finite_p(mf.get("mass"))
        comps_used, fb = [pt], {"mass": pt["mass"], "mu_hat": pt["mu_hat"], "sigma_hess": pt["sigma_hess"],
                                "source": "MODEL.json mu_fit" + (" scan" if mf.get("mu_scan") else " parabola")}
    else:
        if grid_bad:
            regime, why = "single_mass", "; ".join(grid_bad)
        elif not content_usable:
            regime, why = "single_mass", "grid content unusable: " + ("; ".join(content_flags) or "content checks failed")
        elif _stamp_at_least(stamp, "2.1") and not coverage_ok:
            regime, why = "single_mass", "coverage fails at contract " + str(stamp) + ": " + ", ".join(
                f for f in cflags if f in ("grid_short", "grid_sparse")) or "grid_short"
        if regime == "single_mass":
            j, ties = argmin_point(comps, support=support)
            if j is None:
                return {**out, "status": "failed", "regime": None,
                        "reason": (why or "") + " -- and no grid point has a scan on the mu support"}
            pt = comps[j]
            on_d = [nll_min_on_support(comps[i], support) for i in ties]
            fb = {"mass": pt["mass"], "mu_hat": _finite_p(pt.get("mu_hat")), "sigma_hess": _finite_p(pt.get("sigma_hess")),
                  "source": "grid argmin", "tie_set": [comps[i]["mass"] for i in ties],
                  "tie_spread_2nll": float(2.0 * (max(on_d) - min(on_d))) if ties else 0.0}
            comps_used = [pt]
        else:
            comps_used = comps
    out.update(regime=regime, reason=why, fallback_point=fb, n_components=len(comps_used))

    masses = np.asarray([c["mass"] if c["mass"] is not None else 0.5 * (mass_range[0] + mass_range[1])
                         for c in comps_used], float)
    cells = grid_cells(masses, mass_range) if regime == "marginal" else np.ones(len(comps_used))
    s0, src = None, "flat"
    if regime == "marginal":
        for curve, name in ((prior_curve, "per_dataset"), (task_curve, "task_average")):
            s0 = sigma0_at(curve, masses) if prior_curve_gap(curve, mass_range, ends_only=True) is None else None
            if s0 is not None:
                src = name
                break
        if src == "task_average":
            out["coverage_flags"].append("prior_curve_fallback")
        elif src == "flat":
            out["coverage_flags"].append("prior_flat")
    if s0 is None:
        s0 = np.ones(len(comps_used))
    out["prior_source"] = src if regime == "marginal" else None
    with np.errstate(divide="ignore"):
        log_nu = np.log(cells) - np.log(s0)
    keep = np.isfinite(log_nu)
    if not keep.any():
        return {**out, "status": "pending", "evaluator_fault": True,
                "reason": f"{_core.EVALUATOR_FAULT}: every component has zero prior measure under the reference's "
                          "prior curve (closure.json): re-close the suite"}
    comps_used = [c for c, k in zip(comps_used, keep) if k]
    log_nu = log_nu[keep]

    axis = _mu_axis(comps_used, support)
    nll = np.vstack([component_nll(c, axis) for c in comps_used])
    nll = np.where(np.isnan(nll), np.inf, nll)
    terms = log_nu[:, None] - nll
    with np.errstate(invalid="ignore"):
        log_num = logsumexp(terms, axis=0)
    if not np.isfinite(log_num).any():
        bs = [_point_bounds(c) for c in comps_used]
        above = [b[0] for b in bs if b[0] >= hi_s]
        below = [b[1] for b in bs if b[1] <= lo_s]
        sign = 1.0 if len(above) >= len(below) else -1.0
        bound = (min(above) if above else hi_s) if sign > 0 else max(below)
        return {**out, "status": "excluded", "sign": sign, "bound": float(bound), "support_width": hi_s - lo_s,
                "reason": f"the density vanishes on the whole support (the declared bounds exclude [{lo_s:g}, "
                          f"{hi_s:g}])", "n_components": len(comps_used)}
    log_z = _log_trapz(log_num, axis)
    if not np.isfinite(log_z):
        return {**out, "status": "failed", "reason": "the density does not normalise on the support"}
    log_p = log_num - log_z
    comp_logz = np.array([_log_trapz(t, axis) for t in terms])
    shares = np.exp(comp_logz - log_z)
    j_arg, _ = argmin_point(comps_used, support=support)
    share_arg = float(shares[j_arg]) if j_arg is not None and shares[j_arg] > 0 else None
    out.update(status="ok", mu_axis=axis, log_p=log_p, log_z=float(log_z), n_components=len(comps_used),
               n_eff=(1.0 / share_arg) if share_arg else None, shares=[float(s) for s in shares],
               **density_summary(axis, log_p))
    return out


def log_density_at(fc: dict, mu_t: float) -> float:
    """ln p(mu_t) of a marginal_mu result: linear interpolation of log_p between the axis points around mu_t; -inf
    outside the support or where either neighbour has no density (p = 0 at the truth: the cap)."""
    axis, lp = np.asarray(fc["mu_axis"], float), np.asarray(fc["log_p"], float)
    x = float(mu_t)
    if len(axis) == 0 or x < axis[0] - 1e-12 or x > axis[-1] + 1e-12:
        return -math.inf
    k = int(np.searchsorted(axis, x))
    if k < len(axis) and abs(axis[k] - x) <= 1e-12:
        return float(lp[k])
    if k == 0 or k >= len(axis):
        return float(lp[min(k, len(axis) - 1)])
    a, b = lp[k - 1], lp[k]
    if not (np.isfinite(a) and np.isfinite(b)):
        return -math.inf
    f = (x - axis[k - 1]) / (axis[k] - axis[k - 1])
    return float((1.0 - f) * a + f * b)


def cdf_at(fc: dict, mu_t: float) -> float:
    """F(mu_t) = P(mu <= mu_t) of a marginal_mu result (the PIT of mu_t, a diagnostic): the trapezoid integral of p
    over the axis up to mu_t, the last partial cell closed at p(mu_t) as log_density_at reads it -- the quadrature
    that normalised p, so F reaches 1 at the axis end.  0 at or below the support's lower end, 1 at or above its
    upper end."""
    axis, lp = np.asarray(fc["mu_axis"], float), np.asarray(fc["log_p"], float)
    x = float(mu_t)
    if len(axis) == 0 or x <= axis[0]:
        return 0.0
    if x >= axis[-1]:
        return 1.0
    k = int(np.searchsorted(axis, x, side="right"))
    p = np.exp(lp[:k])
    tail = 0.5 * (p[k - 1] + math.exp(log_density_at(fc, x))) * (x - axis[k - 1])
    return float(min(1.0, max(0.0, _trapz(p, axis[:k]) + tail)))


def marginal_summary(fc: dict) -> dict:
    """A marginal_mu result without its arrays (what info / scores.json keep)."""
    return {k: v for k, v in fc.items() if k not in ("mu_axis", "log_p")}


def closure_baseline(run_dir: Path, task, expected_seeds=None) -> dict:
    """closure.json's grader baseline for a gated task from the reference run's VERIFY_FILE (verify --prior-curve):
    {"mu_rule", "scan_version", "baseline_source": "grader", "baseline_id", "reference_pois": {seed: {"mu": {"value",
    "sigma", "entropy", "prior_curve", ...}, "mH": {"value", "sigma"}}}, "prior_curve_task", "mu_baseline"}.
    mu: the reference's own marginal_mu under its per-dataset prior curve (the suite average, flagged, where its
    own fails), only in the marginal regime (a reference dataset without such a density has no mu baseline and is
    excluded from the mu cases, listed in mu_baseline.excluded with why); mH: grid_profile's value and root width as
    the score reads it, else the reported one from private/scores.json (as before).  sigma_ref is not rewritten:
    closure.json keeps `sbh closure`'s gated medians, for mH and the tripwire only (mu's scalar is never a baseline
    under the gate).  baseline_id fingerprints the mu baseline: every score records the one it read, and the gate
    admits a run only once the closure's reference_score was taken against it (sbh.suite.closure_gate).
    baseline_source "pending" with a baseline_note (and no reference_pois) instead when the file cannot be a whole
    baseline: no VERIFY_FILE, one from before the scan (all or nothing), ``expected_seeds`` (the suite's sealed
    datasets) without a row each (a verification stopped early), a grader-side error on a dataset (rerun), no
    reference curve spanning the range, or a mu baseline on fewer than MU_BASELINE_MIN_FRAC of the datasets."""
    rows = load_verify_rows(run_dir)
    base = {"mu_rule": MU_RULE, "scan_version": SCAN_VERSION}

    def pending(note, **extra):
        return {**base, "baseline_source": "pending", "baseline_note": note, **extra}
    if rows is None:
        return pending(f"no {VERIFY_FILE} yet: sbh post-score --label reference writes the baseline")
    rows = {s: r for s, r in rows.items() if isinstance(r, dict)}
    expected = sorted(set(expected_seeds)) if expected_seeds is not None else sorted(rows)
    rerun = "rerun its grader (sbh post-score --label reference, or verify --stage grid --prior-curve)"
    missing = [s for s in expected if s not in rows]
    if missing:
        return pending(f"{len(missing)} of {len(expected)} sealed datasets have no row in the reference's {VERIFY_FILE} "
                       f"({', '.join(missing[:5])}{', ...' if len(missing) > 5 else ''}; a verification that stopped "
                       f"early leaves a partial file): {rerun}")
    sp = getattr(task, "signal_strength_poi", None) or "mu"
    rng = getattr(task, "mass_range", None) or (110.0, 140.0)
    support = getattr(task, "mu_support", MU_SUPPORT)
    status = {s: _grader_status(rows[s]) for s in expected}
    ours = {s: why for s, (st, why) in status.items() if st == "pending"}
    if ours:
        return pending(f"the grader did not finish {len(ours)} reference dataset(s) ("
                       + ", ".join(f"{s}: {str(w)[:60]}" for s, w in sorted(ours.items())[:3])
                       + f"{', ...' if len(ours) > 3 else ''}; not the model's fault): {rerun}")
    ok_rows = {s: rows[s] for s in expected if status[s][0] == "ok"}
    scanned = {s: r for s, r in ok_rows.items() if _row_scan_version(r) == SCAN_VERSION}
    if not scanned or len(scanned) < len(ok_rows):
        late = sorted(set(ok_rows) - set(scanned))
        return pending(f"{len(late)} of {len(ok_rows)} reference datasets were verified before the mu scan "
                       f"({', '.join(late[:5])}{', ...' if len(late) > 5 else ''}): rerun its grader "
                       "(verify --stage grid --prior-curve)")
    reported: Dict[str, dict] = {}
    try:
        priv = json.loads((Path(run_dir) / "private" / "scores.json").read_text())
        reported = {s["seed"]: s.get("pois") or {} for s in priv.get("per_seed") or []}
    except (OSError, ValueError, TypeError, KeyError):
        pass
    curves, curve_why = {}, {}
    for s, r in scanned.items():
        curves[s], curve_why[s] = prior_curve_from_points((r.get("mass_grid") or {}).get("points"), rng)
    task_curve = prior_curve_average([c for c in curves.values() if c], rng)
    if task_curve is None:
        return pending("no reference dataset carries a sigma0_ref curve spanning the mass range (verified without "
                       "--prior-curve, on pyhf, which does not compute it, or on grids that do not reach both ends): "
                       "rerun its grader on the full-range reference grid (verify --stage grid --prior-curve, JAX twin)",
                       mu_baseline={"prior_curve_failed": {s: w for s, w in curve_why.items() if w}})
    excluded = {s: f"reference record: {status[s][1][:120]}" for s in expected if status[s][0] == "failed"}
    ref_pois, fallback_curve = {}, []
    for s, r in sorted(ok_rows.items()):
        entry = {}
        mh = model_mh(r, rng)
        if mh["status"] == "ok":
            entry["mH"] = {"value": mh["value"], "sigma": mh["sigma"]}
        elif isinstance(reported.get(s, {}).get("mH"), dict):
            v = reported[s]["mH"]
            entry["mH"] = {"value": v["value"], "sigma": (v["total"] if v.get("total") else
                                                          math.hypot(v.get("stat", 0.0), v.get("syst", 0.0)))}
        if not isinstance(r.get("mass_grid"), dict):
            excluded[s] = "no MODEL_GRID: no marginal density"
        else:
            own = curves.get(s)
            try:
                rec = json.loads((Path(run_dir) / "runs" / s / "record.json").read_text())
            except (OSError, ValueError):
                rec = None
            fc = marginal_mu(r["mass_grid"], rng, own, task_curve if own is None else None, contract_stamp(rec),
                             support=support, model_fit=r.get("mu_fit"))
            if own is None and curve_why.get(s):
                fallback_curve.append(s)
            if fc["status"] == "ok" and fc["regime"] == "marginal":
                entry[sp] = {"value": fc["mean"], "sigma": fc["sigma"], "entropy": fc["entropy"],
                             **({"prior_curve": own} if own is not None else {}), "prior_source": fc["prior_source"],
                             "regime": fc["regime"], "n_components": fc["n_components"], "n_eff": fc["n_eff"],
                             "z_g": fc["z_g"], "mode": fc["mode"], "p_lt_0p1": fc["p_lt_0p1"],
                             "coverage_flags": fc["coverage_flags"], "content_flags": fc["content_flags"]}
            else:
                excluded[s] = (fc["reason"] or f"{fc['status']} ({fc['regime']})")
        if entry:
            ref_pois[s] = entry
    n_mu = sum(1 for e in ref_pois.values() if sp in e)
    mu_baseline = {"n_datasets": n_mu, "n_expected": len(expected), "excluded": excluded,
                   "min_frac": MU_BASELINE_MIN_FRAC, "prior_curve_fallback": fallback_curve,
                   "prior_curve_failed": {s: w for s, w in curve_why.items() if w},
                   "note": "mu: the reference's H per dataset, its marginal density's mean -ln p "
                           "(sbh.score.forecast_case_loss); sigma_ref['mu'] is kept for the tripwire only, never a "
                           "baseline"}
    if not n_mu or n_mu < MU_BASELINE_MIN_FRAC * len(expected) - 1e-9:
        return pending(f"the reference has a mu density in the marginal regime on {n_mu} of {len(expected)} datasets "
                       f"(the floor is {MU_BASELINE_MIN_FRAC:.0%}); excluded: "
                       + "; ".join(f"{s}: {str(w)[:80]}" for s, w in sorted(excluded.items())[:4])
                       + ("; ..." if len(excluded) > 4 else ""), mu_baseline=mu_baseline)
    blob = json.dumps({"rule": MU_RULE, "scan": SCAN_VERSION, "task_curve": task_curve,
                       "mu": {s: e[sp] for s, e in sorted(ref_pois.items()) if sp in e}}, sort_keys=True, default=str)
    return {**base, "baseline_source": "grader", "baseline_id": hashlib.sha1(blob.encode()).hexdigest()[:16],
            "reference_pois": ref_pois, "prior_curve_task": task_curve, "mu_baseline": mu_baseline}


def declared_yield_response(run_dir: Path, records: List[dict], truths, task) -> Optional[dict]:
    """Response b of the declared signal yield (MODEL.json at POI = 1, nuisances at init) to the injected
    task.yield_response_effect across signal datasets (every dataset under Task.mu_marginal_over_mass): log Y = a +
    b log(effect) + other POIs + other injected effects.  b = 1: the model carries the effect; 0: ignored.  Needs
    pyhf."""
    eff = getattr(task, "yield_response_effect", None)
    if not eff:
        return None
    from ... import use_grader_lib
    use_grader_lib()
    try:
        import pyhf
    except ImportError:
        return {"status": "pyhf not importable"}
    poi_mu = getattr(task, "signal_strength_poi", None) or task.pois[-1]
    others = [p for p in task.pois if p != poi_mu]
    all_datasets = bool(getattr(task, "mu_marginal_over_mass", False))
    rows = []
    for rec in records:
        t = truths[rec["seed"]]
        if not record_ok(rec) or (not all_datasets and float(t.get(poi_mu, 0.0)) <= 0):
            continue
        mj = Path(run_dir) / "runs" / rec["seed"] / "out" / "MODEL.json"
        if not os.path.lexists(mj):
            continue
        try:
            spec = load_output_json(mj)
            spec.pop("_meta", None)
            model = pyhf.Workspace(spec).model()
            name = model.config.poi_name
            sig = {x["name"] for ch in spec["channels"] for x in ch["samples"]
                   if any(m.get("name") == name and m.get("type") == "normfactor" for m in x.get("modifiers", []))}
            pars = np.asarray(model.config.suggested_init(), dtype=float)
            pars[model.config.poi_index] = 1.0
            by = np.asarray(model.main_model.expected_data(pars, return_by_sample=True))
            y = float(sum(by[i].sum() for i, smp in enumerate(model.config.samples) if smp in sig))
        except Exception:
            continue
        if not (y > 0 and np.isfinite(y)):
            continue
        ct = task.event_level_truth(t)
        rep = {x["name"]: float(x["value"]) for x in rec["result"]["pois"]}
        rows.append([np.log(y), np.log(ct[eff])] + [rep.get(p, float(t[p])) for p in others]
                    + [ct[k] for k in sorted(ct) if k != eff])
    if len(rows) < 6:
        return {"status": f"{len(rows)} usable datasets (need 6)"}
    a = np.array(rows, dtype=float)
    X = np.c_[np.ones(len(a)), a[:, 1:]]
    X = X[:, [0] + [j for j in range(1, X.shape[1]) if np.ptp(X[:, j]) > 0]]
    beta, se, chi2 = _wls(X, a[:, 0], np.ones(len(a)))
    se = se * np.sqrt(chi2)
    return {"status": "ok", "effect": eff, "b": float(beta[1]), "se": float(se[1]),
            "n": len(a), "resid": float(np.sqrt(chi2))}


FIT_RULE = "score-from-fit"


V4_W_TRUTH = 0.5

V4_W_CLAIMS = 0.05


TRUTH_TAU_REL, TRUTH_TAU_ABS = 0.25, 0.05


TRUTH_RHO = 0.9


TRUTH_CAP = V2_CAP_FLOOR


Z_DISCOVERY = 5.0


LEAK_TOL, YIELD_TOL = 0.2, 0.3


_MODEL_SIDE = ("no MODEL.json", "invalid workspace", "fit failed")


def load_verify_rows(run_dir: Path) -> Optional[Dict[str, dict]]:
    """{seed: grader record} from model_verify_toys.json, or None when the grader has not run."""
    p = Path(run_dir) / VERIFY_FILE
    if not p.exists():
        return None
    try:
        return {r.get("seed"): r for r in (json.loads(p.read_text()).get("per_seed") or [])}
    except (OSError, ValueError):
        return None


def evaluator_fault(blk) -> Optional[str]:
    """The pending reason of a grader record or block the grader marks as its own failure ("evaluator_error", a
    string): "evaluator fault: <msg>" (sbh.score.EVALUATOR_FAULT); None where it carries none."""
    msg = blk.get("evaluator_error") if isinstance(blk, dict) else None
    if not msg:
        return None
    return f"{_core.EVALUATOR_FAULT}: {str(msg)[:200]}"


ERRORS_MARKED = "errors_marked"

_SHIPPED_READING = "shipped_reading"


LINK_CHECK_FAILED = "linkage check failed"


def _unmarked_error(b, model_side: bool = False) -> bool:
    """Whether ``b`` is a block carrying an "error" and no "evaluator_error"; with ``model_side``, also one whose text
    does not say the model's fit failed ("fit failed: ...", the grader's text for the model's failure)."""
    return (isinstance(b, dict) and bool(b.get("error")) and not b.get("evaluator_error")
            and not (model_side and str(b["error"]).startswith("fit failed")))


def regrade_marked(vrow):
    """A grader record as the standard rule reads it: one graded before the grader marked its own errors (no
    ERRORS_MARKED stamp) with each unmarked error that a grader writing no stamp wrote for any exception, its own code's
    included, marked "evaluator_error" = sbh.score.REGRADE, so its cases are left out as an evaluator fault until the
    dataset is graded again, never charged and never a failure:
      - a "verify error" status (the record);
      - an error block on a toy stage (GOF_STAGES), except on a record whose GOF the model failed ("gof_failed") or
        where its text says the model's fit failed ("fit failed: ...");
      - a mu_fit error block whose text does not say the model's fit failed;
      - a mass_grid error block not marked unusable by the grader ("usable" False: the model's grid with fewer than
        three usable points);
      - a mass_grid whose link the grader could not check (its "why" starts LINK_CHECK_FAILED), marked on the block.
    A stamped record, one the shipped rule reads (_as_shipped) and anything not a dict come back as they are; ``vrow``
    is never modified."""
    if not isinstance(vrow, dict) or vrow.get(ERRORS_MARKED) or vrow.get(_SHIPPED_READING):
        return vrow
    out = None

    def put(k, b):
        nonlocal out
        out = dict(vrow) if out is None else out
        out[k] = dict(b, evaluator_error=_core.REGRADE)
    if str(vrow.get("status") or "").startswith("verify error") and not vrow.get("evaluator_error"):
        out = dict(vrow, evaluator_error=_core.REGRADE)
    if not vrow.get("gof_failed"):
        for k in GOF_STAGES:
            if _unmarked_error(vrow.get(k), model_side=True):
                put(k, vrow[k])
    if _unmarked_error(vrow.get("mu_fit"), model_side=True):
        put("mu_fit", vrow["mu_fit"])
    mg = vrow.get("mass_grid")
    if isinstance(mg, dict) and not mg.get("evaluator_error"):
        link = mg.get("link")
        if ((_unmarked_error(mg) and mg.get("usable") is not False)
                or (isinstance(link, dict) and str(link.get("why") or "").startswith(LINK_CHECK_FAILED))):
            put("mass_grid", mg)
    return vrow if out is None else out


def record_fault(vrow) -> Optional[str]:
    """The evaluator-fault reason of a grader record as a whole, read as _grader_status reads it: evaluator_fault of the
    record as regrade_marked gives it (an unstamped "verify error" is the fault sbh.score.REGRADE; on a stamped record
    only the grader's mark "evaluator_error" is a fault), or None."""
    r = regrade_marked(vrow)
    return evaluator_fault(r) if isinstance(r, dict) else None


def _fault(why: str) -> dict:
    """A pending measurement on an evaluator fault: never scored on the reported value (model_records)."""
    return {"status": "pending", "reason": why, "evaluator_fault": True}


def row_faults(vrow) -> List[str]:
    """Where a grader record carries an evaluator error on what the score reads, by name: the record itself ("record")
    and its scored blocks, the mass grid's link among them (verify.evaluator_errors over verify.SCORED_BLOCKS; never a
    report-only block, verify.REPORT_ONLY_BLOCKS, which no score term reads).  A record graded before the grader
    marked its own errors is read through regrade_marked."""
    from . import verify as V
    return [e.split(":", 1)[0] for e in V.evaluator_errors(regrade_marked(vrow))]


def run_systemic_places(rows: Optional[Dict[str, dict]], seeds) -> Dict[str, int]:
    """{place: number of datasets} for each place (row_faults) where the records ``rows`` of the run's completed
    datasets ``seeds`` carry an evaluator error on more than half of them: verify.systemic_places over the records as
    the score reads them (regrade_marked), as they stand; {} where none is (and for no datasets)."""
    from . import verify as V
    rows = rows or {}
    seeds = sorted(set(seeds or ()))
    return V.systemic_places({s: regrade_marked(rows.get(s)) for s in seeds}, seeds)


def fault_split(rows: Optional[Dict[str, dict]], seeds=None) -> Tuple[List[str], List[str]]:
    """(reproduced, systemic): the run's datasets (``seeds``, its completed datasets; default every row) whose grader
    record carries an evaluator error on what the score reads (row_faults), split by what the run's records say as
    they stand, whatever order the datasets were graded in (in parallel, or one by one with the earliest faults
    retried before a majority shows):
      - systemic: every place the dataset's record faults lies where the records of more than half of those datasets
        fault (verify.systemic_places, over the records as the score reads them): the evaluator's code or machine,
        never reproduced (sbh.kinds.binned_likelihood.verify.SYSTEMIC_FAULT; the run provisional, never unranked by
        it), whether or not the post stage marked it ("systemic_fault") or retried it;
      - reproduced: any other such dataset whose fault the grader saw again on a rerun (grader_attempts
        "n_evaluator_error" >= 2, verify.count_attempt): a rerun does not clear it, the evaluator's code needs a fix
        (sbh.score.REPRODUCED_FAULT; the run unranked).
    A dataset with no record counts as no fault; a report-only block's error never counts.  The rule is the grader's
    own (verify.is_systemic) on the records as the score reads them."""
    from . import verify as V
    rows = rows or {}
    done = sorted(set(rows) if seeds is None else set(seeds))
    places = run_systemic_places(rows, done)
    repro, systemic = [], []
    for s in done:
        r = rows.get(s)
        if not row_faults(r):
            continue
        att = r.get("grader_attempts") if isinstance(r, dict) else None
        if V.is_systemic(regrade_marked(r), places):
            systemic.append(s)
        elif isinstance(att, dict) and int(att.get("n_evaluator_error") or 0) >= 2:
            repro.append(s)
    return repro, systemic


def reproduced_faults(rows: Optional[Dict[str, dict]], seeds=None) -> List[str]:
    """The reproduced evaluator faults of fault_split: the datasets (of ``seeds``, the run's completed datasets;
    default every row) whose fault the grader saw again on a rerun and that is not systemic (sbh.score.REPRODUCED_FAULT:
    the evaluator's code needs a fix)."""
    return fault_split(rows, seeds)[0]


def systemic_faults(rows: Optional[Dict[str, dict]], seeds=None) -> List[str]:
    """The systemic evaluator faults of fault_split: the datasets (of ``seeds``, the run's completed datasets; default
    every row) whose faults all sit where the same evaluator fault sits on more than half of those datasets, decided
    over the records as they stand: their cases are left out as any evaluator fault's, the run provisional with the
    reason verify.SYSTEMIC_FAULT, never unranked by it."""
    return fault_split(rows, seeds)[1]


def systemic_reason(seeds: List[str], n_records: int) -> Optional[str]:
    """The provisional reason of a run with a systemic evaluator fault (systemic_faults): a named reason and the count,
    no dataset; None without one."""
    from .verify import SYSTEMIC_FAULT
    return (f"{SYSTEMIC_FAULT} on {len(seeds)} of {n_records} datasets: the same evaluator fault on most of the run's "
            "completed datasets, until the evaluator is fixed and the run verified again") if seeds else None


CLAIMS_STAGES = ("quality_crosscheck", "z_exp_asimov")


def claims_stage_faults(rows: Optional[Dict[str, dict]], seeds=None) -> List[str]:
    """The datasets (of ``seeds``, default every row) whose grader record carries an evaluator error on a stage
    Delta_claims reads (CLAIMS_STAGES): those checks are left out, the score provisional until rerun."""
    return [s for s, r in sorted((rows or {}).items()) if (seeds is None or s in seeds) and isinstance(r, dict)
            and r.get("status") == "ok" and set(row_faults(r)) & set(CLAIMS_STAGES)]


def reproduced_reason(seeds: List[str]) -> Optional[str]:
    """The provisional reason of a run with reproduced evaluator faults (reproduced_faults): a named reason and the
    count, no dataset (sbh.score.REPRODUCED_FAULT); None without one."""
    return (f"{_core.REPRODUCED_FAULT} on {len(seeds)} datasets: the evaluator's code needs a fix (a rerun of the "
            "grader reproduced it)") if seeds else None


def _grader_status(vrow: Optional[dict]):
    """("ok" | "failed" | "pending", reason): failed = the model's fault; pending = ours (unverified, grader raised; a
    record carrying "evaluator_error": "evaluator fault: <msg>", evaluator_fault).  A record graded before the grader
    marked its own errors is read through regrade_marked: its "verify error" is the evaluator fault sbh.score.REGRADE
    (never the model's, strict or not); the shipped rule's reading (_as_shipped) keeps it pending as it shipped."""
    if vrow is None:
        return "pending", "not verified yet"
    vrow = regrade_marked(vrow)
    ef = evaluator_fault(vrow)
    if ef:
        return "pending", ef
    st = str(vrow.get("status") or "")
    if st == "ok":
        return "ok", ""
    if st.startswith(_MODEL_SIDE):
        return "failed", st[:200]
    return "pending", st[:200]


def _status_block(vrow: Optional[dict]) -> Optional[dict]:
    """model_mu / model_mh's block for a record that is not "ok" (_grader_status), with evaluator_fault set on an
    evaluator fault; None for an ok record."""
    st, why = _grader_status(vrow)
    if st == "ok":
        return None
    return _fault(why) if why.startswith(_core.EVALUATOR_FAULT) else {"status": st, "reason": why}



STALE_COMBINE_GRID = ("graded before the grader kept the declared grid apart from the combine declaration's own MH "
                      "scan: rerun verify --stage grid")


def _combine_graded(vrow: Optional[dict]) -> bool:
    return isinstance(vrow, dict) and vrow.get("status") == "ok" and vrow.get("engine") == "combine"


def _combine_grid(vrow: dict) -> Optional[dict]:
    """A combine-graded record's grid as the score reads it: None when its mass_grid is a declared grid (score it);
    else the status block -- NO_MEASUREMENT where the grader's stamp says no grid was declared (its mass_grid absent,
    stamped "declared_grid" False, or combine's own scan under the stamp), pending where an unstamped mass_grid is
    combine's own scan (STALE_COMBINE_GRID)."""
    mg = vrow.get("mass_grid")
    stamped = any(isinstance(b, dict) and (b.get("combine_grid_rule") is not None or "declared_grid" in b)
                  for b in (vrow, mg))
    if isinstance(mg, dict) and mg.get("engine") == "combine":
        if not stamped:
            return {"status": "pending", "reason": STALE_COMBINE_GRID}
        return {"status": _core.NO_MEASUREMENT, "reason": "no grid: the combine declaration's own MH scan is not a "
                                                          "declared mass grid"}
    if not isinstance(mg, dict) or (stamped and mg.get("declared_grid") is False):
        return {"status": _core.NO_MEASUREMENT, "reason": "no grid: graded through a combine declaration with no "
                                                          "declared mass grid"}
    return None


def model_mu(vrow: Optional[dict]) -> dict:
    """The scored mu from the grader's mu_fit (value, sigma = profile half-width, sigma_min = smaller of profile /
    Hessian, sigma_stat, z_obs), on every dataset (one graded in combine alike).  status ok / failed (the model's) /
    pending (ours; "evaluator_fault" where the grader marks its own failure; a record without a mu_fit was verified
    before the measurement: rerun the grader)."""
    bad = _status_block(vrow)
    if bad is not None:
        return bad
    mf = vrow.get("mu_fit")
    if mf is None:
        return {"status": "pending", "reason": "verified before the model's mu was measured: rerun the grader"}
    ef = evaluator_fault(mf)
    if ef:
        return _fault(ef)
    if not isinstance(mf, dict) or mf.get("error"):
        return {"status": "failed", "reason": f"the model's mu fit failed: {(mf or {}).get('error')}"[:200]}
    v, s = _finite_p(mf.get("value")), _finite_p(mf.get("sigma"))
    base = {"engine": mf.get("engine"), "sigma_source": mf.get("sigma_source"), "profile": mf.get("profile"),
            "sigma_hess": _finite_p(mf.get("sigma_hess")), "sigma_stat": _finite_p(mf.get("sigma_stat")),
            "z_obs": _finite_p(mf.get("z_obs"))}
    if v is None or s is None or s <= 0:
        return {"status": "failed", "reason": "the model gives no mu uncertainty", **base}
    smin = _finite_p(mf.get("sigma_min"))
    return {"status": "ok", "value": v, "sigma": s, "sigma_min": min(s, smin) if smin and smin > 0 else s, **base}


GRID_OVER_CAP = "MODEL_GRID over the evaluator's size cap"


def grid_over_cap(vrow: Optional[dict]) -> bool:
    """Whether the grader refused the declared grid for its size (its mass_grid block's "over_cap")."""
    mg = vrow.get("mass_grid") if isinstance(vrow, dict) else None
    return bool(isinstance(mg, dict) and mg.get("over_cap"))


def model_mh(vrow: Optional[dict], mh_range=None) -> dict:
    """The scored mH: grid_profile of the declared grid within ``mh_range``.  No usable grid (none, one not linked to
    MODEL.json, one over its size cap where the contract counts it as no grid: model_records handles the other
    case) -> sbh.score.NO_MEASUREMENT, the contract's "no mass measurement".  A dataset graded in combine reads the
    declared grid too (the grader grades a declared grid on every dataset, keeping a combine model's own MH scan
    beside it as combine_mass_grid): its grid as _combine_grid reads it -- combine's own scan is never scored (no mass
    measurement), and pending on a record graded before the grader's stamp.  A grid block the grader marks
    "evaluator_error": pending on an evaluator fault; so is one whose profile at MODEL.json's own workspace the grader
    could not make (its link's "model_nll" carries "evaluator_error": the lower-2NLL rule at the identical point,
    verify_grid._profile_at_model, was not applied, so the profile as fitted is not the one the rule scores)."""
    bad = _status_block(vrow)
    if bad is not None:
        return bad
    mg = vrow.get("mass_grid")
    if _combine_graded(vrow):
        cg = _combine_grid(vrow)
        if cg is not None:
            return cg
    ef = evaluator_fault(mg)
    if ef:
        return _fault(ef)
    link = mg.get("link") if isinstance(mg, dict) and not vrow.get(_SHIPPED_READING) else None
    ef = evaluator_fault(link.get("model_nll")) if isinstance(link, dict) else None
    if ef:
        return _fault(ef)
    if isinstance(mg, dict) and mg.get("over_cap"):
        return {"status": _core.NO_MEASUREMENT, "reason": f"no grid: {GRID_OVER_CAP}"}
    gp = grid_profile(mg, mh_range)
    if not gp["usable"]:
        return {"status": _core.NO_MEASUREMENT, "reason": gp["reason"], "grid": gp}
    return {"status": "ok", "value": gp["mh"], "sigma": gp["sigma_root"], "sigma_min": gp["sigma_min"],
            "spacing": gp["spacing"], "grid": gp}


def _row_scan_version(vrow: Optional[dict]) -> Optional[str]:
    """The scan version a grader record was verified under: its mass_grid's (a grid block without one was verified
    before the scan, whatever else the record carries), else (a record without a grid) its mu_fit's (MODEL.json's own
    scan, verify_fit.mu_fit_block, which every task's verification stamps) or its --stage grid stamp ("regrid");
    None = verified before the scan."""
    if not isinstance(vrow, dict):
        return None
    mg = vrow.get("mass_grid")
    if isinstance(mg, dict):
        return str(mg["scan_version"]) if mg.get("scan_version") is not None else None
    for blk in (vrow.get("mu_fit"), vrow.get("regrid")):
        if isinstance(blk, dict) and blk.get("scan_version") is not None:
            return str(blk["scan_version"])
    return None


class GraderForecast(dict):
    """A marginal_mu result as model_records puts it on the scored POI entry ("forecast").  The kind's case_loss
    scores a density only of this type: a RESULT.json entry is plain JSON, so a "forecast" key an agent writes into
    its own POI entry is never read as the grader's density."""


def model_mu_marginal(vrow: Optional[dict], rec: dict, task, prior_curve: Optional[dict],
                      task_curve: Optional[dict]) -> dict:
    """The scored mu of a gated task (Task.mu_marginal_over_mass): two blocks.  "own" = model_mu(vrow), MODEL.json's
    fixed-mass fit, which truth_cases and claims_rows read (RESULT.json reports what MODEL.json gives); "scored" =
    marginal_mu's summary, the density the case loss reads, with "forecast" the full result (arrays; the POI entry
    carries it, info keeps the summary).  Top level: the scored status / reason, value = mean, sigma = sd (what
    _poi_cases, the tripwire and the report card consume), z_obs = Z_G (discovery), regime, stamp.  The scored block
    reads the grid, not MODEL.json's own fit (the run is measured on the grid it delivered): an own fit that failed at
    MODEL.json's mass leaves a scanned grid scored (own = failed, for truth and claims only).  pending when the record
    itself is pending or was verified before the mu scan (no scan_version), pending on an evaluator fault where the grid
    block carries "evaluator_error" (or where the reference's prior curve leaves no component any measure, marginal_mu);
    failed when the record is the model's failure.  A dataset graded through a combine declaration is scored on the
    declared pyhf grid's scan like every other (the verify stage scans it for a gated task; its grid as _combine_grid
    reads it); without a usable one -- none declared, one over its size cap (the contract: "a grid above its cap counts
    as no grid"), one without a scan, or one with no in-range point that has a usable scan -- its mu case has no
    measurement (sbh.score.NO_MEASUREMENT, charged max(F, 0): the contract's "mu is not evaluated (a failed case)" on a
    dataset that does not fail), never MODEL.json's own fit or the reported value (the point rule against the density
    baseline would hand back the fixed-mass lever to anyone declaring MODEL_COMBINE/)."""
    own = model_mu(vrow)
    st, why = _grader_status(vrow)
    if st != "ok":
        return {**own, "own": own}
    mg = vrow.get("mass_grid")
    ef = evaluator_fault(mg)
    stamp = contract_stamp(rec)
    if ef:
        return {**_fault(ef), "z_obs": None, "regime": None, "stamp": stamp, "own": own, "scored": None}
    combine = _combine_graded(vrow)
    no_grid = None
    if combine:
        cg = _combine_grid(vrow)
        if cg is not None and cg["status"] == "pending":
            return {**cg, "z_obs": None, "regime": None, "stamp": stamp, "own": own, "scored": None}
        if cg is not None:
            no_grid = cg["reason"]
        elif mg.get("over_cap"):
            no_grid = f"no grid: {GRID_OVER_CAP}"
        elif "scan_version" not in mg:
            no_grid = ("graded through a combine declaration without a scanned MODEL_GRID: the mass-marginal mu reads "
                       "the declared grid's per-point scan")
    if no_grid:
        return {"status": _core.NO_MEASUREMENT, "reason": no_grid, "z_obs": None, "regime": None, "stamp": stamp,
                "own": own, "scored": None}
    mg = dict(mg) if isinstance(mg, dict) else {}
    if "scan_version" not in mg:
        if own["status"] != "ok":
            return {**own, "own": own}
        sv = _row_scan_version(vrow)
        if sv is not None and not mg:
            mg["scan_version"] = sv
    fc = marginal_mu(mg, getattr(task, "mass_range", None) or (110.0, 140.0), prior_curve, task_curve, stamp,
                     support=getattr(task, "mu_support", MU_SUPPORT),
                     model_fit=vrow.get("mu_fit") if (own["status"] == "ok" and not combine) else None)
    if combine and fc["status"] == "failed" and fc.get("regime") is None and not fc.get("n_components"):
        return {"status": _core.NO_MEASUREMENT, "reason": "graded through a combine declaration whose grid has no "
                                                          "in-range point with a usable mu scan",
                "z_obs": None, "regime": None, "stamp": stamp, "own": own, "scored": marginal_summary(fc)}
    z = fc["z_g"] if fc["z_g"] is not None else own.get("z_obs")
    top = {"status": fc["status"], "reason": fc["reason"], "z_obs": z, "regime": fc["regime"],
           "stamp": stamp, "own": own, "scored": marginal_summary(fc),
           **({"evaluator_fault": True} if fc.get("evaluator_fault") else {})}
    if fc["status"] == "excluded":
        top.update(status="ok", value=fc["bound"], sigma=fc["support_width"], sigma_min=fc["support_width"],
                   forecast=fc)
    elif fc["status"] == "ok":
        sd = max(float(fc["sigma"]), 1e-12)
        top.update(value=fc["mean"], sigma=sd, sigma_min=sd, forecast=fc)
    return top


def model_records(records: List[dict], rows: Optional[Dict[str, dict]], task, closure: Optional[dict] = None):
    """(records with the GRADER's numbers as POIs, per-seed info).  Per POI of task.model_poi_sources ("fit":
    model_mu, "grid": model_mh): ok -> model value, sigma as total; failed -> omitted (a failed case: the run fails);
    sbh.score.NO_MEASUREMENT -> omitted (charged max(F, 0), the dataset not failed); pending -> the reported POI,
    provisionally (none usable: the case is left out until the grader finishes, case_status), except on an evaluator
    fault ("evaluator_fault") and, under Task.grader_strict, on a completed dataset with no grader record
    (sbh.score.NO_GRADER_RECORD: the evaluation lost its grading): left out, never the reported value, never failed;
    under Task.grader_strict any other pending case fails (the shipped rule's strict reading fails a missing record
    too, as it shipped).  A grid over its size cap where the contract makes
    that a failed run (Task.grid_over_cap_fails) fails every case of the dataset.  significance_obs / discovery become
    Z_obs and Z_obs >= Z_DISCOVERY.
    Under Task.mu_marginal_over_mass the signal POI is model_mu_marginal: its POI entry carries the density as
    "forecast" (the kind's case_loss scores it), info[seed][poi] the two blocks, and discovery reads Z_G.
    ``closure``: sbh.score.load_closure_prior_curves' {"per_dataset": {seed: curve}, "task": curve}."""
    src = getattr(task, "model_poi_sources", None) or {}
    gated = bool(getattr(task, "mu_marginal_over_mass", False))
    sp = getattr(task, "signal_strength_poi", None) or "mu"
    cap_fails = bool(getattr(task, "grid_over_cap_fails", False))
    shipped = _core.score_rule(task) == _core.SHIPPED_RULE
    strict = bool(getattr(task, "grader_strict", False))
    curves = closure or {}
    out, info = [], {}
    for rec in records:
        if not record_ok(rec):
            out.append(rec)
            continue
        vr = (rows or {}).get(rec["seed"])
        lost = vr is None and strict and not shipped
        vr = _as_shipped(vr) if shipped else regrade_marked(vr)
        m = {}
        if shipped and _combine_graded(vr):
            m = {p: {"status": "combine", "reason": SHIPPED_COMBINE} for p in src}
        for p, how in (() if m else src.items()):
            if how == "grid":
                m[p] = model_mh(vr, getattr(task, "mass_range", None))
            elif gated and p == sp:
                m[p] = model_mu_marginal(vr, rec, task, (curves.get("per_dataset") or {}).get(rec["seed"]),
                                         curves.get("task"))
            else:
                m[p] = model_mu(vr)
        if cap_fails and grid_over_cap(vr) and _grader_status(vr)[0] == "ok":
            for p in m:
                m[p] = {"status": "failed", "reason": f"{GRID_OVER_CAP}: a failed run on the dataset"}
        if shipped:
            for mm in m.values():
                if mm.get("status") == _core.NO_MEASUREMENT:
                    mm["status"] = "failed"
        pois = []
        for p in task.pois:
            rp = _reported_poi(rec, p)
            if rp is not None and gated:
                rp = {k: rp[k] for k in ("name", *POI_FIELDS) if k in rp}
            mm = m.get(p)
            if mm is None:
                if rp is not None:
                    pois.append(dict(rp))
            elif mm["status"] == "ok":
                entry = {"name": p, "value": mm["value"], "stat": mm["sigma"], "syst": 0.0, "total": mm["sigma"]}
                if "forecast" in mm:
                    entry["forecast"] = GraderForecast(mm.pop("forecast"))
                pois.append(entry)
            elif mm["status"] == "pending" and lost:
                mm["reason"] = _core.NO_GRADER_RECORD
            elif mm["status"] == "pending" and strict and not mm.get("evaluator_fault"):
                mm.update(status="failed", reason=f"strict: {mm.get('reason')}")
            elif mm["status"] in ("pending", "combine") and rp is not None and not mm.get("evaluator_fault"):
                pois.append(dict(rp))
        info[rec["seed"]] = m
        res = dict(rec["result"], pois=pois)
        z = next((mm.get("z_obs") for mm in m.values() if mm.get("z_obs") is not None), None)
        if z is not None:
            res["significance_obs"], res["discovery"] = z, bool(z >= Z_DISCOVERY)
        out.append(dict(rec, result=res))
    return out, info


SHIPPED_COMBINE = "graded in combine: reported POIs (score-from-fit on combine is open)"

_SHIPPED_UNREAD = {None: ("evaluator_error",), "mu_fit": ("evaluator_error",),
                   "mass_grid": ("evaluator_error", "over_cap")}


def _as_shipped(vrow: Optional[dict]) -> Optional[dict]:
    """A grader record as the shipped rule reads it: without the markers it did not read (_SHIPPED_UNREAD: the record's
    and its mu_fit block's "evaluator_error", its mass_grid block's "evaluator_error" and "over_cap"), flagged so that
    no unmarked error of it is read as an evaluator fault (regrade_marked)."""
    if not isinstance(vrow, dict):
        return vrow
    out = {k: v for k, v in vrow.items() if k not in _SHIPPED_UNREAD[None]}
    out[_SHIPPED_READING] = True
    for b in ("mu_fit", "mass_grid"):
        if isinstance(out.get(b), dict) and any(k in out[b] for k in _SHIPPED_UNREAD[b]):
            out[b] = {k: v for k, v in out[b].items() if k not in _SHIPPED_UNREAD[b]}
    return out


def mark_provisional(v2: dict, v2_cases: List[dict], info: Dict[str, dict]) -> None:
    """Flag the physics cases scored on the reported value while the grader's numbers are pending
    (info[seed][poi]["status"]; a pending case without a usable reported value is not scored but left out, v2
    "unscored", sbh.score._poi_cases): v2["provisional"] = {"n_cases", "cases": ["<seed>:<poi>", ...]} or None."""
    prov = sorted({(c["seed"], c["poi"]) for c in v2_cases
                   if ((info.get(c["seed"]) or {}).get(c["poi"]) or {}).get("status") == "pending"})
    v2["provisional"] = {"n_cases": len(prov), "cases": [f"{s}:{p}" for s, p in prov]} if prov else None


def no_measurement_pois(task) -> set:
    """The POIs this kind can leave without a measurement on a completed dataset (sbh.score.NO_MEASUREMENT): those its
    declared grid measures (Task.model_poi_sources "grid") and, under Task.mu_marginal_over_mass, the signal POI (a
    combine-graded dataset without a usable grid); none under the shipped rule, which knows no such case.  Each needs
    a Task.fallback_forecast: its charge is the no-answer forecast's (sbh.score._poi_cases)."""
    if _core.score_rule(task) == _core.SHIPPED_RULE:
        return set()
    src = getattr(task, "model_poi_sources", None) or {}
    out = {p for p, how in src.items() if how == "grid"}
    if src and getattr(task, "mu_marginal_over_mass", False):
        out.add(getattr(task, "signal_strength_poi", None) or "mu")
    return out


def case_status(info: Optional[Dict[str, dict]]) -> Dict[tuple, tuple]:
    """Kind.case_status from the per-dataset measurement info (model_records' info, scan_likelihood's alike):
    {(seed, poi): (status, reason)} for each "pending", "failed" or sbh.score.NO_MEASUREMENT entry (a strict task has
    no pending one)."""
    keep = ("pending", "failed", _core.NO_MEASUREMENT)
    return {(s, p): (m["status"], str(m.get("reason") or "")) for s, ms in (info or {}).items()
            for p, m in (ms or {}).items() if isinstance(m, dict) and m.get("status") in keep}


def own_fit(mm: Optional[dict]) -> dict:
    """The MODEL.json-fit block of an info entry: its "own" block under the mass-marginal gate, the entry itself
    otherwise (truth_cases, claims_rows compare RESULT.json with MODEL.json's own fit, never with the marginal)."""
    mm = mm or {}
    return mm["own"] if isinstance(mm.get("own"), dict) else mm


def truth_cases(records: List[dict], info: Dict[str, dict], truths: Dict[str, dict], task) -> List[dict]:
    """Truthfulness per case with a reported and a model number:
    T = min(max(0, z - tau) + max(0, ln(RHO sigma_min / s_rep)), TRUTH_CAP), z = |x_rep - x_model| / sigma_model,
    tau_mu = 0.25 + 0.05 / sigma, tau_mH = max(grid spacing, 0.25 sigma + 0.05) / sigma; sigma_min the smaller of the
    grader's two conventions."""
    cases = []
    for rec in records:
        if not record_ok(rec) or rec["seed"] not in info:
            continue
        t = truths[rec["seed"]]
        for p, mm in info[rec["seed"]].items():
            mm = own_fit(mm)
            if p not in t or not task.poi_scored_v2(p, t) or mm["status"] != "ok":
                continue
            rp = _reported_poi(rec, p)
            if rp is None:
                continue
            x_r = _finite_p(rp.get("value"))
            s_r = PoiResult(rp.get("value"), rp.get("stat", 0.0), rp.get("syst", 0.0), rp.get("total", 0.0)).sigma()
            if x_r is None or not (np.isfinite(s_r) and s_r > 0):
                continue
            s_m = mm["sigma"]
            tau = (max(mm.get("spacing") or 0.0, TRUTH_TAU_REL * s_m + TRUTH_TAU_ABS) / s_m if "spacing" in mm
                   else TRUTH_TAU_REL + TRUTH_TAU_ABS / s_m)
            z = abs(x_r - mm["value"]) / s_m
            c_val = max(0.0, z - tau)
            c_sig = max(0.0, math.log(TRUTH_RHO * mm["sigma_min"] / s_r))
            cases.append({"seed": rec["seed"], "poi": p, "reported": x_r, "reported_sigma": s_r,
                          "model": mm["value"], "model_sigma": s_m, "z": z, "tau": tau, "ratio": s_r / s_m,
                          "c_val": c_val, "c_sig": c_sig, "T": min(c_val + c_sig, TRUTH_CAP),
                          "cluster": cluster_of(task, t, rec["seed"])})
    return cases


def truth_term(cases: List[dict]) -> Optional[dict]:
    """Delta_truth: per POI the mean T, then the mean over POIs (Delta_phys's aggregation)."""
    if not cases:
        return None
    per = {}
    for p in sorted({c["poi"] for c in cases}):
        cs = [c for c in cases if c["poi"] == p]
        d = np.array([c["T"] for c in cs])
        per[p] = {"n": len(cs), "delta": float(d.mean()), "n_charged": int(sum(c["T"] > 0 for c in cs)),
                  "se": _cluster_se(d, [c["cluster"] for c in cs])}
    return {"delta": float(np.mean([v["delta"] for v in per.values()])),
            "se": float(np.sqrt(sum(v["se"] ** 2 for v in per.values())) / len(per)),
            "n": len(cases), "n_charged": int(sum(c["T"] > 0 for c in cases)), "per_poi": per}


def claims_rows(records: List[dict], rows: Optional[Dict[str, dict]], info: Dict[str, dict],
                truths: Dict[str, dict], task=None) -> List[dict]:
    """Per completed dataset, {check: failed?} of the other model-checkable claims: significance_exp vs Asimov Z_exp,
    significance_obs vs Z_obs (within max(0.3, 0.15 Z)); discovery vs Z_obs >= 5 (beyond that tolerance); mu stat vs
    sigma_stat and mu syst vs sqrt(sigma^2 - sigma_stat^2) (within max(0.3 x, 0.1 sigma)); gof_pvalue and coverage
    vs quality_crosscheck's agree.  significance_exp is read as the grader judged it (its significance_exp_crosscheck
    "agree"; verify_fit._claim reads the claim through finite_number); a record without that block reads the claim
    through finite_number too: a value that is not a finite JSON number (a string, a bool, null aside) is a failed
    check whether or not the evaluator has a Z_exp, as the grader charges it.  The shipped rule reads it as it shipped
    (a reported number against z_exp_asimov).  The model-side mu numbers are MODEL.json's own fit (own_fit); under the
    mass-marginal gate the discovery claim is checked against Z_G, the grid's own best significance (the run is
    charged for the grid it delivered), the observed significance still against the fit's."""
    poi = getattr(task, "signal_strength_poi", None) or "mu"
    shipped = task is not None and _core.score_rule(task) == _core.SHIPPED_RULE
    out = []
    for rec in records:
        if not record_ok(rec):
            continue
        vr = (rows or {}).get(rec["seed"]) or {}
        top = (info.get(rec["seed"]) or {}).get(poi) or {}
        mu = own_fit(top)
        res = rec["result"]
        ch = {}
        zexp = vr.get("z_exp_asimov")
        zexp = _finite_p(zexp) if not isinstance(zexp, dict) else None
        xc = vr.get("significance_exp_crosscheck")
        if shipped:
            sexp = _finite_p(res.get("significance_exp"))
            if zexp is not None and sexp is not None:
                ch["significance_exp"] = bool(abs(sexp - zexp) > max(0.3, 0.15 * zexp))
        elif isinstance(xc, dict) and xc.get("agree") is not None:
            ch["significance_exp"] = not bool(xc["agree"])
        elif res.get("significance_exp") is not None:
            from .verify_fit import finite_number
            sexp = finite_number(res["significance_exp"])
            if sexp is None:
                ch["significance_exp"] = True
            elif zexp is not None:
                ch["significance_exp"] = bool(abs(sexp - zexp) > max(0.3, 0.15 * zexp))
        zobs = mu.get("z_obs") if mu.get("status") == "ok" else None
        if zobs is not None:
            tol = max(0.3, 0.15 * zobs)
            sobs = _finite_p(res.get("significance_obs"))
            if sobs is not None:
                ch["significance_obs"] = bool(abs(sobs - zobs) > tol)
        zdisc = _finite_p(top.get("z_obs")) if "own" in top else None
        zdisc = zdisc if zdisc is not None else zobs
        if zdisc is not None:
            tol = max(0.3, 0.15 * zdisc)
            disc = res.get("discovery")
            if isinstance(disc, bool):
                ch["discovery"] = bool((disc and zdisc < Z_DISCOVERY - tol) or (not disc and zdisc > Z_DISCOVERY + tol))
        rp = _reported_poi(rec, poi)
        sst = mu.get("sigma_stat") if mu.get("status") == "ok" else None
        if rp is not None and sst:
            s_m = mu["sigma"]
            syst_m = math.sqrt(max(s_m ** 2 - sst ** 2, 0.0))
            st, sy = _finite_p(rp.get("stat")), _finite_p(rp.get("syst"))
            if st is not None:
                ch["mu_stat"] = bool(abs(st - sst) > max(0.3 * sst, 0.1 * s_m))
            if sy is not None:
                ch["mu_syst"] = bool(abs(sy - syst_m) > max(0.3 * syst_m, 0.1 * s_m))
        qc = vr.get("quality_crosscheck") if isinstance(vr.get("quality_crosscheck"), dict) else {}
        for k in ("gof_pvalue", "coverage"):
            if isinstance(qc.get(k), dict) and qc[k].get("agree") is not None:
                ch[k] = not bool(qc[k]["agree"])
        if ch:
            out.append({"seed": rec["seed"], "checks": ch, "delta": sum(ch.values()) / len(ch),
                        "cluster": cluster_of(task, truths[rec["seed"]], rec["seed"])})
    return out


def claims_term(rows_: List[dict]) -> Optional[dict]:
    """Delta_claims: the mean over datasets of the failed fraction of their evaluable checks."""
    if not rows_:
        return None
    d = np.array([r["delta"] for r in rows_])
    fails: Dict[str, List[int]] = {}
    for r in rows_:
        for k, b in r["checks"].items():
            fails.setdefault(k, [0, 0])
            fails[k][0] += int(b)
            fails[k][1] += 1
    return {"delta": float(d.mean()), "se": _cluster_se(d, [r["cluster"] for r in rows_]), "n": len(rows_),
            "fails": {k: {"failed": a, "n": b} for k, (a, b) in sorted(fails.items())}}


def _forfeit(k, se, tol):
    """(f, why): calibration credit withdrawn, f = clip((|k| - thr) / (1 - thr), 0, 1), thr = max(tol, 2 se);
    0 when the check is missing or thr >= 1."""
    k, se = _finite_p(k), _finite_p(se)
    if k is None or se is None:
        return 0.0, "no check"
    thr = max(tol, 2.0 * se)
    if thr >= 1.0:
        return 0.0, f"uninformative (threshold {thr:.2f})"
    return float(np.clip((abs(k) - thr) / (1.0 - thr), 0.0, 1.0)), f"|k| {abs(k):.2f}, threshold {thr:.2f}"


def calibration_gate(leak: Dict[str, dict], yld: Optional[dict], task) -> Dict[str, dict]:
    """Per effect of task.calibration_gates, credit withdrawn where the model shows the calibration is not applied:
    {effect: {"f", "why", "check", "k", "se"}}.  "leakage": the leakage probe k on the grader's numbers; "yield":
    k = max(0, 1 - b) (one-sided: b >> 1 is a regression artifact)."""
    out = {}
    for eff, (kind, key) in (getattr(task, "calibration_gates", None) or {}).items():
        tol = LEAK_TOL if kind == "leakage" else YIELD_TOL
        if kind == "leakage":
            lk = (leak or {}).get(key)
            k, se = (lk or {}).get("k"), (lk or {}).get("se")
        else:
            ok = isinstance(yld, dict) and yld.get("status") == "ok"
            k = max(0.0, 1.0 - float(yld["b"])) if ok else None
            se = yld.get("se") if ok else None
        f, why = _forfeit(k, se, tol)
        out[eff] = {"f": f, "why": why, "check": f"{kind}:{key}" if key else kind, "k": _finite_p(k),
                    "se": _finite_p(se)}
    return out



def completeness_term(syst_cases: Optional[List[dict]]) -> Optional[dict]:
    """Delta_syst: the mean fraction of units not covered, plus missing / rejected counts, units and floors."""
    if not syst_cases:
        return None
    d = np.array([c["delta"] for c in syst_cases])
    syst = {"delta": float(d.mean()), "se": _cluster_se(d, [c["cluster"] for c in syst_cases]), "n": len(syst_cases)}
    miss = [m for c in syst_cases for m in c["missing"]]
    syst["missing"] = {m: miss.count(m) for m in sorted(set(miss))}
    rej = [f"{e}: {f}" for c in syst_cases for e, fs in (c.get("rejected") or {}).items() for f in fs]
    syst["rejected"] = {r: rej.count(r) for r in sorted(set(rej))}
    syst["units"] = syst_cases[0].get("units")
    syst["floors"] = syst_cases[0].get("floors")
    syst["rule"] = SPLIT_RULE if syst["units"] is not None else None
    return syst


def v3_terms(v2: dict, cal_cases: List[dict], gof: Optional[List[dict]], truths, n_records: int,
             syst_cases: Optional[List[dict]] = None, truth: Optional[dict] = None, claims: Optional[dict] = None,
             cal_gate: Optional[Dict[str, dict]] = None, rule: Optional[str] = None, task=None,
             gof_left_out: Optional[Dict[str, str]] = None, reproduced: Optional[List[str]] = None,
             claims_faults: Optional[List[str]] = None, systemic: Optional[List[str]] = None):
    """(terms, pending, extra): the kind's weighted S_v3 terms in summation order (bit-identity), the reasons the
    score is not final, and extra output fields.  ``gof_left_out`` (gof_cases' {seed: reason}): the GOF's provisional
    reasons by cause (gof_pending_reasons; its term's "n_evaluator_fault"); ``reproduced`` (reproduced_faults): the
    reproduced evaluator faults' reason (reproduced_reason) and v3 "reproduced_faults" {"n_datasets"};
    ``claims_faults`` (claims_stage_faults): the claims checks left out on an evaluator fault; ``systemic``
    (systemic_faults): the systemic fault's reason (systemic_reason) and v3 "systemic_faults" {"n_datasets"}.
    Without them (the shipped rule) the reasons it shipped with."""
    for c in gof or []:
        c["cluster"] = cluster_of(task, truths[c["seed"]], c["seed"])
    cal = calibration_term_gated(cal_cases, cal_gate)
    syst = completeness_term(syst_cases)
    g = gof_term(gof)
    if g:
        g["n_pending"] = n_records - len(gof)
    terms = [("calibration", _core.cal_weight(task), cal), ("gof", _core.V3_W_GOF, g), ("completeness", _core.V3_W_SYST, syst)]
    pending = []
    faults = sum(1 for w in (gof_left_out or {}).values() if str(w).startswith(_core.EVALUATOR_FAULT))
    if g is not None and gof_left_out is not None:
        g["n_evaluator_fault"] = faults
    if g is None and not faults:
        pending.append("GOF pending (no grader record)")
    elif gof_left_out is not None:
        pending += gof_pending_reasons(gof_left_out, n_records)
    elif g.get("n_pending"):
        pending.append(f"GOF pending on {g['n_pending']} of {n_records} datasets")
    extra = {"gof_pending": g is None}
    if claims_faults:
        pending.append(f"claims checks left out on an {_core.EVALUATOR_FAULT} on {len(claims_faults)} of {n_records} "
                       "datasets (a stage after the measurement), until it is fixed and the evaluation rerun")
    if systemic:
        pending.append(systemic_reason(systemic, n_records))
        extra["systemic_faults"] = {"n_datasets": len(systemic)}
    if reproduced:
        pending.append(reproduced_reason(reproduced))
        extra["reproduced_faults"] = {"n_datasets": len(reproduced)}
    if rule == FIT_RULE:
        terms += [("truthfulness", V4_W_TRUTH, truth), ("claims", V4_W_CLAIMS, claims)]
        extra.update(rule=FIT_RULE, score_source="model", provisional=v2.get("provisional"))
    return terms, pending, extra


def score_v3(v2: dict, cal_cases: List[dict], gof: Optional[List[dict]], truths, n_records: int,
             comp: Optional[dict] = None, syst_cases: Optional[List[dict]] = None,
             truth: Optional[dict] = None, claims: Optional[dict] = None,
             cal_gate: Optional[Dict[str, dict]] = None, rule: Optional[str] = None,
             comp_gate: Optional[Tuple[float, float]] = None, comp_gate_source: Optional[str] = None,
             task=None, gof_left_out: Optional[Dict[str, str]] = None,
             reproduced: Optional[List[str]] = None, claims_faults: Optional[List[str]] = None,
             systemic: Optional[List[str]] = None) -> dict:
    """S_v3 of this kind's terms (sbh.score.score_v3 over v3_terms)."""
    from ...score import score_v3 as core_score_v3
    terms, pending, extra = v3_terms(v2, cal_cases, gof, truths, n_records, syst_cases=syst_cases, truth=truth,
                                     claims=claims, cal_gate=cal_gate, rule=rule, task=task,
                                     gof_left_out=gof_left_out, reproduced=reproduced, claims_faults=claims_faults,
                                     systemic=systemic)
    return core_score_v3(v2, terms, n_records, comp=comp, comp_gate=comp_gate, comp_gate_source=comp_gate_source,
                         pending=pending, extra=extra)


FALLBACKS = {
    "nominal_fit_fallback": "the nominal fit came from a later attempt: the first one failed (cabinetry, or the JAX "
                            "twin first on a model with many free parameters; fit_attempts)",
    "twin_not_used": "the stages after the nominal fit ran on pyhf, not on the JAX twin: the twin did not build or its "
                     "observed-data fit was invalid (engine_fallback)",
    "combine_not_graded": "a CMS combine declaration could not be graded: MODEL.json was graded instead "
                          "(engine_fallback \"combine: ...\")",
    "stage_on_pyhf": "a stage of a dataset graded on the JAX twin ran on pyhf: the truth-pinned toys (the twin's "
                     "truth-pinned fit failed) or the Asimov Z_exp (the twin's discovery fits were invalid)",
    "mu_width_fallback": "the model's mu width is not the two-sided profile: a side's profile scan failed (the other "
                         "side's half-width) or neither side crossed (the Hessian width; mu_fit sigma_source)",
    "grid_points_on_pyhf": "grid points the JAX twin could not build were fitted on pyhf (mass_grid engine_fallback)",
    "grid_link_by_fit": "the grid was linked to MODEL.json by refitting (its point at the reported mass reproduces "
                        "MODEL.json's fit), not as the identical workspace (mass_grid link.by)",
    "grid_link_cross_engine": "the grid link compared the likelihoods across engines: the grid point and MODEL.json "
                              "were fitted on different ones (link.cross_engine)",
    "model_off_grid_minimum": "MODEL.json's point lies above the grid's likelihood minimum by more than the link "
                              "tolerance (content.model_off_minimum)",
    "grid_point_from_model_fit": "the grid point that is MODEL.json's workspace was scored at MODEL.json's own fit, "
                                 "lower than the point's own (a stalled grid fit; link.model_nll used): its neighbours "
                                 "keep their own fits, so a stall they share narrows the profile; the detail is the "
                                 "drop in twice the negative log-likelihood substituted",
    "yield_band_clip": "grid points above the yield band were scored at its edge (the mass-marginal mu; "
                       "content.n_yield_clipped)",
    "mu_single_mass": "the mass-marginal mu came from one mass point: the grid is unusable for the marginal (not one "
                      "model, not MODEL.json's, its content or coverage failing: the grid's best point) or there is "
                      "none (MODEL.json's own fit)",
    "prior_curve_task_average": "the mass-marginal mu's prior read the suite-average reference curve: the "
                                "reference's own curve on the dataset is unusable",
    "prior_flat": "the mass-marginal mu's prior was flat: no reference curve spans the mass range",
    "gof_at_fit_stage": "the GOF p came from the toys at the fit: the truth-pinned stage gave none",
    "gof_global_only": "the GOF p is the global saturated p alone: the grader record has no combined p (verified "
                       "before the signal-region GOF)",
}

_PROFILE_FAILED = ("failed", "not converged")


def _nominal_fallback(r: dict) -> bool:
    """The nominal fit came from a later attempt (fit_attempts: the used one is not the first); a record written
    before fit_attempts: the twin first above PYHF_NOMINAL_MAX_FREE failed, or cabinetry failed where it is present
    (its absence in a grader runtime is that runtime's procedure, the twin then first)."""
    att = r.get("fit_attempts")
    if isinstance(att, list) and att:
        used = next((i for i, a in enumerate(att) if isinstance(a, dict) and a.get("used")), None)
        return bool(used)
    nf = r.get("nominal_fit_fallback")
    return bool((r.get("nominal_fit_twin") or {}).get("failed")) or bool(nf and "no cabinetry here" not in str(nf))


def _grid_link_by_fit(link: dict) -> bool:
    """A linked grid linked by the fit route, from the link fields the grader writes (verify_grid._grid_link): "by"
    names the route (anything with "fit" in it is the refit comparison); without it, "identical" True is the
    identical-workspace route and "fit_linked" True the refit; a record with none of them was linked by the refit
    comparison, the only route _grid_link then had."""
    if link.get("linked") is not True:
        return False
    by = link.get("by")
    if by is not None:
        return "fit" in str(by).lower()
    if link.get("identical") is True:
        return False
    return True


def fallbacks(records: List[dict], rows: Optional[Dict[str, dict]], info: Optional[Dict[str, dict]],
              gof: Optional[List[dict]], task) -> Dict[str, dict]:
    """{kind: sbh.score.fallback_entry} of FALLBACKS taken on the completed datasets of ``records``: the grader's
    from its records (``rows``, {seed: grader record}; status ok; the grid link's route from its "by", "identical" and
    "fit_linked" fields), the scorer's from the measurement ``info`` (model_records') and the GOF cases (gof_cases).
    A kind not taken is absent.  A case with no measurement is the core's kind (sbh.score.fallback_block)."""
    from ...score import fallback_entry
    sp = getattr(task, "signal_strength_poi", None) or "mu"
    gated = bool(getattr(task, "mu_marginal_over_mass", False))
    done = {rec["seed"] for rec in records if record_ok(rec)}
    hit: Dict[str, set] = {k: set() for k in FALLBACKS}
    detail: Dict[str, Dict[str, float]] = {}
    for s in sorted(done):
        r = (rows or {}).get(s)
        if not isinstance(r, dict) or r.get("status") != "ok":
            continue
        if _nominal_fallback(r):
            hit["nominal_fit_fallback"].add(s)
        for part in str(r.get("engine_fallback") or "").split("; "):
            if part:
                hit["combine_not_graded" if part.startswith("combine:") else "twin_not_used"].add(s)
        ft = r.get("freq_toys") if isinstance(r.get("freq_toys"), dict) else {}
        zi = r.get("z_exp_info") if isinstance(r.get("z_exp_info"), dict) else {}
        if r.get("engine") == "jax" and (ft.get("engine") == "pyhf" or zi.get("engine_fallback")):
            hit["stage_on_pyhf"].add(s)
        mf = r.get("mu_fit") if isinstance(r.get("mu_fit"), dict) else {}
        prof = mf.get("profile") if isinstance(mf.get("profile"), dict) else {}
        if not mf.get("error") and (mf.get("sigma_source") == "hessian" or (
                mf.get("sigma_source") == "profile_one_side"
                and any(prof.get(f"status_{t}") in _PROFILE_FAILED for t in ("lo", "hi")))):
            hit["mu_width_fallback"].add(s)
        mg = r.get("mass_grid") if isinstance(r.get("mass_grid"), dict) else {}
        if mg.get("engine_fallback"):
            hit["grid_points_on_pyhf"].add(s)
        link = mg.get("link") if isinstance(mg.get("link"), dict) else {}
        if _grid_link_by_fit(link):
            hit["grid_link_by_fit"].add(s)
        if link.get("cross_engine"):
            hit["grid_link_cross_engine"].add(s)
        mn = link.get("model_nll") if isinstance(link.get("model_nll"), dict) else {}
        if mn.get("used") is True:
            hit["grid_point_from_model_fit"].add(s)
            t_g, t_m = _finite_p(mn.get("twice_nll_grid")), _finite_p(mn.get("twice_nll_model"))
            if t_g is not None and t_m is not None:
                detail.setdefault("grid_point_from_model_fit", {})[s] = float(t_g - t_m)
        content = mg.get("content") if isinstance(mg.get("content"), dict) else {}
        if (content.get("model_off_minimum") or {}).get("flag"):
            hit["model_off_grid_minimum"].add(s)
        if gated and (content.get("n_yield_clipped") or 0) > 0:
            hit["yield_band_clip"].add(s)
    for s, ms in (info or {}).items():
        if s not in done:
            continue
        for p, m in (ms or {}).items():
            if not isinstance(m, dict):
                continue
            if gated and p == sp and m.get("status") == "ok":
                sc = m.get("scored") if isinstance(m.get("scored"), dict) else {}
                if m.get("regime") == "single_mass":
                    hit["mu_single_mass"].add(s)
                elif sc.get("prior_source") == "task_average":
                    hit["prior_curve_task_average"].add(s)
                elif sc.get("prior_source") == "flat":
                    hit["prior_flat"].add(s)
    for c in gof or []:
        if c.get("seed") in done and c.get("p") is not None and not c.get("charged"):
            if c.get("stage") == "at_fit":
                hit["gof_at_fit_stage"].add(c["seed"])
            if c.get("rule") == "legacy_global":
                hit["gof_global_only"].add(c["seed"])
    return {k: fallback_entry(hit[k], FALLBACKS[k], detail=detail.get(k)) for k in FALLBACKS if hit[k]}
