"""The score's error band (docs/METRICS.md "Error bands"): a paired bootstrap over the pseudo-datasets.

A band resamples the suite's sealed datasets with replacement, BAND_REPLICAS times from the fixed seed BAND_SEED, by
cluster (Task.cluster_key: datasets that share a draw move together; without one each dataset is its own cluster), and
recomputes the whole score on every resample from the per-dataset cases the scorer kept, a dataset drawn twice
counting twice everywhere:
  physics       per POI the mean charge of its charged cases (answered, or charged max(F, 0) for no measurement) plus
                beta B_p, the bias term of those cases' misses (sbh.score.multiworld), then the charge of the mean
                score over POIs (grouped roles: over groups, each the mean charge of its roles), as v2_aggregate does.
                A case drawn c_j times enters the combination c_j times, so its z_star^2 has the expectation
                sum_j c_j^2 e_j^2 / sum_j c_j e_j^2 (e_j = sigma_min / w_j its relative weight) for independent unit
                misses, not 1: each replica's B_p subtracts that (1 on the full sample), so a dataset drawn twice is
                not counted as two independent measurements
  calibration   the mean of the cases' charges, each with its effect's gate forfeit (sbh.hep.calibration)
  gof           Fisher's ensemble test over the resample's usable p's, the others at the maximum charge
  completeness, claims, peak   the mean of the cases' charges
  truthfulness  per POI the mean, then over POIs
  compute       the mean of the completed datasets' charges, its speed bonus gated on the resample's own rest of
                Delta_total (sbh.score.compute_gate; none while the score is provisional)
S's band is the BAND_Q (16 / 84 %) quantiles over the replicas of S; Delta_total's and each recomputed term's
likewise.  The relative score's band is the quantiles of S / S_ref, the reference's score recomputed on the same
resamples from the cases its closure recorded (paired: a dataset that fluctuates moves the run and the reference
alike and cancels in the ratio).  Every interval is widened about the full-sample estimate by
sqrt(n / (n - 1)) t_{n-1}(0.84) / z(0.84), n the number of clusters drawn (widen_factor, _widen): the bootstrap's
plug-in variance of a mean over n clusters is (n - 1) / n of the unbiased one, and that mean, its variance estimated
from the n clusters, is Student-t with n - 1 degrees of freedom; the two together are what leave the quantiles
under-covering at few clusters (1.10 x 1.11 = 1.22 at 6 clusters, 1.15 x 1.20 = 1.38 at 4).  The factor
is derived from n, not a free constant.  S and S / S_ref are widened in their logarithm (S = exp(-gamma Delta_total):
the same as widening Delta_total), the Delta terms linearly.  The intervals are the replicas' percentiles: their misses
are uneven (at 4 to 10 clusters E Delta_phys lies above the Delta band on 23-28 % of runs and below it on 6-11 %, the
true ratio below the relative band somewhat more often than above it: both flatter the run), and no tested variant
removes that imbalance: a bias-corrected and accelerated (BCa) interval, its acceleration from the cluster jackknife,
leaves the sides about as uneven (18.5-28 % above, 6.5-12.6 % below) and covers less at three clusters (55-61 %
against 64-68 % for the Delta band); a studentized bootstrap evens the sides but covers 51-56 % (57-65 % widened).  So
the band keeps the percentiles, with the t factor (BCa unwidened covers 45-65 %); the remaining remedy would be a
variance-stabilising transform of the pull part.

Held at its full-sample value on every replica, and named in the band's "held":
  the calibration gate's forfeits (sbh.kinds.binned_likelihood.scoring.calibration_gate): they come from
                 regressions over the whole run, not from per-dataset cases; the term is recomputed with them
  compute        when it is set by rule: 0 for the closure's reference timed by the yardstick and for a run under
                 its own budget
  any term whose cases do not give back its scored value (a task's own extra term, a kind's term of another form)
A failed run has no band: S = 0 is final.  A task kept on the rule it shipped with (Task.score_rule SHIPPED_RULE) has
none either: its relative score keeps the analytic range it shipped with (sbh.score.relative_score).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

BAND_REPLICAS = 1000

BAND_SEED = 1

BAND_Q = (0.16, 0.84)

BAND_RULE = ("paired bootstrap over the datasets by cluster (Task.cluster_key), the full score recomputed per "
             "resample, a case drawn c times weighed c times in the bias term's expectation; 16 / 84 % quantiles "
             "widened about the estimate by sqrt(n / (n - 1)) t_{n-1}(0.84) / z(0.84), n clusters")

_TOL = 1e-9

TERM_FORMS = {"calibration": ("v3_calibration_cases", "calibration"), "gof": ("v3_gof_cases", "fisher"),
              "completeness": ("v3_completeness_cases", "mean"), "claims": ("v4_claims", "mean"),
              "peak": ("v4_peak_cases", "mean"), "truthfulness": ("v4_truth_cases", "poi_mean")}

EFFORT_HELD = "S_effort at its full-sample value (an effort share; its inputs are not among the terms' cases)"


def _close(a, b) -> bool:
    return (a is not None and b is not None and math.isfinite(float(a))
            and abs(float(a) - float(b)) <= _TOL * max(1.0, abs(float(b))))


def _term_spec(name: str, term: dict, private: dict) -> Optional[dict]:
    """What a band recomputes a weighted term from: its cases in the form the term aggregates them, None for a term of
    no known form or without cases."""
    key, form = TERM_FORMS.get(name, (None, None))
    cases = private.get(key) if key else None
    if not cases:
        return None
    if form == "mean":
        return {"form": form, "rows": [[c["seed"], float(c["delta"])] for c in cases]}
    if form == "poi_mean":
        return {"form": form, "rows": [[c["seed"], c["poi"], float(c["T"])] for c in cases]}
    if form == "fisher":
        return {"form": form, "rows": [[c["seed"], c.get("measurement"),
                                        float(c["p"]) if c.get("p") is not None and not c.get("charged") else None]
                                       for c in cases]}
    if form == "calibration":
        gate = {e: float(pe["gate"]["f"]) for e, pe in (term.get("per_effect") or {}).items()
                if isinstance(pe.get("gate"), dict) and pe["gate"]}
        return {"form": form, "gate": gate,
                "rows": [[c["seed"], c["effect"], float(c["delta"]), float(c["unlogged"])] for c in cases]}
    return None


def _compute_rows(records: List[dict], comp: dict) -> Optional[List[list]]:
    """The compute term's per-dataset charges d' (sbh.score.compute_charges) of the completed datasets, from their
    walls and the term's reference wall and limit; None without a reference wall."""
    from .score import compute_charges
    ref = comp.get("reference_wall_s")
    done = [r for r in records if r.get("exit_code") == 0 and not r.get("timed_out") and r.get("wall_s") is not None]
    if not ref or not done:
        return None
    _, dp = compute_charges(np.array([r["wall_s"] for r in done], dtype=float), float(ref), comp.get("limit_s"))
    return [[r["seed"], float(x)] for r, x in zip(done, dp)]


def inputs(public: dict, private: dict, records: List[dict], clusters: Dict[str, str],
           unit: str) -> Optional[dict]:
    """What a band resamples, from a scored run (its summary, its private scores, its records and each dataset's
    cluster): every sealed dataset's cluster, the physics term's answered cases, each weighted term's cases (or its
    value, held), the compute term's per-dataset charges and gate, and S's constants.  None without a v3 score and
    under the shipped rule (v2 without a physics_rule).  The closure records its reference's (reference_record)."""
    from .score import V3_W_COMP
    v2, v3 = public.get("v2") or {}, public.get("v3") or {}
    if not v3 or not v2 or v2.get("physics_rule") is None:
        return None
    eps = float(v2.get("eps") or 0.0)
    held: Dict[str, str] = {"effort": EFFORT_HELD} if eps else {}
    rows = []
    for c in private.get("v2_cases") or []:
        if c.get("delta") is None:
            continue
        u = c.get("mw_miss")
        ln_w = float(c["ref_loss"]) - 0.5 + float(c["delta_precision"]) if u is not None else None
        rows.append([c["seed"], c["poi"], float(c["delta"]), ln_w, None if u is None else float(u),
                     float(c.get("mw_spread") or 0.0)])
    groups = {r: g for g, b in (v2.get("groups") or {}).items() for r in b.get("roles") or ()} or None
    phys = {"roles": list((v2.get("per_poi") or {}).keys()), "groups": groups,
            "multiworld": v2.get("multiworld_rule") is not None, "beta": float(v2.get("bias_beta") or 0.0),
            "gamma": float(v2["gamma"]), "rows": rows}
    answered = sorted({r[0] for r in rows})
    once = _phys(phys, {s: i for i, s in enumerate(answered)}, np.ones((1, len(answered))))[0]
    if v2.get("delta_mean") is not None and not _close(once, v2["delta_mean"]):
        phys = {"held": float(v2["delta_mean"])}
        held["phys"] = "its cases do not give back its scored value"
    weights = v3.get("weights") or {}
    terms = []
    for name, w in weights.items():
        t = v3.get(name)
        if name in ("phys", "compute") or not w or not isinstance(t, dict) or t.get("delta") is None:
            continue
        try:
            spec = _term_spec(name, t, private)
            once = _term_once(spec) if spec is not None else None
        except (KeyError, TypeError, ValueError, IndexError):
            spec = once = None
        if spec is None:
            held[name] = "no per-dataset cases of a form the band recomputes"
        elif not _close(once, t["delta"]):
            held[name] = "its cases do not give back its scored value"
            spec = None
        elif spec.get("gate") and any(spec["gate"].values()):
            held["calibration_gate"] = "the gate's forfeits at their full-sample values (regressions over the run)"
        terms.append({"name": name, "weight": float(w), "delta": float(t["delta"]), "spec": spec})
    comp = v3.get("compute") or {}
    compute = None
    if comp.get("delta") is not None:
        crow = None
        if comp.get("delta_measured") is not None:
            held["compute"] = "set by rule (the yardstick's reference, a run under its own budget)"
        else:
            crow = _compute_rows(records, comp)
            if crow is None or not _close(float(np.mean([r[1] for r in crow])), comp.get("delta_raw")):
                held["compute"] = "its per-dataset charges do not give back its scored value"
                crow = None
        compute = {"weight": float(V3_W_COMP), "delta": float(comp["delta"]), "rows": crow,
                   "rest0": comp.get("gate_rest0"), "width": comp.get("gate_width"),
                   "provisional": bool(comp.get("gate_provisional"))}
    return {"unit": unit, "datasets": {r["seed"]: str(clusters[r["seed"]]) for r in records},
            "gamma": float(v2["gamma"]), "eps": eps,
            "s_effort": float(v3["s_effort"]) if eps else None,
            "failed": bool(v3.get("failed")), "s_v3": float(v3["s_v3"]), "delta_total": v3.get("delta_total"),
            "phys": phys, "terms": terms, "compute": compute, "held": held}



def _cols(rows, col) -> np.ndarray:
    return np.array([col[r[0]] for r in rows], dtype=int)


def _rowmean(m: np.ndarray) -> np.ndarray:
    """The mean of each row over its finite entries, NaN where it has none."""
    ok = np.isfinite(m)
    n = ok.sum(1)
    return np.where(n > 0, np.where(ok, m, 0.0).sum(1) / np.maximum(n, 1), np.nan)


def _score_mean(m: np.ndarray, gamma: float) -> np.ndarray:
    """Per row the charge of the mean score over its finite entries (sbh.score.score_mean_charge), NaN where it has
    none; one column: the column."""
    if m.shape[1] == 1:
        return m[:, 0]
    ok = np.isfinite(m)
    n = ok.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        lse = np.logaddexp.reduce(np.where(ok, -gamma * m, -np.inf), axis=1)
        return np.where(n > 0, -(lse - np.log(np.maximum(n, 1))) / gamma, np.nan)


def _mean(rows, col, W) -> np.ndarray:
    """The mean of rows [seed, value] over the drawn ones."""
    w = W[:, _cols(rows, col)]
    n = w.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (w @ np.array([r[1] for r in rows], dtype=float)) / n, np.nan)


def _poi_mean(spec, col, W) -> np.ndarray:
    by: Dict[str, list] = {}
    for s, p, x in spec["rows"]:
        by.setdefault(p, []).append([s, x])
    return _rowmean(np.column_stack([_mean(rs, col, W) for rs in by.values()]))


def _calibration(spec, col, W) -> np.ndarray:
    """Delta_cal per replica: the mean over the drawn cases of their gated charges (sbh.hep.calibration.gated_charge,
    each effect's forfeit held at its full-sample value)."""
    from .hep.calibration import gated_charge
    return _mean([[r[0], gated_charge(r[2], r[3], spec["gate"].get(r[1]))] for r in spec["rows"]], col, W)


def _fisher(spec, col, W) -> np.ndarray:
    from .kinds.binned_likelihood.scoring import GOF_CHARGE_MAX, V3_GOF_P_FLOOR, _chi2_logsf_even, gof_charge
    per: Dict[str, list] = {}
    for i, (s, meas, p) in enumerate(spec["rows"]):
        e = per.setdefault(s, [0, 0, {}])
        e[0] += 1
        if p is not None:
            e[1] += 1
            e[2][meas if meas is not None else ("case", i)] = p
    seeds = list(per)
    w = W[:, np.array([col[s] for s in seeds], dtype=int)]
    n = w @ np.array([per[s][0] for s in seeds], dtype=float)
    n_ok = w @ np.array([per[s][1] for s in seeds], dtype=float)
    k = w @ np.array([len(per[s][2]) for s in seeds], dtype=float)
    x = w @ np.array([-2.0 * sum(math.log(max(p, V3_GOF_P_FLOOR)) for p in per[s][2].values()) for s in seeds])
    out = np.full(W.shape[0], np.nan)
    for r in range(W.shape[0]):
        if n[r] > 0:
            df = gof_charge(_chi2_logsf_even(float(x[r]), int(round(k[r])))) if k[r] > 0 else 0.0
            out[r] = (n_ok[r] * df + (n[r] - n_ok[r]) * GOF_CHARGE_MAX) / n[r]
    return out


_FORMS = {"mean": lambda spec, col, W: _mean(spec["rows"], col, W), "poi_mean": _poi_mean,
          "calibration": _calibration, "fisher": _fisher}


def _term(spec: dict, col, W) -> np.ndarray:
    return _FORMS[spec["form"]](spec, col, W)


def _term_once(spec: dict) -> float:
    """A term's value from all its cases once (the full sample)."""
    seeds = sorted({r[0] for r in spec["rows"]})
    return float(_term(spec, {s: i for i, s in enumerate(seeds)}, np.ones((1, len(seeds))))[0])


def _phys(ph: dict, col, W) -> np.ndarray:
    """Delta_phys per replica: per role the mean charged-case charge plus beta B_p over the drawn cases (B_p =
    (z_star^2 + s_star^2 - E_c) / (2 K), sbh.score.multiworld's with E_c = sum c_j^2 e_j^2 / sum c_j e_j^2 for the
    times c_j case j is drawn in place of 1, e_j = sigma_min / w_j each case's weight relative to the role's narrowest
    so no width overflows), then the charge of the mean score over roles, or over groups of roles, each group's
    charge the mean of its roles' (as v2_aggregate)."""
    if "held" in ph:
        return np.full(W.shape[0], ph["held"])
    per = {}
    for p in ph["roles"]:
        rs = [r for r in ph["rows"] if r[1] == p]
        if not rs:
            continue
        w = W[:, _cols(rs, col)]
        n = w.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            dp = (w @ np.array([r[2] for r in rs])) / n
            m = [i for i, r in enumerate(rs) if r[4] is not None]
            if ph["multiworld"] and m:
                lw = np.array([rs[i][3] for i in m])
                e = np.exp(lw.min() - lw)
                u, s = np.array([rs[i][4] for i in m]), np.array([rs[i][5] for i in m])
                wm = w[:, m]
                a = wm @ (e * e)
                z = (wm @ (u * e)) / np.sqrt(a)
                spread = (wm @ (s * s * e * e)) / a
                expect = ((wm * wm) @ (e * e)) / a
                dp = dp + ph["beta"] * np.where(a > 0, (z * z + spread - expect) / (2.0 * n), 0.0)
        per[p] = np.where(n > 0, dp, np.nan)
    if not per:
        return np.full(W.shape[0], np.nan)
    g = ph["gamma"]
    if ph["groups"] is None:
        return _score_mean(np.column_stack(list(per.values())), g)
    by: Dict[str, list] = {}
    for p, x in per.items():
        by.setdefault(ph["groups"][p], []).append(x)
    return _score_mean(np.column_stack([_rowmean(np.column_stack(xs)) for xs in by.values()]), g)


def _evaluate(inp: dict, col, W) -> dict:
    """The run's score per replica: Delta_phys, each weighted term (NaN: no cases drawn), the gated compute term,
    Delta_total and S, as sbh.score.score_v3 sums them."""
    R = W.shape[0]
    phys = _phys(inp["phys"], col, W)
    terms = {}
    rest = phys.copy()
    for t in inp["terms"]:
        x = np.full(R, t["delta"]) if t["spec"] is None else _term(t["spec"], col, W)
        terms[t["name"]] = x
        rest = rest + t["weight"] * np.where(np.isfinite(x), x, 0.0)
    dt = rest
    cp = inp["compute"]
    if cp is not None:
        if cp["rows"] is None:
            c = np.full(R, cp["delta"])
        else:
            d = _mean(cp["rows"], col, W)
            r0, gw = float(cp["rest0"] or 0.0), float(cp["width"] or 0.0)
            excess = np.maximum(0.0, rest - r0)
            g = (np.where(excess <= 0, 1.0, 0.0) if gw <= 0 else np.clip(1.0 - excess / gw, 0.0, 1.0))
            g = np.where(np.isfinite(rest), g, 0.0) * (0.0 if cp["provisional"] else 1.0)
            c = np.maximum(0.0, d) + g * np.minimum(0.0, d)
        terms["compute"] = c
        dt = rest + cp["weight"] * np.where(np.isfinite(c), c, 0.0)
    s_q = np.where(np.isfinite(dt), np.exp(-inp["gamma"] * np.where(np.isfinite(dt), dt, 0.0)), 0.0)
    eps = float(inp.get("eps") or 0.0)
    s = (1.0 - eps) * s_q + eps * inp["s_effort"] if eps else s_q
    return {"phys": phys, "terms": terms, "delta_total": dt, "s_v3": np.zeros(R) if inp["failed"] else s}


def _draw(units: Dict[str, str], replicas: int, seed: int) -> Tuple[List[str], np.ndarray, int]:
    """(datasets, W, number of clusters): the datasets in a fixed order and W[r, j] the times replica r draws dataset
    j's cluster (its clusters drawn with replacement, as many as there are)."""
    seeds = sorted(units, key=lambda s: (units[s], s))
    cl = sorted({units[s] for s in seeds})
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(cl), np.full(len(cl), 1.0 / len(cl)), size=replicas).astype(float)
    ix = {c: i for i, c in enumerate(cl)}
    return seeds, counts[:, [ix[units[s]] for s in seeds]], len(cl)


def _q(x: np.ndarray) -> Optional[List[float]]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return [float(v) for v in np.quantile(x, BAND_Q)] if x.size else None


def _t_cdf(x: float, nu: int) -> float:
    """P(T <= x) for Student's t with an integer nu >= 1 degrees of freedom, in closed form (theta = atan(x / sqrt nu):
    odd nu, (2 / pi) (theta + sin theta cos theta (1 + 2/3 cos^2 + 2 4 / (3 5) cos^4 + ...)), the series to nu - 3;
    even nu, sin theta (1 + 1/2 cos^2 + 1 3 / (2 4) cos^4 + ...), to nu - 2; then (1 + A) / 2)."""
    th = math.atan(x / math.sqrt(nu))
    c2, sn = math.cos(th) ** 2, math.sin(th)
    if nu % 2:
        a, term = 1.0, 1.0
        for k in range(1, (nu - 1) // 2):
            term *= c2 * (2 * k) / (2 * k + 1)
            a += term
        A = (2.0 / math.pi) * (th + (sn * math.cos(th) * a if nu > 1 else 0.0))
    else:
        a, term = 1.0, 1.0
        for k in range(1, nu // 2):
            term *= c2 * (2 * k - 1) / (2 * k)
            a += term
        A = sn * a
    return 0.5 * (1.0 + A)


def t_quantile(p: float, nu: int) -> float:
    """The p quantile (p > 1/2) of Student's t with nu degrees of freedom, by bisection on _t_cdf."""
    lo, hi = 0.0, 1.0
    while _t_cdf(hi, nu) < p:
        lo, hi = hi, 2.0 * hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _t_cdf(mid, nu) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def widen_factor(n_units: int) -> float:
    """sqrt(n / (n - 1)) t_{n-1}(q) / z(q), n the clusters a replica draws and q = BAND_Q[1] the band's upper
    quantile: the bootstrap's plug-in variance of a mean over n clusters is (n - 1) / n of the unbiased one, and a
    mean over n clusters with its variance estimated from them is Student-t with n - 1 degrees of freedom, not
    Gaussian.  Both derived from n; 1 below two clusters."""
    if n_units <= 1:
        return 1.0
    from statistics import NormalDist
    q = BAND_Q[1]
    return math.sqrt(n_units / (n_units - 1.0)) * t_quantile(q, n_units - 1) / NormalDist().inv_cdf(q)


def _widen(q: Optional[List[float]], est, n_units: int, log: bool = False) -> Optional[List[float]]:
    """The interval ``q`` widened about the estimate ``est`` by widen_factor(n_units): est + f (q - est), or for a
    positive quantity (``log``: S, S / S_ref) est (q / est)^f, its logarithm widened alike.  q unchanged without an
    estimate."""
    if q is None or est is None or not math.isfinite(float(est)):
        return q
    f, est = widen_factor(n_units), float(est)
    if log and est > 0 and all(v > 0 for v in q):
        return [float(est * (v / est) ** f) for v in q]
    return [float(est + f * (v - est)) for v in q]


def _reproduces(inp: dict) -> bool:
    """The inputs give back the run's own S and Delta_total on the full sample (every dataset once)."""
    seeds = sorted(inp["datasets"])
    ev = _evaluate(inp, {s: i for i, s in enumerate(seeds)}, np.ones((1, len(seeds))))
    dt = inp.get("delta_total")
    return _close(ev["s_v3"][0], inp["s_v3"]) and (dt is None or _close(ev["delta_total"][0], dt))


def _header(inp: dict, replicas: int, seed: int, n_units: int, n_datasets: int) -> dict:
    return {"rule": BAND_RULE, "replicas": int(replicas), "seed": int(seed), "q": list(BAND_Q), "unit": inp["unit"],
            "n_units": int(n_units), "n_datasets": int(n_datasets), "widen": widen_factor(n_units)}


def run_band(inp: Optional[dict], replicas: int = BAND_REPLICAS, seed: int = BAND_SEED) -> Optional[dict]:
    """A run's own band (v3 "band"): the 16 / 84 % quantiles of S, of Delta_total and of each recomputed term
    ("terms"; held ones in "held") over the replicas, widened about the full-sample estimate (_widen).  None without
    inputs (the shipped rule) or for a failed run; intervals None, with "why", over fewer than two clusters or where
    the inputs do not give back the run's score."""
    if inp is None or inp["failed"]:
        return None
    seeds, W, n_units = _draw(inp["datasets"], replicas, seed)
    out = {**_header(inp, replicas, seed, n_units, len(seeds)), "s_v3": None, "delta_total": None, "terms": {},
           "held": dict(inp["held"])}
    if n_units < 2:
        return {**out, "why": "fewer than two clusters to resample"}
    if not _reproduces(inp):
        return {**out, "why": "the inputs do not give back the run's score"}
    col = {s: j for j, s in enumerate(seeds)}
    ev, ev0 = _evaluate(inp, col, W), _evaluate(inp, col, np.ones((1, len(seeds))))

    def band(x, x0, log=False):
        return _widen(_q(x), float(x0[0]) if np.isfinite(x0[0]) else None, n_units, log)
    terms = {} if "held" in inp["phys"] else {"phys": band(ev["phys"], ev0["phys"])}
    terms.update({t["name"]: band(ev["terms"][t["name"]], ev0["terms"][t["name"]]) for t in inp["terms"]
                  if t["spec"] is not None})
    if inp["compute"] is not None and inp["compute"]["rows"] is not None:
        terms["compute"] = band(ev["terms"]["compute"], ev0["terms"]["compute"])
    return {**out, "s_v3": band(ev["s_v3"], ev0["s_v3"], log=True),
            "delta_total": band(ev["delta_total"], ev0["delta_total"]), "terms": terms}


def paired(run: dict, ref: dict, replicas: int = BAND_REPLICAS, seed: int = BAND_SEED) -> dict:
    """The relative score's band: the run's and the reference's scores on the same resamples of the union of their
    datasets, the 16 / 84 % quantiles of S / S_ref ("ratio") and of S_ref ("reference_s_v3"), widened about their
    full-sample estimates (_widen, in the logarithm); intervals None, with "why", over fewer than two clusters or
    where either side's inputs do not give back its score."""
    units = {**ref["datasets"], **run["datasets"]}
    seeds, W, n_units = _draw(units, replicas, seed)
    out = {**_header(run, replicas, seed, n_units, len(seeds)), "ratio": None, "reference_s_v3": None}
    if n_units < 2:
        return {**out, "why": "fewer than two clusters to resample"}
    for who, inp in (("run", run), ("reference", ref)):
        if not _reproduces(inp):
            return {**out, "why": f"the {who}'s band inputs do not give back its score"}
    col = {s: j for j, s in enumerate(seeds)}
    a, b = _evaluate(run, col, W), _evaluate(ref, col, W)
    one = np.ones((1, len(seeds)))
    a0, b0 = _evaluate(run, col, one)["s_v3"][0], _evaluate(ref, col, one)["s_v3"][0]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(b["s_v3"] > 0, a["s_v3"] / b["s_v3"], np.nan)
    return {**out, "ratio": _widen(_q(r), a0 / b0 if b0 > 0 else None, n_units, log=True),
            "reference_s_v3": _widen(_q(b["s_v3"]), b0, n_units, log=True)}
