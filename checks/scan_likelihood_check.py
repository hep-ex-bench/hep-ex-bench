"""The scan_likelihood kind on a synthetic two-POI task (runs in `pixi run check`, seconds): the POI scaling's gate
rules, the interval form of a POI entry (inside the POI's range; err_minus = 0 only on the lower end, err_plus = 0
only on the upper; stat / syst as {err_minus, err_plus}; the symmetric form alone fails), the 68 % set in several
pieces (scan.pieces_of, the RESULT.json form), the run gate and `sbh check` on a scaled MODEL.json with its refit,
the kind's contract pieces and render, h4l_width's participant stub; the two-piece maths (the symmetric unbounded
case is the Gaussian loss and baseline; the truncated density integrates to 1; the baseline is the forecast's own
expected loss); the scan density (a calibrated pipeline at the reference's scales sits at Delta ~ 0, deflating or
inflating the scan or failing strictly loses; the Gaussian limit; pulls on the truth's side; the
closure's reference entry); the likelihood pull (the side pull on a parabolic scan, the boundary, a truth outside
the scan, an unreadable scan; the bias and width gates; the closure's criteria on it, any other POI on the core's
linear gates); the score from the grader's records: the measured entries (ok, failed, pending, the strict rule,
another scan rule's record), the case losses, truthfulness on a two-piece scan and on an interval, the claims, the
net size of the systematics probes; the grader's scan rules on an exact engine (one minimum for the best fit, its
piece and its errors; a piece reaching a scan end off the range bound) and the closure's gate.

--grader (smoke, a minute): the contract's fit conventions in plain numpy against the grader's engine; the grader
on a model whose scanned POI has two minima: its scan against a brute-force grid of multi-start conditional fits at
every point, both pieces found, `sbh check`'s refit against it; the post stage's record end to end (pins, fits,
cross-checks); what is the model's failure and what the evaluator's, never a pyhf fit; a failed truth-pinned fit;
the verification's cap; a closure-like run of six datasets graded and scored end to end (its physics charge from a
derivation: about -1/2 per case at the density's mode, plus the bias term at its floor -1 / (2 K)), its closure
check, and one dataset whose grading was lost, faulted or left pending, with and without the strict rule (strict
fails only the grader's pending status; a lost record and an evaluator fault stay provisional).
"""
from __future__ import annotations

import contextlib
import io
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sbh.contract import contract_pieces, render_contract  # noqa: E402
from sbh.kinds import get_kind  # noqa: E402
from sbh.kinds.binned_likelihood import model as BLM  # noqa: E402
from sbh.kinds.binned_likelihood.model import ENGINE_MODIFIERS  # noqa: E402
from sbh.kinds.scan_likelihood import scaling  # noqa: E402
from sbh.kinds.scan_likelihood import scan as SC  # noqa: E402
from sbh.task import Task  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(("  ok    " if cond else "  FAIL  ") + name + (f": {detail}" if detail and not cond else ""))
    if not cond:
        fails.append(name)


class ScanTask(Task):
    """A two-POI task of the kind: xi scanned on [0, 100], mu_on fitted on [0, 5]."""
    name = "scan_likelihood_check"
    kind = "scan_likelihood"
    pois = ["xi", "mu_on"]
    poi_bounds = {"xi": (0.0, 100.0), "mu_on": (0.0, 5.0)}
    unified_contract = True
    gof_window = (220.0, 1000.0)
    nuisance_prefix = "ana"

    def contract_values(self):
        return {**super().contract_values(), "scan_poi": "xi",
                "interval_pois": "| `xi` | the width ratio, from 0 to 100 |\n| `mu_on` | the on-shell signal strength, "
                                 "from 0 to 5 |"}


T = ScanTask()


class OpenTask:
    """A one-POI task of the kind whose range has an open upper end (xi >= 0)."""
    name = "scan_likelihood_open"
    kind = "scan_likelihood"
    pois = ["xi"]
    result_type = "measurement"
    poi_bounds = {"xi": (0.0, None)}
    requires_model = False
    nuisance_prefix = "ana"
    fallback_forecast = {"xi": (50.0, 28.868)}
    calibration_scored = {}
    cluster_key = "profile"

    def poi_applicable(self, poi, truth):
        return True

    def poi_scored_v2(self, poi, truth):
        return True

    def calibration_truth(self, truth, log=None):
        return {}

    def calibration_spec(self, log=None):
        return {}


TO = OpenTask()


def pscan(v, em, ep, lo=0.0, hi=None, step=0.25):
    """A two-piece parabolic scan q = ((x - v) / s_side)^2 on [lo, hi] (the scan of the interval (v, em, ep))."""
    import numpy as np
    hi = hi if hi is not None else max(3.0 * v, v + 4.0 * ep, 10.0)
    xs = np.arange(lo, hi + 1e-9, step)
    q = ((xs - v) / np.where(xs > v, ep, max(em, 1e-12))) ** 2
    return {"points": [[float(x), float(y)] for x, y in zip(xs, q)]}


def spec(**kw):
    """A scaled workspace in the positive-template basis: off-shell SBI x sqrt(g), B x (1 - sqrt(g)) as B and B x
    -sqrt(g), S x g, g = mu_on xi; on-shell H x mu_on."""
    nf = [{"name": "mu_on", "type": "normfactor", "data": None}, {"name": "xi", "type": "normfactor", "data": None}]
    lumi = {"name": "lumi", "type": "normsys", "data": {"hi": 1.02, "lo": 0.98}}
    sp = {"channels": [
        {"name": "off", "samples": [
            {"name": "sbi", "data": [8.5, 7.6], "modifiers": nf + [lumi]},
            {"name": "b", "data": [8.0, 6.0], "modifiers": [lumi]},
            {"name": "bneg", "data": [8.0, 6.0], "modifiers": nf + [lumi]},
            {"name": "s", "data": [1.0, 2.0], "modifiers": nf + [lumi]}]},
        {"name": "on", "samples": [{"name": "h", "data": [5.0, 6.0], "modifiers": nf[:1] + [lumi]}]}],
        "observations": [{"name": "off", "data": [9, 8]}, {"name": "on", "data": [5, 7]}],
        "measurements": [{"name": "m", "config": {"poi": "xi", "parameters": []}}], "version": "1.0.0",
        "_meta": {"binning": {"off": {"observable": "m4l", "edges": [220.0, 400.0, 1000.0], "data_derived": False},
                              "on": {"observable": "m4l", "edges": [118.0, 125.0, 132.0], "data_derived": False}},
                  "poi_scaling": {"off": {"sbi": {"factor": 1.0, "powers": {"mu_on": 0.5, "xi": 0.5}},
                                          "bneg": {"factor": -1.0, "powers": {"mu_on": 0.5, "xi": 0.5}},
                                          "s": {"factor": 1.0, "powers": {"mu_on": 1, "xi": 1}}}}}}
    sp["_meta"]["poi_scaling"]["off"].update(kw.pop("entries", {}))
    for (ch, smp), mods in kw.pop("modifiers", {}).items():
        next(s for c in sp["channels"] if c["name"] == ch for s in c["samples"] if s["name"] == smp)["modifiers"] = mods
    return sp


def result(**pois):
    """A RESULT.json with xi and mu_on intervals and every modifier of spec() logged."""
    base = {"xi": {"value": 1.2, "err_minus": 0.8, "err_plus": 2.1}, "mu_on": {"value": 1.0, "err_minus": 0.2,
                                                                                "err_plus": 0.25}}
    base.update(pois)
    return {"pois": [{"name": n, **v} for n, v in base.items()], "nuisances": {"lumi": {"value": 1.0, "unc": 0.02}}}


def check_scaling():
    check("a valid scaled model passes the scaling rules", scaling.problems(spec(), T) == [],
          scaling.problems(spec(), T))
    check("the sizing point's scale is each scaled sample's factor",
          scaling.sm_scale(spec()) == {("off", "sbi"): 1.0, ("off", "bneg"): -1.0, ("off", "s"): 1.0})
    nf = [{"name": "mu_on", "type": "normfactor", "data": None}, {"name": "xi", "type": "normfactor", "data": None}]
    bad = {
        "grammar": spec(entries={"s": {"factor": 0.0, "powers": {"mu_on": 1, "xi": 1}}}),
        "not a POI": spec(entries={"s": {"factor": 1.0, "powers": {"mu_on": 1, "xi": 1, "kappa": 0}}}),
        "twice": spec(modifiers={("off", "b"): [{"name": "lumi", "type": "normsys",
                                                  "data": {"hi": 1.02, "lo": 0.98}}] * 2}),
        "modifier type": spec(modifiers={("off", "b"): [{"name": "ana_shape", "type": "shapefactor", "data": None}]}),
        "negative nominal": spec(),
        "lone negative power": spec(entries={"s": {"factor": 1.0, "powers": {"mu_on": -1, "xi": 1}}}),
        "POI not a normfactor": spec(modifiers={("off", "sbi"): [], ("off", "bneg"): [], ("off", "s"): [],
                                                ("on", "h"): []}, entries={}),
        "not an object": dict(spec(), _meta={"binning": {}, "poi_scaling": ["x"]}),
    }
    bad["negative nominal"]["channels"][0]["samples"][3]["data"] = [-1.0, 2.0]
    bad["non-finite data"] = spec()
    bad["non-finite data"]["channels"][0]["samples"][1]["data"] = [8.0, float("nan")]
    bad["POI not a normfactor"]["_meta"]["poi_scaling"] = {}
    want = {"grammar": "factor 0.0", "not a POI": "not a POI", "twice": "lumi (normsys)", "negative nominal": ">= 0",
            "modifier type": "modifiers of type shapefactor: the evaluator reads only normfactor, normsys, histosys, "
                             "staterror, shapesys, lumi",
            "lone negative power": "min_of", "POI not a normfactor": "no normfactor", "not an object": "an object",
            "non-finite data": "off/b: a sample's data must be finite numbers"}
    got = {k: scaling.problems(s, T) for k, s in bad.items()}
    check("each rule refuses its violation", all(any(want[k] in p for p in got[k]) for k in bad),
          {k: v for k, v in got.items() if not any(want[k] in p for p in v)})
    ok_min = spec(entries={"x": {"factor": -1.0, "min_of": [{"mu_on": 0.5, "xi": 0.5}, {"mu_on": -0.5, "xi": -0.5}]}})
    ok_min["channels"][0]["samples"].append({"name": "x", "data": [0.5, 0.4], "modifiers": nf})
    check("a negative power inside a min_of beside a monomial without one passes", scaling.problems(ok_min, T) == [],
          scaling.problems(ok_min, T))


def check_interval_form():
    from sbh.contract import check_result

    def res(v, em, ep, **kw):
        return {"pois": [{"name": "xi", "value": v, "err_minus": em, "err_plus": ep, **kw}]}

    def ok(r, task=TO):
        return not check_result(r, task)
    upper = type("UpperTask", (OpenTask,), {"poi_bounds": {"xi": (0.0, 100.0)}})()
    cases = {
        "a valid interval passes": ok(res(10.0, 3.0, 4.0)),
        "value on the lower end with err_minus 0 passes": ok(res(0.0, 0.0, 2.0)),
        "err_minus 0 away from the lower end fails": not ok(res(5.0, 0.0, 2.0)),
        "value - err_minus below the range fails": not ok(res(1.0, 1.5, 2.0)),
        "value - err_minus on the lower end passes": ok(res(1.0, 1.0, 2.0)),
        "err_plus 0 fails away from an upper end": not ok(res(5.0, 1.0, 0.0)),
        "value on the upper end with err_plus 0 passes": ok(res(100.0, 5.0, 0.0), upper),
        "value + err_plus above the range fails": not ok(res(90.0, 5.0, 20.0), upper),
        "a negative value fails": not ok(res(-0.1, 0.0, 2.0)),
        "a missing err_plus fails": not ok({"pois": [{"name": "xi", "value": 1.0, "err_minus": 0.5}]}),
        "a non-finite err fails": not ok(res(1.0, 0.5, float("inf"))),
        "the symmetric form (stat/syst/total) alone fails": not ok(
            {"pois": [{"name": "xi", "value": 1.0, "stat": 0.5, "syst": 0.0, "total": 0.5}]}),
        "stat / syst as {err_minus, err_plus} pass": ok(res(10.0, 3.0, 4.0, stat={"err_minus": 2.0, "err_plus": 3.0},
                                                         syst={"err_minus": 1.0, "err_plus": 1.5})),
        "stat as a plain number fails": not ok(res(10.0, 3.0, 4.0, stat=2.0)),
        "the task's POI missing fails": not ok({"pois": [{"name": "mu", "value": 1.0, "err_minus": 0.5,
                                                          "err_plus": 0.5}]}),
    }
    check("the interval form of a POI entry: inside the range, err_minus = 0 only on the lower end and err_plus = 0 "
          "only on the upper, stat / syst as sides, never the symmetric form alone",
          all(cases.values()), [k for k, v in cases.items() if not v])


def check_two_piece():
    import math
    import numpy as np
    from sbh.kinds.scan_likelihood import density as D
    sym = all(abs(D.loss(z * sg, 0.0, sg, sg) - (math.log(sg) + 0.5 * z * z)) < 1e-12
              and abs(D.expected_loss(0.0, sg, sg) - (math.log(sg) + 0.5)) < 1e-12
              for sg, z in ((0.7, 0.3), (2.0, -1.7), (5.0, 2.5)))
    check("two-piece: symmetric and unbounded, the loss is ln sigma + z^2 / 2 and the baseline ln sigma + 1/2", sym)
    off = {}
    for m, sm, sp, lo in ((10.0, 3.0, 4.0, 0.0), (1.0, 1.5, 2.0, 0.0), (0.0, 0.0, 2.0, 0.0), (0.3, 0.3, 0.9, 0.0),
                          (5.0, 1.0, 1.0, None)):
        x = np.linspace(lo if lo is not None else m - 12 * sm, m + 14 * sp, 400001)[::50]
        loss = np.array([D.loss(float(v), m, sm, sp, lo) for v in x])
        dens = np.exp(-(loss + D.LN_SQRT_2PI))
        norm = float(np.sum(0.5 * (dens[1:] + dens[:-1]) * np.diff(x)))
        el = float(np.sum(0.5 * ((dens * loss)[1:] + (dens * loss)[:-1]) * np.diff(x))) / norm
        if abs(norm - 1.0) >= 2e-4 or abs(el - D.expected_loss(m, sm, sp, lo)) >= 2e-4:
            off[(m, sm, sp, lo)] = (norm, el, D.expected_loss(m, sm, sp, lo))
    check("two-piece: the truncated density integrates to 1 and the baseline is its own expected loss (numerical "
          "integrals; at the lower end too); a truth outside the range has an infinite loss",
          not off and D.loss(-1.0, 1.0, 1.0, 1.0, 0.0) == math.inf, off)


def interval_runs(rng, truths, ref, mode="accurate"):
    """(records, scans) of a pipeline whose intervals are calibrated two-piece forecasts at the reference's scales
    (mode deflate / inflate rescales the errors and the scan; fail drops every other dataset), each scan the parabolic
    one of its interval."""
    recs, scans = [], {}
    for k, (seed, t) in enumerate(truths.items()):
        sm, sp = ref[seed]["xi"]["err_minus"], ref[seed]["xi"]["err_plus"]
        z = abs(rng.normal())
        v = max(t["xi"] - z * sp if rng.random() < sp / (sm + sp) else t["xi"] + z * sm, 0.0)
        f = {"accurate": 1.0, "deflate": 1 / 3, "inflate": 3.0, "fail": 1.0}[mode]
        em, ep = (0.0 if v == 0.0 else min(sm * f, v)), sp * f
        r = {"seed": seed, "exit_code": 0, "timed_out": False, "result_errors": [],
             "result": {"pois": [{"name": "xi", "value": v, "err_minus": em, "err_plus": ep}]}}
        scans[seed] = pscan(v, em, ep, step=0.05, hi=v + 8 * ep)
        if mode == "fail" and k % 2:
            r.update(exit_code=1, result=None)
        recs.append(r)
    return recs, scans


def check_density():
    import math
    from dataclasses import asdict
    import numpy as np
    from sbh import score as S
    from sbh.kinds.binned_likelihood.scoring import log_density_at
    from sbh.kinds.scan_likelihood import density as D
    kind = get_kind(TO)

    def forecast(scan):
        return D.ScanForecast(D.scan_density(scan, 0.0, scan["points"][-1][0]))
    rng = np.random.default_rng(7919)
    truths = {f"seed_{i:03d}": {"xi": float(rng.choice([8.0, 16.0, 24.0, 32.0])), "profile": f"p{i % 6}"}
              for i in range(1, 201)}
    iv = {s_: {"value": t["xi"], "err_minus": 0.8 + 0.15 * t["xi"], "err_plus": 1.0 + 0.22 * t["xi"]}
          for s_, t in truths.items()}
    ref = {s_: {"xi": kind.reference_poi(e, {"name": "xi", "forecast": forecast(pscan(
        e["value"], e["err_minus"], e["err_plus"], step=0.05, hi=e["value"] + 8 * e["err_plus"]))})}
        for s_, e in iv.items()}
    runs, scored = {}, {}
    for m in ("accurate", "deflate", "inflate", "fail"):
        recs, scans = interval_runs(np.random.default_rng(7), truths, ref, m)
        scored[m] = [r if r["result"] is None else dict(r, result={"pois": [dict(
            r["result"]["pois"][0], forecast=forecast(scans[r["seed"]]))]}) for r in recs]
        runs[m] = S.score_v2(scored[m], truths, TO, {"xi": 4.0}, ref)
    h = runs["accurate"]
    check("the scan density: a calibrated pipeline at the reference's scales sits at Delta ~ 0 (|Delta| < 3 SE); "
          "deflating or inflating the interval and its scan x3 strictly loses; failing half the datasets charges "
          "each as no measurement (the failed datasets named, the run keeps a lower score)",
          abs(h["delta_mean"]) < 3 * h["delta_se"] + 0.02 and runs["deflate"]["delta_mean"] > h["delta_mean"] + 0.5
          and runs["inflate"]["delta_mean"] > h["delta_mean"] + 0.5 and h["failed"] is None
          and runs["fail"]["failed"] is None and (runs["fail"]["failed_datasets"] or {}).get("n_datasets") == 100
          and runs["fail"]["n_failed_charged"] == 100
          and all(c["delta"] >= 0.0 for c in runs["fail"]["cases"] if c.get("failed_charged"))
          and runs["fail"]["delta_mean"] > h["delta_mean"] + 0.5
          and 0.0 < S.score_v3(runs["fail"], [], 0)["s_v3"] < 0.5 * S.score_v3(h, [], 0)["s_v3"]
          and S.score_v3(h, [], 0)["s_v3"] > 0.5,
          {m: (r["delta_mean"], r["delta_se"], (r["failed_datasets"] or {}).get("n_datasets")) for m, r in runs.items()})
    c0 = h["cases"][0]
    fc0 = scored["accurate"][0]["result"]["pois"][0]["forecast"]
    t0 = truths[c0["seed"]]["xi"]
    good = [c for c in h["cases"] if not c.get("fallback")]
    check("the scanned POI's case: loss = -ln p(truth) - ln sqrt(2 pi) against the reference scan's H (its mean -ln p) "
          "- ln sqrt(2 pi); it keeps the density's summary, and its precision piece is its H over the reference's",
          abs(c0["loss"] - (-log_density_at(fc0, t0) - D.LN_SQRT_2PI)) < 1e-12
          and abs(c0["ref_loss"] - (ref[c0["seed"]]["xi"]["entropy"] - D.LN_SQRT_2PI)) < 1e-12
          and all(c0.get(k) is not None for k in ("value", "sigma", "entropy", "pit", "support"))
          and all(abs(c["delta_precision"] - (c["entropy"] - c["ref_loss"] - D.LN_SQRT_2PI)) < 1e-12 for c in good),
          {k: c0.get(k) for k in ("loss", "ref_loss", "value", "sigma", "entropy", "pit", "support")})
    g = D.scan_density(pscan(30.0, 2.0, 2.0, step=0.05, lo=10.0, hi=50.0), 0.0, 100.0)
    check("a symmetric parabolic scan's density: loss ln sigma + pull^2 / 2, H ln sigma + 1/2 + ln sqrt(2 pi); "
          "its support is the range",
          all(abs(-log_density_at(g, t) - D.LN_SQRT_2PI - (math.log(2.0) + 0.5 * ((t - 30.0) / 2.0) ** 2)) < 1e-9
              for t in (27.0, 31.0, 34.0)) and abs(g["entropy"] - D.LN_SQRT_2PI - (math.log(2.0) + 0.5)) < 1e-9
          and g["support"] == [0.0, 100.0])
    rec = {"seed": "seed_001", "exit_code": 0, "timed_out": False, "result_errors": [],
           "result": {"pois": [{"name": "xi", "value": 10.0, "err_minus": 2.0, "err_plus": 5.0}]}}
    up = S.seed_results([rec], {"seed_001": {"xi": 15.0}}, TO)[0].pois["xi"]
    dn = S.seed_results([rec], {"seed_001": {"xi": 6.0}}, TO)[0].pois["xi"]
    kept = kind.reference_poi(asdict(up))
    with_scan = kind.reference_poi(asdict(up), {"name": "xi", "forecast": fc0})
    check("pulls divide by err_plus for a truth above the value and err_minus below; the closure keeps the "
          "reference's interval and, from its scored entry, its scan density's H (its mean -ln p); a symmetric reference entry "
          "keeps the Gaussian baseline",
          up.sigma() == 5.0 and dn.sigma() == 2.0
          and kept == {"value": 10.0, "sigma": 3.5, "err_minus": 2.0, "err_plus": 5.0}
          and with_scan == {**kept, "entropy": fc0["entropy"]}
          and kind.reference_poi({"value": 1.0, "stat": 0.3, "syst": 0.4, "total": 0.5}) == {"value": 1.0, "sigma": 0.5},
          (up.sigma(), dn.sigma(), kept, with_scan))
    two = type("OpenTask2", (OpenTask,), {"pois": ["xi", "mu"]})()
    cl = kind.case_loss({"name": "mu", "value": 1.0, "err_minus": 0.2, "err_plus": 0.3}, "mu", 1.1, two, (1.0, 1.0),
                        {"value": 1.0, "sigma": 0.25, "err_minus": 0.2, "err_plus": 0.3}, 0.0)
    check("any other POI: the two-piece loss of its interval against the reference's two-piece baseline",
          abs(cl["loss"] - D.loss(1.1, 1.0, 0.2, 0.3)) < 1e-12
          and abs(cl["ref_loss"] - D.expected_loss(1.0, 0.2, 0.3)) < 1e-12, cl)


def check_likelihood_pulls():
    import math
    from types import SimpleNamespace
    import numpy as np
    from sbh.kinds.scan_likelihood import closure as CL
    from sbh.kinds.scan_likelihood.verify import SCAN_VERSION
    sc = pscan(10.0, 2.0, 3.0, step=0.01, hi=40.0)
    lp_up, lp_dn = CL.likelihood_pull(sc, 10.0, 13.0, 0.0), CL.likelihood_pull(sc, 10.0, 8.0, 0.0)
    lb = CL.likelihood_pull(pscan(0.0, 0.0, 4.0, step=0.05, hi=30.0), 0.0, 2.0, 0.0)
    sh = {"points": [[x, q + 0.7] for x, q in sc["points"]]}
    check("the likelihood pull: on a parabolic two-piece scan the side pull (value - truth) / s_side; at the xi = 0 "
          "end (value 0 +4, truth 2) -0.5; q from the scan's own minimum; a truth outside the scan -inf / +inf; an "
          "unreadable scan None",
          abs(lp_up + 1.0) < 1e-6 and abs(lp_dn - 1.0) < 1e-6 and abs(lb + 0.5) < 1e-6
          and abs(CL.likelihood_pull(sh, 10.0, 13.0, 0.0) + 1.0) < 1e-6
          and CL.likelihood_pull(sc, 10.0, 50.0, 0.0) == -math.inf
          and CL.likelihood_pull(pscan(30.0, 2.0, 3.0, lo=20.0, hi=50.0), 30.0, 10.0, 0.0) == math.inf
          and CL.likelihood_pull({"points": [[0, 1]]}, 1.0, 1.0, 0.0) is None
          and CL.likelihood_pull(None, 1.0, 1.0, 0.0) is None, (lp_up, lp_dn, lb))
    rng = np.random.default_rng(20261019)
    n = 120
    truths = {f"seed_{i:03d}": {"xi": float(rng.choice([8.0, 16.0, 24.0, 32.0])), "profile": f"p{i % 6}"}
              for i in range(1, n + 1)}
    ref = {s_: {"xi": {"err_minus": 0.8 + 0.15 * t["xi"], "err_plus": 1.0 + 0.22 * t["xi"]}}
           for s_, t in truths.items()}
    recs, scans = interval_runs(np.random.default_rng(3), truths, ref)
    accurate = CL.likelihood_pulls(recs, truths, TO, scans)["xi"]
    recs_d, scans_d = interval_runs(np.random.default_rng(3), truths, ref, "deflate")
    defl = CL.likelihood_pulls(recs_d, truths, TO, scans_d)["xi"]
    nopull = CL.likelihood_pulls(recs, truths, TO, {**scans, "seed_001": None})["xi"]
    check("calibrated runs pass the likelihood pulls' bias and width gates (coverage ~ 68 %); errors deflated x1/3 "
          "fail the width gate (width ~ 3); a dataset without a scan has no pull",
          accurate["bias_pass"] and accurate["width_pass"] and 0.55 < accurate["coverage"] < 0.80 and accurate["n"] == n
          and not defl["width_pass"] and defl["width"] > 2.0 and nopull["n"] == n - 1,
          {k: accurate[k] for k in ("mean", "width", "coverage")} | {"deflated width": defl["width"]})
    a = SimpleNamespace(max_pull=None, max_leakage=None, leakage_nsigma=None, leakage_se_cap=None,
                        calibration_slope_tol=None)
    pub = {"recovery": 1.0, "pois": {"xi": {"sensitivity_pass": True, "calibration_pass": False, "bias_pass": True}},
           "cluster_bias": {"xi": {"pass": True}}}
    lp = {"xi": {**{k: v for k, v in accurate.items() if k != "per_seed"}, "max_abs": 3.2}}
    priv = {"per_seed": [{"pulls": {"xi": -6.19}}, {"pulls": {"xi": 0.4}}], "likelihood_pulls": lp,
            "reference_pois": {"seed_001": {"xi": {"entropy": 2.0}}, "seed_002": {"xi": {"entropy": 2.1}}}}
    with contextlib.redirect_stdout(io.StringIO()):
        crit, mp, fields = CL.closure_criteria(pub, priv, TO, a)
        bad = CL.closure_criteria(pub, {**priv, "likelihood_pulls": {"xi": {**lp["xi"], "width_pass": False}}}, TO, a)
    lin = fields["linear_pull_diagnostic"]
    check("the closure grades the likelihood pull: it passes where the linear pull fails, and fails on a failing "
          "likelihood-pull width; its scan_baseline is the grader's",
          all(crit.values()) and mp == 3.2 and not all(lin["criteria"].values()) and lin["max_abs_pull"] == 6.19
          and not all(bad[0].values())
          and fields["scan_baseline"] == {"source": "grader", "n": 2, "n_datasets": 2, "missing": [],
                                          "scan_version": SCAN_VERSION}, (crit, lin))
    two = type("OpenTask2", (OpenTask,), {"pois": ["xi", "mu"]})()
    recs2 = [dict(r, result={"pois": r["result"]["pois"] + [{"name": "mu", "value": 1.0, "err_minus": 0.1,
                                                              "err_plus": 0.1}]}) for r in recs]
    lp2 = CL.likelihood_pulls(recs2, {s_: {**t, "mu": 1.0} for s_, t in truths.items()}, two, scans)
    pub2 = {**pub, "pois": {**pub["pois"], "mu": {"sensitivity_pass": True, "calibration_pass": True,
                                                   "bias_pass": True}},
            "cluster_bias": {**pub["cluster_bias"], "mu": {"pass": True}}}
    priv2 = {**priv, "per_seed": [{"pulls": {"xi": -6.19, "mu": 0.5}}, {"pulls": {"xi": 0.4, "mu": -1.1}}]}
    off = {**pub2, "cluster_bias": {**pub2["cluster_bias"], "mu": {"pass": False}}}
    nocb = {**pub2, "pois": {**pub2["pois"], "mu": {**pub2["pois"]["mu"], "bias_pass": False}},
            "cluster_bias": {"xi": {"pass": True}}}

    def passes(pub_):
        with contextlib.redirect_stdout(io.StringIO()):
            return all(CL.closure_criteria(pub_, priv2, two, a)[0].values())
    with contextlib.redirect_stdout(io.StringIO()):
        crit2, mp2, _ = CL.closure_criteria(pub2, priv2, two, a)
    check("likelihood pulls only for the scanned POI; a second POI passes on the core's linear gates and fails on a "
          "failing cluster bias (no cluster-bias entry: not gated; no cluster bias: the per-seed bias_pass)",
          set(lp2) == {"xi"} and all(crit2.values()) and mp2 == 3.2 and not passes(off) and passes(nocb)
          and not passes({**nocb, "cluster_bias": {}}) and passes({**pub2, "cluster_bias": {}}), (sorted(lp2), crit2))


def check_pieces():
    xs = [i * 0.5 for i in range(61)]
    qs = [min((x - 3.0) ** 2 / 4.0, 0.4 + (x - 20.0) ** 2 / 9.0) for x in xs]
    ps = SC.pieces_of(xs, qs, 0.0, 30.0)
    iv = SC.interval_of(xs, qs, 0.0, 30.0)
    check("pieces_of: both q <= 1 pieces, their minima and q = 1 ends",
          len(ps) == 2 and ps[0]["x_min"] == 3.0 and ps[1]["x_min"] == 20.0 and abs(ps[0]["lo"] - 1.0) < 1e-9
          and abs(ps[0]["hi"] - 5.0) < 1e-9 and abs(ps[1]["lo"] - (20 - 5.4 ** 0.5)) < 0.02
          and abs(ps[1]["hi"] - (20 + 5.4 ** 0.5)) < 0.02,
          ps)
    check("interval_of is the piece holding the minimum", iv == ps[0], iv)
    kind = get_kind(T)
    good = {"name": "xi", "value": 3.0, "err_minus": 2.0, "err_plus": 2.0, "intervals": [[1.0, 5.0], [17.8, 22.2]]}
    probs = {
        "valid": kind.poi_problems(good, T),
        "unsorted": kind.poi_problems(dict(good, intervals=[[17.8, 22.2], [1.0, 5.0]]), T),
        "overlapping": kind.poi_problems(dict(good, intervals=[[1.0, 5.0], [4.0, 6.0]]), T),
        "outside": kind.poi_problems(dict(good, intervals=[[1.0, 5.0], [90.0, 120.0]]), T),
        "value's piece": kind.poi_problems(dict(good, intervals=[[1.0, 5.5], [17.8, 22.2]]), T),
        "malformed": kind.poi_problems(dict(good, intervals=[[1.0]]), T),
        "other POI": kind.poi_problems({"name": "mu_on", "value": 1.0, "err_minus": 0.2, "err_plus": 0.2,
                                        "intervals": [[0.8, 1.2]]}, T),
    }
    want = {"unsorted": "sorted", "overlapping": "disjoint", "outside": "inside", "value's piece": "holding value",
            "malformed": "pairs", "other POI": "only on xi"}
    check("RESULT.json pieces: a valid set passes; each violation is refused",
          probs["valid"] == [] and all(any(want[k] in p for p in probs[k]) for k in want),
          {k: v for k, v in probs.items()})


MALFORMED = {
    "a channel without name": lambda sp, ch, smp: ch.pop("name"),
    "a sample without name": lambda sp, ch, smp: smp.pop("name"),
    "a modifier without name": lambda sp, ch, smp: smp["modifiers"][0].pop("name"),
    "a modifier without type": lambda sp, ch, smp: smp["modifiers"][0].pop("type"),
    "a null in the data": lambda sp, ch, smp: smp.update(data=[8.5, None]),
    "a string in the data": lambda sp, ch, smp: smp.update(data=[8.5, "x"]),
    "channels a list of names": lambda sp, ch, smp: sp.update(channels=["sr"]),
    "samples a string": lambda sp, ch, smp: ch.update(samples="x"),
    "a modifier named by a list": lambda sp, ch, smp: smp["modifiers"][0].update(name=["a"]),
    "a measurement's config a string": lambda sp, ch, smp: sp["measurements"][0].update(config="x"),
    "a measurement's poi a list": lambda sp, ch, smp: sp["measurements"][0]["config"].update(poi=["xi"]),
}


def check_gate():
    kind = get_kind(T)
    check("the kind: model piece scaled_model, no variant pieces",
          kind.model_piece == "scaled_model" and contract_pieces(T) == ("intervals", "scaled_model", "nuisances"))
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        check("no MODEL.json fails the gate", kind.output_gate(out, result(), T)[0] ==
              ["MODEL.json not produced (required by the contract)"])
        (out / "MODEL.json").write_text(json.dumps(spec()))
        g_ok = kind.output_gate(out, result(), T)
        (out / "MODEL.json").write_text(json.dumps(spec(entries={"s": {"factor": 1.0, "powers": {"mu_on": 1}}})))
        g_bad = kind.output_gate(out, result(), T)
        unlogged = dict(result(), nuisances={})
        (out / "MODEL.json").write_text(json.dumps(spec()))
        g_log = kind.output_gate(out, unlogged, T)
        g_shape = {}
        for name, damage in MALFORMED.items():
            sp = spec()
            damage(sp, sp["channels"][0], sp["channels"][0]["samples"][0])
            (out / "MODEL.json").write_text(json.dumps(sp))
            try:
                g_shape[name] = [kind.output_gate(out, result(), T)[0], BLM.gate(out / "MODEL.json", result(), T)[0]]
            except Exception as e:  # noqa: BLE001
                g_shape[name] = [f"raised {type(e).__name__}: {e}"]
        (out / "MODEL.json").write_text(json.dumps(spec()))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = kind.check_outputs(out, result(), T, None)
    check("a scaled MODEL.json passes the gate, the POIs exempt from the naming rules", g_ok == ([], []), g_ok)
    check("a scaled sample carrying a POI's normfactor its entry does not scale with is a contract failure",
          any(p.startswith("contract: poi_scaling off/s") for p in g_bad[0]), g_bad)
    check("an unlogged nuisance is a contract failure, the POIs not", g_log[0] and all("lumi" in p and "xi" not in p
                                                                                         for p in g_log[0]), g_log)
    check("a MODEL.json out of pyhf's shape (a channel, sample or modifier without its name or type, a null or a "
          "string in a scaled sample's data, channels that are names, samples that are a string, a modifier named by "
          "a list, a measurement whose config or poi is no object or string) is refused with a contract problem by "
          "this kind's gate and binned_likelihood's, never an exception",
          all(len(g) == 2 and all(any(p.startswith(("contract: channel", "contract: measurements")) for p in probs)
                                  for probs in g) for g in g_shape.values()), g_shape)
    lines = buf.getvalue()
    check("`sbh check` passes it and prints its refit with the POI scaling beside RESULT.json's numbers",
          rc == 0 and "refit with the POI scaling (a diagnostic): xi = " in lines and "xi's 68 % set [" in lines
          and "RESULT.json reports:" in lines and "xi = 1.2, mu_on = 1; xi's 68 % set [0.4, 3.3]" in lines, lines)


def check_event_list_messages():
    """The event list of a dataset whose data files are .npy arrays: its messages name the file and its rows."""
    from sbh.kinds.binned_likelihood import eventlist as E
    sp = {"observations": [{"name": "c", "data": [2]}], "_meta": {"binning": {"c": {"edges": [0, 10]}}}}

    def problem(files, entries):
        return E.check({"channels": {"c": {"file": files, "entry": [9, 1], "x": [1, 2]}}}, sp, entries)[0]
    npy, root = {"events.npy": 5}, {"part_0000.root": 5}
    check("the event list's messages on a .npy dataset name its file and rows, on a ROOT dataset the Events tree",
          "e.g. 'events.npy'" in problem([1, 2], npy) and "beyond its file's rows" in problem(["events.npy"] * 2, npy)
          and "0-based row index in that file's array" in problem(["events.npy"] * 2, npy)
          and "e.g. 'part_0000.root'" in problem([1, 2], root)
          and "0-based entry index in that file's Events tree" in problem(["part_0000.root"] * 2, root),
          (problem([1, 2], npy), problem(["events.npy"] * 2, npy)))


def check_refit():
    """scaling.refit, the participant-side fit with the POI scaling: against the scaled likelihood's own minimum on
    a model whose data sit at its expectation, and never the plain product of normfactors pyhf alone would fit."""
    import numpy as np
    import pyhf
    sp = spec()
    g = 2.0 * 1.5
    off = [8.5 * g ** 0.5 + 8.0 - 8.0 * g ** 0.5 + 1.0 * g, 7.6 * g ** 0.5 + 6.0 - 6.0 * g ** 0.5 + 2.0 * g]
    sp["observations"] = [{"name": "off", "data": [100 * round(v, 2) for v in off]},
                          {"name": "on", "data": [750.0, 900.0]}]
    for c in sp["channels"]:
        for smp in c["samples"]:
            smp["data"] = [100.0 * v for v in smp["data"]]
    r = scaling.refit(sp, T, pyhf)
    ws = pyhf.Workspace({k: v for k, v in sp.items() if k != "_meta"})
    plain = ws.model()
    xi_plain = float(np.asarray(pyhf.infer.mle.fit(ws.data(plain), plain))[plain.config.par_slice("xi").start])
    check("the refit with the POI scaling finds the scaled likelihood's minimum (xi 2, mu_on 1.5 on data at their "
          "expectation) and its one 68 % piece around it; a pyhf fit of the plain product lands elsewhere",
          abs(r["pois"]["xi"] - 2.0) < 0.05 and abs(r["pois"]["mu_on"] - 1.5) < 0.02 and len(r["pieces"]) == 1
          and r["pieces"][0][0] < 2.0 < r["pieces"][0][1] and r["pieces"][0][1] - r["pieces"][0][0] < 2.0
          and abs(xi_plain - 2.0) > 0.2, (r, xi_plain))
    const = json.loads(json.dumps(sp))
    const["channels"][0]["samples"][1]["data"] = [400.0, 300.0]
    const["_meta"]["poi_scaling"]["off"]["b"] = {"factor": 2.0, "powers": {}}
    rc = scaling.refit(const, T, pyhf)
    check("the refit applies a constant-factor entry (no powers): half the nominal x 2 fits as the model itself",
          all(abs(rc["pois"][p] - r["pois"][p]) < 1e-3 for p in r["pois"]), (rc, r))
    big = scaling.REFIT_MAX_FREE
    scaling.REFIT_MAX_FREE = 1
    try:
        note = scaling.refit(sp, T, pyhf)
    finally:
        scaling.REFIT_MAX_FREE = big
    check("a model beyond the refit's parameter count is not refitted, with a note", set(note) == {"note"}, note)


def check_contract():
    text = render_contract(T)
    heads = re.findall(r"^## (\d+)\. (.+)$", text, flags=re.M)
    check("the contract renders: sections 1..n, MODEL.json with the POI scaling, the pieces",
          [int(n) for n, _ in heads] == list(range(1, len(heads) + 1)) and not re.search(r"\{\{|<!-- /?if", text)
          and any("MODEL.json" in h for _, h in heads) and "poi_scaling" in text and '"intervals"' in text
          and "`MODEL.json` and `EVENTS.json`" in text, [h for _, h in heads])
    check("the contract states the refit's parameter limit as scaling.py holds it",
          f"at most {scaling.REFIT_MAX_FREE} free\nparameters" in text)
    flat = " ".join(text.split())
    types = [f"`{t}`" for t in ENGINE_MODIFIERS]
    check("the contract's gate states the modifier types the evaluator reads as model.ENGINE_MODIFIERS holds them",
          "every modifier is of a type the evaluator reads: " + ", ".join(types[:-1]) + f" or {types[-1]};" in flat)
    check("the contract states the fit conventions: the floors (scaling.RATE_MIN), the staterror width, the ranges",
          scaling.RATE_MIN == 1e-9 and "floored at 1e-9 before its multiplicative modifiers, and each bin's total "
          "expected rate at 1e-9" in flat and "√(Σ δ²) / Σ nominal" in flat
          and "a `fixed` flag or other bounds on it are not read" in flat)


def check_stub():
    import importlib.util
    from sbh.hep.nuisances import check_nuisance_log
    from sbh.task import load_task, write_participant_task_stub
    task = load_task("h4l_width")
    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp) / "h4l_width_stub"
        write_participant_task_stub(task, pkg)
        sp = importlib.util.spec_from_file_location("h4l_width_stub", pkg / "__init__.py")
        mod = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(mod)
        st = mod.TASK
    log = {"nuisances": {"cross_section_EWVV": {"value": 1.0, "unc": 0.1}}}
    check("h4l_width's participant stub carries the kind, the POIs with their ranges, the model and event-list "
          "requirements and the task's process names: its nuisance-log gate takes cross_section_EWVV, another "
          "task's refuses it",
          st.kind == "scan_likelihood" and st.pois == ["xi", "mu_on"] and st.poi_bounds == task.poi_bounds
          and st.requires_model and st.event_list_from == task.event_list_from and st.processes == ("EWVV",)
          and check_nuisance_log(log, st) == [] and check_nuisance_log(log, T) != [],
          (st.kind, st.pois, st.poi_bounds, st.processes))


def two_piece_row(x_hat=30.0, em=2.0, ep=6.0, island=(70.0, 0.5)):
    """A grader record whose scan has a main piece [x_hat - em, x_hat + ep] (an asymmetric parabola) and an island at
    island[0] with lowest q island[1]; mu_on 1.0 -0.1 +0.15 (Hessian 0.12, stat-only 0.08 / 0.1)."""
    from sbh.kinds.scan_likelihood import verify as V
    xs = [0.25 * i for i in range(401)]
    qs = [min(((x - x_hat) / (em if x < x_hat else ep)) ** 2, island[1] + ((x - island[0]) / 3.0) ** 2) for x in xs]
    pieces, iv = V.scan_pieces(xs, qs, 0.0, 100.0, x_hat)
    return {"seed": "seed_001", "status": "ok", "scan_version": V.SCAN_VERSION,
            "scan": {"points": [[x, q] for x, q in zip(xs, qs)], "x_hat": x_hat, "nll_min": 0.0, "pieces": pieces,
                     "interval": iv, "n_failed": 1, "n_sweep_disagree": 2},
            "poi_fits": {"xi": {"value": x_hat, "err_minus": x_hat - iv["lo"], "err_plus": iv["hi"] - x_hat,
                                "sigma_hess": 3.0, "stat_minus": 1.5, "stat_plus": 5.0},
                         "mu_on": {"value": 1.0, "err_minus": 0.1, "err_plus": 0.15, "sigma_hess": 0.12,
                                   "stat_minus": 0.08, "stat_plus": 0.1}},
            "quality_crosscheck": {"gof_pvalue": {"agree": True}, "coverage": {"agree": False}}}


def check_scoring():
    import math
    from sbh.kinds.binned_likelihood.scoring import nuisance_probes
    from sbh.kinds.scan_likelihood import scoring as SS
    from sbh.kinds.scan_likelihood.density import ScanForecast
    from sbh.kinds.scan_likelihood.verify import VERIFY_FILE
    from sbh.score import HALF_LN_2PI
    kind = get_kind(T)
    row = two_piece_row()
    rep = {"pois": [{"name": "xi", "value": 30.0, "err_minus": 2.0, "err_plus": 6.0, "intervals": [[28.0, 36.0],
                                                                                                    [67.9, 72.1]]},
                    {"name": "mu_on", "value": 1.0, "err_minus": 0.1, "err_plus": 0.15}], "nuisances": {}}
    recs = [{"seed": s, "exit_code": 0, "timed_out": False, "result_errors": [], "result": rep}
            for s in ("seed_001", "seed_002", "seed_003")]
    truths = {s: {"xi": 31.0, "mu_on": 1.05, "profile": "p0"} for s in ("seed_001", "seed_002", "seed_003")}
    from sbh.kinds.scan_likelihood.verify import SCAN_VERSION
    strict_task = type("ScanTaskStrict", (ScanTask,), {"grader_strict": True})()
    with tempfile.TemporaryDirectory() as d:
        for s, r in (("seed_001", row), ("seed_002", {"status": "fit failed: no convergence",
                                                      "scan_version": SCAN_VERSION})):
            (Path(d) / "runs" / s).mkdir(parents=True)
            (Path(d) / "runs" / s / VERIFY_FILE).write_text(json.dumps(r))
        inp = kind.score_inputs(Path(d), recs, truths, T)
        inp_strict = kind.score_inputs(Path(d), recs, truths, strict_task)
        (Path(d) / "runs" / "seed_002" / VERIFY_FILE).write_text(json.dumps(
            {"status": "not graded: the grader's queue", "scan_version": SCAN_VERSION, "errors_marked": 1}))
        inp_queue = kind.score_inputs(Path(d), recs, truths, strict_task)
        (Path(d) / "runs" / "seed_002" / VERIFY_FILE).write_text(json.dumps({"status": "ok", "scan_version": "scan-v0"}))
        try:
            kind.score_inputs(Path(d), recs, truths, T)
            stale = None
        except SystemExit as e:
            stale = str(e)
    check("a record graded under another scan rule stops the score, naming the file and the post stage",
          stale is not None and "seed_002" in stale and "scan-v0" in stale and "--phase post" in stale, stale)
    by_strict = {r["seed"]: r["result"]["pois"] for r in inp_strict["records"]}
    check("Task.grader_strict: a dataset the grader left pending is failed, never scored on its reported entries; a "
          "completed dataset with no grader record (its grading lost) stays pending with the reason \"no grader "
          "record\", never the reported value, never failed (strict fails only what the contract fails)",
          not {r["seed"]: r["result"]["pois"] for r in inp_queue["records"]}["seed_002"]
          and inp_queue["info"]["seed_002"]["xi"]["status"] == "failed"
          and inp_queue["info"]["seed_002"]["xi"]["reason"].startswith("strict: ")
          and by_strict["seed_003"] == [] and all(m["status"] == "pending" and m["reason"] == "no grader record"
                                                  for m in inp_strict["info"]["seed_003"].values()),
          (inp_strict["info"]["seed_003"], inp_queue["info"]["seed_002"]))
    by = {r["seed"]: {p["name"]: p for p in r["result"]["pois"]} for r in inp["records"]}
    xi = by["seed_001"]["xi"]
    check("the scanned POI's info keeps the scan's failed points and disagreeing sweeps",
          (inp["info"]["seed_001"]["xi"]["n_failed"], inp["info"]["seed_001"]["xi"]["n_sweep_disagree"]) == (1, 2),
          inp["info"]["seed_001"]["xi"])
    check("score_inputs: ok -> the grader's entries (the scanned POI's density and both pieces); failed -> none; "
          "pending -> the reported ones",
          isinstance(xi.get("forecast"), ScanForecast) and xi["forecast"]["status"] == "ok"
          and abs(xi["err_minus"] - 2.0) < 0.01 and abs(xi["err_plus"] - 6.0) < 0.01 and len(xi["intervals"]) == 2
          and by["seed_001"]["mu_on"]["err_plus"] == 0.15 and by["seed_002"] == {}
          and "forecast" not in by["seed_003"]["xi"]
          and by["seed_003"]["xi"]["intervals"] == rep["pois"][0]["intervals"]
          and [inp["info"][s]["xi"]["status"] for s in ("seed_001", "seed_002", "seed_003")] == ["ok", "failed",
                                                                                               "pending"],
          {s: {p: v.get("status") for p, v in m.items()} for s, m in inp["info"].items()})
    ref = kind.reference_poi({"value": 99.0, "err_minus": 1.0, "err_plus": 1.0}, xi)
    ref_mu = kind.reference_poi({"value": 9.0, "err_minus": 1.0, "err_plus": 1.0}, by["seed_001"]["mu_on"])
    check("the closure's baseline is the grader's entry: the scanned POI's interval and its density's H, the other's "
          "interval", abs(ref["value"] - 30.0) < 1e-9 and ref.get("entropy") is not None and ref_mu["value"] == 1.0,
          (ref, ref_mu))
    fb = T.fallback_forecast = {"xi": (50.0, 28.868), "mu_on": (2.5, 1.44)}
    c_ok = kind.case_loss(xi, "xi", 31.0, T, fb["xi"], ref, 0.0)
    c_pend = kind.case_loss(by["seed_003"]["xi"], "xi", 31.0, T, fb["xi"], ref, 0.0)
    c_noref = kind.case_loss(xi, "xi", 31.0, T, fb["xi"], {"value": 30.0, "sigma": 4.0}, 0.0)
    c_fail = kind.case_loss(None, "xi", 31.0, T, fb["xi"], ref, 0.0)
    c_mu = kind.case_loss(by["seed_001"]["mu_on"], "mu_on", 1.3, T, fb["mu_on"], ref_mu, 0.0)
    rl = ref["entropy"] - HALF_LN_2PI
    check("case losses: the density against the reference's H (its mean -ln p); pending, the reported main piece's "
          "two-piece loss against it; no H -> excluded; no answer -> failed; mu_on two-piece, its pull on the "
          "truth's side",
          c_ok["good"] and math.isfinite(c_ok["loss"]) and abs(c_ok["ref_loss"] - rl) < 1e-12
          and c_pend["good"] and abs(c_pend["ref_loss"] - rl) < 1e-12 and c_pend["loss"] != c_ok["loss"]
          and c_noref["ref_loss"] is None and not c_fail["good"] and abs(c_mu["mw_pull"] + 2.0) < 1e-9,
          (c_ok, c_pend["ref_loss"], c_noref["ref_loss"], c_mu["mw_pull"]))
    scan = row["scan"]
    a = (30.0, 2.0, 6.0)

    import numpy as np

    def weight(a0, a1, q):
        x = np.linspace(a0, a1, 200001)
        return float(np.trapezoid(np.exp(-0.5 * q(x)), x))
    w_main = weight(28.0, 36.0, lambda x: ((x - 30.0) / np.where(x < 30.0, 2.0, 6.0)) ** 2)
    w_isl = {d: weight(70.0 - 3.0 * math.sqrt(1.0 - d), 70.0 + 3.0 * math.sqrt(1.0 - d),
                       lambda x, d=d: d + ((x - 70.0) / 3.0) ** 2) for d in (0.5, 0.0)}
    isl = [70.0 - 3.0 * math.sqrt(0.5), 70.0 + 3.0 * math.sqrt(0.5)]
    accurate = SS.scan_truth(a, [[28.0, 36.0], isl], scan, 0.0, 100.0)
    cut = SS.scan_truth(a, [[28.0, 31.0], [31.0, 36.0], isl], scan, 0.0, 100.0)
    one = SS.scan_truth(a, [[28.0, 36.0]], scan, 0.0, 100.0)
    deep = SS.scan_truth(a, [[28.0, 36.0]], two_piece_row(island=(70.0, 0.0))["scan"], 0.0, 100.0)
    wide = SS.scan_truth(a, [[20.0, 50.0], [60.0, 80.0]], scan, 0.0, 100.0)
    off = SS.scan_truth((36.0, 8.0, 0.0001), [[28.0, 36.0001], isl], scan, 0.0, 100.0)
    want = {k: w_main / (w_main + w_isl[k]) for k in w_isl}
    check("Delta_truth, the scanned POI: the reported set covering every model piece costs 0, however it is cut; a set "
          "too wide costs 0; a missed island costs ln(0.9 / c), c the covered share of the model set's likelihood "
          "weight (more for a deeper island); a value off the minimum costs sqrt(q(value)) - tau",
          accurate["T"] == 0.0 and abs(cut["T"] - accurate["T"]) < 1e-12
          and abs(cut["coverage"] - accurate["coverage"]) < 1e-12
          and wide["T"] == 0.0 and abs(one["coverage"] - want[0.5]) < 2e-3 and abs(deep["coverage"] - want[0.0]) < 2e-3
          and abs(one["c_set"] - max(0.0, math.log(0.9 / one["coverage"]))) < 1e-12 and deep["c_set"] > one["c_set"]
          and abs(off["c_val"] - (1.0 - max(0.25, 0.25 * 6 + 0.05) / 6.0)) < 1e-3,
          (accurate["T"], cut["coverage"], one["coverage"], want, deep["c_set"], one["c_set"], off["c_val"]))
    sym = two_piece_row(em=2.0, ep=2.0, island=(70.0, 9.0))["scan"]
    for s_r, v_r in ((1.0, 30.0), (1.5, 30.0), (2.0, 30.75)):
        f = SS.scan_truth((v_r, s_r, s_r), [[v_r - s_r, v_r + s_r]], sym, 0.0, 100.0)
        h4l_val = max(0.0, abs(v_r - 30.0) / 2.0 - (0.25 + 0.05 / 2.0))
        c = math.erf(s_r / (2.0 * math.sqrt(2.0))) / math.erf(1.0 / math.sqrt(2.0)) if v_r == 30.0 else None
        ok = abs(f["c_val"] - h4l_val) < 1e-9 and (c is None or abs(f["coverage"] - c) < 2e-3)
        check(f"Delta_truth on a parabolic single piece: h4l's value term, the Gaussian coverage (value {v_r}, "
              f"error {s_r})", ok, f)
    info = inp["info"]["seed_001"]["mu_on"]
    i_ok = SS.interval_truth((1.0, 0.1, 0.15), info)
    i_shift = SS.interval_truth((1.2, 0.1, 0.15), info)
    i_narrow = SS.interval_truth((1.0, 0.05, 0.15), info)
    check("Delta_truth, an interval POI: accurate 0; shifted by 0.2 above costs 0.2 / 0.15 - tau; a lower side half the "
          "model's costs ln(0.9 x 0.1 / 0.05) / 2",
          i_ok["T"] == 0.0 and abs(i_shift["c_val"] - (0.2 / 0.15 - (0.25 + 0.05 / 0.15))) < 1e-9
          and abs(i_narrow["c_sig"] - math.log(0.9 * 0.1 / 0.05) / 2) < 1e-9, (i_ok, i_shift, i_narrow))
    rep2 = json.loads(json.dumps(rep))
    rep2["pois"][1].update(stat={"err_minus": 0.08, "err_plus": 0.1}, syst={"err_minus": 0.2, "err_plus": 0.11})
    cl = SS.claims_rows([dict(recs[0], result=rep2)], {"seed_001": row}, inp["info"], truths, T)
    check("Delta_claims: gof_pvalue / coverage as the grader's cross-checks say; mu_on's stat agrees, its syst does "
          "not",
          cl and cl[0]["checks"] == {"gof_pvalue": False, "coverage": True, "mu_on_stat": False, "mu_on_syst": True},
          cl)
    nf = [{"name": "mu_on", "type": "normfactor", "data": None}, {"name": "xi", "type": "normfactor", "data": None}]
    th = {"name": "ana_xs", "type": "normsys", "data": {"hi": 1.1, "lo": 0.9}}
    sp = {"channels": [{"name": "c", "samples": [{"name": "a", "data": [5.0, 5.0], "modifiers": nf + [th]},
                                                {"name": "aneg", "data": [5.0, 5.0], "modifiers": nf + [th]},
                                                {"name": "b", "data": [10.0, 10.0], "modifiers": []}]}],
          "measurements": [{"name": "m", "config": {"poi": "xi", "parameters": []}}]}
    scale = {("c", "a"): 1.0, ("c", "aneg"): -1.0}
    sz_net, sz_sum = {}, {}
    p_net = nuisance_probes(sp, {"ana_xs"}, "ana", {"norm": 3e-4}, sz_net, scale, ("xi", "mu_on"), True)
    nuisance_probes(sp, {"ana_xs"}, "ana", {"norm": 3e-4}, sz_sum, scale, ("xi", "mu_on"))
    check("the systematics probes' net size: a change on a template and its negative copy cancel (0, below the floor); "
          "binned_likelihood's sum of |change| counts both",
          sz_net["ana_xs"]["norm"] == 0.0 and p_net["ana_xs"] == ["below_floor"]
          and abs(sz_sum["ana_xs"]["norm"] - 0.1) < 1e-12, (sz_net, sz_sum))


def check_scan_rules():
    """The grader's scan rules on an exact one-parameter engine, and the closure's gate: the best fit, its piece and
    its errors from one minimum; a piece that reaches a scan end off the range bound; no run scored before the
    grader measured the reference everywhere."""
    import numpy as np
    from sbh.kinds.scan_likelihood import scoring as SS
    from sbh.kinds.scan_likelihood import verify as V

    class Engine:
        """One parameter on [0, 100] with a minimum at 60 (nll 0) and one at ``b`` (nll ``depth``); a free fit lands
        on the minimum nearer its start."""
        bounds, fixed = [(0.0, 100.0)], np.array([False])

        def __init__(self, b, depth):
            self.b, self.depth = b, depth

        def fit(self, main, aux, init=None, fixed=None, want_sigma=False, starts=()):
            x = float(init[0])
            if fixed is None:
                x = 60.0 if abs(x - 60.0) < abs(x - self.b) else self.b
            return {"valid": True, "x": np.array([x]),
                    "nll": min((x - 60.0) ** 2 / 8.0, (x - self.b) ** 2 / 8.0 + self.depth)}
    at60 = {"x": np.array([60.0]), "nll": 0.0}
    near = V.profile_scan(Engine(25.0, -5e-4), None, None, at60, 0)
    deep = V.profile_scan(Engine(25.3, -0.3), None, None, at60, 0)
    for name, sc, x_hat, restarts in (("a second minimum 5e-4 below the global fit's, less than a restart takes", near,
                                       25.0, 0), ("a second minimum 0.3 below, between scan points", deep, 25.3, 1)):
        q = dict(map(tuple, sc["points"]))
        check(f"the scan, {name}: the best fit is the scan's lowest point, q = 0 there and >= 0 everywhere, its "
              "piece holds it, the other minimum is a second piece",
              sc["x_hat"] == x_hat and q[x_hat] == 0.0 and min(q.values()) == 0.0 and sc["n_restarts"] == restarts
              and sc["interval"]["lo"] < x_hat < sc["interval"]["hi"] and abs(sc["interval"]["lo"] - (x_hat - 2.0)) < 0.02
              and abs(sc["interval"]["hi"] - (x_hat + 2.0)) < 0.02 and len(sc["pieces"]) == 2
              and sc["pieces"][1]["lo"] < 60.0 < sc["pieces"][1]["hi"],
              {k: sc[k] for k in ("x_hat", "interval", "n_restarts")} | {"pieces": len(sc["pieces"])})
    xs = [0.5 + 0.25 * i for i in range(399)]
    qs = [min(0.3 + 0.2 * x * x, ((x - 30.0) / 3.0) ** 2) for x in xs]
    pieces, iv = V.scan_pieces(xs, qs, 0.0, 100.0, 30.0)
    scan = {"points": [[x, q] for x, q in zip(xs, qs)], "x_hat": 30.0, "pieces": pieces, "interval": iv}
    part = SS.scan_truth((30.0, 3.0, 3.0), [[27.0, 33.0]], scan, 0.0, 100.0)
    both = SS.scan_truth((30.0, 3.0, 3.0), [[0.0, 1.8], [27.0, 33.0]], scan, 0.0, 100.0)
    open_end = dict(scan, pieces=[dict(pieces[0], lo=None), pieces[1]])
    check("a piece whose q <= 1 run reaches the scan's first point off the range bound starts at that point, and "
          "the truth term reads it: a report covering it costs nothing, one leaving it out its weight; a piece "
          "with an open end is read the same",
          pieces[0]["lo"] == 0.5 and abs(pieces[0]["hi"] - 1.8667) < 1e-3 and iv == pieces[1] and both["T"] == 0.0
          and 0.0 < part["c_set"] < 0.2 and SS.scan_truth((30.0, 3.0, 3.0), [[27.0, 33.0]], open_end, 0.0, 100.0) == part,
          (pieces[0], both["T"], part["c_set"]))
    with tempfile.TemporaryDirectory() as d:
        suite = Path(d)
        got = {"none": SS.closure_gate(suite, T, "agent_x")}
        for name, sb in (("grader", {"source": "grader", "scan_version": V.SCAN_VERSION}),
                         ("pending", {"source": "pending", "scan_version": V.SCAN_VERSION, "missing": ["seed_002"]}),
                         ("stale", {"source": "grader", "scan_version": "scan-v0"})):
            (suite / "closure.json").write_text(json.dumps({"scan_baseline": sb}))
            got[name] = SS.closure_gate(suite, T, "agent_x")
        got["reference"] = SS.closure_gate(suite, T, "reference")
        from sbh.suite import closure_gate
        via_core = closure_gate(suite, T, "agent_x")
    check("the closure's gate: a run is scored only on a suite whose closure holds the grader's density of the "
          "reference on every dataset under this scan rule; the closure's own run always; the core's gate asks the "
          "kind",
          got["grader"] is None and got["reference"] is None and "no closure.json" in got["none"]
          and "seed_002" in got["pending"] and "scan-v0" in got["stale"] and via_core == got["stale"], got)


def bimodal_spec():
    """Off-shell bins whose rate is a parabola in t = sqrt(g), g = mu_on xi, B + (SBI - B - S) t + S t^2, observed where
    two values of t fit (xi near 1 and near 25 at mu_on near 1); the on-shell bins fix mu_on."""
    nf = [{"name": "mu_on", "type": "normfactor", "data": None}, {"name": "xi", "type": "normfactor", "data": None}]
    lumi = {"name": "lumi", "type": "normsys", "data": {"hi": 1.02, "lo": 0.98}}
    half = {"mu_on": 0.5, "xi": 0.5}
    return {"channels": [
        {"name": "off", "samples": [
            {"name": "sbi", "data": [5.0, 4.2], "modifiers": nf + [lumi]},
            {"name": "b", "data": [10.0, 8.0], "modifiers": [lumi]},
            {"name": "bneg", "data": [10.0, 8.0], "modifiers": nf + [lumi]},
            {"name": "s", "data": [1.0, 0.8], "modifiers": nf + [lumi]},
            {"name": "sneg", "data": [1.0, 0.8], "modifiers": nf + [lumi]}]},
        {"name": "on", "samples": [{"name": "h", "data": [20.0, 30.0], "modifiers": nf[:1] + [lumi]},
                                   {"name": "zz", "data": [5.0, 5.0], "modifiers": [lumi]}]}],
        "observations": [{"name": "off", "data": [5, 5]}, {"name": "on", "data": [25, 35]}],
        "measurements": [{"name": "m", "config": {"poi": "xi", "parameters": []}}], "version": "1.0.0",
        "_meta": {"binning": {"off": {"observable": "m4l", "edges": [220.0, 400.0, 1000.0], "data_derived": False},
                              "on": {"observable": "m4l", "edges": [118.0, 125.0, 132.0], "data_derived": False}},
                  "poi_scaling": {"off": {"sbi": {"factor": 1.0, "powers": half},
                                          "bneg": {"factor": -1.0, "powers": half},
                                          "s": {"factor": 1.0, "powers": {"mu_on": 1, "xi": 1}},
                                          "sneg": {"factor": -1.0, "powers": half}}}}}


def conventions_spec():
    """A scaled model with every convention the contract states: a staterror shared by scaled (either sign) and
    unscaled samples, normsys, a histosys that takes a sample below 0, a POI the measurement fixes inside its own
    bounds."""
    nf = [{"name": "mu_on", "type": "normfactor", "data": None}, {"name": "xi", "type": "normfactor", "data": None}]
    lumi = {"name": "lumi", "type": "normsys", "data": {"hi": 1.02, "lo": 0.97}}
    bnorm = {"name": "ana_bnorm", "type": "normsys", "data": {"hi": 1.5, "lo": 0.5}}
    half = {"mu_on": 0.5, "xi": 0.5}

    def stat(d):
        return {"name": "mcstat_off", "type": "staterror", "data": d}
    return {"channels": [
        {"name": "off", "samples": [
            {"name": "sbi", "data": [5.0, 4.2], "modifiers": nf + [lumi, stat([0.5, 0.4])]},
            {"name": "b", "data": [10.0, 8.0], "modifiers": [lumi, bnorm, stat([1.0, 0.6])]},
            {"name": "bneg", "data": [10.0, 8.0], "modifiers": nf + [lumi, stat([1.0, 0.6])]},
            {"name": "s", "data": [1.0, 0.8], "modifiers": nf + [lumi]},
            {"name": "red", "data": [0.5, 0.5], "modifiers": [
                {"name": "ana_shape", "type": "histosys", "data": {"hi_data": [1.5, 0.6], "lo_data": [-0.5, 0.4]}}]}]},
        {"name": "on", "samples": [{"name": "h", "data": [20.0, 30.0], "modifiers": nf[:1] + [lumi]},
                                   {"name": "zz", "data": [5.0, 5.0], "modifiers": [lumi]}]}],
        "observations": [{"name": "off", "data": [5, 5]}, {"name": "on", "data": [25, 35]}],
        "measurements": [{"name": "m", "config": {"poi": "xi", "parameters": [
            {"name": "xi", "fixed": True, "bounds": [[2.0, 10.0]], "inits": [7.0]}]}}], "version": "1.0.0",
        "_meta": {"binning": {"off": {"observable": "m4l", "edges": [220.0, 400.0, 1000.0], "data_derived": False},
                              "on": {"observable": "m4l", "edges": [118.0, 125.0, 132.0], "data_derived": False}},
                  "poi_scaling": {"off": {"sbi": {"factor": 1.0, "powers": half},
                                          "bneg": {"factor": -1.0, "powers": half},
                                          "s": {"factor": 1.0, "powers": {"mu_on": 1, "xi": 1}}}}}}


def contract_nll(sp, par, gam):
    """-ln L of conventions_spec() as the contract's fit conventions give it, in plain numpy, at nuisances with
    |value| >= 1 (where normsys code 4 is hi^a or lo^-a and histosys code 4p is linear): ``par`` the POIs and the
    constrained nuisances, ``gam`` the staterror factors of the off channel's bins."""
    import numpy as np
    scal = sp["_meta"]["poi_scaling"]
    nll = 0.0
    for ch, obs in zip(sp["channels"], sp["observations"]):
        nu = np.zeros(len(obs["data"]))
        for smp in ch["samples"]:
            base, fac = np.array(smp["data"], float), np.ones(len(obs["data"]))
            ent = scal.get(ch["name"], {}).get(smp["name"])
            for m in smp["modifiers"]:
                a = par.get(m["name"], 0.0)
                if m["type"] == "histosys":
                    nom = np.array(smp["data"], float)
                    base = base + a * ((np.array(m["data"]["hi_data"]) - nom) if a >= 0
                                       else (nom - np.array(m["data"]["lo_data"])))
                elif m["type"] == "normsys":
                    fac = fac * (m["data"]["hi"] ** a if a >= 0 else m["data"]["lo"] ** -a)
                elif m["type"] == "staterror":
                    fac = fac * gam
                elif ent is None:
                    fac = fac * par[m["name"]]
            if ent is not None:
                fac = fac * ent["factor"] * np.prod([par[k] ** pw for k, pw in ent["powers"].items()])
            nu += np.maximum(base, scaling.RATE_MIN) * fac
        nu = np.maximum(nu, scaling.RATE_MIN)
        nll += float(np.sum(nu - np.array(obs["data"]) * np.log(nu)))
    carriers = [s for s in sp["channels"][0]["samples"] if any(m["type"] == "staterror" for m in s["modifiers"])]
    delta2 = sum(np.array(next(m["data"] for m in s["modifiers"] if m["type"] == "staterror")) ** 2 for s in carriers)
    width = np.sqrt(delta2) / sum(np.array(s["data"]) for s in carriers)
    return nll + 0.5 * float(np.sum(((gam - 1.0) / width) ** 2)) + 0.5 * sum(par.get(k, 0.0) ** 2
                                                                          for k in ("lumi", "ana_bnorm", "ana_shape"))


def check_conventions():
    import numpy as np
    from sbh.kinds.scan_likelihood import verify as V
    sp = conventions_spec()
    model, data, built, meta, idx = V.build_model(sp, T)
    ev = V.twin(built, model, meta["poi_scaling"])
    main, aux = data[:ev.n_main], data[ev.n_main:]
    check("the contract's ranges: a POI the measurement fixes inside its own bounds floats in the task's range from "
          "its declared init", ev.bounds[idx["xi"]] == (0.0, 100.0) and not ev.fixed[idx["xi"]]
          and ev.init[idx["xi"]] == 7.0, (ev.bounds[idx["xi"]], ev.fixed[idx["xi"]], ev.init[idx["xi"]]))
    g = [i for i, n in enumerate(ev.par_names) if n.startswith("mcstat_off")]
    worst, floors = 0.0, []
    for par, gam in (({"xi": 4.0, "mu_on": 1.3, "lumi": 1.5, "ana_shape": 1.0}, [1.1, 0.9]),
                     ({"xi": 0.4, "mu_on": 0.7, "lumi": -1.2, "ana_shape": -1.5}, [0.8, 1.2]),
                     ({"xi": 0.0, "mu_on": 1.0, "lumi": 0.0, "ana_shape": -2.0}, [1.0, 1.05]),
                     ({"xi": 6.25, "mu_on": 1.0, "lumi": 0.0, "ana_bnorm": -3.0, "ana_shape": 0.0}, [1.0, 1.0])):
        x = ev.init.copy()
        for k, v in par.items():
            x[ev.par_names.index(k)] = v
        x[g] = gam
        want = contract_nll(sp, par, np.array(gam))
        worst = max(worst, abs(float(ev.nll(x, main, aux)) - want) / max(abs(want), 1.0))
        floors.append(float(np.min(ev.expected_main(x))))
    check("the contract's fit conventions give the grader's -ln L (numpy against the engine: scaled samples of either "
          "sign under a shared staterror with its declared width, normsys, a histosys taking a sample below 0, xi = 0, "
          "a bin total at the floor)", worst <= 1e-12 and min(floors) == scaling.RATE_MIN,
          f"max rel {worst:.1e}, lowest rates {floors}")


def check_grader():
    import numpy as np
    from sbh.kinds.binned_likelihood import verify_fit
    from sbh.kinds.scan_likelihood import verify as V
    sp = bimodal_spec()
    model, data, built, meta, idx = V.build_model(sp, T)
    check("the grader's model: the coverage POI is the measurement's, the task's ranges bound the POIs",
          model.config.poi_name == "mu_on" and model.config.suggested_bounds()[idx["xi"]] == [0.0, 100.0]
          and model.config.suggested_bounds()[idx["mu_on"]] == [0.0, 5.0], model.config.poi_name)
    ev = V.twin(built, model, meta["poi_scaling"])
    main, aux = data[:ev.n_main], data[ev.n_main:]
    fit = V.global_fit(ev, main, aux, idx, "xi")
    scan = V.profile_scan(ev, main, aux, fit, idx["xi"])
    xs = np.array([x for x, _ in scan["points"]])
    q = np.array([v for _, v in scan["points"]])
    fx = ev.fixed.copy()
    fx[idx["xi"]] = True
    nll_bf = []
    for x in xs:
        best = np.inf
        for mu in (0.5, 0.8, 1.0, 1.3, 2.0):
            for a in (-1.0, 0.0, 1.0):
                x0 = ev.init.copy()
                x0[idx["xi"]], x0[idx["mu_on"]], x0[ev.par_names.index("lumi")] = x, mu, a
                r = ev.fit(main, aux, init=x0, fixed=fx, want_sigma=False)
                best = min(best, r["nll"]) if r["valid"] else best
        nll_bf.append(best)
    nll_bf = np.array(nll_bf)
    q_bf = 2.0 * (nll_bf - min(nll_bf.min(), scan["nll_min"]))
    pieces_bf = SC.pieces_of(list(xs), list(q_bf), 0.0, 100.0)
    ends = [(p["lo"], p["hi"]) for p in scan["pieces"]]
    check("the scan matches a brute-force grid of multi-start fits at every point (|dq| <= 0.02) and finds both pieces",
          float(np.max(np.abs(q - q_bf))) <= 0.02 and len(scan["pieces"]) == 2 == len(pieces_bf)
          and all(abs(a - p["lo"]) < 0.05 and abs(b - p["hi"]) < 0.05 for (a, b), p in zip(ends, pieces_bf))
          and ends[0][0] < 1.0 < ends[0][1] < 10.0 < ends[1][0] < 25.0 < ends[1][1],
          f"max |dq| {np.max(np.abs(q - q_bf)):.3g}, pieces {ends} vs {[(p['lo'], p['hi']) for p in pieces_bf]}")
    print(f"        scan: {scan['n_points']} points, pieces {[(round(a, 2), round(b, 2)) for a, b in ends]}, "
          f"q_min {[round(p['q_min'], 3) for p in scan['pieces']]}, failed {scan['n_failed']}, sweeps disagreeing "
          f"{scan['n_sweep_disagree']}, restarts {scan['n_restarts']}")
    with tempfile.TemporaryDirectory() as d:
        run = Path(d) / "run"
        seed = run / "runs" / "seed_001"
        (seed / "out").mkdir(parents=True)
        res = result(xi={"value": 25.0, "err_minus": 4.0, "err_plus": 4.0})
        res["quality"] = {"gof_pvalue": {"value": 0.5, "note": "toys"}, "coverage": {"value": 0.68, "note": "toys"}}
        (seed / "out" / "MODEL.json").write_text(json.dumps(sp))
        (seed / "out" / "RESULT.json").write_text(json.dumps(res))
        (seed / "record.json").write_text(json.dumps({"exit_code": 0, "result": res}))
        suite = Path(d) / "suite"
        suite.mkdir()
        (suite / "manifest.json").write_text(json.dumps({"seeds": [{"seed": "seed_001",
                                                                     "truth": {"xi": 1.0, "mu_on": 1.0}}]}))
        import os
        os.environ["SBH_VERIFY_TOYS"] = "20"
        affinity = getattr(os, "sched_getaffinity", lambda _pid: None)
        cpus = affinity(0)
        get_kind(T).post_dataset(run, suite, T, "seed_001")
        cpus_after = affinity(0)
        rec = json.loads((seed / V.VERIFY_FILE).read_text())
        bad = dict(sp, channels=[dict(sp["channels"][0], samples=sp["channels"][0]["samples"] + [
            {"name": "sf", "data": [1.0, 1.0], "modifiers": [{"name": "kf", "type": "shapefactor", "data": None}]}]),
            sp["channels"][1]])
        truth = {"xi": 1.0, "mu_on": 1.0}

        def graded(model_spec):
            (seed / "out" / "MODEL.json").write_text(json.dumps(model_spec))
            return V.verify_dataset(seed, T, truth, 0)
        refused = graded(bad)
        kappa = json.loads(json.dumps(sp))
        kappa["channels"][1]["samples"][1]["modifiers"][0]["data"] = {"hi": 1.02, "lo": 0.0}
        spec_refused = graded(kappa)
        nf = [{"name": "mu_on", "type": "normfactor", "data": None}, {"name": "xi", "type": "normfactor", "data": None}]
        giant = json.loads(json.dumps(sp))
        giant["channels"][0]["samples"] += [{"name": "bigp", "data": [1e13, 1e13], "modifiers": nf},
                                            {"name": "bign", "data": [1e13, 1e13], "modifiers": nf}]
        half = {"mu_on": 0.5, "xi": 0.5}
        giant["_meta"]["poi_scaling"]["off"].update(bigp={"factor": 1.0, "powers": half},
                                                    bign={"factor": -1.0, "powers": half})
        cancelling = graded(giant)
        (seed / "out" / "MODEL.json").write_text(json.dumps(sp))
        eng = verify_fit._evt()
        evt, verify_fit._evt = verify_fit._evt, lambda: (_ for _ in ()).throw(ImportError("no jax here"))
        try:
            nojax = V.verify_dataset(seed, T, truth, 0)
        finally:
            verify_fit._evt = evt
        mismatch, eng.pyhf_mismatch = eng.pyhf_mismatch, lambda *a, **k: (_ for _ in ()).throw(ValueError("ours"))
        try:
            ours = V.verify_dataset(seed, T, truth, 0)
        finally:
            eng.pyhf_mismatch = mismatch
        from sbh.kinds.binned_likelihood import verify as MV
        cover, MV.freq_coverage = MV.freq_coverage, lambda *a, **k: (_ for _ in ()).throw(MV._TwinFitError("pinned"))
        try:
            unpinned = V.verify_dataset(seed, T, truth, 20)
        finally:
            MV.freq_coverage = cover
        side = verify_fit._profile_side
        verify_fit._profile_side = lambda cond, nll, v, bound, sgn, s0: (
            (None, "failed", 1) if sgn < 0 else side(cond, nll, v, bound, sgn, s0))
        try:
            one_sided = V.verify_dataset(seed, T, truth, 0)
        finally:
            verify_fit._profile_side = side
        from sbh.kinds.binned_likelihood.scoring import GOF_CHARGE_MAX, gof_cases
        ok_rec = [{"seed": "seed_001", "exit_code": 0, "timed_out": False, "result_errors": [], "result": res}]
        (Path(d) / "one_sided" / "runs" / "seed_001").mkdir(parents=True)
        (Path(d) / "one_sided" / "runs" / "seed_001" / V.VERIFY_FILE).write_text(json.dumps(one_sided))
        inp_one = get_kind(T).score_inputs(Path(d) / "one_sided", ok_rec, {"seed_001": {**truth, "profile": "p0"}}, T)
        gof_unpinned = gof_cases(run, ok_rec, combined=True, rows={"seed_001": unpinned})
        capped = type("ScanTaskCapped", (ScanTask,), {"model_verify_cap": True})()
        from sbh.kinds.binned_likelihood.post import CAP_KEY
        ref_run = Path(d) / "reference"
        for r_ in (ref_run, Path(d) / "agent_x"):
            (r_ / "runs" / "seed_001" / "out").mkdir(parents=True)
            for f in ("record.json", "out/MODEL.json", "out/RESULT.json"):
                (r_ / "runs" / "seed_001" / f).write_text((seed / f).read_text())
        get_kind(capped).post_dataset(ref_run, suite, capped, "seed_001")
        ref_rec = json.loads((ref_run / "runs" / "seed_001" / V.VERIFY_FILE).read_text())
        cap_blk = get_kind(capped).closure_fields(ref_run, suite, capped)[CAP_KEY]
        (suite / "closure.json").write_text(json.dumps({CAP_KEY: dict(cap_blk, cap_s=0.05)}))
        get_kind(capped).post_dataset(Path(d) / "agent_x", suite, capped, "seed_001")
        out_rec = json.loads((Path(d) / "agent_x" / "runs" / "seed_001" / V.VERIFY_FILE).read_text())
        uncapped_fields = get_kind(T).closure_fields(ref_run, suite, T)
    check("the post stage leaves the caller's CPU affinity as it found it (the grader's one-CPU pin is its own)",
          cpus_after == cpus, (cpus, cpus_after))
    check("a truth-pinned fit that does not converge is the model's failed GOF: the record says gof_failed and the "
          "GOF case takes the maximal charge, its measurement kept",
          unpinned["status"] == "ok" and unpinned.get("gof_failed") is True and "poi_fits" in unpinned
          and gof_unpinned and gof_unpinned[0]["charged"] and gof_unpinned[0]["delta_raw"] == GOF_CHARGE_MAX,
          (unpinned.get("status"), unpinned.get("gof_failed"), gof_unpinned))
    fits_one = one_sided.get("poi_fits") or {}
    check("a POI whose profile side gives no result: the dataset's record is ok with that side empty, its report-only "
          "consistency fails, and the scorer fails that POI and keeps the scanned one",
          one_sided["status"] == "ok" and fits_one["mu_on"]["err_minus"] is None
          and fits_one["mu_on"]["status_lo"] == "failed" and fits_one["mu_on"]["err_plus"] > 0
          and one_sided["consistency"] == {"pass": False}
          and inp_one["info"]["seed_001"]["mu_on"]["status"] == "failed"
          and inp_one["info"]["seed_001"]["xi"]["status"] == "ok"
          and [p["name"] for p in inp_one["records"][0]["result"]["pois"]] == ["xi"],
          (one_sided.get("status"), fits_one.get("mu_on"), one_sided.get("consistency")))
    check("the cap of a capped task: its reference run is graded uncapped and its slowest dataset gives closure.json's "
          "cap (twice its wall); a dataset of another run not graded within the cap is failed, with ran_out; a task "
          "without the cap records none",
          ref_rec["status"] == "ok" and ref_rec["verify_cap"]["cap_s"] is None
          and "closure's reference run" in ref_rec["verify_cap"]["why"]
          and abs(cap_blk["cap_s"] - 2.0 * ref_rec["verify_wall_s"]) <= 0.1 and cap_blk["toys"] == 20
          and out_rec["status"].startswith("fit failed: the model verification's cap of 0.05 s") and out_rec["ran_out"]
          and out_rec["verify_cap"]["cap_s"] == 0.05 and uncapped_fields == {} and "verify_cap" not in rec,
          (ref_rec.get("verify_cap"), cap_blk, out_rec.get("status")))
    import pyhf
    refit = scaling.refit(sp, T, pyhf)
    check("`sbh check`'s refit with the POI scaling finds the grader's two pieces (ends within 0.1) and its best fit",
          len(refit["pieces"]) == 2 and all(abs(a - lo_) < 0.1 and abs(b - hi_) < 0.1
                                            for (a, b), (lo_, hi_) in zip(refit["pieces"], ends))
          and abs(refit["pois"]["xi"] - scan["x_hat"]) < 0.1, (refit, ends, scan["x_hat"]))
    ft, pf = rec.get("freq_toys") or {}, rec.get("poi_fits") or {}
    check("the post stage's record: status ok, both pieces, every POI's interval and stat-only sides, the GOF toys "
          "with xi pinned at its truth, the cross-checks, the reported values consistent with the model's",
          rec.get("status") == "ok" and len(rec["scan"]["pieces"]) == 2 and ft.get("pins") == {"xi": 1.0}
          and ft.get("n_toys", 0) > 0 and ft.get("mu_truth") == 1.0 and ft.get("gof_p_comb") is not None
          and all(pf[p].get(k) is not None for p in ("xi", "mu_on") for k in ("err_minus", "err_plus",
                                                                                "stat_minus", "stat_plus"))
          and set(rec.get("quality_crosscheck") or {}) == {"gof_pvalue", "coverage"} and rec.get("verify_wall_s")
          and rec.get("consistency") == {"pass": True},
          {k: rec.get(k) for k in ("status", "poi_fits", "quality_crosscheck")})
    check("a model the twin does not read is the model's failure, no pyhf fallback: an unsupported modifier, a spec "
          "its constructor refuses",
          refused["status"].startswith("invalid workspace: unsupported modifiers")
          and spec_refused["status"] == "invalid workspace: non-positive normsys kappa on lumi",
          (refused["status"], spec_refused["status"]))
    check("two samples of 1e13 events that cancel pass the rate cross-check (its tolerance is on the gross rate): the "
          "dataset is graded or failed as the model's, never pending",
          cancelling["status"] == "ok" or cancelling["status"].startswith(("invalid workspace", "fit failed")),
          cancelling["status"])
    check("the evaluator's own failures leave the dataset pending, never fitted with pyhf: no jax, an exception in "
          "its code (a ValueError too)",
          nojax["status"].startswith("verify error: jax engine unavailable")
          and ours["status"] == "verify error: ValueError: ours", (nojax["status"], ours["status"]))


def check_end_to_end():
    """Six datasets of spec()'s model at truths spread over xi and mu_on, observed at their expectation (rounded):
    graded by the post stage, reported as the grader's records give them, scored as a closure's run."""
    import os
    import numpy as np
    from sbh.kinds.scan_likelihood import verify as V
    from sbh.score import score_run
    task = type("ScanTaskE2E", (ScanTask,), {"requires_model": True,
                                              "fallback_forecast": {"xi": (50.0, 28.868), "mu_on": (2.5, 1.44)}})()
    truths = {f"seed_{i:03d}": {"xi": xi, "mu_on": mu, "profile": f"p{i % 3}"}
              for i, (xi, mu) in enumerate(((0.5, 1.0), (1.0, 1.2), (2.0, 0.8), (4.0, 1.0), (8.0, 1.1), (16.0, 0.9)),
                                           start=1)}
    os.environ["SBH_VERIFY_TOYS"] = "20"
    with tempfile.TemporaryDirectory() as d:
        run, suite = Path(d) / "run" / "reference", Path(d) / "suite"
        suite.mkdir(parents=True)
        (suite / "manifest.json").write_text(json.dumps({"suite": "scan_likelihood_check", "master_seed": 1, "seeds": [
            {"seed": s_, "truth": t, "public": False} for s_, t in truths.items()]}))
        for s_, t in truths.items():
            sp = spec()
            model, data, built, meta, idx = V.build_model(sp, task)
            ev = V.twin(built, model, meta["poi_scaling"])
            x = ev.init.copy()
            x[idx["xi"]], x[idx["mu_on"]] = t["xi"], t["mu_on"]
            nu = np.asarray(ev.expected_main(x))
            sp["observations"] = [{"name": "off", "data": [int(round(v)) for v in nu[:2]]},
                                  {"name": "on", "data": [int(round(v)) for v in nu[2:]]}]
            out = run / "runs" / s_ / "out"
            out.mkdir(parents=True)
            (out / "MODEL.json").write_text(json.dumps(sp))
            (out / "RESULT.json").write_text(json.dumps(result()))
            (out.parent / "record.json").write_text(json.dumps({"seed": s_, "exit_code": 0, "timed_out": False,
                                                                "result": result(), "result_errors": [],
                                                                "wall_s": 1.0}))
            get_kind(task).post_dataset(run, suite, task, s_)
            row = json.loads((out.parent / V.VERIFY_FILE).read_text())
            pf = row["poi_fits"]
            accurate = {"pois": [{"name": p, **{k: pf[p][k] for k in ("value", "err_minus", "err_plus")}}
                               for p in task.pois],
                      "nuisances": {"lumi": {"value": 1.0, "unc": 0.02}}}
            accurate["pois"][0]["intervals"] = [[pc["lo"], pc["hi"]] for pc in row["scan"]["pieces"]]
            if len(accurate["pois"][0]["intervals"]) == 1:
                del accurate["pois"][0]["intervals"]
            (out / "RESULT.json").write_text(json.dumps(accurate))
            (out.parent / "record.json").write_text(json.dumps({"seed": s_, "exit_code": 0, "timed_out": False,
                                                                "result": accurate, "result_errors": [],
                                                                "wall_s": 1.0}))
        import argparse
        from sbh.kinds.scan_likelihood import scoring as SS
        with contextlib.redirect_stdout(io.StringIO()):
            pub = score_run(run, suite, task, "reference", making_baseline=True)
            priv = json.loads((run / "private" / "scores.json").read_text())
            a = argparse.Namespace(max_pull=None, max_leakage=None, leakage_nsigma=None, leakage_se_cap=None,
                                   calibration_slope_tol=None)
            crit, _mp, fields = get_kind(task).closure_criteria(pub, priv, task, a)
            holed = json.loads(json.dumps(priv))
            holed["reference_pois"]["seed_002"]["xi"].pop("entropy")
            crit_h, _mp, fields_h = get_kind(task).closure_criteria(pub, holed, task, a)
            vf = run / "runs" / "seed_003" / V.VERIFY_FILE
            kept = vf.read_text()
            strict_task = type("ScanTaskStrict", (type(task),), {"grader_strict": True})()
            variants = {"lost": None,
                        "fault": {"seed": "seed_003", "scan_version": V.SCAN_VERSION, "errors_marked": 1,
                                  "status": "verify error: RuntimeError: the evaluator's",
                                  "evaluator_error": "RuntimeError: the evaluator's"},
                        "pending": {"seed": "seed_003", "scan_version": V.SCAN_VERSION, "errors_marked": 1,
                                    "status": "not graded: the grader's queue"}}
            other = {}
            for how, content in variants.items():
                if content is None:
                    vf.unlink()
                else:
                    vf.write_text(json.dumps(content))
                for strict_ in (False, True):
                    v3_ = score_run(run, suite, strict_task if strict_ else task, "reference",
                                    making_baseline=True)["v3"]
                    pv_ = json.loads((run / "private" / "scores.json").read_text())
                    other[(how, strict_)] = (v3_, pv_["fit_info"]["seed_003"], len(pv_["v4_truth_cases"]),
                                             (pv_.get("v2_provisional") or {}).get("cases"))
            vf.write_text(kept)
    key = "the grader measured xi's density on every dataset"
    check("the closure checks the grader's baseline: the criterion holds and closure.json's scan_baseline is the "
          "grader's on 6 datasets; a dataset without the reference's density fails it and is named",
          crit[key] is True and fields["scan_baseline"] == {"source": "grader", "n": 6, "n_datasets": 6, "missing": [],
                                                           "scan_version": V.SCAN_VERSION}
          and crit_h[key] is False and fields_h["scan_baseline"]["source"] == "pending"
          and fields_h["scan_baseline"]["missing"] == ["seed_002"], (fields, fields_h))
    def gof_n(v):
        return (v.get("gof") or {}).get("n")


    def why(v):
        return " | ".join(v.get("provisional_reasons") or [])
    lost, lost_s = other[("lost", False)], other[("lost", True)]
    check("a completed dataset whose grading was lost (no grader record): pending, scored provisionally on its reported "
          "entries (the closure's own run has no baseline for its scanned POI there) and without a GOF case; under "
          "Task.grader_strict pending all the same, never failed: its cases left out (never the reported value) with "
          "the reason \"no grader record\", its GOF left out",
          lost[3] == ["seed_003:mu_on"] and gof_n(lost[0]) == 5 and not lost_s[0].get("failed")
          and not lost_s[3] and gof_n(lost_s[0]) == 5 and "no grader record" in why(lost_s[0])
          and all(m["status"] == "pending" and m["reason"] == "no grader record" for m in lost_s[1].values()),
          (lost[3], lost[0].get("provisional_reasons"), lost_s[1], why(lost_s[0])))
    fault, fault_s = other[("fault", False)], other[("fault", True)]
    check("a dataset the grader's own code failed on (a stamped record carrying \"evaluator_error\"): its cases left "
          "out as an evaluator fault and its GOF left out, strict or not, never failed",
          all(not v[0].get("failed") and gof_n(v[0]) == 5 and not v[3] and "evaluator fault" in why(v[0])
              and all(m["status"] == "pending" and m.get("evaluator_fault") for m in v[1].values())
              for v in (fault, fault_s)), (fault[1], fault_s[1]))
    pend, pend_s = other[("pending", False)], other[("pending", True)]
    check("a dataset the grader left pending for any other reason (a stamped status that is neither ok, the model's "
          "nor the grader's own error): scored provisionally on its reported entries; under Task.grader_strict failed "
          "(S = 0), its GOF charged and nothing provisional",
          pend[3] == ["seed_003:mu_on"] and pend_s[0].get("failed") and pend_s[0]["s_v3"] == 0.0
          and gof_n(pend_s[0]) == 6 and not pend_s[0].get("provisional") and pend_s[1]["xi"]["status"] == "failed"
          and pend_s[2] == 10,
          (pend[3], pend_s[0].get("failed"), gof_n(pend_s[0]), pend_s[1], pend_s[2]))
    from sbh.score import score_gamma, score_mean_charge
    v3 = pub.get("v3") or {}
    lp = priv.get("likelihood_pulls", {}).get("xi", {})
    tr = v3.get("truthfulness") or {}
    pp = (pub.get("v2") or {}).get("per_poi") or {}
    mws = {p: (pp.get(p) or {}).get("multiworld") or {} for p in task.pois}
    d_phys = v3.get("phys", {}).get("delta", 9.0)
    check("a closure-like run graded and scored end to end: every case measured against its own baseline (data at "
          "their expectation put each truth at the density's mode: a mean case charge of about -1/2 per POI, and the "
          "bias term at its floor -1 / (2 K), K = 6: Delta_phys in (-0.6 - 1/12, -0.4 - 0.75/12)), truthfulness 0 for "
          "reports equal to the grader's, GOF scored, the likelihood pulls on the grader's scans",
          pub.get("recovery") == 1.0 and -0.6 - 1 / 12 < d_phys < -0.4 - 0.75 / 12
          and all(-0.6 < (pp.get(p) or {}).get("delta_mean", 9.0) < -0.4 and m.get("k") == 6
                  and abs(m.get("zstar", 9.0)) < 0.5
                  and abs(m["bias"] - (m["zstar"] ** 2 + m["spread"] - 1.0) / (2 * m["k"])) < 1e-12
                  for p, m in mws.items())
          and abs(d_phys - score_mean_charge([pp[p]["delta_mean"] + mws[p]["bias"] for p in task.pois],
                                             score_gamma(task))) < 1e-12
          and tr.get("delta") == 0.0
          and (v3.get("gof") or {}).get("n") == 6 and lp.get("n") == 6 and not v3.get("provisional"),
          {"delta_mean": {p: (pp.get(p) or {}).get("delta_mean") for p in task.pois}, "recovery": pub.get("recovery")}
          | {k: v3.get(k) for k in ("phys", "truthfulness", "gof", "provisional")} | {"pulls": lp, "multiworld": mws})


if __name__ == "__main__":
    check_scaling()
    check_interval_form()
    check_two_piece()
    check_density()
    check_likelihood_pulls()
    check_pieces()
    check_gate()
    check_event_list_messages()
    check_refit()
    check_contract()
    check_stub()
    check_scoring()
    check_scan_rules()
    if "--grader" in sys.argv:
        check_conventions()
        check_grader()
        check_end_to_end()
    print(f"scan_likelihood check: {'FAIL' if fails else 'PASS'}")
    sys.exit(1 if fails else 0)
