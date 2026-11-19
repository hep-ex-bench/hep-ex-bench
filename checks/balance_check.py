"""The jet balance closure (sbh/kinds/corrections/balance.py) on toy datasets with a known jet response
(checks/balance_toy.py), through the toy's own spec (balance.spec) and the scorer's Delta_peak rule
(scoring.function_peak_cases); seconds.

  0. the candidate cache returns the toy's candidates bit for bit;
  1. the right corrections against themselves as the reference: Delta 0, and the toy closes (the residuals' chi2 per
     cell, simulation and data variance, near 1);
  2. nominal (no correction) on the shifted dataset is penalised, in each stream;
  3. a scale stated twice too large is penalised, in each stream;
  4. the simulation's normalisation against the data (the luminosity x 1.37) leaves every loss unchanged;
  5. the deterministic smear: a second comparison reproduces the first bit for bit, another dataset index differs;
  6. a negative extra variance is continued through zero (the loss at -1e-12, 0 and +1e-12 agree; with fixed normals
     the difference shrinks as sqrt|V|) and stays finite;
  7. a failed run's dataset has no peak case (it fails the run); a comparison not made within its budget has none
     either, counted as the fallback peak_not_in_budget (no failure premium).

    python checks/balance_check.py
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f": {detail}" if not ok and detail else ""))
    if not ok:
        fails.append(name)


def main() -> int:
    t0 = time.time()
    from checks import balance_toy as T
    from sbh.kinds.corrections import balance as B
    from sbh.kinds.corrections.post import PEAK_NOT_IN_BUDGET
    from sbh.kinds.corrections.scoring import fallbacks as peak_fallbacks, function_peak_cases
    from sbh.score import V2_CAP_FLOOR

    request = T.REQUEST
    spec = T.spec({"scale": {"central": 0.05, "endcap": 0.03, "hf": 0.10},
                   "res_pt": {"central": 0.008, "endcap": 0.004, "hf": 0.015}})
    ok_rec = {"exit_code": 0, "timed_out": False, "result": {}, "result_errors": []}
    with tempfile.TemporaryDirectory(prefix="balance_check_") as tmp:
        tmp = Path(tmp)
        ds = T.write_dataset(tmp / "ds", seed=1)
        ds_norm = T.write_dataset(tmp / "ds_norm", seed=1, lumi_scale=1.37)
        cand = B.candidates(ds, spec, "jet", tmp / "cache", jobs=1)
        again = B.candidates(ds, spec, "jet", tmp / "cache", jobs=1)
        same = all(cand[s][k].tobytes() == again[s][k].tobytes() for s in cand for k in cand[s])
        check(f"0. the candidate cache returns them bit for bit ({len(cand['data']['bits'])} data, "
              f"{len(cand['sim']['bits'])} simulated events)", same)
        cand_n = B.candidates(ds_norm, spec, "jet", tmp / "cache_norm", jobs=1)

        def run(cset, c=cand, seed=1):
            return B.closure(spec, c, cset, {}, request, "jet", seed, {})
        right, nominal, double = (T.corrections(T.TRUTH), T.corrections(T.NOMINAL),
                                  T.corrections(T.TRUTH, scale_factor=2.0))
        ref = run(right)
        fb = B.fallback_loss(spec, cand, request, "jet", 1, tmp / "cache")
        base = {"s1": {"jet": {"loss": ref["loss"], "loss_fallback": fb}}}

        def peak_cases(r, rec=ok_rec):
            return function_peak_cases([dict(rec, seed="s1")], {"s1": {"peak": {"jet": r}}}, base,
                                       {"s1": {"profile": "toy"}})

        def delta(r, rec=ok_rec):
            return peak_cases(r, rec)[0][0]
        cap = max(fb - ref["loss"], V2_CAP_FLOOR)
        d_ref = delta(ref)
        fit = ref["source_fit"]
        chi2 = fit["chi2_prefit"] / fit["n_cells"]
        check(f"1. the right corrections against themselves: Delta {d_ref['delta']:+.3g}, over "
              f"{ref['n_categories']} categories (per stream {json.dumps(ref['per_stream'])}); the cap C {cap:.3f}",
              d_ref["delta"] == 0.0 and ref["n_categories"] >= 10 and all(v is not None
                                                                        for v in ref["per_stream"].values()))
        check(f"1. the toy closes: residual chi2 per cell {chi2:.3f} over {fit['n_cells']} cells", 0.7 < chi2 < 1.3)
        losses = {}
        for label, cset in (("2. nominal (no correction) on the shifted dataset", nominal),
                            ("3. a scale stated twice too large", double)):
            r = run(cset)
            losses[label[0]] = r["loss"]
            d = delta(r)
            worse = {k: r["per_stream"][k] - ref["per_stream"][k] for k in r["per_stream"]}
            check(f"{label} is penalised: Delta {d['delta']:+.3f}, per stream "
                  f"{json.dumps({k: round(v, 3) for k, v in worse.items()})}",
                  d["delta"] > 0.1 and all(v > 0.02 for v in worse.values()))
        rel = [abs(run(right, cand_n)["loss"] - ref["loss"]) / abs(ref["loss"]),
               abs(run(nominal, cand_n)["loss"] - losses["2"]) / abs(losses["2"])]
        w_ratio = float(cand_n["sim"]["weight"].sum() / cand["sim"]["weight"].sum())
        check(f"4. the simulation's normalisation x {w_ratio:.2f} leaves the loss unchanged (relative {max(rel):.1e})",
              max(rel) < 1e-9 and abs(w_ratio - 1.37) < 1e-9)
        again = run(right)
        other = run(right, seed=2)
        check("5. the smear is deterministic: the comparison reproduces bit for bit, another dataset index differs",
              json.dumps(again, sort_keys=True) == json.dumps(ref, sort_keys=True) and other["loss"] != ref["loss"])
        def at_var(v):
            per = {"scale": T.TRUTH["scale"], "res_var": {r: v for r in T.TRUTH["res_var"]}}
            return run(T.corrections(per))["loss"]
        l0, lp, lm, ln = at_var(0.0), at_var(1e-12), at_var(-1e-12), at_var(-0.004)
        check(f"6. a negative extra variance is continued through zero (loss {lm:.6f} at -1e-12, {l0:.6f} at 0, "
              f"{lp:.6f} at +1e-12) and stays finite ({ln:.3f} at -0.004, above the right answer's {ref['loss']:.3f})",
              abs(lp - l0) < 1e-3 and abs(lm - l0) < 1e-3 and math.isfinite(ln) and ln > ref["loss"])
        failed_cases, _ = peak_cases(None, dict(ok_rec, exit_code=1))
        over, missed = peak_cases({"why": f"{PEAK_NOT_IN_BUDGET} (alarm)"})
        check("7. a failed run's dataset has no peak case; a comparison not made within its budget has none either, "
              "counted as the fallback peak_not_in_budget",
              failed_cases == [] and over == [] and [m["seed"] for m in missed["not_in_budget"]] == ["s1"]
              and peak_fallbacks(missed)["peak_not_in_budget"]["seeds"] == ["s1"], (failed_cases, over, missed))
    print(f"balance check ({time.time() - t0:.0f} s): {'FAIL' if fails else 'PASS'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
