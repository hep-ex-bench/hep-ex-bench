"""Sequential GOF toys (sbh/kinds/binned_likelihood/seq_toys.py, seq_gof.py): smoke checks (`pixi run smoke`, ~15 s).

1. check_validity: the sequential p is conservative and stops early on a synthetic model.
2. check_rule: the stopping rule (h-th exceedance, cap, n_min, floor, wall), determinism, batch invariance.
3. check_maps / check_moments: MinP's t, rank_minp, the Poisson deviance moments (OOM guard).
4. check_stage: the grader stage on a tiny twin model and the SBH_GOF_SEQ hook.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("SBH_TOY_ENGINE", "jax")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from sbh.kinds.binned_likelihood import seq_toys as st  # noqa: E402

fails = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)


NU = 3.0 + 12.0 * np.exp(-0.5 * ((np.arange(30) - 14.5) / 6.0) ** 2)
WIN = np.zeros(30, bool)
WIN[10:20] = True


def _dev(n, mu):
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(n > 0, n * np.log(n / mu), 0.0)
    return 2.0 * ((mu - n) + t)


def synth_stats(N):
    """(global deviance, window deviance) of each row of counts N at its closed-form best fit s * NU."""
    N = np.atleast_2d(N)
    s = N.sum(axis=1, keepdims=True) / NU.sum()
    d = _dev(N, s * NU)
    return np.c_[d.sum(axis=1), d[:, WIN].sum(axis=1)]


def check_validity():
    """With data drawn from a 30-bin model (global + window deviance via MinP), P(p <= 0.05 / 0.01) is at most nominal
    within 3 sigma, the one-sided KS D+ above U(0, 1) is small, and the mean toy count is << n_max."""
    print("1. validity on the synthetic model")
    maps = st.MinP([29.0, 10.0], eps=1e-9)
    rng = np.random.default_rng(1234)
    R, h, n_max = 2000, 10, 200
    ps, ns = [], []
    for r in range(R):
        obs = rng.poisson(NU)
        ext = maps.extreme(synth_stats(obs)[0])
        res = st.sequential_pvalue(lambda g, n: g.poisson(NU, size=(n, 30)), synth_stats, ext,
                                   h=h, n_max=n_max, batch=40, seed=[5, r])
        ps.append(res["p"])
        ns.append(res["n_toys"])
    ps = np.asarray(ps)
    for u in (0.05, 0.01):
        frac = float(np.mean(ps <= u))
        check(frac <= u + 3 * np.sqrt(u * (1 - u) / R), f"P(p <= {u}) = {frac:.4f} (nominal {u}, R = {R})")
    grid = np.linspace(0.005, 1, 200)
    dplus = float(np.max(np.mean(ps[None, :] <= grid[:, None], axis=1) - grid))
    check(dplus <= 1.36 / np.sqrt(R),
          f"one-sided KS D+ = {dplus:.4f} (CDF above U(0,1); 5% bound {1.36 / np.sqrt(R):.4f})")
    check(np.mean(ns) < 0.3 * n_max, f"mean toys {np.mean(ns):.1f} of n_max {n_max} (early stop)")


def check_rule():
    """p = h/L at the h-th exceedance, (k+1)/(n+1) at the cap / n_min / floor / wall stops; failed toys skipped;
    ``simulate`` == the runner; deterministic given the seed and invariant under the batch size."""
    print("2. the stopping rule")
    ext = np.zeros(300, bool)
    ext[[3, 10, 50]] = True
    r = st.simulate(ext, h=3, n_max=1000)
    check(r["stop"] == "h" and r["n_toys"] == 51 and abs(r["p"] - 3 / 51) < 1e-12, f"h-th exceedance: {r}")
    r = st.simulate(ext[:40], h=3, n_max=40)
    check(r["stop"] == "n_max" and abs(r["p"] - 3 / 41) < 1e-12, f"cap: {r}")
    r = st.simulate(ext, h=3, n_max=1000, n_min=100)
    check(r["stop"] == "n_min" and r["n_toys"] == 100 and abs(r["p"] - 4 / 101) < 1e-12, f"n_min: {r}")
    r = st.simulate(np.zeros(2000, bool), h=20, n_max=2000, floor_stop=(0.005, 0.05))
    check(r["stop"] == "floor" and r["n_toys"] < 700 and r["p"] < 0.005, f"floor stop at k = 0: {r}")
    val = np.ones(300, bool)
    val[[0, 3, 7]] = False
    r = st.simulate(ext, val, h=2, n_max=1000)
    check(r["n_toys"] == 48 and r["n_drawn"] == 51 and abs(r["p"] - 2 / 48) < 1e-12,
          f"failed toys skipped (toy 3's exceedance lost, 3 of 51 drawn failed): {r}")

    def draw(g, n):
        return g.chisquare(4, size=n)

    runs = [st.sequential_pvalue(draw, lambda t: t, lambda S: S >= 9.0, h=15, n_max=600, batch=b, seed=3)
            for b in (1, 37, 600)]
    same = all(x["p"] == runs[0]["p"] and x["n_toys"] == runs[0]["n_toys"] and x["k"] == runs[0]["k"]
               for x in runs)
    check(same, f"batch 1 / 37 / 600 identical: p {[x['p'] for x in runs]}, toys {[x['n_toys'] for x in runs]}")
    again = st.sequential_pvalue(draw, lambda t: t, lambda S: S >= 9.0, h=15, n_max=600, batch=37, seed=3)
    check(again["p"] == runs[1]["p"] and np.array_equal(again["stats"], runs[1]["stats"]),
          "deterministic given the seed")
    sim = st.simulate(runs[1]["extreme"][:runs[1]["n_drawn"]], h=15, n_max=600)
    check(sim["p"] == runs[1]["p"] and sim["n_toys"] == runs[1]["n_toys"], "simulate == runner")
    check(runs[0]["n_discarded"] == 0 and 0 <= runs[1]["n_discarded"] < 37,
          f"toys fitted past the stop: batch 1 -> 0, batch 37 -> {runs[1]['n_discarded']}")
    import time as _time

    def slow(t):
        _time.sleep(0.002 * len(t))
        return t
    w = st.sequential_pvalue(draw, slow, lambda S: S >= 30.0, h=15, n_max=100000, batch=10, seed=3, max_wall_s=0.2)
    check(w["stop"] == "wall" and 10 <= w["n_toys"] < 1000 and abs(w["p"] - (w["k"] + 1) / (w["n_toys"] + 1)) < 1e-12,
          f"wall budget: stop 'wall' after {w['n_toys']} toys, p = (k+1)/(L+1)")
    fixed = st.sequential_pvalue(draw, lambda t: t, lambda S: S >= 9.0, h=None, n_max=500, batch=50, seed=3)
    check(fixed["n_toys"] == 500 and fixed["stop"] == "n_max"
          and abs(fixed["p"] - st.fixed_pvalue(fixed["extreme"])) < 1e-12, "h=None: the fixed ensemble's (k+1)/(N+1)")


def check_maps():
    """MinP's t ~ the min of the chi2 tails, its tie tolerance, moment_map, and rank_minp == a brute-force min-p."""
    print("3. the min-p maps")
    from scipy.stats import chi2
    rng = np.random.default_rng(4)
    Q = np.c_[rng.chisquare(40, 200), rng.chisquare(8, 200)]
    m = st.MinP([40, 8])
    t = m.nominal_p(Q)
    ref = np.minimum(chi2.sf(Q[:, 0], 40), chi2.sf(Q[:, 1], 8))
    dl = float(np.max(np.abs(np.log(t) - np.log(ref))))
    check(dl < 0.25, f"t ~ min chi2 tails (max |dln t| {dl:.3f})")
    obs = np.array([50.0, 9.0])
    e0 = st.MinP([40, 8], eps=0.0).extreme(obs)
    e1 = st.MinP([40, 8], eps=1e-3).extreme(obs)
    check(bool(e0(obs[None, :])[0]) and bool(e1((obs - 5e-4)[None, :])[0]) and not bool(e0((obs - 5e-4)[None, :])[0]),
          "the observed row counts as extreme; a row within eps below it only with the tolerance")
    nu, sc = st.moment_map(30.0, 90.0)
    check(abs(nu * sc - 30.0) < 1e-9 and abs(2 * sc * sc * nu - 90.0) < 1e-9, "moment_map matches mean and variance")
    Qt = rng.normal(size=(60, 2))
    qo = np.array([0.8, 1.1])
    pool = np.vstack([qo, Qt])
    pj = [np.array([np.mean(pool[:, j] >= pool[i, j]) for i in range(len(pool))]) for j in range(2)]
    mm = np.minimum(pj[0], pj[1])
    brute = float(np.mean(mm <= mm[0] + 1e-12))
    rk = st.rank_minp(Qt, qo)
    check(abs(rk["p_comb"] - brute) < 1e-12 and abs(rk["p_each"][0] - pj[0][0]) < 1e-12,
          f"rank_minp == brute force ({rk['p_comb']:.4f})")
    check_moments()


def check_moments():
    """seq_gof.poisson_deviance_moments: exact below POISSON_MOMENTS_BIG, asymptotic (mean 1 + 1/(6 nu), var 2)
    above, continuous across the switch; a bin at nu = 1e9 must not allocate the ~GiB summation window (OOM)."""
    import time
    import tracemalloc
    import scipy.stats  # noqa: F401
    from sbh.kinds.binned_likelihood import seq_gof
    big = seq_gof.POISSON_MOMENTS_BIG
    nu = np.array([0.01, 0.5, 5.0, big - 1.0, big + 1.0, 1e5, 1e9] + [3.0] * 600)
    tracemalloc.start()
    t0 = time.perf_counter()
    m, v = seq_gof.poisson_deviance_moments(nu)
    wall = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    check(peak < 256 * 2 ** 20 and wall < 10.0 and np.all(np.isfinite(m)) and np.all(np.isfinite(v)),
          f"moments: a bin at nu = 1e9 next to 600 ordinary ones costs {peak / 2 ** 20:.1f} MiB, {wall:.2f} s "
          "(the summation window: ~3 GiB per temporary)")
    check(abs(m[6] - 1.0) < 1e-6 and abs(v[6] - 2.0) < 1e-9 and abs(m[5] - (1 + 1 / 6e5)) < 1e-12,
          f"moments: asymptotic above {big:g} (nu = 1e9: mean {m[6]:.6f}, var {v[6]:.3f})")
    check(abs(m[3] - m[4]) < 2e-3 and abs(v[3] - v[4]) < 0.02,
          f"moments: continuous across the switch (mean {m[3]:.5f} | {m[4]:.5f}, var {v[3]:.4f} | {v[4]:.4f})")
    check(abs(m[0] - 0.09) < 0.01 and abs(m[2] - 1.0) < 0.15,
          f"moments: exact below it (nu = 0.01: mean {m[0]:.3f}; nu = 5: {m[2]:.3f})")


def check_stage():
    """The grader stage: batch-invariant, sequential toys == the fixed ensemble's first ones, t- vs rank-based p,
    no-window / 0-dof fallbacks, the pyhf engine, fallback copies == the grader's, and the SBH_GOF_SEQ hook."""
    print("4. the grader stage on a tiny twin model")
    import pyhf
    from sbh.kinds.binned_likelihood import jax_engine as EVT, seq_gof
    from sbh.kinds.binned_likelihood import verify as MV
    from checks.jax_engine_check import _spec, _spec_1bin
    MV.GOF_WINDOW = (105.0, 140.0)
    pyhf.set_backend("numpy", "minuit")
    spec = _spec()
    spec["observations"][0]["data"] = [20, 8, 16, 11, 19, 6, 9, 2]
    meta = {"binning": {"sr": {"edges": list(np.linspace(100, 148, 9))}, "cr": {"edges": [60, 70, 80, 90]}}}
    ws = pyhf.Workspace(spec)
    model = ws.model()
    data = np.asarray(ws.data(model), float)
    n_main = model.config.nmaindata
    ev, _ = EVT.build(spec, model)
    obs = ev.fit(data[:n_main], data[n_main:], want_sigma=False)
    kw = dict(ev=ev, obs_fit=obs, meta=meta, grader=MV, return_arrays=True)
    a = seq_gof.seq_scored_gof(model, data, 1.0, h=10, n_max=300, batch=1, **kw)
    b = seq_gof.seq_scored_gof(model, data, 1.0, h=10, n_max=300, batch=7, **kw)
    check(a["gof_p_comb"] == b["gof_p_comb"] and a["n_toys"] == b["n_toys"] and a["gof_p_comb"] is not None,
          f"batch 1 vs 7: p {a['gof_p_comb']:.4f}, {a['n_toys']} toys, stop {a['gof_seq']['stop']}")
    f = seq_gof.seq_scored_gof(model, data, 1.0, h=None, n_max=150, batch=50, **kw)
    n = a["gof_seq"]["n_drawn"]
    Sa, Sf = a["_arrays"]["stats"], f["_arrays"]["stats"]
    m = min(n, len(Sf))
    check(np.allclose(Sa[:m], Sf[:m], equal_nan=True), f"the sequential toys are the fixed ensemble's first {m}")
    A = f["_arrays"]
    ok = A["valid"]
    rk = st.rank_minp(A["stats"][ok][:, :2], A["q_obs"][:2], eps=1e-3)
    check(abs(np.log(f["gof_p_comb"]) - np.log(rk["p_comb"])) < 0.5,
          f"fixed 150, same toys: t-based p (poisson maps) {f['gof_p_comb']:.3f} vs rank-based {rk['p_comb']:.3f}")
    check(a["gof_n_sr_bins"] == 6 and a["gof_aux"] == "held" and a["coverage_68"] is not None,
          "window of 6 bins, aux held, coverage from the same toys")
    nw = seq_gof.seq_scored_gof(model, data, 1.0, ev=ev, obs_fit=obs, meta=None, h=10, n_max=200, grader=MV)
    check(nw["gof_p_comb"] is None and nw["gof_p_missing"] == "no_window" and nw["gof_p_glob"] is not None,
          f"no window: p_comb None (no_window), global p {nw['gof_p_glob']}")
    s1 = _spec_1bin()
    ws1 = pyhf.Workspace(s1)
    m1 = ws1.model()
    d1 = np.asarray(ws1.data(m1), float)
    ev1, _ = EVT.build(s1, m1)
    o1 = ev1.fit(d1[:1], d1[1:], want_sigma=False)
    z = seq_gof.seq_scored_gof(m1, d1, 0.6, ev=ev1, obs_fit=o1, meta=None, n_max=500, grader=MV)
    check(z["gof_p_comb"] is None and z["gof_p_glob"] is None and z["gof_p_missing"] == "dof"
          and z["n_coverage_toys"] >= 100, f"0 dof: no p, {z['n_coverage_toys']} coverage toys")
    py = seq_gof.seq_scored_gof(model, data, 1.0, meta=meta, h=5, n_max=40, grader=MV)
    check(py["engine"] == "pyhf" and py["gof_p_comb"] is not None, f"pyhf engine: p {py['gof_p_comb']}")
    pts = [obs["x"], model.config.suggested_init()]
    same = [seq_gof._gof_dof(model, pts) == MV._gof_dof(model, pts),
            seq_gof._bound_window(0.0, 10.0) == MV._bound_window(0.0, 10.0),
            seq_gof.GOF_TIE_EPS == MV.GOF_TIE_EPS]
    if hasattr(MV, "_window_mask"):
        same.append(np.array_equal(seq_gof._window_mask(model, meta, MV.gof_window())[0],
                                   MV._window_mask(model, meta)[0]))
    if hasattr(MV, "_deviance_terms"):
        tt = np.asarray(ev.expected_main(obs["x"]))
        same.append(np.allclose(seq_gof._deviance_terms(tt, data[:n_main]), MV._deviance_terms(tt, data[:n_main])))
    check(all(same), f"fallback copies == the grader's ({len(same)} compared)")
    check(seq_gof.config_from_env({}) is None and seq_gof.config_from_env({"SBH_GOF_SEQ": "0"}) is None,
          "SBH_GOF_SEQ unset / 0: off")
    try:
        seq_gof.config_from_env({"SBH_GOF_SEQ": "yes"})
        check(False, "a bad SBH_GOF_SEQ raises")
    except ValueError:
        check(True, "a bad SBH_GOF_SEQ raises")
    cfg = seq_gof.config_from_env({"SBH_GOF_SEQ": "shadow", "SBH_GOF_SEQ_NMAX": "200", "SBH_GOF_SEQ_H": "10"})
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"seq_toys_check_{os.getpid()}"
    (tmp / "out").mkdir(parents=True, exist_ok=True)
    import json
    (tmp / "out" / "MODEL.json").write_text(json.dumps({**spec, "_meta": meta}))
    rec = {"freq_toys": {"fixed": True}}
    seq_gof.grader_stage(cfg, rec, tmp, model, data, 1.0, ev=ev, obs_fit=obs, grader=MV)
    check(rec["freq_toys"] == {"fixed": True} and rec["gof_seq"]["gof_p_comb"] == a["gof_p_comb"]
          and rec["gof_seq"]["gof_n_sr_bins"] == 6,
          "shadow: gof_seq filed next to the fixed stage (window from MODEL.json _meta)")
    rec = {}
    seq_gof.grader_stage({**cfg, "mode": "score"}, rec, tmp, model, data, 1.0, ev=ev, obs_fit=obs, grader=MV)
    check(rec.get("freq_toys", {}).get("gof_p_comb") == a["gof_p_comb"] and "gof_seq" not in rec,
          "score: filed as freq_toys")
    check(MV.SEQ_GOF is None or os.environ.get("SBH_GOF_SEQ"), "grader: SEQ_GOF off without SBH_GOF_SEQ")


if __name__ == "__main__":
    check_validity()
    check_rule()
    check_maps()
    check_stage()
    print(f"seq_toys_check: {'FAIL' if fails else 'PASS'} ({len(fails)} failures)")
    sys.exit(1 if fails else 0)
