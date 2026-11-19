#!/usr/bin/env python3
"""Reference pipeline for the `bump_model` task (numpy and uproot: the base environment).

    analysis.py <dataset_dir> <output_dir> [flags]

1. calibrate the lepton momentum scale on the control peak (data vs MC) and correct the data masses
2. binned Poisson likelihood of mu x signal(mH) + b x background on 1 GeV bins of the corrected mass
3. profile it over a mass grid (110-140 GeV, refined near the minimum): mH, its interval, mu at the best mass
4. write RESULT.json, the likelihood as a pyhf workspace (MODEL.json, MODEL_GRID/mH_<value>.json) and EVENTS.json

The flags only build the attacker submissions of the harness stress test; the reference never sets them.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import uproot

EDGES = np.arange(102.0, 158.001, 1.0)    # corrected masses: inside the data window for |scale - 1| < 2 %
MASS_RANGE = (110.0, 140.0)                # the evaluation's mH range: the grid always reaches both ends
COARSE, FINE, FINE_HALF = 1.0, 0.25, 1.5   # grid steps (GeV) and the fine region's half-width
CTRL_WIN = (82.0, 100.0)                   # control-peak window at the nominal scale
MU_BOUNDS, B_BOUNDS = (-1.0, 10.0), (0.0, 10.0)
CHANNEL, BKG_NORM = "mass", "ana_bkg_norm"
FLOOR = 1e-9                               # the evaluator's floor on every bin's expected rate


# ---- calibration -----------------------------------------------------------------------------------------------
def calibrate(dataset):
    """(scale - 1, its error): the control peak's median in CTRL_WIN scaled by s, over the simulation's in CTRL_WIN,
    iterated to its fixed point (a common scale moves the whole distribution, flat part included)."""
    data = np.load(dataset / "control" / "events.npy")
    mc = np.load(dataset / "mc" / "control_mc.npy")
    mc = mc[(mc >= CTRL_WIN[0]) & (mc < CTRL_WIN[1])]
    s = 1.0
    for _ in range(100):
        d = data[(data >= s * CTRL_WIN[0]) & (data < s * CTRL_WIN[1])]
        s, last = float(np.median(d)) / float(np.median(mc)), s
        if abs(s - last) < 1e-10:
            break
    # error of a median: sqrt(pi / 2) x the spread / sqrt(n), data and simulation independent
    rel = [np.sqrt(np.pi / 2) * np.std(x) / np.sqrt(x.size) / np.median(x) for x in (d, mc)]
    return s - 1.0, s * float(np.hypot(*rel))


# ---- templates and likelihood ----------------------------------------------------------------------------------
class Templates:
    def __init__(self, dataset):
        xs = json.loads((dataset / "mc" / "cross_sections.json").read_text())
        tab = sorted((float(k), float(v)) for k, v in xs["signal_expected_SM"].items())
        self.xs_m, self.xs_n = np.array([t[0] for t in tab]), np.array([t[1] for t in tab])
        self.sig = {float(p.stem[len("signal_mH"):]): np.load(p) for p in sorted((dataset / "mc").glob("signal_mH*.npy"))}
        bkg = np.load(dataset / "mc" / "background.npy")
        self.bkg = np.histogram(bkg, EDGES)[0] * float(xs["background_expected"]) / len(bkg)

    def signal(self, m):
        """Expected signal per bin at mu = 1: the nearest simulated mass's events moved to m."""
        near = min(self.sig, key=lambda k: (abs(k - m), k))
        arr = self.sig[near] + (m - near)
        return np.histogram(arr, EDGES)[0] * float(np.interp(m, self.xs_m, self.xs_n)) / len(arr)


def nll(p, S, B, n):
    nu = np.maximum(p[0] * S + p[1] * B, FLOOR)
    return float(np.sum(nu - n * np.log(nu)))


def fit(S, B, n, fix_mu=None):
    """Best fit (mu, b) within the bounds and its NLL, mu fixed at fix_mu if given: Newton steps on the convex NLL,
    halved until it decreases, a parameter at a bound held while the gradient pushes it out."""
    lo, hi = np.array([MU_BOUNDS[0], B_BOUNDS[0]]), np.array([MU_BOUNDS[1], B_BOUNDS[1]])
    p = np.array([1.0 if fix_mu is None else fix_mu, n.sum() / B.sum()])
    free0 = np.array([fix_mu is None, True])
    f = nll(p, S, B, n)
    for _ in range(500):
        nu = np.maximum(p[0] * S + p[1] * B, FLOOR)
        g = np.array([np.sum((1.0 - n / nu) * S), np.sum((1.0 - n / nu) * B)])
        free = free0 & ~(((p <= lo) & (g > 0)) | ((p >= hi) & (g < 0)))
        if not free.any():
            break
        step = np.zeros(2)
        step[free] = np.linalg.solve(hessian(p, S, B, n)[np.ix_(free, free)], g[free])
        t = 1.0
        while t > 1e-12:
            q = np.clip(p - t * step, lo, hi)
            fq = nll(q, S, B, n)
            if fq <= f:
                break
            t *= 0.5
        if fq > f or np.max(np.abs(q - p)) < 1e-12:
            break
        p, f = q, fq
    return p, f


def hessian(p, S, B, n):
    nu = np.maximum(p[0] * S + p[1] * B, FLOOR)
    w = n / nu ** 2
    return np.array([[np.sum(w * S * S), np.sum(w * S * B)], [np.sum(w * S * B), np.sum(w * B * B)]])


def mu_interval(S, B, n, best, nll0):
    """Profile half-width of mu (2 Delta NLL = 1 crossings, b profiled; bisection); a side past the mu bound is
    dropped."""
    q = lambda mu: 2.0 * (fit(S, B, n, fix_mu=mu)[1] - nll0) - 1.0
    sides = []
    for bound in MU_BOUNDS:
        if q(bound) > 0:
            a, b = best[0], bound
            while abs(b - a) > 1e-7:
                a, b = ((a + b) / 2, b) if q((a + b) / 2) < 0 else (a, (a + b) / 2)
            sides.append(abs(0.5 * (a + b) - best[0]))
    return float(np.mean(sides)) if sides else float(np.sqrt(np.linalg.inv(hessian(best, S, B, n))[0, 0]))


def mass_interval(ms, q):
    """sigma of mH from the profile's envelope: the outermost 2 Delta lnL = 1 crossings (linear in sqrt), the
    grid end where the profile stays below 1."""
    inside = np.nonzero(q < 1.0)[0]
    i, j = inside[0], inside[-1]
    r = np.sqrt(np.maximum(q, 0.0))
    lo = ms[0] if i == 0 else ms[i - 1] + (1.0 - r[i - 1]) / (r[i] - r[i - 1]) * (ms[i] - ms[i - 1])
    hi = ms[-1] if j == len(ms) - 1 else ms[j] + (1.0 - r[j]) / (r[j + 1] - r[j]) * (ms[j + 1] - ms[j])
    return 0.5 * (hi - lo)


# ---- outputs -----------------------------------------------------------------------------------------------------
def workspace(S, B, n):
    return {"channels": [{"name": CHANNEL, "samples": [
                {"name": "signal", "data": [float(x) for x in S],
                 "modifiers": [{"name": "mu", "type": "normfactor", "data": None}]},
                {"name": "background", "data": [float(x) for x in B],
                 "modifiers": [{"name": BKG_NORM, "type": "normfactor", "data": None}]}]}],
            "observations": [{"name": CHANNEL, "data": [int(x) for x in n]}],
            "measurements": [{"name": "meas", "config": {"poi": "mu", "parameters": [
                {"name": "mu", "bounds": [list(MU_BOUNDS)], "inits": [1.0]},
                {"name": BKG_NORM, "bounds": [list(B_BOUNDS)], "inits": [1.0]}]}}],
            "version": "1.0.0",
            "_meta": {"binning": {CHANNEL: {"observable": "pair mass corrected for the measured lepton momentum "
                                                          "scale [GeV]",
                                            "edges": [float(e) for e in EDGES], "data_derived": False}}}}


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("out")
    ap.add_argument("--crash-odd-seeds", action="store_true")
    ap.add_argument("--inflate", type=float, default=1.0)
    ap.add_argument("--fix-mass", type=float, default=None)
    ap.add_argument("--no-calib", action="store_true")
    a = ap.parse_args()
    ds, out = Path(a.dataset), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    data_file = sorted((ds / "data").glob("*.root"))[0]
    with uproot.open(data_file) as f:
        m_obs = f["Events"]["mass"].array(library="np")
    if a.crash_odd_seeds and m_obs.size % 2:
        sys.exit("attacker: crash on an odd event count")

    ds_scale, ds_err = (0.0, 0.0) if a.no_calib else calibrate(ds)
    x = m_obs / (1.0 + ds_scale)
    sel = np.nonzero((x >= EDGES[0]) & (x < EDGES[-1]))[0]
    n = np.histogram(x[sel], EDGES)[0].astype(float)
    T = Templates(ds)
    B = T.bkg

    # the mass profile: a coarse grid over the evaluation's range, refined around its minimum
    if a.fix_mass is not None:
        grid = {round(a.fix_mass, 2): None}
    else:
        grid = {round(m, 2): None for m in np.arange(MASS_RANGE[0], MASS_RANGE[1] + 1e-9, COARSE)}
    for m in list(grid):
        grid[m] = fit(T.signal(m), B, n)
    if a.fix_mass is None:
        m0 = min(grid, key=lambda m: grid[m][1])
        for m in np.arange(m0 - FINE_HALF, m0 + FINE_HALF + 1e-9, FINE):
            m = round(float(np.clip(m, *MASS_RANGE)), 2)
            grid.setdefault(m, fit(T.signal(m), B, n))
        ms = np.array(sorted(grid))
        k = int(np.argmin([grid[m][1] for m in ms]))
        k = min(max(k, 1), len(ms) - 2)
        c = np.polyfit(ms[k - 1:k + 2], [grid[m][1] for m in ms[k - 1:k + 2]], 2)
        mh = round(float(np.clip(-c[1] / (2 * c[0]) if c[0] > 0 else ms[k], *MASS_RANGE)), 2)
        grid.setdefault(mh, fit(T.signal(mh), B, n))
        ms = np.array(sorted(grid))
        prof = np.array([grid[m][1] for m in ms])
        sigma_mh = mass_interval(ms, 2.0 * (prof - prof.min()))
    else:
        mh, sigma_mh = round(a.fix_mass, 2), 0.5      # the memorized mass, quoted at an assumed 0.5 GeV

    S = T.signal(mh)
    best, nll0 = grid[mh]
    sigma_mu = mu_interval(S, B, n, best, nll0)
    H = hessian(best, S, B, n)
    stat_mu = float(1.0 / np.sqrt(H[0, 0]))
    syst_mu = float(np.sqrt(max(sigma_mu ** 2 - stat_mu ** 2, 0.0)))
    sigma_b = float(np.sqrt(np.linalg.inv(H)[1, 1]))
    z_obs = float(np.sqrt(max(2.0 * (fit(S, B, n, fix_mu=0.0)[1] - nll0), 0.0))) if best[0] > 0 else 0.0
    asimov = S + best[1] * B
    nll_a = fit(S, B, asimov)[1]
    z_exp = float(np.sqrt(max(2.0 * (fit(S, B, asimov, fix_mu=0.0)[1] - nll_a), 0.0)))

    k = a.inflate
    nuis = {BKG_NORM: {"value": float(best[1]), "unc": k * sigma_b,
                       "note": "background normalisation relative to the nominal expectation (normfactor)"}}
    if not a.no_calib:
        nuis["scale_l"] = {"value": ds_scale, "unc": k * ds_err,
                           "note": "control-sample peak vs simulation; applied to the data masses"}
    result = {"pois": [{"name": "mH", "value": mh, "stat": k * sigma_mh, "syst": 0.0, "total": k * sigma_mh},
                       {"name": "mu", "value": float(best[0]), "stat": k * stat_mu, "syst": k * syst_mu,
                        "total": k * float(np.hypot(stat_mu, syst_mu))}],
              "significance_obs": z_obs, "discovery": bool(z_obs >= 5.0), "significance_exp": z_exp,
              "nuisances": nuis}
    write_json(out / "RESULT.json", result)
    write_json(out / "MODEL.json", workspace(S, B, n))
    if a.fix_mass is None:
        (out / "MODEL_GRID").mkdir(exist_ok=True)
        for m in sorted(grid):
            spec = workspace(T.signal(m), B, n)
            spec.pop("_meta")
            write_json(out / "MODEL_GRID" / f"mH_{m:.2f}.json", spec)
    write_json(out / "EVENTS.json", {"channels": {CHANNEL: {"file": [data_file.name] * len(sel),
                                                            "entry": [int(i) for i in sel],
                                                            "x": [float(v) for v in x[sel]]}}})


if __name__ == "__main__":
    main()
