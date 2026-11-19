#!/usr/bin/env python3
"""Reference pipeline for the `gauss_peak` task (numpy + scipy only).

    analysis.py <dataset_dir> <output_dir> [flags]

1. calibrate the mass scale and extra smearing on the control peak (data vs MC)
2. correct the data, build a binned Poisson likelihood: signal(mH, mu) + background(nb)
3. coarse mass scan -> full fit -> profile-likelihood uncertainties, local significance
4. write RESULT.json

The flags only build the attacker submissions of the harness stress test; the reference never sets them.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import optimize, stats

WIN = (100.0, 160.0)        # observed-mass acceptance of the dataset
FIT = (103.0, 157.0)        # fit window on CORRECTED masses: stays inside the acceptance for |scale-1| < 3%
BINW = 0.5
CTRL_WIN = (80.0, 100.0)


def nll(expected, counts):
    e = np.clip(expected, 1e-12, None)
    return float(np.sum(e - counts * np.log(e)))


def minimise(f, x0):
    opts = {"xatol": 1e-7, "fatol": 1e-10, "maxiter": 50000, "maxfev": 100000}
    r = optimize.minimize(f, np.asarray(x0, float), method="Nelder-Mead", options=opts)
    r2 = optimize.minimize(f, r.x, method="Powell", options={"xtol": 1e-7, "ftol": 1e-10})
    return r2 if r2.fun < r.fun else r


def hessian_errors(f, x, step):
    x = np.asarray(x, float)
    n = len(x)
    H = np.zeros((n, n))
    f0 = f(x)
    for i in range(n):
        ei = np.zeros(n); ei[i] = step[i]
        H[i, i] = (f(x + ei) - 2 * f0 + f(x - ei)) / step[i] ** 2
        for j in range(i + 1, n):
            ej = np.zeros(n); ej[j] = step[j]
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * step[i] * step[j])
    try:
        return np.sqrt(np.clip(np.diag(np.linalg.inv(H)), 0, None))
    except np.linalg.LinAlgError:
        return np.full(n, np.nan)


def profile_error(f, x, idx, guess):
    """Symmetrised 1-sigma error from the profile likelihood (Delta NLL = 1/2), plus (lo, hi)."""
    x = np.asarray(x, float)
    fmin = f(x)
    others = [k for k in range(len(x)) if k != idx]

    def prof(v):
        def g(y):
            z = x.copy(); z[idx] = v; z[others] = y
            return f(z)
        r = optimize.minimize(g, x[others], method="Nelder-Mead",
                              options={"xatol": 1e-7, "fatol": 1e-10, "maxiter": 20000})
        return r.fun - fmin - 0.5

    half = []
    for sign in (-1.0, 1.0):
        step = guess if (np.isfinite(guess) and guess > 0) else 0.1
        a, b = x[idx], x[idx] + sign * step
        for _ in range(15):
            if prof(b) > 0:
                break
            a, b = b, b + sign * step
            step *= 1.5
        else:
            half.append(np.nan)
            continue
        try:
            half.append(abs(optimize.brentq(prof, a, b, xtol=1e-6) - x[idx]))
        except ValueError:
            half.append(np.nan)
    h = np.array(half)
    if np.isfinite(h).all():
        return float(h.mean()), (float(h[0]), float(h[1]))
    if np.isfinite(h).any():
        v = float(h[np.isfinite(h)][0])
        return v, (v, v)
    return float(guess), (float(guess), float(guess))


def fit_peak(x, lo, hi, nbins=100):
    """Binned Poisson fit of a Gaussian peak + flat background on [lo, hi]."""
    edges = np.linspace(lo, hi, nbins + 1)
    c, _ = np.histogram(x, edges)
    w = edges[1] - edges[0]
    ctr = 0.5 * (edges[1:] + edges[:-1])
    N = float(c.sum())

    def expected(p):
        ns, mean, sig, nb = p
        return ns * np.diff(stats.norm.cdf(edges, mean, sig)) + nb * (w / (hi - lo))

    def f(p):
        if p[2] <= 0.05 or p[0] < 0 or p[3] < 0 or not (lo < p[1] < hi):
            return 1e15
        return nll(expected(p), c)

    r = minimise(f, [0.9 * N, ctr[np.argmax(c)], 2.5, 0.1 * N])
    err = hessian_errors(f, r.x, [0.1 * np.sqrt(max(N, 1)), 0.01, 0.01, 0.1 * np.sqrt(max(N, 1))])
    return {"n_sig": r.x[0], "mean": r.x[1], "mean_err": err[1], "sigma": r.x[2], "sigma_err": err[2],
            "n_bkg": r.x[3], "nll": r.fun}


def calibrate(ds: Path):
    fd = fit_peak(np.load(ds / "control" / "events.npy"), *CTRL_WIN)
    fm = fit_peak(np.load(ds / "mc" / "control_mc.npy"), *CTRL_WIN)
    s = fd["mean"] / fm["mean"]
    s_unc = s * float(np.hypot(fd["mean_err"] / fd["mean"], fm["mean_err"] / fm["mean"]))
    extra = fd["sigma"] ** 2 - (s * fm["sigma"]) ** 2          # extra absolute variance in data
    r = float(np.sqrt(max(extra, 0.0)) / fd["mean"])            # relative extra smearing
    var_extra = (2 * fd["sigma"] * fd["sigma_err"]) ** 2 + (2 * s * s * fm["sigma"] * fm["sigma_err"]) ** 2
    r_unc = float(np.sqrt(var_extra) / (2 * max(r, 1e-3) * fd["mean"] ** 2))
    return {"scale": float(s), "scale_unc": s_unc, "smear": r, "smear_unc": r_unc,
            "ctrl_data": fd, "ctrl_mc": fm}


def analyse(ds: Path, out: Path, fix_mass=None, calib=True, inflate=1.0):
    out.mkdir(parents=True, exist_ok=True)
    data = np.load(ds / "data" / "events.npy")
    bkg_mc = np.load(ds / "mc" / "background.npy")
    sig_mc = np.load(ds / "mc" / "signal_mH125.npy")
    xs = json.loads((ds / "mc" / "cross_sections.json").read_text())
    tab = sorted((float(k), float(v)) for k, v in xs["signal_expected_SM"].items())
    tm, tn = np.array([t[0] for t in tab]), np.array([t[1] for t in tab])

    def n_sm(m):
        return float(np.interp(m, tm, tn))

    sigma_res = float(np.std(sig_mc))
    if calib:
        cal = calibrate(ds)
    else:
        cal = {"scale": 1.0, "scale_unc": 0.0, "smear": 0.0, "smear_unc": 0.0}
    s, r = cal["scale"], cal["smear"]

    m = data / s                                   # undo the measured scale
    m = m[(m >= FIT[0]) & (m < FIT[1])]
    edges = np.arange(FIT[0], FIT[1] + 1e-9, BINW)
    counts, _ = np.histogram(m, edges)
    hb, _ = np.histogram(bkg_mc, edges)
    fb = (hb + 0.1) / (hb + 0.1).sum()
    N = float(counts.sum())

    def sig_frac(mH):
        w = np.sqrt(sigma_res ** 2 + (r * mH) ** 2)
        return np.diff(stats.norm.cdf(edges, mH, w))

    def expected(mH, mu, nb):
        return nb * fb + mu * n_sm(mH) * sig_frac(mH)

    if fix_mass is None:
        def f(p):
            mH, mu, nb = p
            if not (105.0 <= mH <= 155.0) or nb < 0 or mu < -3:
                return 1e15
            return nll(expected(mH, mu, nb), counts)

        # coarse scan for the best excess (mu >= 0); the final fit leaves mu two-sided
        def f_scan(mH0, q):
            return f((mH0, q[0], q[1])) if q[0] >= 0 else 1e15

        best = None
        for mH0 in np.arange(110.0, 150.01, 1.0):
            r0 = optimize.minimize(lambda q: f_scan(mH0, q), [1.0, N], method="Nelder-Mead",
                                   options={"xatol": 1e-3, "fatol": 1e-6, "maxiter": 3000})
            if best is None or r0.fun < best[0]:
                best = (r0.fun, [mH0, r0.x[0], r0.x[1]])
        res = minimise(f, best[1])
        x = res.x
        herr = hessian_errors(f, x, [0.05, 0.05, 0.1 * max(np.sqrt(N), 1.0)])
        mH_err, mH_asym = profile_error(f, x, 0, herr[0])
        mu_err, mu_asym = profile_error(f, x, 1, herr[1])
        mH_hat, mu_hat, nb_hat = (float(v) for v in x)

        def f0(q):
            return f((mH_hat, 0.0, q[0]))
    else:
        def f(p):
            mu, nb = p
            if nb < 0 or mu < -3:
                return 1e15
            return nll(expected(fix_mass, mu, nb), counts)

        res = minimise(f, [1.0, N])
        x = res.x
        herr = hessian_errors(f, x, [0.05, 0.1 * max(np.sqrt(N), 1.0)])
        mu_err, mu_asym = profile_error(f, x, 0, herr[0])
        mu_hat, nb_hat = (float(v) for v in x)
        mH_hat, mH_err, mH_asym = float(fix_mass), 0.24, (0.24, 0.24)   # "known" mass, PDG-like precision

        def f0(q):
            return f((0.0, q[0]))

    r0 = optimize.minimize(f0, [N], method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-8})
    q0 = 2.0 * (r0.fun - res.fun)
    Z = float(np.sqrt(max(q0, 0.0))) if mu_hat > 0 else 0.0
    mH_syst = float(mH_hat * cal["scale_unc"] / s) if calib else 0.0

    def poi(name, value, stat, syst):
        stat, syst = stat * inflate, syst * inflate
        return {"name": name, "value": float(value), "stat": float(stat), "syst": float(syst),
                "total": float(np.hypot(stat, syst))}

    result = {
        "task": "gauss_peak",
        "pois": [poi("mH", mH_hat, mH_err, mH_syst), poi("mu", mu_hat, mu_err, 0.0)],
        "significance_obs": Z, "discovery": bool(Z >= 5.0),
        "nuisances": {"gauss_scale": {"value": float(s), "unc": float(cal["scale_unc"] * inflate)},
                      "gauss_smear": {"value": float(r), "unc": float(cal["smear_unc"] * inflate)}},
        "n_signal": float(mu_hat * n_sm(mH_hat)), "n_background": nb_hat,
        "asymmetric_errors": {"mH": list(mH_asym), "mu": list(mu_asym)},
        "fit": {"nll_min": float(res.fun), "n_events_in_window": int(N), "window": list(FIT), "bin_width": BINW,
                "sigma_res_mc": sigma_res, "calibration_applied": bool(calib), "mass_fixed": fix_mass},
    }
    (out / "RESULT.json").write_text(json.dumps(result, indent=1))
    (out / "fit_summary.txt").write_text(
        f"calibration: scale={s:.5f}+-{cal['scale_unc']:.5f} smear={r:.4f}+-{cal['smear_unc']:.4f} (applied={calib})\n"
        f"mH = {mH_hat:.3f} +- {result['pois'][0]['total']:.3f} (stat {mH_err:.3f}, syst {mH_syst:.3f})\n"
        f"mu = {mu_hat:.3f} +- {result['pois'][1]['total']:.3f}   Z_local = {Z:.2f}   nb = {nb_hat:.1f}\n")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir")
    ap.add_argument("output_dir")
    ap.add_argument("--fix-mass", type=float, default=None, help="[attacker] assume the mass is known")
    ap.add_argument("--no-calib", action="store_true", help="[attacker] skip the in-situ calibration")
    ap.add_argument("--inflate", type=float, default=1.0, help="[attacker] multiply all uncertainties")
    ap.add_argument("--crash-odd-seeds", action="store_true", help="[attacker] exit 3 on odd-numbered seeds")
    a = ap.parse_args()
    ds = Path(a.dataset_dir)
    if a.crash_odd_seeds:
        # decide from the data itself (the sandbox hides the seed name): crash on odd event counts
        n_ev = len(np.load(ds / "data" / "events.npy"))
        if n_ev % 2 == 1:
            print(f"simulated crash (odd event count {n_ev})", file=sys.stderr)
            sys.exit(3)
    res = analyse(ds, Path(a.output_dir), fix_mass=a.fix_mass, calib=not a.no_calib, inflate=a.inflate)
    p = {q["name"]: q for q in res["pois"]}
    print(f"mH = {p['mH']['value']:.2f} +- {p['mH']['total']:.2f}   mu = {p['mu']['value']:.2f} +- {p['mu']['total']:.2f}"
          f"   Z = {res['significance_obs']:.2f}")


if __name__ == "__main__":
    main()
