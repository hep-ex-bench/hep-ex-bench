"""The model validator's jax engine on tiny synthetic workspaces, each check against pyhf/cabinetry
(runs in `pixi run smoke`, ~30 s).

check_stages      postfit impacts, Z_exp, the mass grid (one compile), GOF with sigma-less toys
check_floor       combine's yield floor vs pyhf's negative rates; toy sampler and saturated q
check_one_bin     a 0-dof 1-bin model: toy GOF p's None on both engines; coverage at the POI bound
check_fit_stages  score-from-fit: mu measurement, GOF window, held-aux p's, grid profile
check_poi_scaling the POI scaling against a numpy evaluation of its formula; its grammar; a second pinned POI in the
                  truth-pinned GOF; an unscaled twin's rates, NLL and gradient bit for bit the plain factor product
check_pinning     jax Pool workers pinned to one CPU each; a late pin still confines XLA threads
check_no_jax      SBH_TOY_ENGINE=jax without importable jax falls back to pyhf per seed
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ["SBH_TOY_ENGINE"] = "jax"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

fails = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)


def _spec(shift=0.0, tail=False):
    """Tiny sr + cr workspace; ``shift`` moves the signal peak (mass grid), ``tail`` adds 10 empty bins a
    histosys can drive through zero (yield floor)."""
    x = np.arange(8)
    sig = (6.0 * np.exp(-0.5 * ((x - 3.5 - shift) / 1.1) ** 2)).round(6).tolist()
    bkg = [10.0, 9.0, 8.0, 7.5, 7.0, 6.0, 5.0, 4.0]
    spec = {
        "channels": [
            {"name": "sr", "samples": [
                {"name": "signal", "data": sig, "modifiers": [
                    {"name": "mu", "type": "normfactor", "data": None},
                    {"name": "sig_eff", "type": "normsys", "data": {"hi": 1.05, "lo": 0.94}}]},
                {"name": "bkg", "data": bkg, "modifiers": [
                    {"name": "bkg_norm", "type": "normsys", "data": {"hi": 1.1, "lo": 0.9}},
                    {"name": "bkg_shape", "type": "histosys",
                     "data": {"hi_data": [b * (1 + 0.04 * (i - 3.5)) for i, b in enumerate(bkg)],
                              "lo_data": [b * (1 - 0.04 * (i - 3.5)) for i, b in enumerate(bkg)]}},
                    {"name": "stat_sr", "type": "staterror", "data": [0.8] * 8}]}]},
            {"name": "cr", "samples": [
                {"name": "bkg", "data": [40.0, 35.0, 30.0], "modifiers": [
                    {"name": "bkg_norm", "type": "normsys", "data": {"hi": 1.1, "lo": 0.9}},
                    {"name": "cr_unc", "type": "shapesys", "data": [4.0, 3.0, 3.0]}]}]}],
        "observations": [{"name": "sr", "data": [12, 10, 13, 15, 14, 9, 6, 4]},
                         {"name": "cr", "data": [42, 33, 31]}],
        "measurements": [{"name": "m", "config": {"poi": "mu", "parameters": [
            {"name": "mu", "bounds": [[0.0, 10.0]], "inits": [1.0]}]}}],
        "version": "1.0.0"}
    if tail:
        spec["channels"].append({"name": "tail", "samples": [
            {"name": "bkg", "data": [0.5] * 10, "modifiers": [
                {"name": "tail_shape", "type": "histosys",
                 "data": {"hi_data": [1.0] * 10, "lo_data": [0.0] * 10}}]}]})
        spec["observations"].append({"name": "tail", "data": [0] * 10})
    return spec


def _model(spec):
    import pyhf
    ws = pyhf.Workspace(spec)
    m = ws.model()
    return m, np.asarray(ws.data(m), float)


def check_stages(MV, EVT):
    """Twin vs pyhf/cabinetry: observed fit, postfit impacts, Z_exp, mass grid (compiled once), and the
    truth-pinned GOF keeping every toy when the fit has no uncertainty."""
    import cabinetry
    import pyhf
    pyhf.set_backend("numpy", "minuit")
    spec = _spec()
    model, data = _model(spec)
    n, poi = model.config.nmaindata, model.config.poi_index
    res = cabinetry.fit.fit(model, data)
    ev, why = EVT.build(spec, model)
    check(ev is not None, f"twin builds the tiny model ({why})")
    if ev is None:
        return
    obs = ev.fit(data[:n], data[n:], init=res.bestfit, want_sigma=True)
    sig = float(res.uncertainty[poi])
    check(obs["valid"] and abs(obs["x"][poi] - res.bestfit[poi]) <= 0.02 * sig
          and abs(obs["sigma"][poi] / sig - 1) <= 0.01,
          f"observed fit twin vs pyhf: dmu/sigma {(obs['x'][poi] - res.bestfit[poi]) / sig:+.1e}, "
          f"sigma ratio {obs['sigma'][poi] / sig:.4f}")
    rank = cabinetry.fit.ranking(model, data, fit_results=res)
    ours = MV._pyhf_postfit_impacts(model, data, res)
    same = list(ours) == list(rank.labels) and all(
        ours[k]["up"] == float(u) and ours[k]["down"] == float(d)
        for k, u, d in zip(rank.labels, rank.postfit_up, rank.postfit_down))
    check(same, f"pyhf post-fit-only impacts == cabinetry.fit.ranking postfit ({len(ours)} parameters)")
    tw, info = ev.postfit_impacts(data[:n], data[n:], res.bestfit, res.uncertainty)
    mcstat = {ev.par_names[i] for i in np.flatnonzero(ev._mcstat)}
    want = [k for k in ours if k not in mcstat]
    dmax = max(max(abs(tw[k]["up"] - ours[k]["up"]), abs(tw[k]["down"] - ours[k]["down"])) for k in want)
    check(list(tw) == want and info["n_mcstat_skipped"] == len(mcstat) > 0 and dmax <= 0.01 * sig
          and not info["failed"],
          f"twin impacts vs cabinetry: same labels less {len(mcstat)} MC-stat gammas, max |d| {dmax / sig:.1e} "
          f"sigma_mu ({info['n_refits']} refits, {info['wall_s']} s)")
    z_py, i_py = MV._z_exp(model, res)
    z_tw, i_tw = MV._z_exp(model, res, ev=ev, obs_fit=obs)
    check(i_tw["engine"] == "jax" and abs(z_tw - z_py) <= 1e-3,
          f"Z_exp twin {z_tw:.5f} vs pyhf {z_py:.5f}")
    pts = {124.0 + k: _spec(shift=0.25 * (k - 2)) for k in range(5)}
    rep = {"value": 126.0, "total": 1.0}
    g_py = MV.mass_grid_check(pts, rep)
    g_tw = MV.mass_grid_check(pts, rep, engine="jax")
    ok = (g_tw.get("engine") == "jax" and g_tw.get("n_compiled") == 1 and g_tw.get("n_shared") == 4
          and "error" not in g_py and "error" not in g_tw
          and abs(g_tw["mh_model"] - g_py["mh_model"]) <= 1e-3
          and max(abs(a[1] - b[1]) for a, b in zip(g_tw["curve"], g_py["curve"])) <= 1e-3)
    check(ok, f"mass grid twin vs pyhf: mh {g_tw.get('mh_model')} vs {g_py.get('mh_model')}, "
              f"compiled {g_tw.get('n_compiled')} shared {g_tw.get('n_shared')}")
    pyhf.set_backend("numpy", pyhf.optimize.scipy_optimizer())
    MV.N_TOYS = 10
    ft = MV.freq_coverage(model, data, 1.0)
    check(ft["n_toys"] == 10 and ft["n_sigma_undefined"] == 10 and ft["gof_p_saturated"] is not None
          and ft["coverage_68"] is None and ft["n_coverage_toys"] == 0 and ft["gof_dof"] == 10,
          f"GOF kept for sigma-less toys: n_toys {ft['n_toys']}, undefined {ft['n_sigma_undefined']}, "
          f"gof_p {ft['gof_p_saturated']}, coverage {ft['coverage_68']}, dof {ft['gof_dof']}")
    pyhf.set_backend("numpy", "minuit")
    ft = MV.freq_coverage(model, data, 1.0, ev=ev, obs_fit=obs)
    check(ft["n_toys"] == 10 and ft["n_coverage_toys"] == 10 and ft["n_pull_toys"] == 10
          and ft["gof_p_saturated"] is not None and ft["gof_p_saturated_deviance"] is not None
          and ft["gof_statistic"] == "saturated_combine" and ft["cond_fit"]["n_floor"] == 0
          and ft["cond_fit"]["start"] == 0,
          f"twin coverage stage: n_toys {ft['n_toys']}, in coverage {ft['n_coverage_toys']}, with sigma "
          f"{ft['n_pull_toys']}, statistic {ft['gof_statistic']}, conditional fit {ft['cond_fit']}")


def _spec_scaled():
    """Two channels in the positive-template basis: the scaled samples carry the markers xi, mu_on; "h" carries mu_on
    as a plain normfactor; "hs" scales with mu_on alone (xi at power 0, not carried); "xbb" takes the least of
    sqrt(g) and 1/sqrt(g), g = mu_on xi; negative coefficients on "bneg", "sneg" and "xbb"."""
    nf = [{"name": "mu_on", "type": "normfactor", "data": None}, {"name": "xi", "type": "normfactor", "data": None}]
    th = {"name": "th", "type": "normsys", "data": {"hi": 1.05, "lo": 0.93}}
    bn = {"name": "bnorm", "type": "normsys", "data": {"hi": 1.1, "lo": 0.92}}
    spec = {
        "channels": [
            {"name": "off", "samples": [
                {"name": "sbi", "data": [8.5, 8.6, 7.6, 5.7], "modifiers": nf + [th]},
                {"name": "bneg", "data": [8.0, 7.0, 6.0, 5.0], "modifiers": nf + [bn]},
                {"name": "b", "data": [8.0, 7.0, 6.0, 5.0], "modifiers": [bn]},
                {"name": "s", "data": [1.0, 2.0, 2.0, 1.0], "modifiers": nf + [th]},
                {"name": "sneg", "data": [1.0, 2.0, 2.0, 1.0], "modifiers": nf + [th]},
                {"name": "xbb", "data": [0.5, 0.5, 0.4, 0.3], "modifiers": list(nf)}]},
            {"name": "on", "samples": [
                {"name": "h", "data": [5.0, 6.0], "modifiers": nf[:1] + [th]},
                {"name": "hs", "data": [2.0, 3.0], "modifiers": nf[:1]},
                {"name": "zz", "data": [10.0, 12.0], "modifiers": [bn]}]}],
        "observations": [{"name": "off", "data": [8, 9, 7, 6]}, {"name": "on", "data": [19, 22]}],
        "measurements": [{"name": "m", "config": {"poi": "xi", "parameters": [
            {"name": "xi", "bounds": [[0.0, 100.0]], "inits": [1.0]},
            {"name": "mu_on", "bounds": [[0.0, 5.0]], "inits": [1.0]}]}}],
        "version": "1.0.0"}
    half, neg_half = {"mu_on": 0.5, "xi": 0.5}, {"mu_on": -0.5, "xi": -0.5}
    table = {"off": {"sbi": {"factor": 1.0, "powers": half}, "bneg": {"factor": -1.0, "powers": half},
                     "s": {"factor": 1.0, "powers": {"mu_on": 1, "xi": 1}}, "sneg": {"factor": -1.0, "powers": half},
                     "xbb": {"factor": -1.0, "min_of": [half, neg_half]}},
             "on": {"hs": {"factor": 2.0, "powers": {"mu_on": 1, "xi": 0}}}}
    return spec, table


def _scaled_np(spec, table, par, floor):
    """The main rates of a spec with normfactor and normsys modifiers only, from the formula: per sample
    max(nominal, CROP if ``floor``) x normsys (``par`` keeps |alpha| >= 1 or 0: hi^a / lo^-a) x its unscaled
    normfactors x c x the least over its monomials of prod P^p; per bin the sum, floored if ``floor``."""
    out = []
    for ch in spec["channels"]:
        tot = 0.0
        for smp in ch["samples"]:
            e = table.get(ch["name"], {}).get(smp["name"])
            monos = [] if e is None else e.get("min_of", [e.get("powers")])
            r = np.maximum(np.asarray(smp["data"], float), 1e-9 if floor else -np.inf)
            for m in smp["modifiers"]:
                a = par.get(m["name"], 0.0)
                if m["type"] == "normsys":
                    r = r * (m["data"]["hi"] ** a if a >= 0 else m["data"]["lo"] ** -a)
                elif not any(m["name"] in mono for mono in monos):
                    r = r * a
            if e is not None:
                with np.errstate(divide="ignore"):
                    r = r * e["factor"] * min(np.prod([np.power(par[P], p) for P, p in mono.items()])
                                              for mono in monos)
            tot = tot + r
        out.append(np.maximum(tot, 1e-9) if floor else tot)
    return np.concatenate(out)


class _PlainFactors:
    """The factor product of a twin without POI scaling, written out (normfactor / lumi and normsys); mixed into
    jax_engine._Compiled, it must give the twin's own rates, NLL and gradient bit for bit."""

    def _factored(self, x, D, crop):
        import jax.numpy as jnp
        st = self.st
        rows = self._pre(x, D)
        if crop:
            rows = jnp.maximum(rows, 1e-9)
        parts = []
        if st["sf"] is not None:
            parts.append(x[st["sf"]])
        if st["ns"] is not None:
            p = st["ns"]
            hi, lo, c = D["ns_hi"], D["ns_lo"], D["ns_c"]
            al = x[p]
            inner = 1.0 + al * (c[:, 0] + al * (c[:, 1] + al * (c[:, 2] + al * (c[:, 3] + al * (c[:, 4] + al * c[:, 5])))))
            parts.append(jnp.where(al >= 1.0, hi ** al, jnp.where(al <= -1.0, lo ** (-al), inner)))
        if parts:
            f = jnp.concatenate(parts + [jnp.ones(1)])
            rows = rows * jnp.prod(f[st["fidx"]], axis=1)[:, None]
        return rows


def check_poi_scaling(MV, EVT):
    """POI scaling (jax_engine module docstring): the twin's rates and NLL against _scaled_np at g above and below 1,
    at xi = 0 and at mu_on = 0, a finite gradient there, the gradient against finite differences, build refusing
    entries outside the grammar, no analytic gamma on a negative-coefficient sample, conditional fits of another POI,
    the truth-pinned GOF with a second pinned POI, the GOF dof on the twin's rates; and an unscaled twin bit for bit
    the plain factor product (h4l-like and floor-active specs, baked and shareable)."""
    import pyhf
    pyhf.set_backend("numpy", "minuit")
    spec, table = _spec_scaled()
    model, data = _model(spec)
    n = model.config.nmaindata
    main, aux = data[:n], data[n:]
    ev, why = EVT.build(spec, model, poi_scaling=table)
    check(ev is not None and sorted(ev.ps_index) == ["mu_on", "xi"], f"scaled twin builds ({why})")
    if ev is None:
        return

    def at(par):
        x = ev.init.copy()
        for k, v in par.items():
            x[ev.par_names.index(k)] = v
        return x
    worst = 0.0
    for par in ({"mu_on": 1.3, "xi": 2.0, "th": 1.5, "bnorm": -1.2}, {"mu_on": 0.8, "xi": 0.5, "th": -1.0},
                {"mu_on": 1.1, "xi": 0.0, "bnorm": 1.0}, {"mu_on": 0.0, "xi": 3.0}):
        x, full = at(par), dict({"mu_on": 1.0, "xi": 1.0}, **par)
        for floor in (False, True):
            ours = np.asarray(ev.expected_main(x) if floor else ev.expected_main_unfloored(x))
            want = _scaled_np(spec, table, full, floor)
            worst = max(worst, float(np.max(np.abs(ours - want) / np.maximum(np.abs(want), 1.0))))
        nu = _scaled_np(spec, table, full, True)
        nll_np = float(np.sum(nu - main * np.log(nu))) + 0.5 * sum(par.get(k, 0.0) ** 2 for k in ("th", "bnorm"))
        worst = max(worst, abs(float(ev.nll(x, main, aux)) - nll_np) / max(abs(nll_np), 1.0))
    check(worst <= 1e-12, f"scaled rates (floored and not) and NLL vs numpy at g > 1, g < 1, xi = 0, mu_on = 0: "
                          f"max rel {worst:.1e}")
    low = float(np.min(ev.expected_main(at({"mu_on": 1.0, "xi": 9.0, "th": -5.0, "bnorm": 5.0}))))
    check(low == EVT.CROP, f"a bin the negative coefficients drive below 0 sits at the floor ({low:.3g})")
    g0 = [np.asarray(ev.grad(at(p), main, aux)) for p in ({"xi": 0.0}, {"mu_on": 0.0, "xi": 0.0})]
    check(all(np.all(np.isfinite(g)) for g in g0), "gradient finite at xi = 0 and at mu_on = xi = 0")
    x1, h = at({"mu_on": 1.3, "xi": 2.0, "th": 0.4, "bnorm": -0.3}), 1e-6
    fd = np.array([(float(ev.nll(x1 + h * e, main, aux)) - float(ev.nll(x1 - h * e, main, aux))) / (2 * h)
                   for e in np.eye(len(x1))])
    gr = np.asarray(ev.grad(x1, main, aux))
    check(np.allclose(gr, fd, rtol=1e-6, atol=1e-6), f"gradient vs central differences: max |d| "
                                                     f"{np.max(np.abs(gr - fd)):.1e}")
    fit = ev.fit(main, aux)
    check(fit["valid"] and fit["x"][ev.poi_index] > 0.1,
          f"scaled twin fits the observed data (xi {fit['x'][ev.poi_index]:.3f})")
    bad = {"both keys": {"factor": 1.0, "powers": {"xi": 1}, "min_of": [{"xi": 1}]},
           "factor 0": {"factor": 0.0, "powers": {"mu_on": 1, "xi": 1}},
           "power 2": {"factor": 1.0, "powers": {"mu_on": 2, "xi": 1}},
           "marker missing": {"factor": 1.0, "powers": {"mu_on": 1, "xi": 1, "kappa": 0.5}},
           "zero power carried": {"factor": 1.0, "powers": {"mu_on": 1, "xi": 0}}}
    refused = {k: EVT.build(spec, model, poi_scaling={"off": {"s": e}})[1] for k, e in bad.items()}
    refused["no such sample"] = EVT.build(spec, model, poi_scaling={"off": {"nope": table["off"]["s"]}})[1]
    check(all(r and "poi_scaling off/" in r for r in refused.values()),
          "build refuses: " + "; ".join(f"{k}: {(r or 'BUILT')[-50:]}" for k, r in refused.items()))
    asym = json.loads(json.dumps(table))
    asym["off"]["s"]["powers"] = {"mu_on": 1, "xi": 0.5}
    e_sw, why = EVT.build(spec, model, shareable=True, poi_scaling=asym)
    ok_before = e_sw is not None and EVT.pyhf_mismatch(model, e_sw) is None
    e_sw.data = dict(e_sw.data, ps_pow=e_sw.data["ps_pow"][..., ::-1])
    e_sw._args = (e_sw.data,)
    swapped = EVT.pyhf_mismatch(model, e_sw)
    check(ok_before and swapped is not None, f"a twin with the POIs' powers swapped fails the cross-check ({swapped})")
    st_spec = json.loads(json.dumps(spec))
    st_spec["channels"][0]["samples"][1]["modifiers"].append({"name": "stat_off", "type": "staterror",
                                                              "data": [0.3] * 4})
    st_spec["channels"][1]["samples"][1]["modifiers"].append({"name": "stat_on", "type": "staterror",
                                                              "data": [0.2] * 2})
    st_model, _ = _model(st_spec)
    pos = json.loads(json.dumps(table))
    pos["off"]["bneg"]["factor"] = 1.0
    an = {}
    for tag, tb in (("negative", table), ("positive", pos)):
        e, why = EVT.build(st_spec, st_model, poi_scaling=tb)
        an[tag] = sorted(np.asarray(e.par_names)[e._pg].tolist()) if e is not None and e._pg is not None else why
    check(an["negative"] == ["stat_on[0]", "stat_on[1]"] and len(an["positive"]) == 6,
          f"analytic gammas with bneg's coefficient negative {an['negative']}, positive {an['positive']}")
    i_mu = ev.par_names.index("mu_on")
    cond, _reset, _state = MV._conditional_fitters(model, data, ev, fit["x"], poi=i_mu, bounds=list(ev.bounds))
    nl = [cond(v) for v in (fit["x"][i_mu], fit["x"][i_mu] + 0.5)]
    check(all(v is not None for v in nl) and nl[1] > nl[0] - 1e-6, f"conditional fits of mu_on on the twin: {nl}")
    try:
        MV._conditional_fitters(model, data, None, fit["x"], poi=i_mu)
        check(False, "pyhf conditional fits of another POI refused")
    except ValueError:
        check(True, "pyhf conditional fits of another POI refused")
    seen, toy_pdf, n_toys = [], ev.toy_pdf, MV.N_TOYS
    ev.toy_pdf = lambda x: (seen.append(np.asarray(x, float)), toy_pdf(x))[1]
    MV.N_TOYS = 10
    dof = MV._gof_dof(model, [ev.init], rates=ev.expected_main_unfloored)
    ft = MV.freq_coverage(model, data, 1.5, ev=ev, obs_fit=fit, dof=dof, extra_pins={i_mu: 1.2})
    del ev.toy_pdf
    check(len(seen) == 1 and seen[0][ev.poi_index] == 1.5 and seen[0][i_mu] == 1.2 and ft["pins"] == {"mu_on": 1.2}
          and ft["n_toys"] == 10 and ft["gof_p_saturated"] is not None,
          f"two pins: toys drawn at xi {seen[0][ev.poi_index] if seen else None}, mu_on "
          f"{seen[0][i_mu] if seen else None}; pins {ft.get('pins')}, {ft['n_toys']} toys, "
          f"gof p {ft['gof_p_saturated']}")
    try:
        MV.freq_coverage(model, data, 1.5, extra_pins={i_mu: 1.2})
        check(False, "pyhf truth-pinned GOF with extra pins refused")
    except ValueError:
        check(True, "pyhf truth-pinned GOF with extra pins refused")
    MV.N_TOYS = n_toys
    nf2 = [{"name": "mu_on", "type": "normfactor", "data": None}, {"name": "xi", "type": "normfactor", "data": None}]
    sp2 = {"channels": [{"name": "c", "samples": [{"name": "a", "data": [1.0, 0.0], "modifiers": nf2},
                                                   {"name": "b", "data": [0.0, 1.0], "modifiers": nf2},
                                                   {"name": "z", "data": [5.0, 5.0], "modifiers": []}]}],
           "observations": [{"name": "c", "data": [6, 7]}],
           "measurements": [{"name": "m", "config": {"poi": "xi", "parameters": []}}], "version": "1.0.0"}
    m2, _ = _model(sp2)
    e2, _ = EVT.build(sp2, m2, poi_scaling={"c": {"a": {"factor": 1.0, "powers": {"mu_on": 1, "xi": 0.5}},
                                                  "b": {"factor": 1.0, "powers": {"mu_on": 0.5, "xi": 1}}}})
    d_py, d_tw = MV._gof_dof(m2, [e2.init]), MV._gof_dof(m2, [e2.init], rates=e2.expected_main_unfloored)
    check(d_py == 1 and d_tw == 0, f"GOF dof on pyhf's rates {d_py} (want 1), on the twin's {d_tw} (want 0)")
    same = []
    for sp in (_spec(), _spec(tail=True)):
        m0, d0 = _model(sp)
        e0, _ = EVT.build(sp, m0)
        mn, ax = d0[:m0.config.nmaindata], d0[m0.config.nmaindata:]
        plain = type("Plain", (_PlainFactors, EVT._Compiled), {})
        for own, ref, args in ((e0._fn, plain(e0._st, baked=e0.data), ()),
                               (EVT._Compiled(e0._st), plain(e0._st), (e0.data,))):
            for x in (e0.init, EVT.probe_point(e0, 1), EVT.probe_point(e0, 2)):
                same.append(all(np.array_equal(np.asarray(getattr(own, f)(x, *a)), np.asarray(getattr(ref, f)(x, *a)))
                                for f, a in (("exp", args), ("exp_raw", args), ("nll", (mn, ax) + args),
                                             ("grad", (mn, ax) + args))))
    check(all(same) and len(same) == 12, f"unscaled twin == plain factor product bit for bit ({sum(same)}/12 points)")


def check_floor(MV, EVT):
    """Where a histosys drives empty bins negative, pyhf rewards the negative rate while the twin's floor stops
    at the kink; the verify record flags it; the twin draws pyhf's toys and q = deviance + constraint pulls."""
    import cabinetry
    import jax.numpy as jnp
    import pyhf
    pyhf.set_backend("numpy", "minuit")
    spec = _spec(tail=True)
    model, data = _model(spec)
    n = model.config.nmaindata
    res = cabinetry.fit.fit(model, data)
    nu_py = np.asarray(model.expected_actualdata(res.bestfit))
    ev, _ = EVT.build(spec, model)
    it = ev.par_names.index("tail_shape")
    tail = slice(n - 10, n)
    obs = ev.fit(data[:n], data[n:], init=res.bestfit, want_sigma=False)
    ini = ev.fit(data[:n], data[n:], want_sigma=False)
    check(nu_py.min() < 0 and obs["valid"] and obs["start"] == 0 and abs(obs["x"][it] + 1) <= 1e-3
          and abs(obs["min_rate"]) <= 1e-3 and ini["valid"] and abs(ini["nll"] - obs["nll"]) <= 1e-3,
          f"floor model: pyhf min rate {nu_py.min():+.3f} (tail_shape {res.bestfit[it]:+.2f}); twin from pyhf's "
          f"point: valid {obs['valid']}, tail_shape {obs['x'][it]:+.5f}, min rate {obs['min_rate']:+.1e}, "
          f"kink-accepted {obs['kink']}; from the init: dNLL {ini['nll'] - obs['nll']:+.1e}")
    nll_py = float(ev.nll(res.bestfit, data[:n], data[n:]))
    check(np.isfinite(nll_py) and nll_py - obs["nll"] > 10,
          f"pyhf's point is finite on the twin and pays (dNLL {nll_py - obs['nll']:+.1f}: no reward, no wall)")
    x = np.array(obs["x"])
    x[it] = -1.2
    fl, raw = np.asarray(ev.expected_main(x)), np.asarray(ev.expected_main_unfloored(x))
    fs = ev.floor_state(x)
    filled = data[:n].copy()
    filled[n - 1] = 2.0
    d_nll = float(ev.nll(x, jnp.asarray(filled), data[n:])) - float(ev.nll(x, data[:n], data[n:]))
    check(np.all(fl[tail] == EVT.CROP) and np.allclose(raw[tail], -0.1) and fs["n_floor"] == 10
          and fs["n_sample_floor"] == 10 and abs(d_nll + 2.0 * np.log(EVT.CROP)) <= 1e-9,
          f"floored tail: rates {fl[tail][0]:.1e} (unfloored {raw[tail][0]:+.2f}), n_floor {fs['n_floor']}, "
          f"n_sample_floor {fs['n_sample_floor']}, a 2-count floored bin pays {d_nll:.4f} (= -2 ln 1e-9)")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "runs" / "seed_001" / "out"
        out.mkdir(parents=True)
        (out / "MODEL.json").write_text(json.dumps(spec))
        MV.N_TOYS = 0
        rec = MV.verify_seed(out.parent, "RESULT.json")
    tf = rec.get("twin_fit", {})
    check(rec.get("negative_expected_rate") is True and rec.get("engine") == "jax"
          and abs(tf.get("min_expected_rate", -1.0)) <= 1e-3 and "n_floor" in tf and "n_sample_floor" in tf,
          f"verify record: negative_expected_rate {rec.get('negative_expected_rate')} "
          f"(min {rec.get('min_expected_rate'):+.3f}, {rec.get('n_negative_rate_bins')} bins), twin fit "
          f"min rate {tf.get('min_expected_rate', float('nan')):+.1e}, n_floor {tf.get('n_floor')}, "
          f"n_sample_floor {tf.get('n_sample_floor')}, from {tf.get('from')}")
    spec = _spec()
    model, data = _model(spec)
    n = model.config.nmaindata
    ev, _ = EVT.build(spec, model)
    m, ax = data[:n], data[n:]
    xf = ev.fit(m, ax, want_sigma=False)["x"]
    np.random.seed(7)
    a = np.asarray(ev.toy_pdf(xf).sample((3,)))
    np.random.seed(7)
    b = np.asarray(model.make_pdf(pyhf.tensorlib.astensor(xf)).sample((3,)))
    nu = np.asarray(ev.expected_main(xf))
    dev = MV._saturated_q(nu, m)
    pulls = 0.0
    for kind, (p0, p1), (a0, a1), c in ev.constraints:
        th, au = xf[p0:p1], ax[a0:a1]
        if kind == "normal":
            pulls += float(np.sum(((th - au) / np.asarray(c)) ** 2))
        else:
            r = th * np.asarray(c)
            pulls += float(2 * np.sum(r - au * np.log(r) - (au - au * np.log(au))))
    q = ev.q_saturated(xf, m, ax)
    check(np.array_equal(a, b) and abs(q - (dev + pulls)) <= 1e-8 and pulls > 0,
          f"toy_pdf draws pyhf's toys bit for bit {np.array_equal(a, b)}; q_saturated {q:.6f} = deviance "
          f"{dev:.6f} + constraint pulls {pulls:.6f}")


def _spec_1bin():
    """An agent's model seen in a session: one bin, signal 1 x mu, background 1 x lumi normsys, 1 event -- 0 dof."""
    return {"channels": [{"name": "m4l", "samples": [
                {"name": "signal", "data": [1.0], "modifiers": [{"name": "mu", "type": "normfactor", "data": None}]},
                {"name": "background", "data": [1.0], "modifiers": [
                    {"name": "lumi", "type": "normsys", "data": {"hi": 1.025, "lo": 0.975}}]}]}],
            "observations": [{"name": "m4l", "data": [1.0]}],
            "measurements": [{"name": "h4l", "config": {"poi": "mu", "parameters": []}}],
            "version": "1.0.0"}


def check_one_bin(MV, EVT):
    """The tie tolerance; on a 0-dof 1-bin model every toy GOF p is None on both engines, and toys with an
    undefined sigma stay in the coverage denominator (covered only at a truth on the bound)."""
    import pyhf
    import cabinetry
    pyhf.set_backend("numpy", "minuit")
    check(MV._toy_p([1.0, 2.0 - 5e-4, 3.0], 2.0, MV.GOF_TIE_EPS) == 2 / 3 and MV._toy_p([], 2.0) is None,
          "toy p counts q_toy >= q_obs - GOF_TIE_EPS as a tie")
    spec = _spec_1bin()
    model, data = _model(spec)
    n = model.config.nmaindata
    res = cabinetry.fit.fit(model, data)
    m2, _ = _model(_spec())
    d1, d2 = MV._gof_dof(model, [res.bestfit]), MV._gof_dof(m2, [m2.config.suggested_init()])
    check(d1 == 0 and d2 == 10, f"saturated-statistic dof: 1-bin model {d1} (want 0), tiny 2-channel model "
                                f"{d2} (11 bins - mu; staterror/shapesys/normsys are constrained: want 10)")
    ev, _ = EVT.build(spec, model)
    obs = ev.fit(data[:n], data[n:], init=res.bestfit, want_sigma=False)
    MV.N_TOYS = 40
    for mu_truth, want_cov in ((0.6, 0), (0.0, None)):
        for eng, kw in (("pyhf", {}), ("jax", {"ev": ev, "obs_fit": obs})):
            ft = MV.freq_coverage(model, data, mu_truth, **kw)
            ty = MV.toy_ensembles(model, data, res, float(res.bestfit[model.config.poi_index]), **kw)
            none = (ft["gof_p_saturated"] is None
                    and ty["gof_p_toys"] is None and ft["gof_dof"] == 0 and ty["gof_dof"] == 0
                    and "gof_p_undefined" in ft and "gof_p_undefined" in ty)
            cov_ok = ft["n_coverage_toys"] == ft["n_toys"] == 40 and ft["coverage_68"] is not None
            if eng == "jax":
                und = ft["n_sigma_undefined"]
                cov_ok = (cov_ok and und > 0 and ft["n_sigma_undefined_poi_at_bound"] == und
                          and ft["n_sigma_undefined_covered"] == (und if want_cov is None else want_cov)
                          and ft["n_pull_toys"] == 40 - und and ft["cond_fit"]["n_floor"] == 0)
            check(none and cov_ok,
                  f"1-bin model, {eng}, mu_truth {mu_truth}: toy GOF p's None (dof {ft['gof_dof']}); coverage "
                  f"{ft['coverage_68']} over {ft['n_coverage_toys']}/{ft['n_toys']} toys, sigma undefined "
                  f"{ft['n_sigma_undefined']} (on the bound {ft['n_sigma_undefined_poi_at_bound']}, covered "
                  f"{ft.get('n_sigma_undefined_covered')})")


def check_fit_stages(MV, EVT):
    """Score-from-fit stages: the mu measurement (twin vs pyhf, one-sided at a bound), the GOF window, held-aux
    p's and a calibrated combined p, a mu-only model, and the grid profile."""
    import cabinetry
    import pyhf
    from sbh.kinds.binned_likelihood.scoring import grid_profile
    pyhf.set_backend("numpy", "minuit")
    spec = _spec()
    model, data = _model(spec)
    n, poi = model.config.nmaindata, model.config.poi_index
    res = cabinetry.fit.fit(model, data)
    ev, _ = EVT.build(spec, model)
    obs = ev.fit(data[:n], data[n:], init=res.bestfit, want_sigma=False)
    tw, py = MV.mu_fit_block(model, data, res, ev=ev, obs_fit=obs), MV.mu_fit_block(model, data, res)
    ok = (tw["sigma_source"] == py["sigma_source"] == "profile" and abs(tw["value"] - py["value"]) <= 0.02 * py["sigma"]
          and abs(tw["sigma"] / py["sigma"] - 1) <= 0.01 and abs(tw["z_obs"] - py["z_obs"]) <= 1e-2
          and abs(tw["sigma"] / tw["sigma_hess"] - 1) <= 0.1 and tw["sigma_stat"] < tw["sigma_hess"]
          and tw["sigma_min"] == min(tw["sigma"], tw["sigma_hess"]))
    check(ok, f"mu_fit twin vs pyhf: mu {tw['value']:.4f} / {py['value']:.4f}, profile sigma {tw['sigma']:.4f} / "
              f"{py['sigma']:.4f} (Hessian {tw['sigma_hess']:.4f}, stat-only {tw['sigma_stat']:.4f}), Z_obs "
              f"{tw['z_obs']:.3f} / {py['z_obs']:.3f}")
    lo_b = model.config.suggested_bounds()[poi][0]
    x = np.asarray(obs["x"], float).copy()
    fake = dict(obs, x=x)
    x[poi] = lo_b
    fake["nll"] = float(ev.nll(x, data[:n], data[n:]))
    b = MV.mu_fit_block(model, data, res, ev=ev, obs_fit=fake)
    check(b["profile"]["status_lo"] == "bound" and b["sigma_source"] == "profile_one_side" and b["z_obs"] == 0.0,
          f"mu on its lower bound: lo side {b['profile']['status_lo']}, sigma from {b['sigma_source']}, Z_obs {b['z_obs']}")
    meta = {"binning": {"sr": {"observable": "m4l", "edges": [100.0 + 5 * i for i in range(9)], "data_derived": False},
                        "cr": {"observable": "m4l", "edges": [0.0, 1.0, 2.0, 3.0], "data_derived": False}}}
    mask, info = MV._window_mask(model, meta)
    want = [False] * 3 + [False] + [True] * 7
    none, info0 = MV._window_mask(model, None)
    check(list(model.config.channels) == ["cr", "sr"] and mask.tolist() == want and info["n_bins"] == 7
          and not none.any() and len(info0["notes"]) == 2,
          f"GOF window: {info['n_bins']} bins in [105, 140) (want the 7 sr bins above 105 GeV), no binning -> "
          f"{int(none.sum())} bins, notes {info0['notes']}")
    MV.N_TOYS = 30
    MV.GOF_AUX = "held"
    ft = MV.freq_coverage(model, data, 1.0, ev=ev, obs_fit=obs, window=mask, window_info=info)
    grid = lambda p: p is not None and abs(p * 31 - round(p * 31)) < 1e-9  # noqa: E731
    pm = min(ft["gof_p_glob"], ft["gof_p_sr"])
    ok = (ft["gof_aux"] == "held" and all(grid(ft[k]) for k in ("gof_p_glob", "gof_p_sr", "gof_p_comb"))
          and pm - 1e-12 <= ft["gof_p_comb"] <= min(1.0, 2 * pm + 1 / 31) and ft["gof_n_sr_bins"] == 7
          and ft["gof_p_saturated"] == ft["gof_p_glob"] and ft["gof_p_missing"] is None)
    ft0 = MV.freq_coverage(model, data, 1.0, ev=ev, obs_fit=obs, window=none, window_info=info0)
    check(ok and ft0["gof_p_comb"] is None and ft0["gof_p_missing"] == "no_window" and ft0["gof_p_glob"] is not None,
          f"scored stage: p_glob {ft['gof_p_glob']:.3f}, p_sr {ft['gof_p_sr']:.3f}, p_comb {ft['gof_p_comb']:.3f} "
          f"({ft['n_toys']} toys, aux {ft['gof_aux']}); no window: p_comb {ft0['gof_p_comb']}, missing {ft0['gof_p_missing']}")
    np.random.seed(3)
    pdf = ev.toy_pdf(obs["x"])
    toys = np.array(pdf.sample((5,)), float)
    toys[:, n:] = data[n:]
    check(np.all(toys[:, n:] == data[n:]) and not np.all(np.asarray(pdf.sample((5,)))[:, n:] == data[n:]),
          "held aux: the scored toys carry the observed auxiliary data; the sampler alone re-draws them")
    rng = np.random.default_rng(11)
    pc = []
    for _ in range(400):
        q = rng.chisquare(5, size=(200, 2))
        q[:, 1] += 0.5 * q[:, 0]
        r = MV._gof_pvalues({"saturated": q[0, 0], "window": q[0, 1]},
                            {"saturated": q[1:, 0], "window": q[1:, 1]}, 5, {"n_bins": 3, "window": [105, 140]})
        pc.append(r["gof_p_comb"])
    pc = np.asarray(pc)
    check(abs(pc.mean() - 0.5) <= 0.04 and abs(np.mean(pc <= 0.1) - 0.1) <= 0.04,
          f"combined p calibrated under the model: mean {pc.mean():.3f} (0.5), P(p <= 0.1) {np.mean(pc <= 0.1):.3f} (0.1)")
    mo = {"channels": [{"name": "c", "samples": [
              {"name": "s", "data": [1.0, 4.0, 6.0, 4.0, 1.0], "modifiers": [{"name": "mu", "type": "normfactor",
                                                                             "data": None}]},
              {"name": "b", "data": [10.0, 9.0, 8.0, 7.0, 6.0], "modifiers": []}]}],
          "observations": [{"name": "c", "data": [11, 14, 15, 10, 8]}],
          "measurements": [{"name": "m", "config": {"poi": "mu", "parameters": []}}], "version": "1.0.0"}
    m1, d1 = _model(mo)
    r1 = cabinetry.fit.fit(m1, d1)
    e1, _ = EVT.build(mo, m1)
    o1 = e1.fit(d1[:5], d1[5:], init=r1.bestfit, want_sigma=False)
    b1 = MV.mu_fit_block(m1, d1, r1, ev=e1, obs_fit=o1)
    MV.N_TOYS = 20
    f1 = MV.freq_coverage(m1, d1, 1.0, ev=e1, obs_fit=o1)
    check(b1["sigma_source"] == "profile" and b1["z_obs"] is not None and f1["n_toys"] == 20
          and f1["cond_fit"]["engine"] == "jax",
          f"mu-only model: profile {b1['profile']['status_lo']}/{b1['profile']['status_hi']} sigma {b1['sigma']:.3f} "
          f"(Hessian {b1['sigma_hess']:.3f}), Z_obs {b1['z_obs']}, truth-pinned toys {f1['n_toys']} on {f1['cond_fit']['engine']}")
    mh0, s0 = 125.25, 0.31
    pts = [124.0 + 0.25 * k for k in range(9)]
    gp = grid_profile({"grid_uniform": True, "curve": [[m, ((m - mh0) / s0) ** 2] for m in pts]})
    edge = grid_profile({"grid_uniform": True, "curve": [[m, (m - 124.0) ** 2] for m in pts]})
    nonu = grid_profile({"grid_uniform": False, "curve": [[m, (m - 125.0) ** 2] for m in pts]})
    check(gp["usable"] and abs(gp["mh"] - mh0) < 1e-9 and abs(gp["sigma_root"] - s0) < 1e-9 and gp["sigma_lin"] < gp["sigma_root"]
          and not edge["usable"] and not nonu["usable"],
          f"grid profile: mH {gp['mh']:.4f} (want {mh0}), sqrt-interpolated sigma {gp['sigma_root']:.4f} (want {s0}), "
          f"linear {gp['sigma_lin']:.4f}; edge minimum usable {edge['usable']}, non-uniform usable {nonu['usable']}")


def _probe_threads(_=None):
    """Pool worker probe: affinity, thread masks, and CPU/wall over a matmul XLA:CPU would spread."""
    from sbh.kinds.binned_likelihood import jax_engine as EVT
    from sbh.kinds.binned_likelihood.cpu_pin import thread_masks
    import jax.numpy as jnp
    f = EVT.jax.jit(lambda v: jnp.sum(jnp.tanh(v @ v.T) @ v))
    v = jnp.asarray(np.random.default_rng(1).normal(size=(600, 600)))
    f(v).block_until_ready()
    w0, c0 = time.perf_counter(), time.process_time()
    for _ in range(40):
        f(v).block_until_ready()
    return (sorted(os.sched_getaffinity(0)), (time.process_time() - c0) / (time.perf_counter() - w0),
            sorted(thread_masks()))


_LATE_PIN = r"""
import os, sys, time, json
sys.path.insert(0, sys.argv[1])
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, jax.numpy as jnp
from sbh.kinds.binned_likelihood.cpu_pin import pin_single_cpu, thread_masks
import numpy as np
f = jax.jit(lambda v: jnp.sum(jnp.tanh(v @ v.T) @ v))      # a matmul XLA:CPU spreads over its pool
v = jnp.asarray(np.random.default_rng(1).normal(size=(800, 800)))
f(v).block_until_ready()                  # the XLA backend and its pools exist before the pin
cpu = pin_single_cpu(0)
w0, c0 = time.perf_counter(), time.process_time()
for _ in range(40):
    f(v).block_until_ready()
print(json.dumps({"cpu": cpu, "masks": sorted(thread_masks()),
                  "ratio": (time.process_time() - c0) / (time.perf_counter() - w0)}))
"""


def check_pinning(MV):
    """jax Pool workers each run on one distinct CPU (every thread, CPU/wall <= 1.1); a pin applied after the
    first JAX computation still confines the running XLA threads."""
    ncpu = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1
    if ncpu < 2:
        print("  skip  thread pinning (one CPU available)")
        return
    with MV.make_pool(2, "minuit", 0) as pool:
        got = pool.map(_probe_threads, range(2))
    cpus = [c for c, _, _ in got]
    ratio = max(r for _, r, _ in got)
    masks = [m for _, _, m in got]
    check(all(len(c) == 1 for c in cpus) and cpus[0] != cpus[1] and ratio <= 1.1
          and all(m == [str(c[0])] for m, c in zip(masks, cpus)),
          f"jax Pool workers pinned: CPUs {cpus}, thread masks {masks}, max CPU/wall {ratio:.2f}")
    import subprocess
    p = subprocess.run([sys.executable, "-c", _LATE_PIN, str(ROOT)], capture_output=True, text=True, timeout=300)
    try:
        r = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        check(False, f"late pin probe ran (rc {p.returncode}: {p.stderr[-300:]})")
        return
    check(r["masks"] == [str(r["cpu"])] and r["ratio"] <= 1.1,
          f"pin after the first JAX computation confines every thread: masks {r['masks']}, "
          f"CPU/wall {r['ratio']:.2f}")


_FAKE_JAX = "raise ImportError('fake jax: not importable on this node (jax_engine_check)')\n"


def check_no_jax():
    """Without importable jax the verify completes on pyhf with a per-seed fallback reason, with --jobs 2 and 1
    (a Pool initializer importing jax would respawn forever)."""
    import subprocess
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "fake" / "jax").mkdir(parents=True)
        (tmp / "fake" / "jax" / "__init__.py").write_text(_FAKE_JAX)
        for s in ("seed_001", "seed_002"):
            out = tmp / "run" / "runs" / s / "out"
            out.mkdir(parents=True)
            (out / "MODEL.json").write_text(json.dumps(_spec()))
        env = dict(os.environ, SBH_TOY_ENGINE="jax",
                   PYTHONPATH=os.pathsep.join([str(tmp / "fake"), str(ROOT), os.environ.get("PYTHONPATH", "")]))
        for jobs in (2, 1):
            name = f"mv_{jobs}.json"
            t0 = time.perf_counter()
            try:
                p = subprocess.run([sys.executable, "-m", "sbh.kinds.binned_likelihood.verify", "--run-dir", str(tmp / "run"),
                                    "--jobs", str(jobs), "--out", name, "--gof-window", "105", "140"],
                                   cwd=str(ROOT), env=env,
                                   capture_output=True, text=True, timeout=150)
                rc = p.returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
            recs = []
            if (tmp / "run" / name).exists():
                recs = json.loads((tmp / "run" / name).read_text())["per_seed"]
            ok = (rc == 0 and len(recs) == 2 and all(r.get("status") == "ok" and r.get("engine") == "pyhf"
                                                   and "jax engine unavailable" in (r.get("engine_fallback") or "")
                                                   for r in recs))
            check(ok, f"no importable jax, --jobs {jobs}: rc {rc} in {time.perf_counter() - t0:.0f} s, "
                      f"engines {[r.get('engine') for r in recs]}, fallback "
                      f"{(recs[0].get('engine_fallback') or '')[:70] if recs else None!r}")


if __name__ == "__main__":
    t0 = time.perf_counter()
    from sbh.kinds.binned_likelihood import verify as MV
    from sbh.kinds.binned_likelihood import jax_engine as EVT
    import logging
    MV.GOF_WINDOW = (105.0, 140.0)
    logging.getLogger("pyhf").setLevel(logging.ERROR)
    check(MV.ENGINE == "jax", f"SBH_TOY_ENGINE=jax selects the jax engine ({MV.ENGINE})")
    check_pinning(MV)
    check_stages(MV, EVT)
    check_floor(MV, EVT)
    check_one_bin(MV, EVT)
    check_fit_stages(MV, EVT)
    check_poi_scaling(MV, EVT)
    check_no_jax()
    print(f"jax engine check: {'FAIL' if fails else 'PASS'} ({time.perf_counter() - t0:.0f} s)")
    sys.exit(1 if fails else 0)
