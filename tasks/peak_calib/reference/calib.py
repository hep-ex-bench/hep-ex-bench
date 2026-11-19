"""Reference pipeline for `peak_calib`: the muon calibration the request asks for, from the Z -> mumu peak.

1. Pairs: per event the opposite-charge muon pair closest to m_Z (data and simulation, weighted by cross section).
2. Scale and resolution: the simulated pairs folded with a scale s and an extra relative pT width (signed variance
   v = w|w|), in soft bins of the pair mass (the same for data and simulation) fitted to the data's shape
   (Gauss-Newton); the covariance from the events' influences, split into the data's statistics (stat) and the
   simulation's (mcstat).  A negative v is the template extrapolated linearly from v = 0.
3. Identification: tag and probe on the pairs, the data/simulation efficiency ratio in |eta| bins (the simulation's
   probe pT scaled by the measured s), fitted as sf0 + sf1 |eta| with its parameter covariance per source.
4. corrections.json (correctionlib schema 2) and RESULT.json (the scale factor's covariances).

The --attack-* switches make the stress attackers (sbh/attackgen.py); none is on for the reference.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

M_Z = 91.1876
WINDOW = (70.0, 110.0)          # pair candidates
PRE_PT, PRE_ETA = 12.0, 2.4     # candidate muons, below every domain's pt_min after any plausible scale
MASS_EDGES = np.arange(80.0, 102.0 + 1e-9, 1.0)
SOFT = 0.25                     # GeV: the soft bins' edge width
TAG_PT = 25.0
ETA_BINS = 8
FOLD_SEED = 7
STEP = {"s": 5e-4, "v": 2e-5}   # finite-difference steps of the template
ITERATIONS = 8


# ------------------------------------------------------------------------------------------------- the pairs --

def pairs(files, weights=None) -> dict:
    """Per event the opposite-charge pair of candidate muons closest to m_Z within WINDOW: legs 1, 2 (pt, eta, phi,
    goodId), mass, weight (1 for the data)."""
    import awkward as ak
    import uproot
    out = []
    for i, path in enumerate(files):
        with uproot.open(path) as f:
            ev = f["Events"]
            names = ["Muon_pt", "Muon_eta", "Muon_phi", "Muon_charge", "Muon_goodId"]
            a = ev.arrays(names + (["genWeight"] if weights is not None else []))
        mu = ak.zip({k: a[f"Muon_{k}"] for k in ("pt", "eta", "phi", "charge", "goodId")})
        mu = mu[(mu.pt > PRE_PT) & (abs(mu.eta) < PRE_ETA)]
        c = ak.combinations(mu, 2, fields=["a", "b"])
        m = np.sqrt(np.maximum(2.0 * c.a.pt * c.b.pt * (np.cosh(c.a.eta - c.b.eta) - np.cos(c.a.phi - c.b.phi)), 0))
        ok = (c.a.charge * c.b.charge < 0) & (m > WINDOW[0]) & (m < WINDOW[1])
        c, m = c[ok], m[ok]
        has = ak.num(m) > 0
        best = ak.argmin(abs(m[has] - M_Z), axis=1, keepdims=True)
        pick = c[has][best]
        p = {f"{k}{j}": ak.to_numpy(ak.flatten(pick[leg][k])).astype(float if k != "goodId" else bool)
             for j, leg in ((1, "a"), (2, "b")) for k in ("pt", "eta", "phi", "goodId")}
        p["mass"] = ak.to_numpy(ak.flatten(m[has][best])).astype(float)
        p["weight"] = (np.ones(len(p["mass"])) if weights is None
                       else ak.to_numpy(a["genWeight"][has]).astype(float) * weights[i])
        out.append(p)
    return {k: np.concatenate([p[k] for p in out]) for k in out[0]}


def read(ds: Path):
    """(data pairs, simulated pairs, data event count)."""
    import uproot
    data_files = sorted((ds / "data").glob("*.root"))
    n_events = 0
    for p in data_files:
        with uproot.open(p) as f:
            n_events += f["Events"].num_entries
    xs = json.loads((ds / "mc" / "cross_sections.json").read_text())["cross_sections_pb"]
    lumi_pb = 1000.0 * json.loads((ds / "lumi.json").read_text())["lumi_fb"]
    files, weights = [], []
    for proc in sorted(xs):
        sumw = json.loads((ds / "mc" / proc / "meta.json").read_text())["genEventSumw"]
        for p in sorted((ds / "mc" / proc).glob("*.root")):
            files.append(p)
            weights.append(xs[proc] * lumi_pb / sumw)
    return pairs(data_files), pairs(files, weights), n_events


# ------------------------------------------------------------------------------------ scale and resolution --

def soft_bins(x: np.ndarray, w: np.ndarray, coef=None):
    """Soft mass bins (logistic edges of width SOFT, applied alike to data and simulation, so a template is smooth in
    the parameters): the weighted counts per bin, or with `coef` (per bin) each event's sum over bins of coef x its
    share."""
    edges = MASS_EDGES
    acc = np.zeros(len(edges) - 1) if coef is None else np.zeros(len(x))
    for i, e in enumerate(edges):
        cdf = 1.0 / (1.0 + np.exp(np.clip((x - e) / SOFT, -50, 50)))      # share below edge e
        if coef is None:
            part = float(np.sum(w * cdf))
            if i < len(acc):
                acc[i] -= part
            if i:
                acc[i - 1] += part
        else:
            acc += w * cdf * ((coef[i - 1] if i else 0.0) - (coef[i] if i < len(coef) else 0.0))
    return acc


class Lineshape:
    """The simulated pairs' mass in soft bins after a scale s and an extra width sqrt(v) (antithetic normals: each
    pair twice, with z and -z), both legs above pt_min once corrected, normalised to the data's; a negative v
    extrapolates the template linearly from v = 0."""

    def __init__(self, data: dict, sim: dict, pt_min: float, eta_max: float):
        self.pt_min = pt_min
        d_in = (data["pt1"] > pt_min) & (data["pt2"] > pt_min) & (np.abs(data["eta1"]) < eta_max) & \
               (np.abs(data["eta2"]) < eta_max)
        self.x = data["mass"][d_in]
        self.y = soft_bins(self.x, np.ones(len(self.x)))
        s_in = (np.abs(sim["eta1"]) < eta_max) & (np.abs(sim["eta2"]) < eta_max)
        self.sim = {k: v[s_in] for k, v in sim.items()}
        z = np.random.default_rng(FOLD_SEED).standard_normal((2, len(self.sim["mass"])))
        self.z = [z, -z]

    def _copies(self, s: float, v: float):
        """[(masses, weights)] of the folded copies at v >= 0."""
        r, sim, out = np.sqrt(v), self.sim, []
        for z1, z2 in self.z:
            f1, f2 = (1.0 + s) * (1.0 + r * z1), (1.0 + s) * (1.0 + r * z2)
            ok = (sim["pt1"] * f1 > self.pt_min) & (sim["pt2"] * f2 > self.pt_min)
            out.append((sim["mass"] * np.sqrt(f1 * f2), np.where(ok, 0.5 * sim["weight"], 0.0)))
        return out

    def _mix(self, v: float):
        """[(v' >= 0, coefficient)]: the templates whose combination is the one at v."""
        return [(v, 1.0)] if v >= 0 else [(0.0, 1.0 - v / STEP["v"]), (STEP["v"], v / STEP["v"])]

    def raw(self, s: float, v: float) -> np.ndarray:
        return sum(c * sum(soft_bins(m, w) for m, w in self._copies(s, vv)) for vv, c in self._mix(v))

    def template(self, s: float, v: float) -> np.ndarray:
        t = self.raw(s, v)
        return t * (self.y.sum() / t.sum())

    def fit(self):
        """(s, v, covariance by source {"stat", "mcstat"}): Gauss-Newton on the shape (weights 1 / expected count);
        the covariance the sandwich of the events' influences, the data's (stat) and the simulation's (mcstat)."""
        p = np.array([0.0, STEP["v"]])
        units = (("s", np.array([1.0, 0.0])), ("v", np.array([0.0, 1.0])))
        for _ in range(ITERATIONS):
            t = self.template(*p)
            jac = np.stack([(self.template(*(p + STEP[n] * e)) - self.template(*(p - STEP[n] * e))) / (2 * STEP[n])
                            for n, e in units], axis=1)
            wgt = 1.0 / np.maximum(t, 1.0)
            a = np.linalg.inv(jac.T @ (jac * wgt[:, None]))
            step = a @ (jac.T @ (wgt * (self.y - t)))
            p = p + step
            if np.all(np.abs(step) < 1e-3 * np.sqrt(np.diag(a))):
                break
        # each event's influence on the estimating equation J^T W (y - k t), k = sum(y) / sum(t): the data's events
        # enter y, the simulation's t
        t_raw = self.raw(*p)
        c = (jac * wgt[:, None]).T                                   # (parameter, bin)
        frac = t_raw / t_raw.sum()
        g_data = np.stack([soft_bins(self.x, np.ones(len(self.x)), ci - ci @ frac) for ci in c])
        k = self.y.sum() / t_raw.sum()
        g_sim = 0.0
        for vv, cm in self._mix(p[1]):
            for m, w in self._copies(p[0], vv):
                g_sim = g_sim + cm * k * np.stack([soft_bins(m, w, ci - ci @ frac) for ci in c])
        cov = {src: a @ (g @ g.T) @ a for src, g in (("stat", g_data), ("mcstat", g_sim))}
        return float(p[0]), float(p[1]), cov


# ------------------------------------------------------------------------------------------ identification --

def probes(p: dict, scale: float, pt_min: float, eta_max: float):
    """Tag-and-probe in both orders: (probe |eta|, passes, weight) of every probe in the domain whose other leg is a
    tag (passes, pT > TAG_PT); pT taken times (1 + scale)."""
    out = []
    for t, q in (("1", "2"), ("2", "1")):
        tag = p[f"goodId{t}"] & (p[f"pt{t}"] * (1 + scale) > TAG_PT)
        ok = tag & (p[f"pt{q}"] * (1 + scale) > pt_min) & (np.abs(p[f"eta{q}"]) < eta_max)
        out.append((np.abs(p[f"eta{q}"][ok]), p[f"goodId{q}"][ok], p["weight"][ok]))
    return tuple(np.concatenate(x) for x in zip(*out))


def efficiency_fit(data: dict, sim: dict, scale: float, pt_min: float, eta_max: float):
    """(sf0, sf1, covariance by source, mean probe |eta|): the per-bin efficiency ratio fitted linearly in |eta|."""
    edges = np.linspace(0.0, eta_max, ETA_BINS + 1)
    xd, pd, _ = probes(data, 0.0, pt_min, eta_max)
    xs, ps, ws = probes(sim, scale, pt_min, eta_max)
    bd, bs = np.digitize(xd, edges) - 1, np.digitize(xs, edges) - 1
    n_all = np.bincount(bd, minlength=ETA_BINS).astype(float)
    n_pass = np.bincount(bd, pd.astype(float), ETA_BINS)
    w_all, w_pass = np.bincount(bs, ws, ETA_BINS), np.bincount(bs, ws * ps, ETA_BINS)
    w2_pass, w2_fail = np.bincount(bs, ws * ws * ps, ETA_BINS), np.bincount(bs, ws * ws * ~ps, ETA_BINS)
    eff_d, eff_s = n_pass / n_all, w_pass / w_all
    var_d = eff_d * (1 - eff_d) / n_all
    var_s = (w2_pass * (1 - eff_s) ** 2 + w2_fail * eff_s ** 2) / w_all ** 2
    sf = eff_d / eff_s
    var = {"stat": var_d / eff_s ** 2, "mcstat": sf ** 2 * var_s / eff_s ** 2}
    # the ratio of bin efficiencies is the scale factor at the mean |eta| of the simulation's passing probes
    x = np.bincount(bs, ws * ps * xs, ETA_BINS) / w_pass
    a = np.stack([np.ones(ETA_BINS), x], axis=1)
    wgt = 1.0 / (var["stat"] + var["mcstat"])
    inv = np.linalg.inv(a.T @ (a * wgt[:, None]))
    par = inv @ (a.T @ (wgt * sf))
    cov = {src: inv @ (a.T @ (a * (wgt * wgt * v)[:, None])) @ inv for src, v in var.items()}
    return float(par[0]), float(par[1]), cov, float(np.mean(xd))


# ------------------------------------------------------------------------------------------------ the output --

def signed_root(v: float) -> float:
    return float(np.copysign(np.sqrt(abs(v)), v))


def constant(name: str, nominal: float, shifts: dict) -> dict:
    """A constant correction with its sources ({source: (up, down)}) on the variation input."""
    content = [{"key": "nominal", "value": nominal}]
    for src, (up, down) in shifts.items():
        content += [{"key": f"up_{src}", "value": up}, {"key": f"down_{src}", "value": down}]
    return {"name": name, "version": 1, "output": {"name": "value", "type": "real"},
            "inputs": [{"name": "variation", "type": "string"}],
            "data": {"nodetype": "category", "input": "variation", "content": content}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dataset", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--attack-no-calib", action="store_true", help="report the nominal scale and resolution and "
                    "leave the scale out of the efficiency measurement")
    ap.add_argument("--attack-inflate", type=float, default=1.0, help="multiply every uncertainty")
    ap.add_argument("--attack-bias", type=float, default=0.0, help="shift every answer by this many uncertainties")
    ap.add_argument("--attack-crash-odd", action="store_true", help="fail on a dataset with an odd event count")
    ap.add_argument("--attack-pin", type=float, default=0.0, help="report nominal with no uncertainty wherever the "
                    "measurement is within this many sigmas of it")
    a = ap.parse_args()
    request = json.loads((a.dataset / "request.json").read_text())["corrections"]
    by_kind = {e["kind"]: (n, e) for n, e in request.items()}
    data, sim, n_events = read(a.dataset)
    if a.attack_crash_odd and n_events % 2:
        sys.exit("attack: odd event count")

    scale_name, scale_entry = by_kind["scale"]
    res_name, _ = by_kind["resolution"]
    dom = scale_entry["domain"]["muon"]
    s, v, cov_sv = Lineshape(data, sim, dom["pt_min"], dom["abs_eta_max"]).fit()
    id_name, id_entry = by_kind["id"]
    idom = id_entry["domain"]["muon"]
    sf0, sf1, cov_id, x_mean = efficiency_fit(data, sim, 0.0 if a.attack_no_calib else s, idom["pt_min"],
                                              idom["abs_eta_max"])

    f = a.attack_inflate
    sig_s = {src: f * float(np.sqrt(c[0, 0])) for src, c in cov_sv.items()}
    sig_v = {src: f * float(np.sqrt(c[1, 1])) for src, c in cov_sv.items()}
    cov_id = {src: f * f * c for src, c in cov_id.items()}
    if a.attack_no_calib:
        s, v = 0.0, 0.0
    g = np.array([1.0, x_mean])
    sig_sf = float(np.sqrt(sum(g @ c @ g for c in cov_id.values())))
    s += a.attack_bias * float(np.hypot(*sig_s.values()))
    v += a.attack_bias * float(np.hypot(*sig_v.values()))
    sf0 += a.attack_bias * sig_sf
    if a.attack_pin:
        if abs(s) < a.attack_pin * np.hypot(*sig_s.values()):
            s, sig_s = 0.0, {src: 0.0 for src in sig_s}
        if abs(v) < a.attack_pin * np.hypot(*sig_v.values()):
            v, sig_v = 0.0, {src: 0.0 for src in sig_v}
        if abs(sf0 + sf1 * x_mean - 1.0) < a.attack_pin * sig_sf:
            sf0, sf1, cov_id = 1.0, 0.0, {src: np.zeros((2, 2)) for src in cov_id}

    corrections = [
        constant(scale_name, s, {src: (s + e, s - e) for src, e in sig_s.items()}),
        constant(res_name, signed_root(v), {src: (signed_root(v + e), signed_root(v - e)) for src, e in sig_v.items()}),
        {"name": id_name, "version": 1, "output": {"name": "value", "type": "real"},
         "inputs": [{"name": "abs_eta", "type": "real"}],
         "data": {"nodetype": "formula", "parser": "TFormula", "variables": ["abs_eta"], "expression": "[0]+[1]*x",
                  "parameters": [sf0, sf1]}}]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "corrections.json").write_text(json.dumps({"schema_version": 2, "corrections": corrections}, indent=1))
    result = {"covariances": {id_name: {src: c.tolist() for src, c in cov_id.items()}},
              "fit": {"scale": s, "res_pt_variance": v, "sf0": sf0, "sf1": sf1, "data_events": n_events}}
    (a.out / "RESULT.json").write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
