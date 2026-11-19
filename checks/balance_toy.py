"""Toy datasets for the jet balance closure's quick check (checks/balance_check.py): data and simulation in the
layout of a jet-level NanoAOD slim (Jet, Muon, Electron, GenJet, PV, RawMET / ChsMET, the single-muon paths and a
single-jet menu HLT_SingleJet<T>) with a known jet response, and the toy's own request and balance selection (spec),
so the closure (sbh/kinds/corrections/balance.py) is tested on its own.

The model, the same in both roles but for the data's jet response:
- Z(mumu)+jet events (process "dy"): a Z of pT(Z) above 25 GeV decaying isotropically to two muons, a recoil jet
  back to back with it at a random eta and a second, soft jet; each muon also makes a jet of its own (the
  reclustering keeps every candidate), never distorted.  The simulation's weights are +-1, 16 % negative.
- Multijet events ("qcd"): pT_avg log-uniform from 40 GeV, two back-to-back jets, one central, the other anywhere,
  and a soft third; an electron next to the third jet in a fifth of them.  Weights uniform in [0.5, 1.5].
- Reconstruction: every jet pT = true x 0.9 x (1 + 0.14 g); the data's non-lepton jets then x (1 + scale) x
  (1 + sqrt(V) g'), scale and V per eta region (TRUTH) -- exactly the closure's folding, so the right corrections
  reproduce the data in law.  Jets above 10 GeV are kept.
- Triggers: the single-muon paths on the leading muon (pT > 24 GeV, 95 %); the jet menu (MENU) on the leading jet's
  online pT (offline x (1 + 0.08 g)), post-prescale in the data (PRESCALES), unprescaled in the simulation; an event
  is kept when a muon path or a jet path fired.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np

REGIONS = (("central", 0.0, 1.3), ("endcap", 1.3, 3.0), ("hf", 3.0, 4.7))
TRUTH = {"scale": {"central": 0.05, "endcap": -0.02, "hf": 0.10},
         "res_var": {"central": 0.008, "endcap": 0.003, "hf": 0.015}}
NOMINAL = {"scale": {r: 0.0 for r, *_ in REGIONS}, "res_var": {r: 0.0 for r, *_ in REGIONS}}
MENU = (40, 60, 80, 110, 140, 200, 260, 320, 400, 500)
PRESCALES = (4, 4, 2, 2, 1, 1, 1, 1, 1, 1)
REQUEST = {"request_version": 1, "corrections": {
    "res_pt": {"kind": "resolution", "domain": {"jet": {"pt_min": 30.0, "abs_eta_max": 4.7}}},
    "scale": {"kind": "scale", "domain": {"jet": {"pt_min": 30.0, "abs_eta_max": 4.7}}}}}
SELECTION = {"muon": {"pt_min": 15.0, "lead_pt_min": 25.0, "abs_eta_max": 2.1, "id": "mediumId", "iso_max": 0.25,
                      "dxy_max": 0.1, "dz_max": 0.3},
             "muon_paths": ["HLT_IsoMu24", "HLT_IsoTkMu24"], "mass_window": [76.0, 106.0],
             "clean": {"dr": 0.4, "lepton_pt_min": 10.0}, "jet_abs_eta_max": 4.7, "jet_pt_min": 25.0,
             "preselect": 0.6,
             "zjet": {"dphi_min": 2.8, "alpha_max": 0.35, "pt_z": [30.0, 60.0, 110.0, 220.0, 800.0],
                      "balance": [round(0.06 * i, 2) for i in range(45)]},
             "dijet": {"dphi_min": 2.8, "alpha_max": 0.25, "tag_abs_eta_max": 1.3,
                       "asymmetry": [round(-0.66 + 0.03 * i, 3) for i in range(45)]}}
PLATEAU = 1.5
SIZES = {"data": {"dy": 8000, "qcd": 16000}, "sim": {"dy": 20000, "qcd": 32000}}
LUMI_FB = 0.1


def _region_values(eta: np.ndarray, per_region: Dict[str, float]) -> np.ndarray:
    a = np.abs(eta)
    out = np.zeros_like(a)
    for r, lo, hi in REGIONS:
        out = np.where((a >= lo) & (a < hi), per_region[r], out)
    return out


def _dy(rng, n: int):
    """True Z(mumu)+jet events: muons (2 per event) and jets [recoil, soft, mu-jet 1, mu-jet 2]."""
    pt_z = 25.0 + rng.exponential(40.0, n)
    y = rng.uniform(-1.8, 1.8, n)
    phi_z = rng.uniform(-np.pi, np.pi, n)
    m = 91.19 + 2.5 * rng.standard_normal(n)
    ct, ph = rng.uniform(-1.0, 1.0, n), rng.uniform(-np.pi, np.pi, n)
    st = np.sqrt(1.0 - ct * ct)
    mt = np.sqrt(m * m + pt_z * pt_z)
    E, px, py, pz = mt * np.cosh(y), pt_z * np.cos(phi_z), pt_z * np.sin(phi_z), mt * np.sinh(y)
    b = np.stack([px, py, pz], 1) / E[:, None]
    g = E / m
    mus = []
    for sign in (1.0, -1.0):
        p = sign * 0.5 * m[:, None] * np.stack([st * np.cos(ph), st * np.sin(ph), ct], 1)
        e = 0.5 * m
        bp = (b * p).sum(1)
        k = (g - 1.0) * bp / np.maximum((b * b).sum(1), 1e-30) + g * e
        q = p + k[:, None] * b
        mus.append((np.hypot(q[:, 0], q[:, 1]), np.arcsinh(q[:, 2] / np.hypot(q[:, 0], q[:, 1])),
                    np.arctan2(q[:, 1], q[:, 0])))
    j_pt = pt_z * (1.0 + 0.08 * rng.standard_normal(n))
    j_phi = phi_z + np.pi + 0.1 * rng.standard_normal(n)
    j_eta = rng.uniform(-4.6, 4.6, n)
    s_pt = pt_z * rng.exponential(0.08, n)
    s_eta, s_phi = rng.uniform(-4.6, 4.6, n), rng.uniform(-np.pi, np.pi, n)
    jets = [(j_pt, j_eta, j_phi, False), (s_pt, s_eta, s_phi, False),
            (mus[0][0], mus[0][1], mus[0][2], True), (mus[1][0], mus[1][1], mus[1][2], True)]
    return mus, jets


def _qcd(rng, n: int):
    """True multijet events: jets [one, two, soft]; an electron next to the soft jet in a fifth."""
    avg = 40.0 * np.exp(rng.uniform(0.0, math.log(1500.0 / 40.0), n))
    d = 0.04 * rng.standard_normal(n)
    c_eta, o_eta = rng.uniform(-1.3, 1.3, n), rng.uniform(-4.6, 4.6, n)
    swap = rng.uniform(size=n) < 0.5
    eta1, eta2 = np.where(swap, o_eta, c_eta), np.where(swap, c_eta, o_eta)
    phi1 = rng.uniform(-np.pi, np.pi, n)
    phi2 = phi1 + np.pi + 0.05 * rng.standard_normal(n)
    s_pt = avg * rng.exponential(0.05, n)
    s_eta, s_phi = rng.uniform(-4.6, 4.6, n), rng.uniform(-np.pi, np.pi, n)
    ele = rng.uniform(size=n) < 0.2
    jets = [(avg * (1 + d), eta1, phi1, False), (avg * (1 - d), eta2, phi2, False), (s_pt, s_eta, s_phi, False)]
    return ele, (s_pt, s_eta + 0.1, s_phi), jets


def _wrap(phi):
    return np.remainder(phi + np.pi, 2.0 * np.pi) - np.pi


def _jagged(cols: Dict[str, np.ndarray], keep: np.ndarray):
    """Per event the entries (columns n x k) where `keep`, as awkward lists (NanoAOD collection fields)."""
    import awkward as ak
    counts = keep.sum(1)
    return {f: ak.unflatten(v[keep], counts) for f, v in cols.items()}


def _events(rng, process: str, n: int, role: str, truth: Dict[str, Dict[str, float]]) -> Dict[str, object]:
    """One process's events of a role in the slim's write form (collections zipped, flat branches)."""
    import awkward as ak
    data = role == "data"
    if process == "dy":
        mus, jets = _dy(rng, n)
        mu_pt = np.stack([mus[0][0], mus[1][0]], 1)
        mu_eta, mu_phi = np.stack([mus[0][1], mus[1][1]], 1), np.stack([mus[0][2], mus[1][2]], 1)
        q = rng.choice([-1, 1], n)
        mu_q = np.stack([q, -q], 1)
        mu_keep = np.ones((n, 2), dtype=bool)
        el = None
    else:
        ele, (e_pt, e_eta, e_phi), jets = _qcd(rng, n)
        mu_pt = mu_eta = mu_phi = np.zeros((n, 0))
        mu_q = np.zeros((n, 0), dtype=int)
        mu_keep = np.zeros((n, 0), dtype=bool)
        el = (ele[:, None], e_pt[:, None], e_eta[:, None], e_phi[:, None])
    pt = np.stack([j[0] for j in jets], 1)
    eta = np.stack([j[1] for j in jets], 1)
    phi = _wrap(np.stack([j[2] for j in jets], 1))
    lepton = np.array([j[3] for j in jets])[None, :] & np.ones((n, 1), dtype=bool)
    reco = np.where(lepton, pt, pt * 0.9 * (1.0 + 0.14 * rng.standard_normal(pt.shape)))
    if data:
        s = _region_values(eta, truth["scale"])
        v = _region_values(eta, truth["res_var"])
        g = rng.standard_normal(pt.shape)
        reco = np.where(lepton, reco, reco * (1.0 + s) * (1.0 + np.sqrt(v) * g))
    keep = reco > 10.0
    order = np.argsort(-np.where(keep, reco, -1.0), axis=1, kind="stable")
    reco, eta, phi, keep = (np.take_along_axis(x, order, 1) for x in (reco, eta, phi, keep))
    lead = np.where(keep[:, 0], reco[:, 0], 0.0)
    online = lead * (1.0 + 0.08 * rng.standard_normal(n))
    bits = {}
    for t, ps in zip(MENU, PRESCALES):
        fired = online >= t
        if data:
            fired &= rng.uniform(size=n) < 1.0 / ps
        bits[f"HLT_SingleJet{t}"] = fired
    mu_lead = mu_pt.max(axis=1) if mu_pt.shape[1] else np.zeros(n)
    mu_trig = (mu_lead > 24.0) & (rng.uniform(size=n) < 0.95)
    stored = mu_trig | np.any(np.stack(list(bits.values()), 1), axis=1)
    if not data:
        w = (np.where(rng.uniform(size=n) < 0.16, -1.0, 1.0) if process == "dy" else rng.uniform(0.5, 1.5, n))
    sel = np.flatnonzero(stored)
    nk = len(sel)
    jc = {"pt": reco, "eta": eta, "phi": phi, "mass": 0.1 * reco, "nConstituents": np.full(reco.shape, 12),
          "chHEF": np.full(reco.shape, 0.6), "neHEF": np.full(reco.shape, 0.1), "chEmEF": np.full(reco.shape, 0.05),
          "neEmEF": np.full(reco.shape, 0.25), "muEF": np.where(lepton, 0.9, 0.0)}
    jc = {k: np.asarray(v)[sel].astype(np.int32 if k == "nConstituents" else np.float32) for k, v in jc.items()}
    m = mu_pt.shape[1]
    mc = {"pt": mu_pt, "eta": mu_eta, "phi": mu_phi, "mass": np.full((n, m), 0.1057), "charge": mu_q,
          "pdgId": -13 * mu_q, "dxy": 0.01 * rng.standard_normal((n, m)), "dz": 0.02 * rng.standard_normal((n, m)),
          "pfRelIso04_all": rng.exponential(0.03, (n, m)), "looseId": np.ones((n, m), bool),
          "mediumId": np.ones((n, m), bool), "tightId": rng.uniform(size=(n, m)) < 0.97,
          "isGlobal": np.ones((n, m), bool), "isTracker": np.ones((n, m), bool), "isPFcand": np.ones((n, m), bool)}
    ints = {"charge", "pdgId"}
    bools = {"looseId", "mediumId", "tightId", "isGlobal", "isTracker", "isPFcand"}
    mc = {k: np.asarray(v)[sel].astype(bool if k in bools else np.int32 if k in ints else np.float32)
          for k, v in mc.items()}
    if el is None:
        ek = np.zeros((nk, 0), dtype=bool)
        ec = {k: np.zeros((nk, 0)) for k in ("pt", "eta", "phi")}
    else:
        ek = el[0][sel]
        ec = {"pt": el[1][sel], "eta": el[2][sel], "phi": _wrap(el[3][sel])}
    e1 = ek.shape[1]
    ec.update({"mass": np.zeros((nk, e1)), "charge": np.ones((nk, e1)), "pdgId": np.full((nk, e1), 11),
               "dxy": np.zeros((nk, e1)), "dz": np.zeros((nk, e1)), "pfRelIso03_all": np.full((nk, e1), 0.02)})
    ec = {k: np.asarray(v).astype(np.int32 if k in ("charge", "pdgId") else np.float32)
          for k, v in ec.items()}
    out = {"Jet": ak.zip(_jagged(jc, keep[sel])),
           "Muon": ak.zip(_jagged(mc, mu_keep[sel])),
           "Electron": ak.zip(_jagged(ec, ek))}
    flat = {"run": np.ones(nk, np.uint32), "luminosityBlock": np.ones(nk, np.uint32),
            "event": rng.integers(0, 1 << 40, nk).astype(np.uint64), "PV_npvs": np.full(nk, 20, np.int32),
            "PV_npvsGood": np.full(nk, 18, np.int32), "PV_x": np.zeros(nk, np.float32),
            "PV_y": np.zeros(nk, np.float32), "PV_z": rng.normal(0, 3, nk).astype(np.float32),
            "RawMET_pt": np.zeros(nk, np.float32), "RawMET_phi": np.zeros(nk, np.float32),
            "RawMET_sumEt": np.zeros(nk, np.float32), "ChsMET_pt": np.zeros(nk, np.float32),
            "ChsMET_phi": np.zeros(nk, np.float32), "ChsMET_sumEt": np.zeros(nk, np.float32),
            "HLT_IsoMu24": mu_trig[sel], "HLT_IsoTkMu24": mu_trig[sel]}
    flat.update({k: v[sel] for k, v in bits.items()})
    if not data:
        flat["genWeight"] = w[sel].astype(np.float32)
        gk = keep[sel] & ~lepton[sel]
        gj = {"pt": pt, "eta": eta, "phi": phi}
        gj = {k: np.take_along_axis(np.asarray(v), order, 1)[sel].astype(np.float32) for k, v in gj.items()}
        gj.update({"mass": np.zeros_like(gj["pt"]), "partonFlavour": np.zeros(gj["pt"].shape, np.int32),
                   "hadronFlavour": np.zeros(gj["pt"].shape, np.int32)})
        out["GenJet"] = ak.zip(_jagged(gj, gk))
    out.update(flat)
    return out, (float(np.sum(w)), float(np.sum(w * w)), n) if not data else None


def _write(path: Path, events: Dict[str, object], runs: Optional[tuple] = None) -> None:
    import awkward as ak
    import uproot
    with uproot.recreate(path) as f:
        t = f.mktree("Events", {k: (v.type if isinstance(v, ak.Array) else v.dtype) for k, v in events.items()})
        t.extend(events)
        if runs is not None:
            sw, sw2, cnt = runs
            arr = {"run": np.ones(1, np.uint32), "genEventCount": np.array([cnt], np.int64),
                   "genEventSumw": np.array([sw]), "genEventSumw2": np.array([sw2])}
            r = f.mktree("Runs", {k: v.dtype for k, v in arr.items()})
            r.extend(arr)


def write_dataset(d: Path, truth: Dict[str, Dict[str, float]] = TRUTH, seed: int = 1, lumi_scale: float = 1.0,
                  sizes: Dict[str, Dict[str, int]] = SIZES) -> Path:
    """A toy dataset under `d` (data/, mc/<process>/, cross sections, lumi.json, meta.json, request.json) whose data
    carry `truth`'s jet response; lumi_scale multiplies the stated luminosity (an offset of the simulation's
    normalisation against the data)."""
    rng = np.random.default_rng([seed, 991])
    d = Path(d)
    (d / "data").mkdir(parents=True, exist_ok=True)
    xs = {}
    for proc in ("dy", "qcd"):
        ev, _ = _events(rng, proc, sizes["data"][proc], "data", truth)
        _write(d / "data" / f"part_{proc}.root", ev)
        (d / "mc" / proc).mkdir(parents=True, exist_ok=True)
        ev, runs = _events(rng, proc, sizes["sim"][proc], "mc", truth)
        _write(d / "mc" / proc / "part_0.root", ev, runs)
        (d / "mc" / proc / "meta.json").write_text(json.dumps({"genEventSumw": runs[0], "genEventCount": runs[2]}))
        xs[proc] = sizes["data"][proc] / (1000.0 * LUMI_FB)
    (d / "mc" / "cross_sections.json").write_text(json.dumps({"cross_sections_pb": xs}))
    (d / "lumi.json").write_text(json.dumps({"lumi_fb": LUMI_FB * lumi_scale, "lumi_unc_rel": 0.025}))
    (d / "meta.json").write_text(json.dumps({"trigger_paths": [{"name": f"HLT_SingleJet{t}", "threshold_GeV": t,
                                                                "prescale": p}
                                                               for t, p in zip(MENU, PRESCALES)]}))
    return d


def spec(widths: Dict[str, Dict[str, float]]) -> dict:
    """The closure's spec on the toy (balance.spec), each correction's forecast nominal +- its per-region width."""
    from sbh.kinds.corrections.balance import spec as balance_spec
    regions = {r: {"abs_eta": [lo, hi]} for r, lo, hi in REGIONS}
    return balance_spec(SELECTION, regions, MENU, PLATEAU, REQUEST, {n: (0.0, w) for n, w in widths.items()})


def corrections(per_region: Dict[str, Dict[str, float]], scale_factor: float = 1.0, sigma=(2e-3, 5e-4),
                sources=(("stat", 0.6), ("model", 0.4))) -> dict:
    """corrections.json of a per-region answer: `scale` (times scale_factor) and `res_pt` (the signed square root of
    the extra variance), each region a bin of |eta|, with nominal / up / down and per-source variations whose shifts
    add up to the band (sigma: the scale's and the extra variance's half-band)."""
    names = [r for r, *_ in REGIONS]
    edges = [REGIONS[0][1]] + [hi for *_, hi in REGIONS]

    def ss(v):
        return math.copysign(math.sqrt(abs(v)), v)

    def node(values, sig, signed):
        f = ss if signed else (lambda v: v)
        content = [{"key": "nominal", "value": [f(v) for v in values]},
                   {"key": "up", "value": [f(v + sig) for v in values]},
                   {"key": "down", "value": [f(v - sig) for v in values]}]
        for src, share in sources:
            content += [{"key": f"up_{src}", "value": [f(v + sig * math.sqrt(share)) for v in values]},
                        {"key": f"down_{src}", "value": [f(v - sig * math.sqrt(share)) for v in values]}]
        return {"nodetype": "category", "input": "variation",
                "content": [{"key": c["key"], "value": {"nodetype": "binning", "input": "abs_eta", "edges": edges,
                                                         "content": c["value"], "flow": "clamp"}} for c in content]}
    scale = [scale_factor * per_region["scale"][r] for r in names]
    res = [per_region["res_var"][r] for r in names]
    corr = []
    for name, vals, sig, signed in (("scale", scale, sigma[0], False), ("res_pt", res, sigma[1], True)):
        corr.append({"name": name, "version": 1, "inputs": [{"name": "abs_eta", "type": "real"},
                                                              {"name": "variation", "type": "string"}],
                     "output": {"name": "value", "type": "real"}, "data": node(vals, sig, signed)})
    return {"schema_version": 2, "corrections": corr}
