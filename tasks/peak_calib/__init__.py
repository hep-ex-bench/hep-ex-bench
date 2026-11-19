"""peak_calib -- a toy task of the corrections kind: a muon calibration measured from a synthetic Z -> mumu peak.

Every dataset holds Z -> mumu events in a small NanoAOD-like ROOT file (data/) and one nominal simulation of them
(mc/zmumu/), identical for every dataset.  The data differ from the simulation by a momentum scale s, an extra relative
pT resolution r and an identification scale factor linear in |eta|, sf0 + sf1 |eta|, drawn per dataset.  The pipeline
answers request.json -- `scale`, `res_pt` and `Muon_goodId` -- as correctionlib functions with uncertainties
(EVAL_CONTRACT.md); each is scored at the evaluation points, simulated Z legs, against the injected truth, and the three
are checked together on the data's Z peak (the kind's Z-peak closure).
Sealed truth: scale, res_pt, sf0, sf1.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from sbh.kinds.corrections.task import CorrectionsTask

M_Z, GAMMA_Z = 91.1876, 2.4952
M_LO, M_HI = 60.0, 120.0                # generated Z mass range
ACC_PT, ACC_ETA = 5.0, 2.5              # a muon outside is not reconstructed
XSEC_PB = 2000.0                        # Z -> mumu cross section times acceptance of the generated range
LUMI_FB = 0.02                          # ~40k data events per dataset
N_SIM = 150_000
SIM_SEED = 20261028                     # the simulation is identical for every dataset and suite
EVAL_SEED = 20261029                    # the evaluation points' own draw
N_EVAL = 3000
PROCESS = "zmumu"
ID_NAME = "Muon_goodId"
DOMAIN = {"scale": {"pt_min": 20.0, "abs_eta_max": 2.4}, "res_pt": {"pt_min": 20.0, "abs_eta_max": 2.4},
          ID_NAME: {"pt_min": 15.0, "abs_eta_max": 2.4}}
KIND_OF = {"scale": "scale", "res_pt": "resolution", ID_NAME: "id"}
ROLE_OF = {"scale": "scale_mu", "res_pt": "res_pt_mu", ID_NAME: "id_mu"}
NOMINAL = {"scale": 0.0, "resolution": 0.0, "id": 1.0}
# the floor added in quadrature to every uncertainty, per kind in the comparison space (a resolution's extra
# variance): a quarter (two digits, rounded down) of the reference's smallest uncertainty over the four sealed
# datasets of truth_grids/grid_v0.json (scale 1.64e-4, extra variance 1.05e-5, identification 1.46e-3)
TAU = {"scale": 4.0e-5, "resolution": 2.6e-6, "id": 3.6e-4}
REQUIRED_SOURCES = ("stat", "mcstat")
# the results page's views: curves in pT and |eta|, an identification's also as a pT x |eta| map
CURVE_PT = [20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 60.0, 80.0, 150.0]
MAP_PT = [20.0, 30.0, 40.0, 50.0, 150.0]
CURVE_ETA_BINS, MAP_ETA_BINS = 8, 4


# ---------------------------------------------------------------------------------------------------- the events --

def sim_resolution(abs_eta):
    """The simulation's relative pT resolution of a muon."""
    return 0.010 + 0.004 * np.asarray(abs_eta, float)


def sim_efficiency(pt, abs_eta):
    """The simulation's probability that a muon passes Muon_goodId (pt and |eta| of the generated muon)."""
    pt, abs_eta = np.asarray(pt, float), np.asarray(abs_eta, float)
    return 0.97 * (1.0 - 0.25 * np.exp(-pt / 12.0)) - 0.03 * (abs_eta / ACC_ETA) ** 2


def id_truth(truth: Dict[str, Any], abs_eta) -> np.ndarray:
    """The injected data/simulation ratio of the Muon_goodId efficiency."""
    return float(truth["sf0"]) + float(truth["sf1"]) * np.asarray(abs_eta, float)


def generated_muons(rng: np.random.Generator, n: int) -> Dict[str, np.ndarray]:
    """n Z -> mumu decays: a Breit-Wigner mass in [M_LO, M_HI], the Z's pT and rapidity, an isotropic decay; per leg
    (columns 0, 1) pt, eta, phi and charge of the generated muon."""
    lo, hi = (np.arctan(2 * (x - M_Z) / GAMMA_Z) for x in (M_LO, M_HI))
    m = M_Z + 0.5 * GAMMA_Z * np.tan(rng.uniform(lo, hi, n))
    pt_z, y_z, phi_z = rng.gamma(1.6, 7.0, n), rng.normal(0.0, 1.3, n), rng.uniform(-np.pi, np.pi, n)
    mt = np.hypot(m, pt_z)
    big_e = mt * np.cosh(y_z)
    beta = np.stack([pt_z * np.cos(phi_z), pt_z * np.sin(phi_z), mt * np.sinh(y_z)]) / big_e
    gamma = big_e / m
    cos_t, phi_d = rng.uniform(-1.0, 1.0, n), rng.uniform(-np.pi, np.pi, n)
    sin_t = np.sqrt(1.0 - cos_t ** 2)
    pstar = 0.5 * m * np.stack([sin_t * np.cos(phi_d), sin_t * np.sin(phi_d), cos_t])
    out = {k: np.empty((n, 2)) for k in ("pt", "eta", "phi")}
    for leg, sign in ((0, 1.0), (1, -1.0)):
        p = sign * pstar
        bp = np.sum(beta * p, axis=0)
        p_lab = p + (gamma ** 2 / (gamma + 1.0) * bp + gamma * 0.5 * m) * beta
        out["pt"][:, leg] = np.hypot(p_lab[0], p_lab[1])
        out["eta"][:, leg] = np.arcsinh(p_lab[2] / out["pt"][:, leg])
        out["phi"][:, leg] = np.arctan2(p_lab[1], p_lab[0])
    q = np.where(rng.random(n) < 0.5, 1, -1)
    out["charge"] = np.stack([q, -q], axis=1)
    return out


def reconstructed(rng: np.random.Generator, gen: Dict[str, np.ndarray], truth: Optional[Dict[str, Any]]) -> dict:
    """The reconstructed muons of the generated events (columns as gen), "ok" where reconstructed: the simulation's
    response, then for the data (truth given) its scale, extra resolution and identification scale factor."""
    abs_eta = np.abs(gen["eta"])
    pt = gen["pt"] * (1.0 + sim_resolution(abs_eta) * rng.standard_normal(gen["pt"].shape))
    eff = sim_efficiency(gen["pt"], abs_eta)
    if truth is not None:
        pt = pt * (1.0 + float(truth["scale"])) * (1.0 + float(truth["res_pt"]) * rng.standard_normal(pt.shape))
        eff = eff * id_truth(truth, abs_eta)
    good = rng.random(pt.shape) < eff
    return {"pt": pt, "eta": gen["eta"], "phi": gen["phi"], "charge": gen["charge"], "goodId": good,
            "ok": (pt > ACC_PT) & (abs_eta < ACC_ETA)}


def write_events(path: Path, reco: dict, weight: Optional[float] = None) -> None:
    """Events tree of the reconstructed muons, NanoAOD style (nMuon, Muon_<field>, pT descending); genWeight when
    `weight` is given (the simulation)."""
    import awkward as ak
    import uproot
    order = np.argsort(-np.where(reco["ok"], reco["pt"], -1.0), axis=1, kind="stable")
    counts = reco["ok"].sum(axis=1)
    keep = np.take_along_axis(reco["ok"], order, axis=1)
    types = {"pt": np.float32, "eta": np.float32, "phi": np.float32, "charge": np.int32, "goodId": np.bool_}
    muon = ak.zip({k: ak.unflatten(np.take_along_axis(reco[k], order, axis=1)[keep].astype(t), counts)
                   for k, t in types.items()})
    branches, kinds = {"Muon": muon}, {"Muon": muon.layout.form.type}
    if weight is not None:
        branches["genWeight"], kinds["genWeight"] = np.full(len(counts), weight, dtype=np.float32), np.float32
    path.parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(path) as f:
        f.mktree("Events", kinds, field_name=lambda outer, inner: f"{outer}_{inner}", counter_name=lambda c: f"n{c}")
        f["Events"].extend(branches)


@lru_cache(maxsize=1)
def simulation() -> dict:
    """The nominal simulation, identical for every dataset."""
    rng = np.random.default_rng(SIM_SEED)
    return reconstructed(rng, generated_muons(rng, N_SIM), None)


# ----------------------------------------------------------------------------------------- the evaluation points --

@lru_cache(maxsize=1)
def eval_muons() -> Dict[str, np.ndarray]:
    """The evaluation population: both muons of simulated Z -> mumu events with both reconstructed (pt, eta, phi,
    charge), from their own fixed draw."""
    rng = np.random.default_rng(EVAL_SEED)
    reco = reconstructed(rng, generated_muons(rng, N_EVAL), None)
    both = reco["ok"].all(axis=1)
    return {k: reco[k][both].ravel() for k in ("pt", "eta", "phi", "charge")}


def case_points(name: str) -> Dict[str, np.ndarray]:
    """The evaluation points of one requested correction: the evaluation muons inside its domain."""
    p, dom = eval_muons(), DOMAIN[name]
    m = (p["pt"] > dom["pt_min"]) & (np.abs(p["eta"]) < dom["abs_eta_max"])
    return {"pt": p["pt"][m], "eta": p["eta"][m], "phi": p["phi"][m], "charge": p["charge"][m].astype(np.int64)}


def case_truth(name: str, truth: Dict[str, Any], points: Dict[str, np.ndarray]) -> np.ndarray:
    """The injected correction at the points, in the comparison space (a resolution as its extra variance)."""
    n = len(points["pt"])
    if name == "scale":
        return np.full(n, float(truth["scale"]))
    if name == "res_pt":
        return np.full(n, float(truth["res_pt"]) ** 2)
    return id_truth(truth, np.abs(points["eta"]))


def fallback_width(name: str, priors: Dict[str, Any]) -> float:
    """The "no answer" forecast's width about nominal, in the comparison space: the RMS over the stated uniform priors
    (and, for the identification, the evaluation points) of the injected correction's deviation from nominal."""
    def moments(lo, hi):            # E[x], E[x^2] of a uniform draw
        return 0.5 * (lo + hi), (lo * lo + lo * hi + hi * hi) / 3.0
    if name == "scale":
        return float(np.sqrt(moments(*priors["scale_range"])[1]))
    if name == "res_pt":
        lo, hi = priors["res_pt_range"]
        return float(np.sqrt((hi ** 5 - lo ** 5) / (5.0 * (hi - lo))))     # sqrt(E[r^4])
    (a1, a2), (b1, b2) = moments(*(x - 1.0 for x in priors["sf0_range"])), moments(*priors["sf1_range"])
    x = np.abs(case_points(name)["eta"])
    return float(np.sqrt(np.mean(a2 + 2.0 * a1 * b1 * x + b2 * x * x)))


def request_json() -> Dict[str, Any]:
    """Every dataset's request.json: the scale and pT resolution of muons and their Muon_goodId scale factor."""
    out = {n: {"kind": KIND_OF[n], "domain": {"muon": DOMAIN[n]}} for n in ("scale", "res_pt")}
    out[ID_NAME] = {"kind": "id", "domain": {"muon": DOMAIN[ID_NAME]}, "selection": ID_NAME}
    return {"request_version": 1, "corrections": {k: out[k] for k in sorted(out)}}


class PeakCalibTask(CorrectionsTask):
    name = "peak_calib"
    pois: List[str] = []
    run_timeout_s = 600
    run_cpus = 1                         # the reference is single-threaded: ~4 s of CPU, ~120 MB
    worker_memory_mb = 2048
    # the compute yardstick: the reference timed on the public dataset in every evaluation (k: sealed and public
    # datasets are alike)
    compute_yardstick_k = 1.0
    # the post stage per dataset: no declared measurement to check (the reference declares none), the Z-peak
    # comparison takes a few times the reference's wall
    post_cpus = 2
    post_memory_mb = 4000
    verify_budget_y = 4.0
    peak_budget_y = 8.0
    # the declared measurement: optional and not refitted
    measurement_required = False
    measurement_refit = False
    evaluation_population = ("both muons of the simulation's Z → μμ events in which both are reconstructed, within "
                             "its domain ({{ref:request}})")
    example_submission = "attacker_constant_peakcalib"
    stress_submissions = ("reference", "attacker_constant_peakcalib", "attacker_nocalib_peakcalib",
                          "attacker_inflator_peakcalib", "attacker_shrinker_peakcalib", "attacker_bias_peakcalib",
                          "attacker_crasher_peakcalib", "attacker_symlink_peakcalib")
    attackers = {
        "attacker_nocalib_peakcalib": ("nocalib", "--attack-no-calib",
                                       "reports the nominal scale and resolution and leaves its scale out of the "
                                       "efficiency measurement"),
        "attacker_inflator_peakcalib": ("inflator", "--attack-inflate 3", "quotes three times every uncertainty"),
        "attacker_shrinker_peakcalib": ("shrinker", "--attack-inflate 0.3333", "quotes a third of every uncertainty"),
        "attacker_bias_peakcalib": ("bias", "--attack-bias 2", "shifts every answer by twice its own uncertainty"),
        "attacker_crasher_peakcalib": ("crasher", "--attack-crash-odd",
                                       "crashes on the datasets with an odd number of data events (about half)"),
        "attacker_symlink_peakcalib": ("symlink", "", "writes RESULT.json as a symlink to its own answer"),
        "attacker_pinner_peakcalib": ("pinner", "--attack-pin 2",
                                      "reports the nominal value with no uncertainty wherever its measurement is "
                                      "compatible with it within two sigmas"),
    }
    hand_attackers = {"attacker_constant_peakcalib": "constant"}
    attack_waivers = {"fabricator": "the declared measurement is optional and not refitted here, and the reference "
                                    "declares none: there is no declaration to fabricate"}
    review_note = ("A review case, not a stress attacker: on the nominal dataset pinning earns the clipped gain, "
                   "elsewhere its zero uncertainty is charged; the scoring's pinned flag marks it.")

    # ---- priors and truths ---------------------------------------------------------------------------------------
    def default_priors(self) -> Dict[str, Any]:
        return {"scale_range": [-0.004, 0.004], "res_pt_range": [0.0, 0.015], "sf0_range": [0.95, 1.0],
                "sf1_range": [-0.03, 0.0], "lumi_fb": LUMI_FB}

    def priors_without_response(self, priors):
        return {**priors, "scale_range": [0.0, 0.0], "res_pt_range": [0.0, 0.0], "sf0_range": [1.0, 1.0],
                "sf1_range": [0.0, 0.0]}

    def public_truth(self, priors) -> Dict[str, Any]:
        return {"scale": 0.001, "res_pt": 0.008, "sf0": 0.98, "sf1": -0.01, "lumi_fb": float(priors["lumi_fb"])}

    def draw_truth(self, rng, priors, fixed: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        f = fixed or {}
        t = {k: float(f[k]) if k in f else float(rng.uniform(*priors[f"{k}_range"]))
             for k in ("scale", "res_pt", "sf0", "sf1")}
        return {**t, "lumi_fb": float(priors["lumi_fb"])}

    # ---- data ------------------------------------------------------------------------------------------------------
    def generate(self, truth, rng, d: Path) -> None:
        n = int(rng.poisson(XSEC_PB * 1000.0 * truth["lumi_fb"]))
        write_events(d / "data" / "events.root", reconstructed(rng, generated_muons(rng, n), truth))
        write_events(d / "mc" / PROCESS / "events.root", simulation(), weight=1.0)
        (d / "mc" / PROCESS / "meta.json").write_text(json.dumps({"genEventSumw": float(N_SIM)}))
        (d / "mc" / "cross_sections.json").write_text(json.dumps({"cross_sections_pb": {PROCESS: XSEC_PB}}))
        (d / "lumi.json").write_text(json.dumps({"lumi_fb": truth["lumi_fb"]}))
        (d / "request.json").write_text(json.dumps(request_json(), indent=1))
        # the dataset's own name: its cluster (no two datasets share data; all share the simulation)
        truth["dataset"], truth["n_data_events"] = d.name, n

    # ---- scoring (the corrections kind) ------------------------------------------------------------------------------
    def function_cases(self, truth: Dict[str, Any]) -> List[Dict[str, Any]]:
        priors = self.default_priors()
        out = []
        for name in ("scale", "res_pt", ID_NAME):
            kind = KIND_OF[name]
            pts = case_points(name)
            t = case_truth(name, truth, pts)
            out.append({"role": ROLE_OF[name], "group": kind, "name": name, "flavour": "muon",
                        "cluster": truth["dataset"], "domain": DOMAIN[name], "points": pts, "truth": t,
                        "space": "signed_variance" if kind == "resolution" else "value", "tau": TAU[kind],
                        "fallback": (NOMINAL[kind], fallback_width(name, priors)), "nominal": NOMINAL[kind],
                        "null": bool(np.all(t == NOMINAL[kind])), "required_sources": REQUIRED_SOURCES})
        return out

    def function_views(self, kind: str, points: Dict[str, np.ndarray], domain: Dict[str, float]) -> Dict[str, list]:
        pt, aeta = points["pt"], np.abs(points["eta"])

        def pt_axis(edges):
            return {"name": "pt", "values": pt, "edges": [domain["pt_min"]] + [e for e in edges if e > domain["pt_min"]]}

        def eta_axis(nb):
            return {"name": "abs_eta", "values": aeta,
                    "edges": [round(float(x), 6) for x in np.linspace(0.0, domain["abs_eta_max"], nb + 1)]}
        out = {"pt": [pt_axis(CURVE_PT)], "abs_eta": [eta_axis(CURVE_ETA_BINS)]}
        if kind == "id":
            out["pt_abs_eta"] = [pt_axis(MAP_PT), eta_axis(MAP_ETA_BINS)]
        return out

    def peak_spec(self, flavour: str, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """The Z-peak closure's selection and histogram (sbh/kinds/corrections/peak.py): per event the opposite-charge
        pair of muons passing Muon_goodId closest to m_Z, both in the scale's domain once corrected, in categories of
        the pair's pT and |y|; per correction its no-answer forecast."""
        entries = (request or {}).get("corrections") or {}
        if flavour != "muon" or any(flavour not in ((entries.get(n) or {}).get("domain") or {}) for n in DOMAIN):
            return None
        priors = self.default_priors()
        return {"id": ID_NAME, "selection": entries[ID_NAME]["selection"], "pt_min": DOMAIN["scale"]["pt_min"],
                "abs_eta_max": DOMAIN["scale"]["abs_eta_max"], "raw_pt_min": 15.0, "window": [60.0, 120.0],
                "mass": [80.0 + 0.5 * i for i in range(45)],
                "categories": {"pt_pair": [0.0, 8.0, 20.0, 1e4], "abs_y_pair": [0.0, 1.2, 3.0]},
                "fallback": {n: (NOMINAL[KIND_OF[n]], fallback_width(n, priors)) for n in DOMAIN}}

    def function_baseline_id(self) -> str:
        p = eval_muons()
        sha = hashlib.sha256(b"".join(np.ascontiguousarray(p[k], dtype=np.float64).tobytes()
                                      for k in ("pt", "eta", "phi", "charge"))).hexdigest()[:16]
        return f"evaluation points sha256 {sha}; tau {json.dumps(TAU, sort_keys=True)}"

    # ---- the results page --------------------------------------------------------------------------------------------
    def report_truth(self, truth: Dict[str, Any]) -> Dict[str, Any]:
        return {k: truth.get(k) for k in ("dataset", "scale", "res_pt", "sf0", "sf1")}

    def report_page(self) -> Dict[str, Any]:
        return {
            "title": "HEP-Ex-Bench peak_calib", "heading": "HEP-Ex-Bench peak_calib (toy)",
            "lede": ("A toy of the corrections kind: a muon momentum scale, pT resolution and identification scale "
                     "factor measured from a synthetic Z → μμ peak and reported as correctionlib functions with "
                     "uncertainties, scored against injected truths."),
            "datasets": f"~{XSEC_PB * LUMI_FB * 1000:,.0f} Z → μμ events each",
            "seeds_intro": ("Every dataset is synthetic Z → μμ data with one nominal simulation shared by all; the "
                            "data carry a momentum scale, an extra pT resolution and an identification scale factor "
                            "linear in |η|, drawn per dataset."),
            "labels": {"scale": "momentum scale", "resolution": "resolution", "id": "identification",
                       "peak": "Z-peak closure", "scale_mu": "μ momentum scale", "res_pt_mu": "μ pT resolution",
                       "id_mu": "μ identification", "pt": "muon pT [GeV]", "abs_eta": "muon |η|",
                       "mass": "m(μμ) [GeV]", "muon": "μ"},
            "glossary": {
                "scored": ("The three requested corrections on every dataset, each at the evaluation points inside "
                           "its domain: simulated Z → μμ muons."),
                "peak": ("## Z-peak closure\nThe reported scale, resolution and identification scale factor applied "
                         "together to the simulation's Z → μμ pairs, against the data's dimuon mass (80–102 GeV) in "
                         "6 categories of the pair's pT and |y|; Δ = the log loss above the reference's on the same "
                         "dataset, capped.  A correction not answered enters at its no-answer forecast.")}}


TASK = PeakCalibTask()
