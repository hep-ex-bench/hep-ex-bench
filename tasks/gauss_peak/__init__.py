"""gauss_peak -- a toy sealed-truth task with the output contract of the H->ZZ*->4l benchmark; its shapes, yields
and priors are its own.

One observable m in [100, 160] GeV: exponential background (~2000 x bkg_norm) plus a 2 GeV Gaussian signal at mH
(mu x 120 exp(-(mH-125)/50) events).  Data and a 200k-event "Z -> ll" control peak share a mass scale s and extra
smearing r, to be measured on the control and transferred; the MC library is nominal and identical for every seed.
Sealed truth: mH, mu, bkg_norm, scale, smear.  POIs: mH, mu.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from sbh.task import Task

M_LO, M_HI = 100.0, 160.0
BKG_DECAY = 30.0
SIG_RES = 2.0
N_BKG_EXP = 2000.0
Z_MASS, Z_WIDTH = 91.19, 2.5
C_LO, C_HI = 70.0, 110.0
CTRL_FLAT_FRAC = 0.05
N_CTRL_EXP = 200_000
SIG_GRID = [115.0, 120.0, 125.0, 130.0, 135.0]
N_MC_SIG, N_MC_BKG, N_MC_CTRL = 20_000, 200_000, 200_000
MC_SEED = 20260917          # the MC library is identical for every seed and suite
# The point-mass tripwire's rule settings (sbh/tripwire.py SETTINGS) for the toy: mH in GeV, reported to 0.01 GeV or
# on a coarser scan; a mass rule counts the datasets whose own significance is 3 or more
TRIPWIRE_SETTINGS = {"alpha": 0.01, "eps": 0.01, "tight": 0.5, "min_tight": 2, "pin_z": 1.0, "z_signal": 3.0,
                     "zero_tol": 1e-6, "half": 1.0, "steps": (1.0, 0.5, 0.25, 0.1, 0.05, 0.01), "min_step_evidence": 3}


def n_sm(m):
    return 120.0 * np.exp(-(np.asarray(m, float) - 125.0) / 50.0)


def expected_significance(truth) -> float:
    """Asimov local significance of the injected signal in a +-2 sigma mass window."""
    mH, mu = float(truth["mH"]), float(truth["mu"])
    if mu <= 0:
        return 0.0
    width = float(np.sqrt(SIG_RES ** 2 + (float(truth.get("smear", 0.0)) * mH) ** 2))
    lo, hi = mH - 2 * width, mH + 2 * width
    norm = 1.0 - np.exp(-(M_HI - M_LO) / BKG_DECAY)
    cdf = lambda x: (1.0 - np.exp(-(x - M_LO) / BKG_DECAY)) / norm
    b = N_BKG_EXP * float(truth.get("bkg_norm", 1.0)) * (cdf(hi) - cdf(lo))
    s = 0.9545 * mu * float(n_sm(mH))
    return float(np.sqrt(2.0 * ((s + b) * np.log1p(s / b) - s)))


def trunc_exp(rng, n, lo, hi, decay):
    u = rng.random(n)
    return lo - decay * np.log1p(-u * (1.0 - np.exp(-(hi - lo) / decay)))


def trunc_norm(rng, n, mean, sigma, lo, hi):
    out = np.empty(0)
    while out.size < n:
        x = rng.normal(mean, sigma, int(1.3 * (n - out.size)) + 16)
        out = np.concatenate([out, x[(x >= lo) & (x < hi)]])
    return out[:n]


def control_sample(rng, n):
    nflat = int(rng.binomial(n, CTRL_FLAT_FRAC))
    z = trunc_norm(rng, n - nflat, Z_MASS, Z_WIDTH, C_LO, C_HI)
    return np.concatenate([z, rng.uniform(C_LO, C_HI, nflat)])


def distort(rng, m, scale, smear):
    return m * scale * (1.0 + smear * rng.normal(size=m.size))


def window(m, lo, hi):
    return m[(m >= lo) & (m < hi)]


@lru_cache(maxsize=1)
def mc_library():
    rng = np.random.default_rng(MC_SEED)
    sig = {m: trunc_norm(rng, N_MC_SIG, m, SIG_RES, M_LO, M_HI) for m in SIG_GRID}
    bkg = trunc_exp(rng, N_MC_BKG, M_LO, M_HI, BKG_DECAY)
    ctrl = control_sample(rng, N_MC_CTRL)
    return sig, bkg, ctrl


class GaussPeakTask(Task):
    kind = "pois"
    signal_strength_poi = "mu"
    name = "gauss_peak"
    # an offline verifier on a fixed runtime takes packages vendored with `pip install --target pylib`, as the toy's
    # hand-written contract says
    vendor_dir = "pylib"
    pois = ["mH", "mu"]
    result_type = "search"
    nuisance_prefix = "gauss"
    # Ignorance forecast from the shipped (participant-facing) priors: center +- width/sqrt(12).
    fallback_forecast = {"mH": (125.0, 5.774), "mu": (1.5, 0.866)}
    attackers = {
        "attacker_crasher": ("crasher", "--crash-odd-seeds",
                             "crashes on the datasets with an odd number of events (about half)"),
        "attacker_inflator": ("inflator", "--inflate 3", "quotes three times the uncertainty (over-coverage)"),
        "attacker_memorizer": ("memorizer", "--fix-mass 125",
                               "assumes the resonance is at 125 GeV (the memorized answer) and only fits mu"),
        "attacker_nocalib": ("nocalib", "--no-calib", "skips the in-situ calibration and trusts the nominal response"),
        "attacker_symlink": ("symlink", "", "writes RESULT.json as a symlink to its own answer"),
    }
    hand_attackers = {"attacker_constant": "constant"}

    # Sealed-safe results page: the steps of the design grid the sealed truths
    # sit on (truth_grids/grid_v0.json: integer masses, mu in halves; a measured value on it is guessable) and the
    # injected values a fallback charge may sit at.
    sealed_value_steps = {"mH": 1.0, "mu": 0.5}

    def sealed_value_grid(self, priors: Dict[str, Any]) -> Dict[str, list]:
        lo, hi = priors.get("mu_range") or [0.0, 3.0]
        return {"mu": [lo + 0.5 * i for i in range(int(round((hi - lo) / 0.5)) + 1)]}

    @property
    def tripwire_spec(self) -> Dict[str, Any]:
        """Point-mass tripwire (flag-only): the sealed masses sit on the design grid, nulls at mu = 0; 125 GeV is the
        value a pipeline could paste in (the memorizer's)."""
        grid = json.loads((Path(__file__).parent / "truth_grids" / "grid_v0.json").read_text())
        return {"poi": "mH", "points": tuple(sorted({float(x["mH"]) for x in grid["seeds"]})), "literature": (125.0,),
                "density_floor": 0.1, "null_poi": "mu", "null_value": 0.0, "settings": TRIPWIRE_SETTINGS}

    def calibration_truth(self, truth, log=None):
        # diagnostics only (no calibration term for the toy)
        return {"gauss_scale": float(truth["scale"]), "gauss_smear": float(truth["smear"])}

    def report_page(self) -> Dict[str, Any]:
        """The base page facts, the POIs' names, and the glossary entries of a score with physics and compute only."""
        return {**super().report_page(),
                "cases": {"mH": {"name": "Resonance mass mH", "symbol": "m<sub>H</sub>", "text": "mH", "unit": "GeV",
                                 "digits": 1, "unscored": f"expected significance below {self.APPLICABLE_Z:g}σ"},
                          "mu": {"name": "Signal strength μ", "symbol": "μ", "text": "μ", "unit": "", "digits": 2}},
                "glossary": {
                    "score": ("## Score\n"
                              "A percentage of the reference's score on the same datasets, with its ±1σ range: 16–84 % "
                              "of the ratio over resamples of the datasets, the run and the reference rescored on each.  "
                              "Two parts, each a mean log-loss Δ (0 = reference "
                              "quality):\n"
                              "- physics results: ln σ + pull²/2 on the reported mH and μ against a forecast at the "
                              "reference's precision; per case between {clip} and max(F, {cap}) (precision credited "
                              "down to σ_ref/{ratio}, the pull always counted), F = the charge for answering the "
                              "stated prior; per POI the datasets' inverse-variance combination, its squared pull "
                              "beyond 1 charged per dataset: a bias b (in σ) shared by all of them costs about b², the "
                              "cases alone b²/2\n"
                              "- compute (weight {w.compute}): log2 of the wall time per dataset over the reference's: "
                              "nothing up to 1.5× the reference's, a charge above it up to the time limit, no reward "
                              "for running faster\n"
                              "S = {score_form}; {gx} is {gratio}.\n"
                              "- failed dataset: the whole score is 0; a dataset the evaluation lost (no record) "
                              "leaves it provisional until rerun"),
                    "scored": ("- μ: every dataset\n"
                               "- mH: every dataset whose injected signal has an expected significance of "
                               f"{self.APPLICABLE_Z:g}σ or more\n"
                               "- the same set for every run: it depends on the truth alone"),
                    "fdr": ("Fraction of μ = 0 datasets on which the run claims a discovery.\n"
                            "~ The claim counted is the run's own, in RESULT.json.")}}

    def poi_scored_v2(self, poi, truth):
        # v2 log-score: mu priced on every seed, mass keeps the gate.
        if poi == "mu":
            return "mu" in truth
        return self.poi_applicable(poi, truth)

    def default_priors(self) -> Dict[str, Any]:
        return {"mH_range": [115.0, 135.0], "mu_range": [0.0, 3.0], "null_fraction": 0.2,
                "bkg_norm_range": [0.9, 1.1], "scale_range": [0.985, 1.015],
                "smear_range": [0.0, 0.015], "lumi_fb": 20.0}

    def priors_without_response(self, priors):
        q = dict(priors)
        q["scale_range"], q["smear_range"] = [1.0, 1.0], [0.0, 0.0]
        return q

    def public_truth(self, priors):
        return {"mH": 125.0, "mu": 1.0, "bkg_norm": 1.0, "scale": 1.0, "smear": 0.0,
                "lumi_fb": float(priors["lumi_fb"])}

    def draw_truth(self, rng, priors, fixed: Optional[Dict[str, Any]] = None):
        f = fixed or {}
        t: Dict[str, Any] = {}
        t["mH"] = float(f.get("mH", rng.uniform(*priors["mH_range"])))
        if "mu" in f:
            t["mu"] = float(f["mu"])
        else:
            t["mu"] = 0.0 if rng.random() < priors["null_fraction"] else float(rng.uniform(*priors["mu_range"]))
        t["bkg_norm"] = float(f.get("bkg_norm", rng.uniform(*priors["bkg_norm_range"])))
        t["scale"] = float(f.get("scale", rng.uniform(*priors["scale_range"])))
        t["smear"] = float(f.get("smear", rng.uniform(*priors["smear_range"])))
        t["lumi_fb"] = float(priors["lumi_fb"])
        return t

    APPLICABLE_Z = 5.0

    def poi_applicable(self, poi, truth):
        return expected_significance(truth) >= self.APPLICABLE_Z

    def generate(self, truth, rng, d: Path):
        for sub in ("data", "control", "mc"):
            (d / sub).mkdir(parents=True, exist_ok=True)
        s, r = truth["scale"], truth["smear"]
        n_sig = int(rng.poisson(truth["mu"] * float(n_sm(truth["mH"]))))
        n_bkg = int(rng.poisson(N_BKG_EXP * truth["bkg_norm"]))
        sig = trunc_norm(rng, n_sig, truth["mH"], SIG_RES, M_LO, M_HI) if n_sig else np.empty(0)
        bkg = trunc_exp(rng, n_bkg, M_LO, M_HI, BKG_DECAY)
        ev = window(distort(rng, np.concatenate([sig, bkg]), s, r), M_LO, M_HI)
        rng.shuffle(ev)
        np.save(d / "data" / "events.npy", ev.astype(np.float64))
        ctrl = window(distort(rng, control_sample(rng, int(rng.poisson(N_CTRL_EXP))), s, r), C_LO, C_HI)
        rng.shuffle(ctrl)
        np.save(d / "control" / "events.npy", ctrl.astype(np.float64))
        sig_mc, bkg_mc, ctrl_mc = mc_library()
        for m, arr in sig_mc.items():
            np.save(d / "mc" / f"signal_mH{int(m)}.npy", arr)
        np.save(d / "mc" / "background.npy", bkg_mc)
        np.save(d / "mc" / "control_mc.npy", ctrl_mc)
        xs = {
            "lumi_fb": truth["lumi_fb"],
            "signal_expected_SM": {f"{m:.1f}": float(n_sm(m)) for m in np.arange(105.0, 150.01, 1.0)},
            "signal_expected_SM_note": "expected signal events in the full dataset at mu = 1 as a function of the "
                                       "resonance mass; interpolate linearly between tabulated masses",
            "signal_mc_note": "mc/signal_mH<M>.npy are nominal (no detector response) templates at the grid masses",
            "background_expected": N_BKG_EXP,
            "background_note": "nominal expectation; the true normalisation differs by up to ~10% and must be "
                               "determined from the data",
            "control": {"expected_events": N_CTRL_EXP, "peak_mass": Z_MASS,
                        "flat_background_fraction": CTRL_FLAT_FRAC,
                        "note": "control/events.npy and data/events.npy carry the same detector response; "
                                "mc/control_mc.npy carries none"},
        }
        (d / "mc" / "cross_sections.json").write_text(json.dumps(xs, indent=1))
        (d / "lumi.json").write_text(json.dumps({"lumi_fb": truth["lumi_fb"]}))
        truth["n_signal_realized"], truth["n_background_realized"] = n_sig, n_bkg
        truth["expected_significance"] = round(expected_significance(truth), 3)


TASK = GaussPeakTask()
