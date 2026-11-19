"""bump_model -- a toy sealed-truth task of the binned_likelihood kind.

One observable, the reconstructed pair mass m in [100, 160] GeV: an exponential background (~1500 x bkg_norm events
in the window) plus a 1.5 GeV Gaussian resonance at mH (mu x 50 exp(-(mH-125)/80) events).  Data and a 50k-event
"Z -> ll" control peak share a lepton momentum scale s, to be measured on the control and transferred; the MC library
is nominal and identical for every seed.  The pipeline declares its binned likelihood (MODEL.json, the mass grid
MODEL_GRID/, EVENTS.json); the evaluator scores mu from its refit of MODEL.json and mH from the grid's profile.
Sealed truth: mH, mu, bkg_norm, scale.  POIs: mH, mu.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from sbh.task import Task

M_LO, M_HI = 100.0, 160.0          # the data's window in reconstructed mass
GEN_LO, GEN_HI = 90.0, 170.0       # true background range: the window stays filled under any scale
BKG_DECAY = 25.0
N_BKG_EXP = 2400.0                 # nominal background events over [GEN_LO, GEN_HI)
SIG_RES = 1.5
Z_MASS, Z_WIDTH = 91.19, 2.5
C_LO, C_HI = 75.0, 105.0
CTRL_FLAT_FRAC = 0.05
N_CTRL_EXP = 50_000
SIG_GRID = [110.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0]
N_MC_SIG, N_MC_BKG, N_MC_CTRL = 20_000, 100_000, 100_000
MC_SEED = 20261028                 # the MC library is identical for every seed and suite
DATA_FILE = "part_0000.root"       # data/<DATA_FILE>, tree Events, branch mass


def n_sm(m):
    """Expected signal events at mu = 1 for a resonance of mass m."""
    return 50.0 * np.exp(-(np.asarray(m, float) - 125.0) / 80.0)


def expected_significance(truth) -> float:
    """Asimov local significance of the injected signal in a +-2 sigma mass window."""
    mH, mu = float(truth["mH"]), float(truth["mu"])
    if mu <= 0:
        return 0.0
    lo, hi = mH - 2 * SIG_RES, mH + 2 * SIG_RES
    norm = 1.0 - np.exp(-(GEN_HI - GEN_LO) / BKG_DECAY)
    cdf = lambda x: (1.0 - np.exp(-(x - GEN_LO) / BKG_DECAY)) / norm
    b = N_BKG_EXP * float(truth.get("bkg_norm", 1.0)) * (cdf(hi) - cdf(lo))
    s = 0.9545 * mu * float(n_sm(mH))
    return float(np.sqrt(2.0 * ((s + b) * np.log1p(s / b) - s)))


def trunc_exp(rng, n, lo, hi, decay):
    u = rng.random(n)
    return lo - decay * np.log1p(-u * (1.0 - np.exp(-(hi - lo) / decay)))


def control_sample(rng, n):
    nflat = int(rng.binomial(n, CTRL_FLAT_FRAC))
    z = rng.normal(Z_MASS, Z_WIDTH, n - nflat)
    return np.concatenate([z, rng.uniform(C_LO - 5.0, C_HI + 5.0, nflat)])


def window(m, lo, hi):
    return m[(m >= lo) & (m < hi)]


@lru_cache(maxsize=1)
def mc_library():
    rng = np.random.default_rng(MC_SEED)
    sig = {m: rng.normal(m, SIG_RES, N_MC_SIG) for m in SIG_GRID}
    bkg = trunc_exp(rng, N_MC_BKG, GEN_LO, GEN_HI, BKG_DECAY)
    ctrl = window(control_sample(rng, N_MC_CTRL), C_LO, C_HI)
    return sig, bkg, ctrl


def write_events(path: Path, mass: np.ndarray) -> None:
    """A ROOT file with one tree, Events, and one branch, mass."""
    import uproot
    with uproot.recreate(path) as f:
        f["Events"] = {"mass": mass.astype(np.float64)}


class BumpModelTask(Task):
    kind = "binned_likelihood"
    signal_strength_poi = "mu"
    name = "bump_model"
    pois = ["mH", "mu"]
    result_type = "search"
    nuisance_prefix = "bump"
    unified_contract = True              # EVAL_CONTRACT.md rendered from tasks/EVAL_CONTRACT.in.md
    requires_model = True
    event_list_from = "0.8"
    cluster_key = None                   # every dataset its own draw
    # Worker spec (single source; the contract's Budgets paragraph states it): the reference takes seconds.
    run_cpus = 2
    worker_memory_mb = 2048
    run_memory_mb = 0
    run_timeout_s = 300
    # Ignorance forecast from what a participant is told: mH in [115, 135] -> 125 +- 20/sqrt(12); mu in [0, 3].
    fallback_forecast = {"mH": (125.0, 5.774), "mu": (1.5, 0.866)}
    # mu scored on the evaluator's fit of MODEL.json, mH on its profile over MODEL_GRID; RESULT.json's POIs enter only
    # the truthfulness and claims terms.
    model_poi_sources = {"mu": "fit", "mH": "grid"}
    # The contract makes any output over its size cap, MODEL_GRID/ among them, "a failed run on that dataset"
    grid_over_cap_fails = True
    # The contract's evaluation mass range for mH (EVAL_CONTRACT.md section 3): a flat profile is bounded by it.
    mass_range = (110.0, 140.0)
    # the model verification's signal-region GOF window (the contract's binning range)
    gof_window = (105.0, 140.0)

    reference_label = "reference_bump"
    attackers = {
        "attacker_bump_crasher": ("crasher", "--crash-odd-seeds",
                                  "crashes on the datasets with an odd number of events (about half)"),
        "attacker_bump_inflator": ("inflator", "--inflate 3",
                                   "reports three times the uncertainties its model gives (over-coverage)"),
        "attacker_bump_memorizer": ("memorizer", "--fix-mass 125",
                                    "assumes the resonance is at 125 GeV (the memorized answer), fits only mu and "
                                    "declares no mass grid"),
        "attacker_bump_nocalib": ("nocalib", "--no-calib",
                                  "skips the in-situ calibration and trusts the nominal response"),
        "attacker_bump_symlink": ("symlink", "", "writes RESULT.json as a symlink to its own answer"),
    }
    hand_attackers = {"attacker_bump_constant": "constant"}
    example_submission = "attacker_bump_constant"

    # Sealed-safe results page: the steps of the design grid the sealed truths
    # sit on (truth_grids/grid_v0.json: integer masses, mu in halves) and the injected values a charge may sit at.
    sealed_value_steps = {"mH": 1.0, "mu": 0.5}

    def sealed_value_grid(self, priors: Dict[str, Any]) -> Dict[str, list]:
        lo, hi = priors.get("mu_range") or [0.0, 3.0]
        return {"mu": [lo + 0.5 * i for i in range(int(round((hi - lo) / 0.5)) + 1)]}

    # Point-mass tripwire (sbh/tripwire.py; flag-only): the sealed masses sit on whole GeV in the prior, nulls at
    # mu = 0; no published value.
    tripwire_spec = {"poi": "mH", "points": tuple(float(m) for m in range(115, 136)), "literature": (),
                     "density_floor": 0.1, "null_poi": "mu", "null_value": 0.0,
                     "settings": {"alpha": 0.01, "eps": 0.005, "tight": 0.5, "min_tight": 2, "pin_z": 0.5,
                                  "z_signal": 3.0, "zero_tol": 1e-6, "half": 1.0,
                                  "steps": (1.0, 0.5, 0.25, 0.1, 0.05, 0.01), "min_step_evidence": 3}}

    def contract_values(self) -> Dict[str, str]:
        return super().contract_values() | {"observable": "the pair mass"}

    def report_page(self) -> Dict[str, Any]:
        """The base page facts, the POIs' names, and the glossary entries of the toy's score."""
        return {**super().report_page(),
                "cases": {"mH": {"name": "Resonance mass mH", "symbol": "m<sub>H</sub>", "text": "mH", "unit": "GeV",
                                 "digits": 1, "unscored": "no signal injected"},
                          "mu": {"name": "Signal strength μ", "symbol": "μ", "text": "μ", "unit": "", "digits": 2}},
                "glossary": {
                    "score": ("## Score\n"
                              "A percentage of the reference's score on the same datasets, with its ±1σ range: 16–84 % "
                              "of the ratio over resamples of the datasets, the run and the reference rescored on each.  "
                              "Four parts, each a mean log-loss Δ (0 = reference "
                              "quality):\n"
                              "- physics results: ln σ + pull²/2 on what the evaluator gets from the run's declared "
                              "model (μ from its fit of MODEL.json, mH from the declared mass grid) against a forecast "
                              "at the reference's precision; per case between {clip} and max(F, {cap}) (precision "
                              "credited down to σ_ref/{ratio}, the pull always counted), F = the charge for answering "
                              "the stated prior; per POI the datasets' inverse-variance combination, its squared pull "
                              "beyond 1 charged per dataset: a bias b (in σ) shared by all of them costs about b², the "
                              "cases alone b²/2\n"
                              "- truthfulness (weight {w.truthfulness}): the reported values and σ against the "
                              "model's\n"
                              "- other claims (weight {w.claims}): significances, discovery and μ stat/syst against "
                              "the model\n"
                              "- model fit (weight {w.gof}): the toy-calibrated goodness of fit of the declared "
                              "models\n"
                              "S = {score_form}; {gx} is {gratio}.\n"
                              "- failed dataset: the whole score is 0; a dataset the evaluation lost (no record) "
                              "leaves it provisional until rerun\n"
                              "- no usable mass grid on a dataset that completed: no mass measurement, charged as "
                              "the no-answer forecast (F), the dataset not failed"),
                    "scored": ("- μ: every dataset\n"
                               "- mH: every dataset with an injected signal (μ > 0), weak ones included: their mass "
                               "interval is bounded by the 110–140 GeV mass range\n"
                               "- the same set for every run: it depends on the truth alone")}}

    def default_priors(self) -> Dict[str, Any]:
        return {"mH_range": [115.0, 135.0], "mu_range": [0.0, 3.0], "null_fraction": 0.2,
                "bkg_norm_range": [0.9, 1.1], "scale_range": [0.99, 1.01], "lumi_fb": 30.0}

    def priors_without_response(self, priors):
        q = dict(priors)
        q["scale_range"] = [1.0, 1.0]
        return q

    def public_truth(self, priors):
        return {"mH": 125.0, "mu": 1.0, "bkg_norm": 1.0, "scale": 1.0, "lumi_fb": float(priors["lumi_fb"])}

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
        t["lumi_fb"] = float(priors["lumi_fb"])
        return t

    APPLICABLE_Z = 5.0

    def poi_applicable(self, poi, truth):
        # diagnostic gate only (the score uses poi_scored_v2)
        return expected_significance(truth) >= self.APPLICABLE_Z

    def poi_scored_v2(self, poi, truth):
        # mu on every dataset; mH wherever a signal was injected (a weak signal's wide grid interval is accurate)
        if poi == "mu":
            return "mu" in truth
        return poi in truth and float(truth.get("mu", 0.0)) > 0.0

    def generate(self, truth, rng, d: Path):
        for sub in ("data", "control", "mc"):
            (d / sub).mkdir(parents=True, exist_ok=True)
        s = truth["scale"]
        n_sig = int(rng.poisson(truth["mu"] * float(n_sm(truth["mH"]))))
        n_bkg = int(rng.poisson(N_BKG_EXP * truth["bkg_norm"]))
        sig = rng.normal(truth["mH"], SIG_RES, n_sig)
        bkg = trunc_exp(rng, n_bkg, GEN_LO, GEN_HI, BKG_DECAY)
        ev = window(s * np.concatenate([sig, bkg]), M_LO, M_HI)
        rng.shuffle(ev)
        write_events(d / "data" / DATA_FILE, ev)
        ctrl = window(s * control_sample(rng, int(rng.poisson(N_CTRL_EXP))), C_LO, C_HI)
        rng.shuffle(ctrl)
        np.save(d / "control" / "events.npy", ctrl.astype(np.float64))
        sig_mc, bkg_mc, ctrl_mc = mc_library()
        for m, arr in sig_mc.items():
            np.save(d / "mc" / f"signal_mH{int(m)}.npy", arr)
        np.save(d / "mc" / "background.npy", bkg_mc)
        np.save(d / "mc" / "control_mc.npy", ctrl_mc)
        xs = {
            "lumi_fb": truth["lumi_fb"],
            "signal_expected_SM": {f"{m:.1f}": round(float(n_sm(m)), 4) for m in np.arange(105.0, 145.01, 1.0)},
            "signal_expected_SM_note": "expected signal events in the dataset at mu = 1 as a function of the "
                                       "resonance mass (all of them inside the data window); interpolate linearly "
                                       "between tabulated masses",
            "signal_mc_note": "mc/signal_mH<M>.npy are nominal (no detector response) signal masses at the grid "
                              "masses",
            "background_expected": N_BKG_EXP,
            "background_note": f"nominal expectation over the whole range of mc/background.npy, [{GEN_LO:g}, "
                               f"{GEN_HI:g}) GeV before the detector response; the data keep the events "
                               f"reconstructed in [{M_LO:g}, {M_HI:g}) GeV.  The true normalisation differs by up "
                               "to ~10% and must be determined from the data",
            "control": {"expected_events": N_CTRL_EXP, "peak_mass": Z_MASS,
                        "flat_background_fraction": CTRL_FLAT_FRAC,
                        "note": "control/events.npy and the data carry the same lepton momentum scale; "
                                "mc/control_mc.npy carries none"},
        }
        (d / "mc" / "cross_sections.json").write_text(json.dumps(xs, indent=1))
        (d / "lumi.json").write_text(json.dumps({"lumi_fb": truth["lumi_fb"]}))
        truth["n_signal_realized"], truth["n_background_realized"] = n_sig, n_bkg
        truth["expected_significance"] = round(expected_significance(truth), 3)


TASK = BumpModelTask()
