"""width_toy -- a fast toy sealed-truth task with the structure of the off-shell Higgs width measurement (h4l_width);
its shapes, yields and priors are its own.

One observable, the "four-lepton mass" m in [100, 1000] GeV, and two POIs: the width ratio xi = Gamma_H / Gamma_SM
and the on-shell signal strength mu_on.  Expected event density (per GeV) at the sealed truth:

    T(m; xi) = bkg_norm * B_qq(m) + B_gg(m) + sqrt(mu_on xi) I(m) + mu_on xi S(m) + mu_on P(m)

  B_qq   "qq -> ZZ" continuum: a Z-pair threshold at ~182 GeV with a falling tail, plus a low-mass tail
  B_gg   "gg -> ZZ" continuum (interferes with the signal)
  S      off-shell "gg -> H* -> ZZ" signal: rising above ~2 m_Z, broad
  I      their interference, I = -KAPPA sqrt(B_gg S) < 0 (KAPPA < 2 keeps T >= 0 for every xi)
  P      the on-shell peak at 125 GeV (scales with mu_on only)

The MC library (identical for every seed, no ``xi != 1`` sample) mirrors the real task's: B_qq, B_gg, S at xi = 1,
the S+B_gg+I sample at xi = 1 and the on-shell peak, unit-weight events plus their expected yields.  The template
formula T(xi) = B + sqrt(xi) (SBI - S - B_gg) + xi S follows from the stated definitions.

Sealed truth per seed: xi (the design file's values), mu_on (uniform in the priors' range unless the design names
it), and bkg_norm (the qq -> ZZ normalisation, logged as cross_section_VV and value-scored by the calibration
term where the declared model carries it).  The output kind is scan_likelihood: RESULT.json carries an interval per
POI, MODEL.json the fitted likelihood with its POI scaling, EVENTS.json the events behind its observed counts; the
evaluator measures both POIs on the declared model.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from sbh.kinds.scan_likelihood import range_text
from sbh.task import Task

M_LO, M_HI = 100.0, 1000.0
KAPPA = 1.6                     # interference strength: I = -KAPPA sqrt(B_gg S); T >= (1 - KAPPA^2 / 4) B_gg > 0
N_QQ, N_GG, N_S, N_ON = 2000.0, 300.0, 15.0, 40.0     # expected events at xi = mu_on = bkg_norm = 1
ON_MASS, ON_RES = 125.0, 1.5
LUMI_UNC_REL = 0.025            # the luminosity's relative uncertainty, stated in lumi.json
N_MC = {"B_qq": 100_000, "B_gg": 50_000, "S": 50_000, "SBI_xi1": 50_000, "H_onshell": 20_000}
MC_SEED = 7919                  # the MC library is identical for every seed and suite
GRID = np.linspace(M_LO, M_HI, 7201)     # fine grid (0.125 GeV) on which the densities are tabulated
_C = 0.5 * (GRID[1:] + GRID[:-1])
_W = np.diff(GRID)


def _thr(m, m0, w):
    return 1.0 / (1.0 + np.exp(-(m - m0) / w))


def _unit(f):
    return f / np.sum(f * _W)


# unit-normalised shapes on the fine grid's cell centres (per GeV)
F_QQ = _unit(_thr(_C, 182.0, 6.0) * np.exp(-(_C - 182.0) / 110.0) + 0.25 * np.exp(-(_C - 100.0) / 25.0))
# B_gg soft, S hard (the off-shell signal's share grows with m4l, as in the real process): the shape difference breaks
# the yield degeneracy N(xi) = c - |N_I| sqrt(xi) + N_S xi between small and large xi (with B_gg and S of similar
# shape the toy's likelihood is bimodal, xi ~ 0 vs ~ 40 on Asimov data)
F_GG = _unit(_thr(_C, 185.0, 8.0) * np.exp(-(_C - 185.0) / 80.0))
F_S = _unit(_thr(_C, 200.0, 12.0) * np.clip((_C - 150.0) / 100.0, 0.0, None) * np.exp(-(_C - 200.0) / 400.0))
F_ON = _unit(np.exp(-0.5 * ((_C - ON_MASS) / ON_RES) ** 2))


def densities(xi: float, mu_on: float = 1.0, bkg_norm: float = 1.0) -> np.ndarray:
    """T(m) per GeV on the fine grid's cells."""
    bq, bg, s = N_QQ * F_QQ, N_GG * F_GG, N_S * F_S
    i = -KAPPA * np.sqrt(bg * s)
    g = mu_on * xi
    return bkg_norm * bq + bg + np.sqrt(g) * i + g * s + mu_on * N_ON * F_ON


def n_interference() -> float:
    return float(np.sum(-KAPPA * np.sqrt(N_GG * F_GG * N_S * F_S) * _W))


def sample(rng: np.random.Generator, dens: np.ndarray, n: int) -> np.ndarray:
    """n events from a non-negative per-GeV density on the fine grid (piecewise constant: a cell by its content, then
    uniform within it)."""
    w = np.clip(dens, 0.0, None) * _W
    cdf = np.cumsum(w)
    cdf /= cdf[-1]
    k = np.searchsorted(cdf, rng.random(n), side="right")
    k = np.minimum(k, len(_W) - 1)
    return GRID[k] + rng.random(n) * _W[k]


@lru_cache(maxsize=1)
def mc_library() -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(MC_SEED)
    bq, bg, s = N_QQ * F_QQ, N_GG * F_GG, N_S * F_S
    sbi = bg + s - KAPPA * np.sqrt(bg * s)
    shapes = {"B_qq": bq, "B_gg": bg, "S": s, "SBI_xi1": sbi, "H_onshell": N_ON * F_ON}
    return {k: sample(rng, shapes[k], N_MC[k]) for k in N_MC}


def mc_yields() -> Dict[str, float]:
    return {"B_qq": N_QQ, "B_gg": N_GG, "S": N_S, "SBI_xi1": N_GG + N_S + n_interference(), "H_onshell": N_ON}


class WidthToyTask(Task):
    name = "width_toy"
    kind = "scan_likelihood"
    unified_contract = True
    pois = ["xi", "mu_on"]               # the evaluator scans xi and fits mu_on
    result_type = "measurement"
    nuisance_prefix = "ana"
    poi_bounds = {"xi": (0.0, 100.0), "mu_on": (0.0, 5.0)}
    requires_model = True
    event_list_from = "2.2"
    # the model verification's GOF window: the off-shell region, above the Z-pair threshold
    gof_window = (182.0, 1000.0)
    model_verify_toys = 200
    # each dataset's model is graded within twice the reference's slowest (closure.json)
    model_verify_cap = True
    # worker spec (the unified contract's Budgets): seconds per dataset on two CPUs
    run_timeout_s = 300
    run_cpus = 2
    worker_memory_mb = 4096
    run_memory_mb = 0
    example_submission = "attacker_width_constant"      # data-blind, no analysis code
    attackers = {
        "attacker_width_crasher": ("crasher", "--crash-odd-seeds",
                                   "crashes on the datasets with an odd number of events (about half)"),
        "attacker_width_inflator": ("inflator", "--inflate 3",
                                    "reports three times the uncertainties its model gives (over-coverage)"),
        "attacker_width_deflator": ("deflator", "--deflate 3", "reports a third of the uncertainties its model gives"),
        "attacker_width_misreport": ("misreport", "--misreport",
                                     "reports each value two upper errors above its model's best fit"),
        "attacker_width_fabricate": ("fabricate", "--fabricate",
                                     "declares the Standard-Model expectation as its observed counts"),
        "attacker_width_token": ("token", "--token", "declares the luminosity at a millionth of its uncertainty"),
        "attacker_width_misfit": ("misfit", "--misfit", "fits and declares a flat qq -> ZZ template"),
        "attacker_width_memorizer": ("memorizer", "--memorize-mu-on",
                                     "holds the on-shell signal strength at 1 (the public dataset's) and fits the rest"),
        "attacker_width_nocalib": ("nocalib", "--no-calib", "holds the qq -> ZZ normalisation at its nominal 1"),
        "attacker_width_symlink": ("symlink", "", "writes RESULT.json as a symlink to its own answer"),
    }
    hand_attackers = {"attacker_width_constant": "constant"}
    # the prompt's envelopes "xi is at most 100" and "mu_on is at most 5" read as uniform: 50 +- 100 / sqrt(12),
    # 2.5 +- 5 / sqrt(12)
    fallback_forecast = {"xi": (50.0, 28.868), "mu_on": (2.5, 1.443)}
    # the qq -> ZZ normalisation, value-scored from the nuisance log where the declared model carries it.
    # target: the reference's typical precision (closure: median unc 0.024); prior: the width of U(0.85, 1.15)
    calibration_scored = {"cross_section_VV": {"nominal": 1.0, "target": 0.025, "prior": 0.0866}}
    # the luminosity is the one unit of the completeness term
    completeness_units = [("luminosity", ("lumi",))]
    nuisance_floors = {"norm": 3e-4, "shape": 1e-5}

    # Sealed-safe results page (and its audit): the step of the design's xi
    # values (truth_grids/width_toy_v0.json: integers; a measured value on it is guessable) and the values an injected
    # xi may take.
    sealed_value_steps = {"xi": 1.0}

    def sealed_value_grid(self, priors: Dict[str, Any]) -> Dict[str, list]:
        lo, hi = priors.get("xi_range") or [0.0, 50.0]
        return {"xi": [lo + 1.0 * i for i in range(int(round(hi - lo)) + 1)]}

    def calibration_truth(self, truth, log=None):
        return {"cross_section_VV": float(truth["bkg_norm"])} if "bkg_norm" in truth else {}

    def contract_values(self):
        return {**super().contract_values(), "interval_pois": "{{hep:poi_xi}}\n{{hep:poi_mu_on}}",
                "scan_poi": "xi", "xi_range": range_text(self, "xi"), "mu_on_range": range_text(self, "mu_on"),
                "observable": "the four-lepton mass"}

    def report_page(self) -> Dict[str, Any]:
        return {**super().report_page(),
                "cases": {"xi": {"name": "Width ratio ξ", "symbol": "ξ", "text": "xi", "unit": "", "digits": 2},
                          "mu_on": {"name": "On-shell signal strength μ_on", "symbol": "μ_on", "text": "mu_on",
                                    "unit": "", "digits": 2}}}

    def default_priors(self) -> Dict[str, Any]:
        return {"xi_range": [0.0, 50.0], "mu_on_range": [0.7, 1.5], "bkg_norm_range": [0.85, 1.15], "lumi_fb": 10.0}

    def priors_without_response(self, priors):
        q = dict(priors)
        q["bkg_norm_range"] = [1.0, 1.0]
        return q

    def public_truth(self, priors):
        return {"xi": 1.0, "mu_on": 1.0, "bkg_norm": 1.0, "lumi_fb": float(priors["lumi_fb"])}

    def draw_truth(self, rng, priors, fixed: Optional[Dict[str, Any]] = None):
        f = fixed or {}
        return {"xi": float(f.get("xi", rng.uniform(*priors["xi_range"]))),
                "mu_on": float(f.get("mu_on", rng.uniform(*priors["mu_on_range"]))),
                "bkg_norm": float(f.get("bkg_norm", rng.uniform(*priors["bkg_norm_range"]))),
                "lumi_fb": float(priors["lumi_fb"])}

    def generate(self, truth, rng, d: Path):
        for sub in ("data", "mc"):
            (d / sub).mkdir(parents=True, exist_ok=True)
        dens = densities(truth["xi"], truth["mu_on"], truth["bkg_norm"])
        n = int(rng.poisson(float(np.sum(dens * _W))))
        ev = sample(rng, dens, n)
        rng.shuffle(ev)
        np.save(d / "data" / "events.npy", ev.astype(np.float64))
        for k, arr in mc_library().items():
            np.save(d / "mc" / f"{k}.npy", arr)
        y = mc_yields()
        xs = {
            "lumi_fb": truth["lumi_fb"],
            "expected_events": y,
            "note": "expected events in the full dataset for each MC sample at its generated point (xi = 1, the "
                    "on-shell signal strength 1, nominal normalisations); every MC event carries weight "
                    "expected_events / number of events in its file",
            "samples": {
                "B_qq": "qq -> ZZ continuum (does not interfere with the Higgs)",
                "B_gg": "gg -> ZZ continuum (interferes with the Higgs)",
                "S": "gg -> H* -> ZZ off-shell signal alone, at xi = 1",
                "SBI_xi1": "gg -> ZZ including the Higgs: signal + gg continuum + their interference, at xi = 1",
                "H_onshell": "on-shell H -> ZZ peak at m_H = 125 GeV",
            },
        }
        (d / "mc" / "cross_sections.json").write_text(json.dumps(xs, indent=1))
        (d / "lumi.json").write_text(json.dumps({"lumi_fb": truth["lumi_fb"], "lumi_unc_rel": LUMI_UNC_REL}))
        truth["n_events_realized"] = n


TASK = WidthToyTask()
