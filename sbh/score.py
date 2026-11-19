
"""Pair run records with sealed truths and compute the metric stack.

The scoring rule
----------------
The score is S_v3 (score_v3), shown as its ratio to the reference's (relative_score) where the closure recorded
one.
S_v3 = exp(-gamma Delta_total), gamma the task's exponent (score_gamma), with
    Delta_total = Delta_phys + V3_W_CAL Delta_cal + V3_W_GOF Delta_gof + V3_W_SYST Delta_syst
                  + V4_W_TRUTH Delta_truth + V4_W_CLAIMS Delta_claims + V3_W_COMP Delta_compute
(weights 1 / 0.20 / 0.30 / 0.10 / 0.5 / 0.05 / 0.1; a term with no cases contributes nothing).  Every term is a
charge on one per-case log-loss scale, 0 = the reference's calibrated level.  A task kept on the rule it shipped with
(SHIPPED_RULE) keeps its effort share: S_v3 = (1 - V2_EPS) exp(-gamma Delta_total) + V2_EPS S_effort (v2 "eps").

Failures.  A dataset a frozen pipeline fails on is charged as no measurement: each of its cases max(F_i, 0) (below),
its goodness of fit at the maximum, the dataset and why in v2 "failed_datasets".  Only a case that cannot be charged
fails the run, S_v3 = 0 (and its relative score 0; v2 / v3 "failed"): a failed dataset's case with no no-answer
forecast or no reference scale, a charge that is not finite, and any failed dataset of a kind that states no
no-measurement charge for one (function outputs).  A dataset fails as the participant contract states a failed run: a
non-zero
exit, a timeout (killed at the stated limit), a missing or malformed RESULT.json (the GATE file; record_ok), an
output-gate failure (MODEL.json, EVENTS.json, a size cap the contract makes a failed run), or a model the grader
cannot fit (Kind.case_status "failed").  Where the contract says a completed dataset still counts but has no
measurement of a quantity (no usable grid: "no mass measurement"; "mu is not evaluated (a failed case)"; a correction
"not reported"), the case is charged max(F_i, 0), F_i the charge of its no-answer forecast (Task.fallback_forecast),
floored at the reference's level so that "don't know" never scores better than the reference
(no_measurement_charge): it counts in K, enters no signed pull, and its dataset is not failed (Kind.case_status
NO_MEASUREMENT; counted as the fallback "no_measurement").  A POI a kind can mark NO_MEASUREMENT needs a
Task.fallback_forecast (a configuration error otherwise).  Provisional, never charged: a dataset without a record.json
(load_scored_records' placeholder: the evaluation lost it, rerun it), a case the grader has not finished
(Kind.case_status "pending"; one on an evaluator fault is left out, never scored on the reported value, one on a
stored record graded before the grader marked its own errors alike, REGRADE; a fault the grader reproduced on a rerun
is named apart, REPRODUCED_FAULT: the evaluator's code needs a fix, as is a closure comparison that missed a larger
budget too, REPRODUCED_BUDGET; either shows the run unranked, unranked_reason) and a case without a reference scale it
would be charged on; the score is then reported with its reasons (v3 "provisional_reasons", named reasons and counts,
never an evaluator's raw message) and is not final.  A failed run's S = 0 is final; its Delta
terms stay as diagnostics of what it answered.

Delta_phys (score_v2): one case per (dataset, POI) Task.poi_scored_v2 admits (h4l: mu on every dataset, mH on every
signal dataset).  Under score-from-fit (Task.model_poi_sources) value and sigma are the grader's, from the declared
model: mu = the MODEL.json fit (best fit, profile half-width; a dataset graded in combine alike); mH = the profile over
MODEL_GRID (parabolic vertex, sigma = half the width between the envelope's 2 DeltalnL = 1 crossings, bounded by
Task.mass_range).  RESULT.json enters only Delta_truth and
Delta_claims.  Per case the loss l = ln sigma + pull^2 / 2 (_v2_loss) against the
reference's calibrated loss ln sigma_ref + 1/2 on the same dataset (closure.json per-seed sigma, else scalar sigma_ref;
the closure's own run against the per-seed baselines it records, own_reference_pois), in two pieces: the precision
piece ln(sigma / sigma_ref) and the pull piece (z^2 - 1) / 2 (case_charge), the plain logarithmic score with no
floor and no cap:  Delta_i = ln(sigma_i / sigma_ref,i) + (z_i^2 - 1) / 2.
A quantity with no measurement is charged max(F_i, 0), F_i = the charge of Task.fallback_forecast
(no_measurement_charge).  A report that gives the injected value zero probability (a charge that is not finite)
fails its dataset (UNBOUNDED_WHY).  A density case splits by H, the density's mean -ln p over itself, a function
case by its geometric-mean width.
Per POI with K cases
    Delta_p = mean_i Delta_i + V2_BIAS_BETA B_p,   B_p = (z_star^2 + s_star^2 - 1) / (2 K)     (multiworld)
the bias term: z_star = sigma_star sum z_i / sigma_i the inverse-variance combination of the pulls z_i at the
reported widths sigma_i, sigma_star^-2 = sum sigma_i^-2, s_star^2 = sigma_star^2 sum s_i^2 / sigma_i^2 the spread of
a function case's point pulls (0 for one value).  With Task.multiworld False, no bias term.
Delta_phys = -ln(mean over POIs of exp(-gamma Delta_p)) / gamma, the charge of the mean per-POI score (one POI:
Delta_p); grouped roles (function outputs) enter as their groups, each with the mean charge of its roles as its
Delta_p.  K counts the answered cases and the cases charged as having no measurement.
A task frozen with its verifier's pass cut keeps the rule it shipped with (Task.score_rule SHIPPED_RULE; _shipped_*),
with its S_effort grading data-responsiveness on the fits (0.4 if the values vary, 0.6 x the graded slope of value on
truth).

Delta_gof (gof_cases, gof_ensemble): per dataset the grader's toy-calibrated combined p of the declared model.
Fisher's X = -2 sum ln max(p_i, V3_GOF_P_FLOOR) over the n_ok datasets with a usable p, p_F = chi2_sf(X, 2 n_ok),
Delta_F = min(GOF_CHARGE_MAX, max(0, z(p_F) - z(V3_GOF_ALPHA))) with z the one-sided Gaussian z of a p (the z-hinge);
Delta_gof = (n_ok Delta_F + n_fail GOF_CHARGE_MAX) / n, n_fail = datasets failed, model-less, p-less or stopped by the grader's wall
(V3_GOF_P_FLOOR 0.005, V3_GOF_ALPHA 0.01, GOF_CHARGE_MAX 4.30 = ln 200 - 1).

Delta_cal (sbh.hep.calibration: calibration_cases, calibration_term): one case per value-scored effect
(Task.calibration_spec, at the dataset's logged working point; only an effect whose prior is wider than the task's
target precision, sbh.hep.calibration.value_scored) and dataset, charged as a physics case is with the effect's
target tau_j in the role of sigma_ref:  Delta_j = ln(s_j / tau_j) + (z_j^2 - 1) / 2,
z_j = (reported - injected) / s_j, s_j the reported uncertainty.  Every valid logged entry is scored on its value,
modelled or not (the contract logs a correction applied without a nuisance parameter the same way); an unlogged
effect, or one logged with no uncertainty, costs max(F_j, 0), F_j the charge of the nominal value at the task's
prior width.
Delta_cal = the plain mean over the cases.  The model enters only through the gate: under score-from-fit,
calibration_gate withdraws a share f of each case's credit against its unlogged charge where the leakage / yield
checks show the calibration is not applied.

Delta_syst (completeness_cases): per dataset the fraction of the never-varied usual nuisances (syst_units) with no
effect passing nuisance_probes in its model (a failed or model-less dataset covers nothing); mean over datasets.
Delta_truth (truth_cases, truth_term) and Delta_claims (claims_rows, claims_term): reported numbers against the
model's own, and the failed fraction of the other model-checkable claims.

Delta_compute (compute_cost, gated_compute): per completed dataset d = clip(log2(wall / reference wall),
V3_COMP_FLOOR, log2(limit / reference)); free up to V3_COMP_ALLOWANCE x the reference wall, above it
log2(wall / reference) - log2(V3_COMP_ALLOWANCE); mean over datasets.  The reference wall is the yardstick's
(sbh/yardstick.py), else Task.compute_reference_wall_s; without a usable yardstick there is no compute term.  The
slowness charge always counts; the speed bonus (d < 0) is scaled by g = compute_gate(Delta_rest): full while the
rest of Delta_total is <= rest0, fading to 0 at rest0 + width, and 0 on a provisional score.  A strict grader
(Task.grader_strict) fails a dataset it did not finish, never scores the reported value; an evaluator fault and a
completed dataset with no grader record (NO_GRADER_RECORD) stay provisional, strict or not.

Error band (sbh.band, write_scores): the 16 / 84 % quantiles of S, Delta_total and each term over a bootstrap of the
datasets resampled by cluster (Task.cluster_key), the whole score recomputed on every resample from the cases it
kept, widened about the estimate by sqrt(n / (n - 1)) t_{n-1}(0.84) / z(0.84) (n the clusters drawn,
sbh.band.widen_factor); the relative score's range the same of
S / S_ref, the reference's score recomputed on the same resamples (paired).  The analytic standard errors (v2 / v3
"delta_se", a term's "se") are report-only.

Fallbacks (fallback_block, v3 "fallbacks"): the fallbacks the evaluation took instead of its primary procedure that a
kind counts (its grader's and scorer's, Kind.score_terms "fallbacks"; the core's: a physics case on the scalar
sigma_ref, a case with no measurement), by kind with its datasets.  Counting charges nothing: a case with no
measurement costs what the rule charges it (max(F, 0)).  None under the shipped rule.  docs/METRICS.md "Fallbacks" lists
them and the ones only recorded.

Writes:
  <run>/private/scores.json   per-seed truths, values, pulls, and the per-dataset lists of the failed datasets'
                              cases, the provisional cases and each fallback (public_split)  -> evaluator only
  <run>/public/summary.json   aggregate metrics only: of those lists, counts and dataset-level reasons -> what a
                              submitter sees
  <run>/public/scoreboard.md  the same, as tables
The summary also carries truth-free review flags with no score effect: the point-mass tripwire (sbh/tripwire.py)
and the missing-standard-background flag.
"""

from __future__ import annotations

import json

import math

import os

import time

from dataclasses import asdict

from pathlib import Path

from typing import Dict, List, Optional, Tuple

import numpy as np

from .attackers import scale_rows


from .metrics import PoiResult, ScoringConfig, SeedResult, score_suite


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def read_json_maybe_shielded(path: Path) -> dict:
    """Read a JSON file a concurrent eval may hold under the chmod-000 shield (sbh/runner.py read_shielded: read
    under the shield's lock)."""
    from .runner import read_shielded
    return json.loads(read_shielded(Path(path)))


def load_records(run_dir: Path) -> List[dict]:
    """Every seed's record.json.  One that does not parse stops the score, naming each: it is the evaluator's fault
    (records are written atomically), so it is neither dropped from the denominator nor charged to the submission."""
    records, bad = [], []
    for p in sorted((Path(run_dir) / "runs").glob("seed_*/record.json")):
        try:
            records.append(json.loads(p.read_text()))
        except (OSError, ValueError) as e:
            bad.append(f"{p.parent.name} ({type(e).__name__}: {str(e)[:80]})")
    if bad:
        raise SystemExit(f"{run_dir}: unreadable record.json for {', '.join(bad)}: rerun those datasets "
                         "(--phase run --seeds ...) before scoring")
    return records


def write_text_atomic(path: Path, text: str) -> None:
    """Write via a temporary file and a rename: a reader, or a writer killed midway, sees the old file or the new
    one, never half of one."""
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


PLACEHOLDER_REASON = "no record.json: the evaluation lost the dataset (infrastructure); rerun it"


def load_scored_records(run_dir: Path, manifest: dict) -> List[dict]:
    """The run's records plus a "placeholder" (record_ok False) for every sealed seed without one, so the
    denominator never shrinks: the score leaves it out as provisional until the dataset is rerun (the shipped rule,
    SHIPPED_RULE, charges it as failed)."""
    records = load_records(run_dir)
    missing = sorted({s["seed"] for s in manifest["seeds"] if not s["public"]} - {r["seed"] for r in records})
    return records + [{"seed": s, "exit_code": None, "timed_out": False, "result": None, "wall_s": None,
                       "result_errors": [PLACEHOLDER_REASON], "placeholder": True}
                      for s in missing]


def failure_reason(rec: dict) -> str:
    """Why a dataset's record is not completed (record_ok False), in one line."""
    if rec.get("placeholder"):
        return PLACEHOLDER_REASON
    if rec.get("timed_out"):
        return "timed out (killed at the time limit)"
    if rec.get("exit_code") != 0:
        return f"exit code {rec.get('exit_code')}"
    if rec.get("result") is None:
        return "no readable RESULT.json"
    errs = [str(e) for e in rec.get("result_errors") or []]
    if not errs:
        return "failed"
    return "; ".join(e[:200] for e in errs[:3]) + (f" (+{len(errs) - 3} more)" if len(errs) > 3 else "")


def cluster_of(task, truth: dict, seed: str) -> str:
    """A dataset's cluster (Task.cluster_key: datasets sharing a draw): the unit the band resamples (sbh.band) and
    the report-only standard errors' cluster."""
    key = getattr(task, "cluster_key", "profile")
    return str(truth.get(key, seed)) if key else str(seed)


def record_ok(rec: dict) -> bool:
    """A completed dataset: exit 0, not timed out, a RESULT.json, and no result_errors (the contract's checks of
    RESULT.json, the kind's output gate).  Anything else is the contract's failed run on the dataset."""
    return (rec.get("exit_code") == 0 and not rec.get("timed_out")
            and rec.get("result") is not None and not rec.get("result_errors"))


def scoring_truth(task, truth: dict) -> Dict[str, float]:
    return {p: float(truth[p]) for p in task.pois if p in truth and task.poi_applicable(p, truth)}


def seed_results(records: List[dict], truths: Dict[str, dict], task) -> List[SeedResult]:
    from .kinds import get_kind
    kind = get_kind(task)
    out = []
    for rec in records:
        st = scoring_truth(task, truths[rec["seed"]])
        if not record_ok(rec):
            out.append(SeedResult(seed=rec["seed"], truth=st, completed=False))
            continue
        res = rec["result"]
        pois = {p["name"]: kind.poi_result(p, st.get(p["name"])) for p in res["pois"]}
        out.append(SeedResult(seed=rec["seed"], truth=st, completed=True, pois=pois,
                              discovery=res.get("discovery"), significance_obs=res.get("significance_obs")))
    return out


def self_sigma_ref(results: List[SeedResult], pois) -> Dict[str, float]:
    """Median quoted uncertainty over the scored seeds (on null or weak seeds a mass fit's error is meaningless)."""
    ref = {}
    for p in pois:
        s = [r.pois[p].sigma() for r in results if r.completed and p in r.truth and p in r.pois and r.pois[p].sigma() > 0]
        if s:
            ref[p] = float(np.median(s))
    return ref



V2_GAMMA = 2.0

V2_CAP_FLOOR = 4.0
V2_CLIP = -(math.log(1.25) + 0.5)

UNBOUNDED_WHY = "the report gives the injected value zero probability (an unbounded charge)"
UNBOUNDED_CHARGE = 1e300

V2_BIAS_BETA = 1.0

SCORE_RULE = "standard"
SHIPPED_RULE = "tbs-shipped"

PHYS_RULE = ("per case ln(sigma / sigma_ref) + (z^2 - 1) / 2, no floor and no cap; a case with no measurement on a "
             "completed dataset charged max(F, 0), F its no-answer forecast's charge; per POI the mean case charge "
             "plus the bias term; over POIs the charge of the mean per-POI score, "
             "-ln(mean_p exp(-gamma Delta_p)) / gamma; a failed dataset's cases charged as no measurement; a case "
             "whose charge is not finite, or that cannot be charged, fails the run (S = 0); S = exp(-gamma Delta_total); a dataset without a record or a pending grader case leaves the "
             "score provisional")

NO_MEASUREMENT = "no_measurement"

EVALUATOR_FAULT = "evaluator fault"

REPRODUCED_FAULT = "reproduced evaluator fault"

REPRODUCED_BUDGET = "reproduced budget miss"

UNRANKED_REASONS = (REPRODUCED_FAULT, REPRODUCED_BUDGET)

REGRADE = "graded before errors were marked: regrade"
REGRADE_FAULT = f"{EVALUATOR_FAULT}: {REGRADE}"

NO_GRADER_RECORD = "no grader record"


def unranked_reason(v3: Optional[dict]) -> Optional[str]:
    """Why a scored run is shown unranked: its provisional reason of a reproduced evaluator fault (REPRODUCED_FAULT) or
    of a comparison that missed a larger budget too (REPRODUCED_BUDGET), else None (a failed run's S = 0 is final and
    ranked)."""
    if not v3 or v3.get("failed"):
        return None
    return next((str(r) for r in v3.get("provisional_reasons") or [] if str(r).startswith(UNRANKED_REASONS)), None)

V2_FAIL_PREMIUM = 0.25

V2_K_MISSING = 4.0

V2_EPS = 0.075


def cal_weight(task) -> float:
    """The calibration term's weight for ``task``: V3_W_CAL, or V3_W_CAL_SHIPPED under the shipped rule."""
    return V3_W_CAL_SHIPPED if task is not None and score_rule(task) == SHIPPED_RULE else V3_W_CAL


def score_rule(task) -> str:
    """The task's physics rule (Task.score_rule): SCORE_RULE, or SHIPPED_RULE for a task kept on the rule it shipped
    with."""
    r = getattr(task, "score_rule", None) or SCORE_RULE
    if r not in (SCORE_RULE, SHIPPED_RULE):
        raise ValueError(f"{getattr(task, 'name', task)}: unknown score_rule {r!r}")
    return r


def case_charge(d_prec: float, d_pull: float) -> float:
    """A case's charge Delta_i = d_prec + d_pull, the plain log score against the reference (its pieces: _loss_pieces,
    _entropy_pieces): ln(sigma / sigma_ref) + (z^2 - 1) / 2 for a Gaussian answer, no floor and no cap.  +inf where
    the sum is not finite (p(truth) = 0, an overflowing pull, a degenerate width): the caller fails the case."""
    x = float(d_prec) + float(d_pull)
    return x if math.isfinite(x) else math.inf


def no_measurement_charge(d_prec: float, d_pull: float) -> float:
    """The charge of a case with no measurement (a no-answer case: Kind.case_status NO_MEASUREMENT, a correction not
    reported, an unlogged calibration nuisance): max(F, 0), F = case_charge of its no-answer forecast's pieces.  The
    floor is the reference's own level (derived, not a constant): where the reference's width exceeds the forecast's
    reach F falls below 0, and "don't know" would otherwise score better than measuring as well as the reference."""
    return max(case_charge(d_prec, d_pull), 0.0)


def _v2_loss(sigma: float, err: float) -> float:
    z = err / sigma
    return math.log(sigma) + 0.5 * z * z


def point_case_loss(pr, truth: float, fallback: tuple, ref: Optional[dict], ref_sigma: float) -> dict:
    """Kind.case_loss's default for a reported PoiResult ``pr`` (None: no usable answer): the loss ln sigma +
    pull^2 / 2 (_v2_loss), the fallback forecast's loss, the baseline of a calibrated forecast at the reference's
    sigma on the dataset, else at the scalar sigma_ref: ln sigma_ref + 1/2, and the bias term's signed
    pull mw_pull = (value - truth) / sigma (None without an answer)."""
    good = bool(pr is not None and math.isfinite(pr.value) and pr.sigma() > 0 and math.isfinite(pr.sigma()))
    fc, fs = fallback
    fb_loss = _v2_loss(fs, fc - truth)
    loss = _v2_loss(pr.sigma(), pr.value - truth) if good else fb_loss
    if ref and float(ref.get("sigma", 0.0)) > 0:
        ref_loss, src = math.log(float(ref["sigma"])) + 0.5, "seed"
    elif ref_sigma > 0:
        ref_loss, src = math.log(ref_sigma) + 0.5, "scalar"
    else:
        ref_loss, src = None, None
    return {"good": good, "loss": loss, "fb_loss": fb_loss, "ref_loss": ref_loss, "ref_source": src,
            "value": pr.value if good else None, "sigma": pr.sigma() if good else None,
            "mw_pull": _signed_pull(pr.value - truth, pr.sigma()) if good else None, "fields": {}}


def _signed_pull(err: float, sigma: float) -> float:
    """err / sigma, +-inf where it overflows (a contract-valid extreme: an unbounded charge)."""
    with np.errstate(over="ignore", divide="ignore"):
        return float(np.float64(err) / np.float64(sigma))


def pit_pull(pit: Optional[float], truth: float, support) -> Optional[float]:
    """A density case's PIT pull (the report card's density pulls, never the bias term's):
    z = -Phi^-1(F(truth)), F = P(x <= truth) under the scored density ((measured - injected) / sigma for a Gaussian
    density), +-inf at F = 1 / 0; None without a PIT or with the truth at an end of the density's support, where F is
    0 or 1 under every density on it and no pull exists."""
    if pit is None:
        return None
    lo, hi = (support or (None, None))
    if (lo is not None and truth <= lo) or (hi is not None and truth >= hi):
        return None
    if pit <= 0.0:
        return math.inf
    if pit >= 1.0:
        return -math.inf
    from statistics import NormalDist
    return -NormalDist().inv_cdf(pit)


def likelihood_pull(fc: dict, truth: float) -> Optional[float]:
    """A density case's signed pull for the bias term (mw_pull): z = sign(mode - truth) sqrt(2 (ln p(mode) -
    ln p(truth))), mode the density's largest value on its axis and p(truth) read as the loss reads it
    (log_density_at).  It does not depend
    on how the POI is parametrised: for a flat-prior scan density p ~ exp(-q / 2) it is sign(x_hat - truth)
    sqrt(q(truth)), and for a Gaussian density (mode - truth) / sigma.  Where p(truth) = 0 (a truth outside the
    density's support) it is +-inf (an unbounded charge).  None for a truth on an end of the support (a bound
    such as mu = 0 that every density on that axis shares): there an accurate density's mode lies on the truth or on one
    side of it, so the pull is >= 0 by construction (about +0.4 on average) and would read as a bias."""
    from .kinds.binned_likelihood.scoring import log_density_at
    lp = np.asarray(fc["log_p"], float)
    axis = np.asarray(fc["mu_axis"], float)
    if any(math.isclose(truth, float(end), rel_tol=0.0, abs_tol=1e-9 * max(1.0, abs(float(end))))
           for end in (axis[0], axis[-1])):
        return None
    k = int(np.argmax(np.where(np.isfinite(lp), lp, -np.inf)))
    mode, top = float(axis[k]), float(lp[k])
    at = log_density_at(fc, truth)
    if not math.isfinite(at):
        return math.copysign(math.inf, mode - truth)
    return math.copysign(math.sqrt(max(2.0 * (top - at), 0.0)), mode - truth)


HALF_LN_2PI = 0.5 * math.log(2.0 * math.pi)


def entropy_ref_loss(ref: Optional[dict]) -> Optional[float]:
    """The density rule's baseline from a reference entry: H[p_ref] - 1/2 ln 2 pi (closure.json
    reference_pois[seed][poi]["entropy"], the grader's); None without one (the case is then excluded)."""
    h = ref.get("entropy") if isinstance(ref, dict) else None
    return float(h) - HALF_LN_2PI if isinstance(h, (int, float)) and math.isfinite(h) else None


def forecast_case_loss(fc: Optional[dict], truth: float, fallback: tuple, ref: Optional[dict],
                       ref_sigma: float, fb_loss: Optional[float] = None) -> dict:
    """point_case_loss for a density forecast ``fc`` (sbh.kinds.binned_likelihood.scoring.marginal_mu, a scan_likelihood
    run's scan density; None or not "ok": no usable answer): loss = -ln p(truth) - 1/2 ln 2 pi (the Gaussian point
    rule's _v2_loss when p is the Gaussian N(value, sigma)), the fallback forecast's loss (``fb_loss`` where the kind
    reads its fallback otherwise, scan_likelihood truncated to the range; else its _v2_loss), and the baseline H[p_ref] -
    1/2 ln 2 pi from the reference's H on the dataset (closure.json reference_pois[seed][poi]["entropy"];
    ln sigma_ref + 1/2 when p_ref is Gaussian).  NEVER the scalar sigma_ref: without an H the case has no
    baseline and is excluded.  value / sigma = the density's mean / sd (diagnostics: fits[], the tripwire, the card);
    fields = regime, n_eff, clipped_null_term (the raw charge loss - ref_loss), the coverage / content flags, and the
    scored density's summary (diagnostics, None without a density): value (mean), sigma (sd), mode, H[p] its mean -ln p
    (which splits the raw charge, _entropy_pieces), pit = F(truth) = P(mu <= truth) under p (cdf_at, on the axis and
    log density the loss reads) and support (the axis' ends).  The bias term's signed pull mw_pull is
    the likelihood pull (likelihood_pull), not the PIT."""
    fc_c, fs = fallback
    fb_loss = _v2_loss(fs, fc_c - truth) if fb_loss is None else float(fb_loss)
    if isinstance(fc, dict) and fc.get("status") == "excluded":
        return _excluded_case_loss(fc, fb_loss, ref)
    good = bool(isinstance(fc, dict) and fc.get("status") == "ok" and fc.get("log_p") is not None)
    pit = support = None
    if good:
        from .kinds.binned_likelihood.scoring import cdf_at, log_density_at
        loss = -log_density_at(fc, truth) - HALF_LN_2PI
        try:
            pit = cdf_at(fc, truth)
            support = [float(fc["mu_axis"][0]), float(fc["mu_axis"][-1])]
        except Exception:  # noqa: BLE001
            pit = support = None
    else:
        loss = fb_loss
    ref_loss = entropy_ref_loss(ref)
    src = "seed" if ref_loss is not None else None
    sigma = max(float(fc["sigma"]), 1e-12) if good else None
    h = _finite(fc.get("entropy")) if good else None
    flags = {"regime": fc.get("regime") if good else None, "n_eff": fc.get("n_eff") if good else None,
             "clipped_null_term": (float(loss - ref_loss) if ref_loss is not None and math.isfinite(loss) else None),
             "coverage_flags": list(fc.get("coverage_flags") or []) if isinstance(fc, dict) else [],
             "content_flags": list(fc.get("content_flags") or []) if isinstance(fc, dict) else [],
             "prior_source": fc.get("prior_source") if good else None,
             "forecast_status": fc.get("status") if isinstance(fc, dict) else None,
             "value": float(fc["mean"]) if good else None, "sigma": sigma,
             "mode": _finite(fc.get("mode")) if good else None, "entropy": h, "pit": pit, "support": support}
    return {"good": good, "loss": float(loss), "fb_loss": fb_loss, "ref_loss": ref_loss, "ref_source": src,
            "value": float(fc["mean"]) if good else None, "sigma": sigma,
            "mw_pull": likelihood_pull(fc, truth) if good else None, "fields": flags}


def _excluded_case_loss(fc: dict, fb_loss: float, ref: Optional[dict]) -> dict:
    """forecast_case_loss of a density that vanishes on the whole support (marginal_mu "excluded": its declared bounds
    claim every value there impossible): an answer, scored on the claim -- p(truth) = 0, an infinite loss, so the case's
    charge is unbounded -- at the reference's own width (its precision piece 0: the claim has no width on the
    support), its signed pull +-inf on the side of its bounds.  value: the bound it claims
    beyond."""
    ref_loss = entropy_ref_loss(ref)
    sigma = math.exp(ref_loss - 0.5) if ref_loss is not None else 1.0
    flags = {"regime": None, "n_eff": None, "clipped_null_term": None, "coverage_flags": [], "content_flags": [],
             "prior_source": None, "forecast_status": "excluded", "value": float(fc["bound"]), "sigma": sigma,
             "mode": None, "entropy": None, "pit": None, "support": None}
    return {"good": True, "loss": math.inf, "fb_loss": fb_loss, "ref_loss": ref_loss,
            "ref_source": "seed" if ref_loss is not None else None, "value": float(fc["bound"]), "sigma": sigma,
            "mw_pull": math.copysign(math.inf, float(fc.get("sign") or 1.0)), "fields": flags}


def _finite(x) -> Optional[float]:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def _loss_pieces(ln_sigma: float, loss: float, ref_loss: float) -> Tuple[float, float]:
    """A case's raw charge loss - ref_loss in two pieces (diagnostics; docs/METRICS.md "Per-case loss"): the
    precision piece ln sigma - ln sigma_ref (ref_loss = ln sigma_ref + 1/2) and the pull piece loss - ln sigma - 1/2
    (= z^2 / 2 - 1/2 for a point case, mean z^2 / 2 - 1/2 for a function case).  Their sum is the raw charge."""
    d_prec = float(ln_sigma - (ref_loss - 0.5))
    return d_prec, float(loss - ln_sigma - 0.5)


def _entropy_pieces(entropy: float, loss: float, ref_loss: float) -> Tuple[float, float]:
    """_loss_pieces of a density case (forecast_case_loss: loss = -ln p(truth) - 1/2 ln 2 pi, ref_loss = H[p_ref] -
    1/2 ln 2 pi): the precision piece H[p] - H[p_ref] = ln(sigma_H / sigma_H,ref), sigma_H = e^H / sqrt(2 pi e) the
    width of a Gaussian of the same H, and the pull piece -ln p(truth) - H[p].  Their sum is the raw charge; for a
    Gaussian p they are ln(sigma / sigma_ref) and (pull^2 - 1) / 2, _loss_pieces' point pieces."""
    return float(entropy - (ref_loss + HALF_LN_2PI)), float(loss + HALF_LN_2PI - entropy)


def _piece_mean(cases: List[dict], key: str) -> Optional[float]:
    """The mean of a diagnostic piece over cases; None when a case lacks it (a kind's own cases without it)."""
    vals = [c.get(key) for c in cases]
    return None if any(v is None for v in vals) else float(np.mean(vals))


def _wls_slope(x: np.ndarray, y: np.ndarray, sigma: np.ndarray) -> tuple:
    """Weighted least squares y = a + b*x, weights 1/sigma^2 (fits wide with accurate uncertainties don't drown
    precise ones); returns (b, se_b)."""
    w = 1.0 / np.clip(sigma, 1e-12, None) ** 2
    sw = w.sum()
    xm, ym = (w * x).sum() / sw, (w * y).sum() / sw
    sxx = float((w * (x - xm) ** 2).sum())
    if sxx == 0.0:
        return float("nan"), float("nan")
    b = float((w * (x - xm) * (y - ym)).sum() / sxx)
    return b, float(np.sqrt(1.0 / sxx))


def _cluster_se(d: np.ndarray, clusters: List[str]) -> float:
    """Cluster-robust SE of the mean (datasets of one cluster share a draw, so their cases are not independent)."""
    m = d.mean()
    groups: Dict[str, float] = {}
    for x, c in zip(d, clusters):
        groups[c] = groups.get(c, 0.0) + (x - m)
    return float(np.sqrt(sum(v ** 2 for v in groups.values())) / len(d))


def cluster_bias_tests(records: List[dict], truths: Dict[str, dict], task) -> Dict[str, dict]:
    """Per-POI bias test with cluster-robust errors (the naive 3/sqrt(N) gate overstates a shared fluctuation).
    `sbh closure`'s PASS criterion; metrics.py's naive gate stays a diagnostic."""
    from .kinds import get_kind
    kind = get_kind(task)
    out = {}
    for p in task.pois:
        pulls, clusters = [], []
        for rec in records:
            t = truths[rec["seed"]]
            if not record_ok(rec) or p not in scoring_truth(task, t):
                continue
            pr = {x["name"]: kind.poi_result(x, float(t[p])) for x in rec["result"]["pois"]}.get(p)
            if pr is None or pr.sigma() <= 0:
                continue
            pulls.append((pr.value - float(t[p])) / pr.sigma())
            clusters.append(cluster_of(task, t, rec["seed"]))
        if len(pulls) < 3:
            continue
        d = np.array(pulls)
        se = _cluster_se(d, clusters)
        z = float(d.mean() / se) if se > 0 else float("inf")
        out[p] = {"mean_pull": float(d.mean()), "se_cluster": se, "n": len(pulls),
                  "n_clusters": len(set(clusters)), "z": z, "pass": bool(abs(z) <= 3.0)}
    return out


def load_closure_baseline(suite_dir: Path):
    """(reference_pois, sigma_ref, passed) from `sbh closure`.  Used even from an unpassed closure (consumers flag
    it via `passed`): self-referencing is degenerate -- an all-fail run would normalise to itself and score 1."""
    p = Path(suite_dir) / "closure.json"
    if p.exists():
        c = json.loads(p.read_text())
        return c.get("reference_pois"), c.get("sigma_ref") or {}, bool(c.get("passed"))
    return None, {}, False


def own_reference_pois(kind, task, results: List[SeedResult], scored_recs: List[dict]) -> Dict[str, dict]:
    """The per-dataset baselines the closure records of its reference run (closure.json reference_pois): per sealed
    dataset, Kind.reference_poi of each reported POI with its entry as the physics term scores it (``scored_recs``,
    Kind.score_inputs' records; {} where the run failed), except the POIs whose baseline the kind's grader writes
    (sbh.suite.baseline_pois), which closure.json receives from Kind.closure_fields.  score_run scores the closure's
    run against exactly these, and `sbh closure` writes them (private scores.json reference_pois), so the reference is
    graded against the baselines every other run is."""
    from .suite import baseline_pois
    grader = set(baseline_pois(task))
    scored = {r["seed"]: {p.get("name"): p for p in r["result"]["pois"] if isinstance(p, dict)}
              for r in scored_recs if record_ok(r)}
    return {r.seed: {p: kind.reference_poi(asdict(v), (scored.get(r.seed) or {}).get(p))
                     for p, v in r.pois.items() if p not in grader} for r in results}


def _numbers_differ(a, b, rel: float = 1e-9) -> bool:
    """True when two JSON-like values differ beyond a relative ``rel`` in any number (keys, lengths, types alike)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() != b.keys() or any(_numbers_differ(a[k], b[k], rel) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) != len(b) or any(_numbers_differ(x, y, rel) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return not (a == b or math.isclose(float(a), float(b), rel_tol=rel, abs_tol=0.0))
    return a != b


def load_closure_prior_curves(suite_dir: Path, poi: str = "mu") -> dict:
    """The mass-marginal rule's prior curves from closure.json (the kind's closure_fields, written by `sbh post-score
    --label reference`): {"per_dataset": {seed: {"masses", "sigma0"}}, "task": the suite-average curve or None,
    "mu_rule", "baseline_source", "baseline_id"}; empty curves without a closure."""
    p = Path(suite_dir) / "closure.json"
    out = {"per_dataset": {}, "task": None, "mu_rule": None, "baseline_source": None, "baseline_id": None}
    if not p.exists():
        return out
    try:
        c = read_json_maybe_shielded(p) or {}
    except (OSError, ValueError):
        return out
    for seed, pois in (c.get("reference_pois") or {}).items():
        cur = ((pois or {}).get(poi) or {}).get("prior_curve") if isinstance(pois, dict) else None
        if isinstance(cur, dict):
            out["per_dataset"][seed] = cur
    out.update(task=c.get("prior_curve_task") if isinstance(c.get("prior_curve_task"), dict) else None,
               mu_rule=c.get("mu_rule"), baseline_source=c.get("baseline_source"), baseline_id=c.get("baseline_id"))
    return out


def _gated(gate, poi: str, truth: dict) -> Optional[bool]:
    """The task's applicability gate on one case (diagnostic): True without a gate, None if not evaluable."""
    if gate is None:
        return True
    try:
        return bool(gate(poi, truth))
    except Exception:  # noqa: BLE001
        return None


def score_v2(records: List[dict], truths: Dict[str, dict], task,
             sigma_ref: Dict[str, float],
             ref_pois: Optional[Dict[str, dict]] = None,
             status: Optional[Dict[Tuple[str, str], Tuple[str, str]]] = None) -> Optional[dict]:
    """The physics term of a POI task (Delta_phys): _poi_cases through v2_aggregate at the task's exponent.
    ``status``: Kind.case_status of the kind's measurement ({(seed, poi): (status, reason)})."""
    return v2_aggregate(_poi_cases(records, truths, task, sigma_ref, ref_pois, status), sigma_ref, ref_pois,
                        score_gamma(task))


def pending_reason(why: Optional[str]) -> str:
    """The reason a pending case is left out (v2 "unscored"): an evaluator fault as the kind states it (it starts with
    EVALUATOR_FAULT), else "grader pending: <why>"."""
    w = str(why or "")
    return w if w.startswith(EVALUATOR_FAULT) else f"grader pending: {w}"


NO_REFERENCE_SCALE = "no reference scale"


def _poi_cases(records: List[dict], truths: Dict[str, dict], task, sigma_ref: Dict[str, float],
               ref_pois: Optional[Dict[str, dict]],
               status: Optional[Dict[Tuple[str, str], Tuple[str, str]]] = None) -> dict:
    """The v2 cases of a POI task, one per scored (dataset, POI), plus what v2_aggregate needs besides them.  A case
    without a usable answer is
      failed            (delta None, "why" its reason; its dataset in "failed", per POI in "failed_cases") where its
                        record is not completed (record_ok) or the kind's measurement failed (``status`` "failed",
                        Kind.case_status: the model's fault);
      no measurement    (``status`` NO_MEASUREMENT on a completed dataset: the contract counts the dataset but gives
                        the case no measurement) charged max(F_i, 0) (no_measurement_charge of its no-answer forecast's
                        pieces), "no_measurement" True, no signed pull (mw_miss None), its dataset not
                        failed; a POI without a Task.fallback_forecast has no F_i: such a case is a configuration
                        error (ValueError);
      provisional       ("unscored", never charged): a dataset without a record (a placeholder), a case whose
                        measurement is pending (``status`` "pending"; pending_reason), and a case with no measurement
                        that has no reference scale to charge F on (NO_REFERENCE_SCALE).
    A failed case without a reference scale still fails its dataset.  Under the shipped rule (SHIPPED_RULE) every case
    without an answer is charged C + V2_FAIL_PREMIUM, a placeholder is a failed dataset and nothing fails the run."""
    from .kinds import get_kind
    kind = get_kind(task)
    fb = task.fallback_forecast
    shipped = score_rule(task) == SHIPPED_RULE
    mw_on = bool(getattr(task, "multiworld", True)) and not shipped
    gate = getattr(task, "poi_applicable", None)
    cases: List[dict] = []
    fits: Dict[str, List[tuple]] = {p: [] for p in task.pois}
    unanswered: List[tuple] = []
    failed: Dict[str, List[str]] = {}
    fatal: set = set()
    failed_cases: Dict[str, List[tuple]] = {}
    unscored: List[dict] = []
    for rec in records:
        seed = rec["seed"]
        t_full = truths[seed]
        if rec.get("placeholder") and not shipped:
            unscored.append({"seed": seed, "poi": None, "reason": PLACEHOLDER_REASON})
            continue
        ok = record_ok(rec)
        if not ok:
            failed.setdefault(seed, []).append(failure_reason(rec))
        res_pois = {p["name"]: p for p in rec["result"]["pois"]} if ok else {}
        n_scored = n_missing = 0
        for p in task.pois:
            if p not in t_full or not task.poi_scored_v2(p, t_full):
                continue
            tv = float(t_full[p])
            fallback = fb.get(p, (tv, sigma_ref.get(p, 1.0)))
            cl = kind.case_loss(res_pois.get(p), p, tv, task, fallback,
                                (ref_pois or {}).get(seed, {}).get(p), sigma_ref.get(p, 0.0))
            good, loss, fb_loss = cl["good"], cl["loss"], cl["fb_loss"]
            ref_loss, ref_src = cl["ref_loss"], cl["ref_source"]
            if good:
                fits[p].append((tv, cl["value"], cl["sigma"]))
            why, no_meas = None, False
            if not good:
                st, st_why = (status or {}).get((seed, p)) or (None, None)
                if ok and st == "pending" and not shipped:
                    unscored.append({"seed": seed, "poi": p, "reason": pending_reason(st_why)})
                    continue
                no_meas = bool(ok and st == NO_MEASUREMENT and not shipped)
                why = failure_reason(rec) if not ok else (st_why if st in ("failed", NO_MEASUREMENT) and st_why else
                                                          f"no usable {p} from the run's outputs")
                if ok and not no_meas:
                    failed_cases.setdefault(seed, []).append((p, why))
            if no_meas and p not in fb:
                raise ValueError(f"{getattr(task, 'name', task)}: {p} on {seed} has no measurement and the task "
                                 f"states no fallback_forecast for {p} (a configuration error: every POI its kind can "
                                 "mark no_measurement needs one)")
            if ref_loss is None:
                if no_meas:
                    unscored.append({"seed": seed, "poi": p, "reason": f"{NO_REFERENCE_SCALE} (no measurement: {why})"})
                elif not good and not shipped:
                    fatal.add(seed)
                continue
            raw = loss - ref_loss
            h = cl["fields"].get("entropy") if good else None
            d_prec, d_pull = (_entropy_pieces(h, loss, ref_loss) if h is not None else
                              _loss_pieces(math.log(cl["sigma"] if good else fallback[1]), loss, ref_loss))
            cap = None
            if shipped:
                cap = max(fb_loss - ref_loss, V2_CAP_FLOOR)
                delta = min(max(raw, V2_CLIP), cap) if good else cap + V2_FAIL_PREMIUM
                capped, clipped = bool(good and raw > cap), bool(good and raw < V2_CLIP)
            else:
                failed_case = bool(not good and not no_meas)
                delta = (case_charge(d_prec, d_pull) if good else
                         no_measurement_charge(d_prec, d_pull) if (no_meas or p in fb) else None)
                capped = clipped = False
                if delta is None:
                    fatal.add(seed)
                elif not math.isfinite(delta):
                    good, no_meas, failed_case, delta, why = False, False, False, None, UNBOUNDED_WHY
                    failed_cases.setdefault(seed, []).append((p, why))
                    fatal.add(seed)
            if not math.isfinite(loss):
                loss = raw = d_pull = None
            if not math.isfinite(d_prec):
                d_prec = None
            mw = mw_miss(cl["mw_pull"]) if good else None
            n_scored += 1
            n_missing += not good
            cases.append({"seed": seed, "poi": p, "loss": loss, "ref_loss": ref_loss,
                          "delta": delta, "delta_raw": raw if good else None, "fallback": not good, "why": why,
                          **({"no_measurement": True} if no_meas else {}),
                          **({"failed_charged": True} if (not shipped and not good and not no_meas
                                                          and delta is not None) else {}),
                          "delta_precision": d_prec, "delta_pull": d_pull, "cap": cap,
                          "capped": capped, "clipped": clipped,
                          "gated": _gated(gate, p, t_full), "ref_source": ref_src, "mw_miss": mw,
                          "cluster": cluster_of(task, t_full, seed), **cl["fields"]})
        if n_scored:
            unanswered.append((n_missing / n_scored, cluster_of(task, t_full, seed)))
    for seed, cs in failed_cases.items():
        failed.setdefault(seed, []).extend(f"{p}: {w}" for p, w in cs)
    return {"cases": cases, "fits": fits, "unanswered": unanswered, "roles": list(task.pois), "groups": None,
            "multiworld": mw_on, "rule": SHIPPED_RULE if shipped else SCORE_RULE,
            "failed": {} if shipped else failed, "failed_cases": {} if shipped else failed_cases,
            "fatal": set() if shipped else fatal,
            "unscored": unscored,
            "scored_truths": {p: {round(float(truths[c["seed"]][p]), 9) for c in cases if c["poi"] == p}
                              for p in fits},
            "effort_scale": {p: sigma_ref.get(p) or fb.get(p, (0.0, 1.0))[1] for p in fits}}


GROUP_RULE = "a group of roles is one POI: its Delta_p the mean of its roles' charges"

MW_RULE = ("bias term: per POI the mean of its K case charges + (z_star^2 + s_star^2 - 1) / (2 K), z_star the answered "
           "cases' inverse-variance combination of their pulls at their reported widths (a function case's point "
           "pulls each, their spread s_star^2 kept)")


def mw_miss(z: Optional[float]) -> Optional[float]:
    """An answered case's miss in the combination (case "mw_miss"): its signed pull z in units of its own width; None
    for none (a truth on an end of a density's support, an undefined or infinite pull)."""
    if z is None or not math.isfinite(float(z)):
        return None
    return float(z)


def multiworld(cases: List[dict]) -> Optional[dict]:
    """The bias term B_p of one POI (role) over its K charged cases (the answered ones and those charged max(F, 0) for
    no measurement; failed ones carry no charge and are not scored).  The combination takes the n of them with a miss
    u_i (case "mw_miss", the signed pull z_i); one without a pull (a case with no measurement, a truth on an end of a
    density's support) counts in K alone.  Each has its width w_i = sigma_i: ln w_i = ln sigma_ref,i + its
    delta_precision (a density's width sigma_H, a two-piece interval's mean half-width, a function case's sigma_bar),
    ln sigma_ref,i = ref_loss - 1/2 (every kind's baseline is ln sigma_ref + 1/2).  A function case's u_i is the mean
    of its point pulls and s_i their sd over the points (case "mw_spread"; 0 for one value).
    sigma_star^-2 = sum w_i^-2, z_star = sigma_star sum u_i / w_i, s_star^2 = sigma_star^2 sum s_i^2 / w_i^2 (in
    logs: no overflow at contract-valid extremes), and B_p = (z_star^2 + s_star^2 - 1) / (2 K): z_star^2 + s_star^2 is
    the point-by-point combination's squared pull averaged over the points, with the datasets' deviations from their
    own mean miss taken as independent, so an accurate run's E B_p = 0 whatever the number of points and their
    correlation; B_p >= -1 / (2 K).  {"n" (combined), "k" (K), "zstar", "spread" (s_star^2), "bias" (B_p)}; None where
    no answered case has a pull (nothing to combine: no term)."""
    charged = [c for c in cases if not c.get("fallback") or c.get("delta") is not None]
    rows = [c for c in charged if c.get("mw_miss") is not None and c.get("delta_precision") is not None]
    if not rows:
        return None
    ln_w, ut, sd = [], [], []
    for c in rows:
        ln_w.append(float(c["ref_loss"]) - 0.5 + float(c["delta_precision"]))
        ut.append(float(c["mw_miss"]))
        sd.append(float(c.get("mw_spread") or 0.0))
    lw = np.array(ln_w)
    ln_s = -0.5 * float(np.logaddexp.reduce(-2.0 * lw))
    a = np.exp(ln_s - lw)
    zstar = float(np.dot(np.array(ut), a))
    spread = float(np.dot(np.square(np.array(sd)), a * a))
    k = len(charged)
    return {"n": len(rows), "k": k, "zstar": zstar, "spread": spread,
            "bias": (zstar * zstar + spread - 1.0) / (2.0 * k)}


def closure_multiworld(v2: Optional[dict]) -> dict:
    """closure.json's multiworld block from the closure reference's v2: the rule it was scored under (v2
    multiworld_rule; None where the task scores no bias term) and per POI (role) the reference's own z_star, B_p
    (multiworld "bias"; None where the term is not scored) and delta_multiworld, the term's part of its charge
    (V2_BIAS_BETA B_p).  A diagnostic: nothing in the rule is set from it."""
    per = {}
    for p, m in ((v2 or {}).get("per_poi") or {}).items():
        mw = m.get("multiworld") or {}
        per[p] = {"zstar": mw.get("zstar"), "bias": mw.get("bias"), "delta_multiworld": m.get("delta_multiworld")}
    return {"rule": (v2 or {}).get("multiworld_rule"), "per_poi": per}


def _group_members(per_poi: dict, groups: Dict[str, str]) -> Dict[str, List[str]]:
    """group -> its roles present in per_poi, both in first-seen order."""
    out: Dict[str, List[str]] = {}
    for p in per_poi:
        out.setdefault(groups[p], []).append(p)
    return out


def _wls_slope_within(x: np.ndarray, y: np.ndarray, sigma: np.ndarray, within: list) -> tuple:
    """_wls_slope with per-group fixed effects, so a constant depending on the request alone earns no slope."""
    w = 1.0 / np.clip(sigma, 1e-12, None) ** 2
    keys = [str(g) for g in within]
    xd, yd = np.array(x, dtype=float), np.array(y, dtype=float)
    for g in sorted(set(keys)):
        m = np.array([k == g for k in keys])
        sw = w[m].sum()
        xd[m] -= (w[m] * xd[m]).sum() / sw
        yd[m] -= (w[m] * yd[m]).sum() / sw
    sxx = float((w * xd * xd).sum())
    if sxx == 0.0:
        return float("nan"), float("nan")
    return float((w * xd * yd).sum() / sxx), float(np.sqrt(1.0 / sxx))


def score_gamma(task) -> float:
    """The task's headline exponent (Task.score_gamma), else the scorer's V2_GAMMA."""
    g = getattr(task, "score_gamma", None)
    return float(g) if g else V2_GAMMA


def score_mean_charge(xs, gamma: float) -> float:
    """The charge of the mean score: -ln(mean_i exp(-gamma x_i)) / gamma.  One entry: itself.  The score of several
    quantities is the mean of their scores, so one measured out of n keeps at least 1 / n of its score."""
    x = np.asarray(list(xs), dtype=float)
    if x.size == 1:
        return float(x[0])
    return float(-(np.logaddexp.reduce(-gamma * x) - math.log(x.size)) / gamma)


def v2_aggregate(built: dict, sigma_ref: Dict[str, float], ref_pois: Optional[Dict[str, dict]],
                 gamma: float = V2_GAMMA) -> Optional[dict]:
    """Aggregate a case set (_poi_cases or a kind's own): per role (POI) the mean charge of its charged cases (the
    answered ones and those with no measurement, charged max(F, 0): the role's K) plus V2_BIAS_BETA x its bias term
    (multiworld; none where built["multiworld"] is False), combined over roles as the charge of their mean score
    (grouped roles: over groups, each the mean charge of its roles), and the cases' cluster-robust SE ("delta_se":
    report-only, the bias term adds none; the score's band resamples the whole score, sbh.band).  A failed dataset's
    cases are charged as no measurement ("failed_datasets"); the cases that cannot be charged (fallback, delta None)
    fail the run, S = 0 (score_v3), and v2 records their datasets ("failed", per dataset its reasons and the cases the
    model cannot give); with nothing charged the charges are None.  Provisional cases (built["unscored"]) are left out
    and listed.  No effort share ("eps" 0, "s_effort" None: S = exp(-gamma Delta_total)).  Under the
    shipped rule (built["rule"] SHIPPED_RULE) _shipped_aggregate."""
    if built.get("rule") == SHIPPED_RULE:
        return _shipped_aggregate(built, sigma_ref, ref_pois, gamma)
    cases, roles, groups = built["cases"], built["roles"], built["groups"]
    all_failed = {s: list(w) for s, w in sorted((built.get("failed") or {}).items())}
    failed_cases = built.get("failed_cases") or {}
    if not cases and not all_failed:
        return None
    fatal = built.get("fatal")
    failed = all_failed if fatal is None else {s: w for s, w in all_failed.items() if s in fatal}
    charged_failed = {s: w for s, w in all_failed.items() if s not in failed}
    answered = [c for c in cases if not c["fallback"]]
    charged = [c for c in cases if c["delta"] is not None]
    per_poi = {}
    for p in roles:
        cs = [c for c in charged if c["poi"] == p]
        if not cs:
            continue
        d = np.array([c["delta"] for c in cs])
        mw = multiworld(cs) if built["multiworld"] else None
        dp = float(d.mean()) + (V2_BIAS_BETA * mw["bias"] if mw else 0.0)
        per_poi[p] = {"n": len(cs), "n_fallback": int(sum(c["fallback"] for c in cases if c["poi"] == p)),
                      "n_no_measurement": int(sum(bool(c.get("no_measurement")) for c in cs)),
                      "delta": dp, "delta_mean": float(d.mean()), "multiworld": mw,
                      "delta_se": _cluster_se(d, [c["cluster"] for c in cs]),
                      "n_capped": int(sum(c["capped"] for c in cs)), "n_clipped": int(sum(c["clipped"] for c in cs)),
                      "n_ungated": int(sum(c["gated"] is False for c in cs)),
                      "n_ungated_scalar_ref": int(sum(c["gated"] is False and c["ref_source"] == "scalar" for c in cs)),
                      "s_quality": float(np.exp(-gamma * dp)),
                      "delta_precision_mean": _piece_mean(cs, "delta_precision"),
                      "delta_pull_mean": _piece_mean(cs, "delta_pull")}
    members = _group_members(per_poi, groups) if groups is not None else None

    def over_roles(key):
        """A per-POI quantity as a plain mean: over the POIs, or over the groups of their roles' means; None when a
        POI lacks it or nothing was answered."""
        if not per_poi or any(m[key] is None for m in per_poi.values()):
            return None
        if groups is None:
            return float(np.mean([m[key] for m in per_poi.values()]))
        return float(np.mean([np.mean([per_poi[p][key] for p in ps]) for ps in members.values()]))

    def score_mean(key):
        """The charge whose score is the mean of the per-POI scores: -ln(mean_p exp(-gamma x_p)) / gamma over POIs;
        grouped roles are quantities of one kind, so a group enters as one POI whose charge is the mean of its roles'
        charges; one POI: its own charge."""
        if not per_poi:
            return None
        if groups is None:
            return score_mean_charge([m[key] for m in per_poi.values()], gamma)
        return score_mean_charge([np.mean([per_poi[p][key] for p in ps]) for ps in members.values()], gamma)

    for p, m in per_poi.items():
        m["delta_multiworld"] = m["delta"] - m["delta_mean"]
    delta_mean = score_mean("delta")
    d_mw = None if delta_mean is None else float(delta_mean - score_mean("delta_mean"))
    se_cases = None
    if per_poi:
        if groups is None:
            se_cases = float(np.mean([m["delta_se"] for m in per_poi.values()]) / np.sqrt(len(per_poi)))
        else:
            se_cases = float(np.mean([np.mean([per_poi[p]["delta_se"] for p in ps]) / np.sqrt(len(ps))
                                      for ps in members.values()]) / np.sqrt(len(members)))
    s_quality = 0.0 if (failed or delta_mean is None) else float(np.exp(-gamma * delta_mean))
    unscored = list(built.get("unscored") or [])
    no_meas = [c for c in charged if c.get("no_measurement")]
    return {"s_quality": s_quality, "s_effort": None,
            "delta_mean": delta_mean, "delta_se": se_cases,
            "delta_cases": delta_mean,
            "delta_multiworld": d_mw,
            "physics_rule": PHYS_RULE, "multiworld_rule": MW_RULE if built["multiworld"] else None,
            **({"group_rule": GROUP_RULE} if groups is not None else {}),
            "bias_beta": V2_BIAS_BETA if built["multiworld"] else None,
            "delta_cases_se": se_cases,
            "delta_precision": over_roles("delta_precision_mean"), "delta_pull": over_roles("delta_pull_mean"),
            "gamma": gamma, "eps": 0.0, "cap_floor": None, "clip": None,
            "n_cases": len(cases), "n_fallback": len(cases) - len(answered),
            "n_no_measurement": len(no_meas),
            "n_capped": int(sum(c["capped"] for c in answered)), "n_clipped": int(sum(c["clipped"] for c in answered)),
            "failed": {"n_datasets": len(failed), "datasets": [
                {"seed": s, "reasons": w,
                 **({"cases": [{"poi": p, "reason": r} for p, r in failed_cases[s]]} if failed_cases.get(s) else {})}
                for s, w in failed.items()]} if failed else None,
            "failed_datasets": {"n_datasets": len(charged_failed), "datasets": [
                {"seed": s, "reasons": w,
                 **({"cases": [{"poi": p, "reason": r} for p, r in failed_cases[s]]} if failed_cases.get(s) else {})}
                for s, w in charged_failed.items()]} if charged_failed else None,
            "n_failed_charged": int(sum(bool(c.get("failed_charged")) for c in charged)),
            "unscored": unscored or None,
            "ref_loss_source": ("reference per-seed" if ref_pois else
                                ("sigma_ref scalar" if sigma_ref else "ignorance forecast")),
            "per_poi": per_poi, "cases": cases,
            **({"groups": _group_blocks(per_poi, members)} if groups is not None else {})}


def _effort(fits: Dict[str, List[tuple]], scored_truths: Dict[str, set], effort_scale: Dict[str, float]) -> float:
    """S_effort, the shipped rule's (_shipped_aggregate): per POI (role) whose truth varies over >= 3 values, 0.4 x
    (its answers move by > 0.1 of the reference scale) + 0.6 x clip(slope / se / 3, 0, 1) of the error-weighted slope
    of answer on truth (grouped cases: within their fit groups); the mean over those POIs, 0 without one."""
    eff_terms = []
    for p, rows in fits.items():
        if len(scored_truths[p]) < 3:
            continue
        scale = effort_scale[p]
        vals = np.array([r[1] for r in rows], dtype=float)
        moves = 1.0 if len(vals) >= 2 and float(vals.std()) > 0.1 * scale else 0.0
        grade = 0.0
        if len(rows) >= 3:
            t = np.array([r[0] for r in rows], dtype=float)
            if len(set(np.round(t, 9))) >= 3:
                sig = np.array([r[2] for r in rows], dtype=float)
                within = [r[3] for r in rows] if len(rows[0]) > 3 else None
                slope, se_b = (_wls_slope(t, vals, sig) if within is None or all(w is None for w in within)
                               else _wls_slope_within(t, vals, sig, within))
                if np.isfinite(slope) and np.isfinite(se_b) and se_b > 0 and slope > 0:
                    grade = float(np.clip(slope / se_b / 3.0, 0.0, 1.0))
        eff_terms.append(0.4 * moves + 0.6 * grade)
    return float(np.mean(eff_terms)) if eff_terms else 0.0


def _group_blocks(per_poi: dict, members: Dict[str, List[str]]) -> dict:
    """v2 "groups": per group its roles and their charges' means."""
    return {g: {"roles": ps, "delta": float(np.mean([per_poi[p]["delta"] for p in ps])),
                "delta_mean": float(np.mean([per_poi[p]["delta_mean"] for p in ps])),
                "delta_multiworld": float(np.mean([per_poi[p]["delta_multiworld"] for p in ps])),
                "delta_precision_mean": _piece_mean([per_poi[p] for p in ps], "delta_precision_mean"),
                "delta_pull_mean": _piece_mean([per_poi[p] for p in ps], "delta_pull_mean")}
            for g, ps in members.items()}


def _shipped_aggregate(built: dict, sigma_ref: Dict[str, float], ref_pois: Optional[Dict[str, dict]],
                       gamma: float) -> Optional[dict]:
    """v2_aggregate under the shipped rule (SHIPPED_RULE: a task frozen with its verifier's pass cut): per POI the
    mean of its case charges (a failed case at C + V2_FAIL_PREMIUM), averaged over POIs, plus completeness
    V2_K_MISSING x the mean over datasets of their unanswered fraction; the SE of the cases and the completeness term
    (with their covariance, from per-cluster influences); effort x the answered-case fraction.  No bias term, no
    failure of the run."""
    cases, fits, unanswered, roles = built["cases"], built["fits"], built["unanswered"], built["roles"]
    scored_truths, effort_scale = built["scored_truths"], built["effort_scale"]
    if not cases:
        return None
    per_poi = {}
    for p in roles:
        cs = [c for c in cases if c["poi"] == p]
        if not cs:
            continue
        d = np.array([c["delta"] for c in cs])
        dp = float(d.mean())
        n_fb = int(sum(c["fallback"] for c in cs))
        per_poi[p] = {"n": len(cs), "n_fallback": n_fb, "unanswered": n_fb / len(cs),
                      "delta": dp, "delta_mean": dp, "multiworld": None,
                      "delta_se": _cluster_se(d, [c["cluster"] for c in cs]),
                      "n_capped": int(sum(c["capped"] for c in cs)), "n_clipped": int(sum(c["clipped"] for c in cs)),
                      "n_ungated": int(sum(c["gated"] is False for c in cs)),
                      "n_ungated_scalar_ref": int(sum(c["gated"] is False and c["ref_source"] == "scalar" for c in cs)),
                      "s_quality": float(np.exp(-gamma * dp)),
                      "delta_precision_mean": _piece_mean(cs, "delta_precision"),
                      "delta_pull_mean": _piece_mean(cs, "delta_pull"), "delta_multiworld": 0.0}

    def over_roles(key):
        if any(m[key] is None for m in per_poi.values()):
            return None
        return float(np.mean([m[key] for m in per_poi.values()]))

    delta_cases = over_roles("delta")
    u_mean = float(np.mean([u for u, _ in unanswered])) if unanswered else 0.0
    delta_missing = V2_K_MISSING * u_mean
    delta_mean = delta_cases + delta_missing
    se_cases = float(np.mean([m["delta_se"] for m in per_poi.values()]) / np.sqrt(len(per_poi)))
    g_miss: Dict[str, float] = {}
    for u, cl in unanswered:
        g_miss[cl] = g_miss.get(cl, 0.0) + V2_K_MISSING * (u - u_mean) / len(unanswered)
    g_case: Dict[str, float] = {}
    for p, m in per_poi.items():
        cs = [c for c in cases if c["poi"] == p]
        for c in cs:
            k = c.get("dataset_cluster", c["cluster"])
            g_case[k] = g_case.get(k, 0.0) + (c["delta"] - m["delta_mean"]) / len(cs) / len(per_poi)
    se_missing = float(np.sqrt(sum(v * v for v in g_miss.values())))
    cov = sum(v * g_case.get(k, 0.0) for k, v in g_miss.items())
    delta_se = float(np.sqrt(max(se_cases ** 2 + se_missing ** 2 + 2.0 * cov, 0.0)))
    genuine_frac = 1.0 - (sum(c["fallback"] for c in cases) / len(cases))
    s_effort = _effort(fits, scored_truths, effort_scale) * genuine_frac
    return {"s_quality": float(np.exp(-gamma * delta_mean)), "s_effort": s_effort,
            "delta_mean": delta_mean, "delta_se": delta_se,
            "delta_cases": delta_cases, "delta_missing": delta_missing,
            "delta_multiworld": over_roles("delta_multiworld"), "multiworld_rule": None,
            "delta_cases_se": se_cases, "delta_missing_se": se_missing,
            "delta_precision": over_roles("delta_precision_mean"), "delta_pull": over_roles("delta_pull_mean"),
            "n_datasets_unanswered": float(sum(u for u, _ in unanswered)),
            "gamma": gamma, "eps": V2_EPS, "cap_floor": V2_CAP_FLOOR, "clip": V2_CLIP,
            "fail_premium": V2_FAIL_PREMIUM, "k_missing": V2_K_MISSING,
            "n_cases": len(cases), "n_fallback": int(sum(c["fallback"] for c in cases)),
            "n_capped": int(sum(c["capped"] for c in cases)), "n_clipped": int(sum(c["clipped"] for c in cases)),
            "ref_loss_source": ("reference per-seed" if ref_pois else
                                ("sigma_ref scalar" if sigma_ref else "ignorance forecast")),
            "per_poi": per_poi, "cases": cases}



V3_W_CAL = 0.10
V3_W_CAL_SHIPPED = 0.20


V3_W_GOF = 0.30


V3_W_SYST = 0.10

V3_W_COMP = 0.1

V3_COMP_FLOOR = -1.0

V3_COMP_ALLOWANCE = 1.5

COMP_GATE_RULE = "compute bonus gated on the rest of the score (from the reference's level, width 0.05)"

V3_COMP_GATE_REST0 = 0.0

V3_COMP_GATE_WIDTH = 0.05

GATE_REST0_STEP = 0.005


def reference_rest(reference: Optional[dict]) -> Optional[float]:
    """The closure reference's Delta_rest (closure.json reference_score: its delta_rest, else its delta_total -- the
    closure's reference is timed by the yardstick in its own evaluation, so its compute term is 0), None without one."""
    if not reference:
        return None
    v = reference.get("delta_rest", reference.get("delta_total"))
    return None if v is None else float(v)


def compute_gate_params(task, reference: Optional[dict] = None) -> Tuple[float, float]:
    """(rest0, width) of the compute bonus gate.  rest0: the task's compute_gate_rest0 when set (an explicit gate),
    else the closure reference's Delta_rest (reference_rest of `reference`, closure.json
    reference_score) rounded up to the next GATE_REST0_STEP, so the reference keeps g = 1; else the scorer's default.
    width: the task's compute_gate_width, else the scorer's."""
    r0 = getattr(task, "compute_gate_rest0", None)
    if r0 is None:
        rest = reference_rest(reference)
        r0 = V3_COMP_GATE_REST0 if rest is None else math.ceil(round(rest / GATE_REST0_STEP, 9)) * GATE_REST0_STEP
    w = getattr(task, "compute_gate_width", None)
    return float(r0), float(V3_COMP_GATE_WIDTH if w is None else w)


def compute_gate_source(task, reference: Optional[dict] = None) -> str:
    """Where the gate's (rest0, width) came from (v3.compute.gate_source): task.compute_gate_source if set (tb_score:
    "explicit", a pass-rule file or the task fallback), else "task", "closure" or "scorer default"."""
    src = getattr(task, "compute_gate_source", None)
    if src:
        return str(src)
    if getattr(task, "compute_gate_rest0", None) is not None:
        return "task"
    return "closure" if reference_rest(reference) is not None else "scorer default"


def compute_gate(delta_rest: float, rest0: float = V3_COMP_GATE_REST0, width: float = V3_COMP_GATE_WIDTH) -> float:
    """g in [0, 1], the share of the compute bonus kept: 1 at delta_rest <= rest0, linear to 0 at rest0 + width."""
    excess = max(0.0, float(delta_rest) - rest0)
    if width <= 0:
        return 1.0 if excess <= 0 else 0.0
    return float(min(1.0, max(0.0, 1.0 - excess / width)))


def gated_compute(delta_compute: float, delta_rest: float, rest0: float = V3_COMP_GATE_REST0,
                  width: float = V3_COMP_GATE_WIDTH) -> Tuple[float, float]:
    """(scored Delta_compute, g): max(0, d) + g * min(0, d), g = compute_gate(delta_rest)."""
    g = compute_gate(delta_rest, rest0, width)
    d = float(delta_compute)
    return float(max(0.0, d) + g * min(0.0, d)), g


def _wls(X: np.ndarray, y: np.ndarray, sigma: np.ndarray):
    w = 1.0 / sigma ** 2
    A, b = X * np.sqrt(w)[:, None], y * np.sqrt(w)
    beta, *_ = np.linalg.lstsq(A, b, rcond=None)
    cov = np.linalg.pinv(A.T @ A)
    r = b - A @ beta
    return beta, np.sqrt(np.clip(np.diag(cov), 0, None)), float(r @ r) / max(len(y) - X.shape[1], 1)


def leakage(records: List[dict], truths, task) -> Dict[str, dict]:
    """Calibration leakage per (POI, effect): weighted fit of the POI residual on the residual an uncalibrated
    analysis would show (task.leakage_probes).  k = 0 calibrated, 1 uncalibrated, < 0 over-corrected.  Errors are
    floored at 10% of their median so a degenerate sigma cannot dominate."""
    from .kinds import get_kind
    kind = get_kind(task)
    out = {}
    for poi, effects in (task.leakage_probes() or {}).items():
        for eff, fn in effects.items():
            rows = []
            for rec in records:
                t = truths[rec["seed"]]
                if not record_ok(rec) or poi not in scoring_truth(task, t):
                    continue
                pr = {x["name"]: x for x in rec["result"]["pois"]}.get(poi)
                if pr is None:
                    continue
                sg = kind.poi_result(pr, float(t[poi])).sigma()
                rows.append((float(pr["value"]) - float(t[poi]), sg, float(fn(t))))
            if len(rows) < 4:
                continue
            r, sg, x = (np.array(c, dtype=float) for c in zip(*rows))
            if np.ptp(x) == 0:
                continue
            sg = np.maximum(sg, 0.1 * np.median(sg[sg > 0]) if np.any(sg > 0) else 1.0)
            beta, se, chi2 = _wls(np.c_[np.ones_like(x), x], r, sg)
            out[f"{poi}:{eff}"] = {"k": float(beta[1]), "se": float(se[1]), "n": len(rows), "chi2_dof": chi2}
    return out


def compute_cost(records: List[dict], task, yardstick: Optional[dict] = None) -> dict:
    """The raw compute term (score_v3 gates its bonus): wall seconds on the task's worker shape are the figure of
    merit, CPU seconds recorded alongside.  Delta_compute = mean of d' (module docstring); a timed-out dataset is a
    failed case, charged elsewhere.  delta_unallowed = mean of d before the allowance.  With a yardstick
    (task.compute_yardstick set by a grading verifier, or ``yardstick`` from run_yardstick) the reference wall is Y x k
    (yardstick.reference_wall); no usable Y means no compute term."""
    yard = yardstick if yardstick is not None else getattr(task, "compute_yardstick", None)
    ys = yref = None
    if yard is not None:
        from .yardstick import reference_wall, summary
        ys, yref = summary(yard), reference_wall(yard)
    done = [r for r in records if r.get("exit_code") == 0 and not r.get("timed_out")]
    if not done:
        return {"n": 0, **({"yardstick": ys} if ys else {})}
    cost = np.array([r["wall_s"] for r in done], dtype=float)
    out = {"n": len(done), "source": "wall", "cost_median_s": float(np.median(cost)),
           "wall_median_s": float(np.median(cost)), "cost_total_s": float(cost.sum())}
    if ys:
        out["yardstick"] = ys
    if all(r.get("cpu_s") is not None for r in done):
        cpu = np.array([r["cpu_s"] for r in done], dtype=float)
        out["cpu_median_s"] = float(np.median(cpu))
        out["cpu_total_s"] = float(cpu.sum())
        out["parallelism"] = float(np.median(cpu / np.maximum(cost, 1e-6)))
        out["maxrss_mb_max"] = float(max(r.get("maxrss_mb") or 0 for r in done))
    ref = yref if yard is not None else getattr(task, "compute_reference_wall_s", None)
    if ref:
        from .yardstick import timeout_scale
        limit = float(getattr(task, "run_timeout_s", 0) or 0) * timeout_scale(task, yard)
        out["reference_wall_s"] = float(ref)
        out["limit_s"] = limit or None
        d, dp = compute_charges(cost, float(ref), limit)
        out["allowance"] = V3_COMP_ALLOWANCE
        out["delta_unallowed"] = float(np.mean(d))
        out["delta"] = float(np.mean(dp))
    return out


def compute_charges(walls: np.ndarray, ref: float, limit: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
    """(d, d') per completed dataset: d = clip(log2(wall / ref), V3_COMP_FLOOR, log2(limit / ref)) (no top without a
    limit above ref) and d' = d below 0, else max(0, d - log2(V3_COMP_ALLOWANCE)).  Delta_compute is the mean of d'
    (compute_cost; the band resamples them, sbh.band)."""
    hi = np.log2(limit / ref) if (limit and limit > ref) else np.inf
    d = np.clip(np.log2(np.maximum(np.asarray(walls, dtype=float), 1e-6) / ref), V3_COMP_FLOOR, hi)
    return d, np.where(d < 0, d, np.maximum(0.0, d - np.log2(V3_COMP_ALLOWANCE)))



def own_budget_compute(comp: dict, records: List[dict], task, yardstick: Optional[dict], label: Optional[str]) -> dict:
    """Compute term of a run under its own budget (Task.own_budget): Delta 0, since its walls were not taken on the
    task's worker shape; delta_measured and delta_scaled (walls x its CPUs / the task's, an estimate) kept for
    display.  Other runs: compute_cost's term unchanged."""
    own = task.own_budget(label) if label and hasattr(task, "own_budget") else {}
    if not own:
        return comp
    base = int(getattr(task, "run_cpus", 0) or 0)
    cpus = int(own.get("run_cpus", base) or 0)
    out = {**comp, "own_budget": own, "cpus": cpus or None, "task_cpus": base or None}
    if comp.get("delta") is not None:
        scale = cpus / base if (cpus and base) else 1.0
        scaled = compute_cost([{**r, "wall_s": r["wall_s"] * scale} for r in records if r.get("wall_s") is not None],
                              task, yardstick)
        out.update(delta=0.0, delta_measured=comp["delta"], cpu_scale=scale,
                   wall_scaled_median_s=scaled.get("wall_median_s"), delta_scaled=scaled.get("delta"))
    return out


def reference_compute(comp: dict, yardstick: Optional[dict], making_baseline: bool = False) -> dict:
    """The closure's reference timed by the yardstick in its own evaluation is the yardstick itself: its compute term
    is 0 by definition (the measured one kept as delta_measured), as a further reference's under its own budget
    (own_budget_compute).  Against a static reference wall it keeps the measured term.  Every kind's rule."""
    if not making_baseline or yardstick is None or comp.get("delta") is None:
        return comp
    return {**comp, "delta": 0.0, "delta_measured": comp["delta"], "yardstick_reference": True}


def reference_record(public: dict, private: dict) -> Optional[dict]:
    """What the closure records of its reference run's score (closure.json reference_score): S, Delta_total, each
    physics case's charge and each answered case's signed pull, the band's inputs (private "band_inputs"), the rules it
    was scored under, the yardstick of every run's relative score (relative_score).  None without a v3 score."""
    v3 = public.get("v3")
    if not v3:
        return None
    cases = {f"{c['seed']}:{c['poi']}": c["delta"] for c in (private.get("v2_cases") or [])
             if c.get("delta") is not None}
    mw_misses = {f"{c['seed']}:{c['poi']}": c["mw_miss"] for c in (private.get("v2_cases") or [])
                 if c.get("mw_miss") is not None and not c.get("fallback")}
    v2 = public.get("v2") or {}
    return {"label": public.get("label"), "s_v3": v3["s_v3"], "delta_total": v3["delta_total"],
            "delta_rest": (v3.get("compute") or {}).get("delta_rest", v3["delta_total"]), "cases": cases,
            "mw_misses": mw_misses,
            "band_inputs": private.get("band_inputs"),
            "physics_rule": v2.get("physics_rule"),
            "multiworld_rule": v2.get("multiworld_rule"), "fail_premium": v2.get("fail_premium"),
            **({"group_rule": v2["group_rule"]} if v2.get("group_rule") else {}),
            **({"calibration_rule": (v3.get("calibration") or {}).get("charge_rule")}
               if (v3.get("calibration") or {}).get("charge_rule") else {}),
            **({"mu_rule": public["mu_rule"]} if public.get("mu_rule") else {}),
            **({"mu_baseline_id": public.get("mu_baseline_id")} if public.get("mu_rule") else {})}


def load_reference_record(suite_dir: Path) -> Optional[dict]:
    """The suite's reference score (closure.json reference_score), None if the closure has not recorded one."""
    p = Path(suite_dir) / "closure.json"
    if not p.exists():
        return None
    return (read_json_maybe_shielded(p) or {}).get("reference_score")


def _physics_rule_mismatch(v2: dict, ref: dict) -> Optional[str]:
    """Why a run's physics term (its public v2) and the closure reference's (reference_record) were scored under
    different rules, None where they were not: another physics rule (v2 physics_rule, None under the shipped rule),
    another bias-term rule (v2 multiworld_rule, None without one), another grouping of roles (v2 group_rule, absent
    without groups) or another failure premium (the shipped rule's).  A
    record kept before the reference recorded them carries none, so it matches only a run under the shipped rule."""
    run_p, ref_p = v2.get("physics_rule"), ref.get("physics_rule")
    if run_p != ref_p:
        return f"the run's physics term was scored under {run_p!r}, the closure's reference under {ref_p!r}"
    run_rule, ref_rule = v2.get("multiworld_rule"), ref.get("multiworld_rule")
    if run_rule != ref_rule:
        return f"the run's physics term was scored under {run_rule!r}, the closure's reference under {ref_rule!r}"
    run_g, ref_g = v2.get("group_rule"), ref.get("group_rule")
    if run_g != ref_g:
        return f"the run's roles were grouped under {run_g!r}, the closure's reference's under {ref_g!r}"
    pr, pf = v2.get("fail_premium"), ref.get("fail_premium")
    if pr is not None and pf is not None and abs(float(pr) - float(pf)) > 1e-12:
        return f"the run's failed cases were charged their cap + {pr}, the closure's reference's + {pf}"
    return None


def relative_score(public: dict, private: dict, ref: Optional[dict]) -> Optional[dict]:
    """A run's score relative to the reference's on the same datasets: ratio = S / S_ref (the reference 1, shown as
    100 %), and its band "range", the 16 / 84 % quantiles of S / S_ref over a paired bootstrap of the datasets (the
    run's and the reference's full scores recomputed on the same resamples, sbh.band.paired; its header in "band");
    a failed run (v3 "failed") 0, its range [0, 0].  The range is None, with "band_missing" saying why, where the
    run's private scores or the closure's reference_score carry no band inputs.  Under the shipped rule (v2 without a
    physics_rule: a task kept on the rule it shipped with) the range it shipped with (_shipped_range).  None without a
    v3 score or a reference record, and (refused, with the reason on stderr) against a reference scored under another
    mu rule, mu baseline, physics rule (_physics_rule_mismatch) or calibration charge rule (where the run has a
    calibration term)."""
    v3 = public.get("v3")
    if not v3 or not ref or not ref.get("s_v3"):
        return None
    if (public.get("mu_rule") or None) != (ref.get("mu_rule") or None):
        import sys
        print(f"relative score refused: the run was scored under mu rule {public.get('mu_rule')!r}, the closure's "
              f"reference under {ref.get('mu_rule')!r} (re-close the suite: sbh closure, then sbh post-score --label "
              "reference)", file=sys.stderr)
        return None
    if (public.get("mu_baseline_id") or None) != (ref.get("mu_baseline_id") or None):
        import sys
        print(f"relative score refused: the run was scored against mu baseline {public.get('mu_baseline_id')!r}, the "
              f"closure's reference against {ref.get('mu_baseline_id')!r} (rescore the run against the current "
              "closure)", file=sys.stderr)
        return None
    v2 = public.get("v2") or {}
    why = _physics_rule_mismatch(v2, ref)
    run_c = (v3.get("calibration") or {}).get("charge_rule")
    if not why and run_c is not None and run_c != ref.get("calibration_rule"):
        why = (f"the run's calibration term was scored under {run_c!r}, the closure's reference under "
               f"{ref.get('calibration_rule')!r}")
    if why:
        import sys
        print(f"relative score refused: {why} (re-score the closure's reference: sbh post-score --label reference)",
              file=sys.stderr)
        return None
    if v3.get("failed"):
        return {"ratio": 0.0, "percent": 0.0, "range": [0.0, 0.0], "band": None, "n_datasets_paired": 0,
                "reference": ref.get("label"), "reference_s_v3": ref["s_v3"], "failed": True}
    ratio = float(v3["s_v3"] / ref["s_v3"])
    out = {"ratio": ratio, "percent": 100.0 * ratio, "reference": ref.get("label"), "reference_s_v3": ref["s_v3"]}
    if v2.get("physics_rule") is None:
        return {**out, **_shipped_range(v3, private, ref, ratio, float(v2.get("gamma") or V2_GAMMA))}
    run_in, ref_in = private.get("band_inputs"), ref.get("band_inputs")
    if not (run_in and ref_in):
        miss = ("the closure's reference_score carries no band inputs (re-score it: sbh post-score --label reference)"
                if run_in else "the run's private scores carry no band inputs (rescore it)")
        return {**out, "range": None, "band": None, "band_missing": miss, "n_datasets_paired": 0}
    from .band import paired

    def answered(inp):
        return {r[0] for r in inp["phys"].get("rows") or ()}
    try:
        b = paired(run_in, ref_in)
    except Exception as e:  # noqa: BLE001
        return {**out, "range": None, "band": None, "n_datasets_paired": 0,
                "band_missing": f"not computed: {type(e).__name__}: {e}"[:300]}
    return {**out, "range": b.pop("ratio"), "band": b, "n_datasets_paired": len(answered(run_in) & answered(ref_in))}


def _shipped_range(v3: dict, private: dict, ref: dict, ratio: float, gamma: float) -> dict:
    """The relative score's range under the shipped rule (SHIPPED_RULE, kept as it shipped): ratio exp(-+gamma se), se
    from the paired per-dataset differences of the physics cases' charges (ref "cases") and the run's quality terms'
    standard errors in quadrature; "se_delta" se, "n_datasets_paired" the datasets with a paired case."""
    per: Dict[str, List[float]] = {}
    for c in private.get("v2_cases") or []:
        k = f"{c['seed']}:{c['poi']}"
        if c.get("delta") is not None and k in ref.get("cases", {}):
            per.setdefault(c["seed"], []).append(c["delta"] - ref["cases"][k])
    d = [float(np.mean(x)) for x in per.values()]
    se = None
    if len(d) > 1:
        se2 = float(np.var(d, ddof=1)) / len(d)
        for name, w in (v3.get("weights") or {}).items():
            t = v3.get(name)
            if name not in ("phys", "compute") and isinstance(t, dict) and t.get("se"):
                se2 += (w * t["se"]) ** 2
        se = math.sqrt(se2)
    return {"range": [ratio * math.exp(-gamma * se), ratio * math.exp(gamma * se)] if se is not None else None,
            "se_delta": se, "n_datasets_paired": len(d)}


def run_yardstick(run_dir: Path, task) -> Optional[dict]:
    """A dev evaluation's compute yardstick: <run>/yardstick.json via sbh.yardstick.load (ok-false if absent).
    None for a task that does not time its reference in the evaluation (no compute_yardstick_k; a grading verifier
    with its own reference wall clears it) and for one whose verifier set its yardstick record
    (Task.compute_yardstick)."""
    if getattr(task, "compute_yardstick_k", None) is None or getattr(task, "compute_yardstick", None) is not None:
        return None
    from .yardstick import RECORD_FILE, load, load_replicas
    seeds = getattr(task, "compute_yardstick_seeds", None)
    if seeds:
        rec = load_replicas(run_dir, seeds, float(task.compute_yardstick_k))
        (Path(run_dir) / RECORD_FILE).write_text(json.dumps(rec, indent=1, default=str))
        return rec
    return load(Path(run_dir) / RECORD_FILE)


def yardstick_weight(v3: Optional[dict], yardstick: Optional[dict]) -> None:
    """With a yardstick but no compute term, report the applied compute weight 0 in v3.weights."""
    if yardstick is not None and v3 and v3.get("weights") and (v3.get("compute") or {}).get("delta") is None:
        v3["weights"]["compute"] = 0.0


def _reported_poi(rec: dict, name: str) -> Optional[dict]:
    for p in ((rec.get("result") or {}).get("pois") or []):
        if isinstance(p, dict) and p.get("name") == name:
            return p
    return None


def discovery_rates(records: List[dict], truths: Dict[str, dict], task, claimed) -> dict:
    """False-discovery rate over the mu = 0 datasets and discovery power over the scored strong-signal ones, with
    claimed(record) -> bool."""
    mu = getattr(task, "signal_strength_poi", None) or "mu"
    nulls = [r for r in records if truths[r["seed"]].get(mu) == 0.0]
    strong = [r for r in records if mu in scoring_truth(task, truths[r["seed"]])
              and truths[r["seed"]].get(mu, 0.0) > 0]
    return {"fdr": (sum(claimed(r) for r in nulls) / len(nulls)) if nulls else None,
            "power": (sum(claimed(r) for r in strong) / len(strong)) if strong else None, "n_null": len(nulls)}


CORE_FALLBACKS = {
    "ref_scalar_sigma": "a physics case's baseline is the scalar sigma_ref: the closure records no per-dataset "
                        "baseline for it (closure.json reference_pois)",
    "no_measurement": "a scored case on a completed dataset has no measurement (no usable or no linked mass grid, a "
                      "quantity the contract leaves unevaluated, a correction not reported): charged its no-answer "
                      "forecast's charge F, never below the reference's level, the dataset not failed",
}


def fallback_entry(seeds, what: str, pois=None, reasons=None, detail: Optional[Dict[str, float]] = None) -> dict:
    """One kind's count: {"n" (datasets), "seeds" (sorted), "what"}, "pois" (sorted) for a kind taken by physics
    cases rather than datasets (its datasets are then the ones scored for those POIs: a sealed-safe page drops the
    count where a POI is scored on part of them, and the public summary its datasets, public_split), "reasons"
    ({reason: number of cases}) where the kind says why per case, and "detail" ({dataset: number}) where it records
    what the fallback changed per dataset (its "what" says which number)."""
    ss = sorted(set(seeds))
    out = {"n": len(ss), "seeds": ss, "what": what, **({"pois": sorted(set(pois))} if pois is not None else {})}
    if reasons is not None:
        rs = [str(r) for r in reasons]
        out["reasons"] = {r: rs.count(r) for r in sorted(set(rs))}
    if detail:
        out["detail"] = {k: float(detail[k]) for k in sorted(detail)}
    return out


def fallback_block(kind_fallbacks: Optional[Dict[str, dict]], cases: Optional[List[dict]]) -> Dict[str, dict]:
    """v3 "fallbacks": the kind's (``kind_fallbacks``, {kind: fallback_entry}) and the core's from the physics cases
    (_poi_cases, a kind's own): an answered case whose baseline is the scalar sigma_ref ("ref_source" "scalar") rather
    than the closure's per-dataset one, and a case charged max(F, 0) for no measurement ("no_measurement", with its
    reasons).
    {} = no fallback taken."""
    out = dict(kind_fallbacks or {})
    scalar = [c for c in cases or [] if not c.get("fallback") and c.get("ref_source") == "scalar"]
    if scalar:
        out["ref_scalar_sigma"] = fallback_entry([c["seed"] for c in scalar], CORE_FALLBACKS["ref_scalar_sigma"],
                                                 pois=[c["poi"] for c in scalar])
    nm = [c for c in cases or [] if c.get("no_measurement") and c.get("delta") is not None]
    if nm:
        out["no_measurement"] = fallback_entry([c["seed"] for c in nm], CORE_FALLBACKS["no_measurement"],
                                               pois=[c["poi"] for c in nm],
                                               reasons=[f"{c['poi']}: {c.get('why') or 'no measurement'}" for c in nm])
    return out


def public_reason(text: str) -> str:
    """A per-case reason as a public summary may quote it: a path or a dataset id in it (a grader's message can quote
    <run>/runs/seed_NNN/out/...) replaced, so it names no dataset."""
    import re
    t = re.sub(r"(?:\S*/){2,}\S*|(?<!\S)/\S+", "<path>", str(text))
    return re.sub(r"\bseed_\w+", "<dataset>", t)


def public_fallbacks(fb: Optional[Dict[str, dict]]) -> Optional[Dict[str, dict]]:
    """v3 "fallbacks" as the public summary carries it: a kind taken by physics cases (its entry has "pois") keeps its
    counts, what it is, its POIs and its reasons (public_reason: no path, no dataset id), never its datasets (which
    ones carry the POI's case is truth-defined: a mass is scored only where a signal is injected); a kind taken by
    datasets keeps them."""
    if not isinstance(fb, dict):
        return fb
    out = {}
    for k, e in fb.items():
        if isinstance(e, dict) and "pois" in e:
            e = {x: v for x, v in e.items() if x != "seeds"}
            if isinstance(e.get("reasons"), dict):
                rs: Dict[str, int] = {}
                for r, n in e["reasons"].items():
                    rs[public_reason(r)] = rs.get(public_reason(r), 0) + n
                e["reasons"] = rs
        out[k] = e
    return out


def public_failed(failed: Optional[dict]) -> Optional[dict]:
    """v2 / v3 "failed" as the public summary carries it: per failed dataset its reasons with the POI of each case the
    model cannot give dropped ("<poi>: <why>" becomes "<why>", once), and no per-case list; the full block stays in
    the private scores (public_split)."""
    if not isinstance(failed, dict):
        return failed
    out = []
    for d in failed.get("datasets") or []:
        case_reasons = {f"{c['poi']}: {c['reason']}": c["reason"] for c in d.get("cases") or []}
        rs = []
        for r in d.get("reasons") or []:
            r = case_reasons.get(r, r)
            if r not in rs:
                rs.append(r)
        out.append({"seed": d.get("seed"), "reasons": rs})
    return {"n_datasets": failed.get("n_datasets"), "datasets": out}


def public_split(public: dict, private: dict) -> None:
    """Move what could tell which datasets carry a POI's case (a truth-defined subset) from the public summary to the
    private scores, in place: the fallbacks' datasets of a kind taken by physics cases (private "fallbacks", the full
    block), the failed datasets' per-case reasons (private "v2_failed", "v2_failed_datasets"), the provisional cases' list (private
    "v2_unscored"; the public summary keeps the datasets lost whole and the number of cases left out by reason) and
    the provisional cases scored on a reported value (private "v2_provisional"; public: their number).  Run once,
    after the band and the relative score are computed, before the files are written."""
    v2, v3 = public.get("v2") or {}, public.get("v3") or {}
    if isinstance(v3.get("fallbacks"), dict):
        private["fallbacks"] = v3["fallbacks"]
        v3["fallbacks"] = public_fallbacks(v3["fallbacks"])
    if v2.get("failed"):
        private["v2_failed"] = v2["failed"]
        v2["failed"] = public_failed(v2["failed"])
        if v3.get("failed"):
            v3["failed"] = v2["failed"]
    if v2.get("failed_datasets"):
        private["v2_failed_datasets"] = v2["failed_datasets"]
        v2["failed_datasets"] = public_failed(v2["failed_datasets"])
    if v2.get("unscored"):
        private["v2_unscored"] = v2["unscored"]
        cases = [u for u in v2["unscored"] if u.get("poi") is not None]
        v2["unscored"] = [u for u in v2["unscored"] if u.get("poi") is None] or None
        if cases:
            v2["unscored_cases"] = {"n": len(cases), "reasons": unscored_reasons(cases)}
    for blk in (v2, v3):
        pv = blk.get("provisional")
        if isinstance(pv, dict) and pv.get("cases"):
            private["v2_provisional"] = pv
            blk["provisional"] = {"n_cases": pv.get("n_cases")}


PRIVATE_TERM_KEYS = ("v3_calibration_cases", "v3_completeness_cases", "v3_gof_cases", "v3_model_coverage", "fit_rule",
                     "fit_info", "v4_truth_cases", "v4_claims")


def unscored_reasons(unscored: Optional[List[dict]]) -> List[str]:
    """The provisional reasons of the physics cases left out (v2 "unscored" entries with a POI; a lost dataset's is the
    score's own pending reason), one line per cause, a named reason and a count: a stored record graded before the
    grader marked its own errors (REGRADE: grade it again), any other evaluator fault (the evaluator must be fixed and
    rerun), no reference scale (the closure gives none), no grader record under a strict grader (NO_GRADER_RECORD:
    the evaluation lost the dataset's grading), the grader not finished.  Never a case's
    own reason text: an evaluator's message can quote a path that names a dataset carrying the case (the private
    scores keep the cases and their reasons, public_split)."""
    left = [u for u in unscored or [] if u.get("poi") is not None]
    causes = ((REGRADE_FAULT, f"({REGRADE}), until the datasets are graded again"),
              (EVALUATOR_FAULT, "on an evaluator fault, until it is fixed and the evaluation rerun"),
              (NO_REFERENCE_SCALE, "without a reference scale to charge their missing measurement on"),
              (pending_reason(NO_GRADER_RECORD), f"with {NO_GRADER_RECORD} (the evaluation lost their grading), until "
                                                 "their datasets are graded"))

    def cause(u):
        why = str(u.get("reason") or "")
        return next((c for c, _ in causes if why.startswith(c)), None)
    out = []
    for c, says in (*causes, (None, "until the grader finishes them")):
        n = sum(1 for u in left if cause(u) == c)
        if n:
            out.append(f"{n} physics cases left out {says}")
    return out


def score_v3(v2: dict, terms, n_records: int, comp: Optional[dict] = None,
             comp_gate: Optional[Tuple[float, float]] = None, comp_gate_source: Optional[str] = None,
             pending=(), extra: Optional[dict] = None) -> dict:
    """S_v3 from the v2 physics term, the kind's weighted terms and compute.  ``terms``: [(name, weight, term)] in
    summation order (bit-identity matters; term None = nothing to score, weight still listed); ``pending``: reasons
    the score is provisional (no compute bonus), to which the physics term's own are added (its cases scored on a
    reported value or left out while the grader is pending); ``extra``: further output fields; comp_gate: (rest0,
    width), None = scorer defaults.  S_v3 = exp(-gamma Delta_total); under the shipped rule (v2 "eps" > 0) the effort
    share blends in: (1 - eps) exp(-gamma Delta_total) + eps S_effort.  A run with a failed dataset (v2 "failed")
    scores S_v3 = 0, final whatever is pending; its Delta terms stay as diagnostics of what it answered (None where it
    answered nothing)."""
    failed = v2.get("failed")
    prov = list(pending)
    if (v2.get("provisional") or {}).get("n_cases"):
        prov.append(f"{v2['provisional']['n_cases']} physics cases provisional (grader pending)")
    prov += unscored_reasons(v2.get("unscored"))
    if failed:
        prov = []
    dc = (comp or {}).get("delta")
    rest = v2["delta_mean"]
    if rest is not None:
        for _, w, t in terms:
            rest = rest + (w * t["delta"] if t else 0.0)
    if dc is not None:
        r0, gw = comp_gate if comp_gate is not None else (V3_COMP_GATE_REST0, V3_COMP_GATE_WIDTH)
        dcs, g_rest = gated_compute(dc, rest, r0, gw) if rest is not None else (float(max(0.0, float(dc))), 0.0)
        gc = 0.0 if prov else g_rest
        if prov:
            dcs = float(max(0.0, float(dc)))
        comp = {**comp, "delta_raw": float(dc), "delta": dcs, "g": gc, "g_rest": g_rest,
                "delta_rest": None if rest is None else float(rest),
                "gate_rest0": r0, "gate_width": gw, "gate_provisional": prov or None, "gate_rule": COMP_GATE_RULE,
                "gate_source": comp_gate_source or ("passed" if comp_gate is not None else "scorer default")}
        dc = dcs
    dt = None if rest is None else rest + (V3_W_COMP * dc if dc is not None else 0.0)
    se = None
    if v2["delta_se"] is not None:
        se2 = v2["delta_se"] ** 2
        for _, w, t in terms:
            se2 = se2 + ((w * t["se"]) ** 2 if t else 0.0)
        se = math.sqrt(se2)
    s_q = 0.0 if (failed or dt is None) else float(np.exp(-v2["gamma"] * dt))
    weights = {"phys": 1.0, **{name: w for name, w, _ in terms}, "compute": V3_W_COMP}
    eps = float(v2.get("eps") or 0.0)
    s_v3 = 0.0 if failed else (float((1.0 - eps) * s_q + eps * v2["s_effort"]) if eps else s_q)
    out = {"s_v3": s_v3, "s_quality": s_q, "s_effort": v2.get("s_effort"),
           "delta_total": None if dt is None else float(dt), "delta_se": None if se is None else float(se),
           "weights": weights,
           "phys": {"delta": v2["delta_mean"], "se": v2["delta_se"],
                    "delta_cases": v2.get("delta_cases"), "delta_missing": v2.get("delta_missing"),
                    "delta_multiworld": v2.get("delta_multiworld")},
           "compute": comp,
           "provisional_reasons": prov or None,
           **({"failed": failed} if failed else {})}
    out.update({name: t for name, _, t in terms})
    out.update(extra or {})
    return out


def score_run(run_dir: Path, suite_dir: Path, task, label: str,
              sigma_ref: Optional[Dict[str, float]] = None, cfg: Optional[ScoringConfig] = None,
              making_baseline: bool = False) -> dict:
    """Score a run against the suite's truths and write its summary, scores, scoreboard and report card.
    making_baseline: this is `sbh closure`'s run (only a kind with its own score_run uses it)."""
    from .kinds import get_kind
    kind = get_kind(task)
    own = kind.score_run(run_dir, suite_dir, task, label, sigma_ref=sigma_ref, cfg=cfg,
                         making_baseline=making_baseline)
    if own is not None:
        return own
    cfg = cfg or ScoringConfig()
    run_dir, suite_dir = Path(run_dir), Path(suite_dir)
    manifest = read_json_maybe_shielded(suite_dir / "manifest.json")
    truths = {s["seed"]: s["truth"] for s in manifest["seeds"]}
    records = load_scored_records(run_dir, manifest)
    yard = run_yardstick(run_dir, task)
    if all(r.get("placeholder") for r in records):
        raise SystemExit(f"no run records under {run_dir}/runs")
    comp = reference_compute(own_budget_compute(compute_cost(records, task, yard), records, task, yard, label),
                             yard, making_baseline)
    inp = kind.score_inputs(run_dir, records, truths, task,
                            **({"suite_dir": suite_dir} if getattr(kind, "score_inputs_takes_suite", False) else {}))
    scored_recs, measured = inp["records"], inp["source"] != "reported"
    results = seed_results(records, truths, task)
    src = ("self (the closure's own per-seed baseline)" if making_baseline else
           "reference closure" if sigma_ref else "self-referenced (no reference sigma; precision term == 1)")
    sigma_ref = sigma_ref or self_sigma_ref(results, task.pois)
    suite = score_suite(results, sigma_ref, cfg)
    nuis = kind.nuisance_metrics(records, truths, task, cfg)
    rng = np.random.default_rng(int(manifest["master_seed"]) + 7)
    att = {name: score_suite(rows, sigma_ref, cfg)
           for name, rows in scale_rows([r.truth for r in results], sigma_ref, rng, task.pois,
                                         getattr(task, "signal_strength_poi", None)).items()}
    ref_pois, closure_sigma, closure_passed = load_closure_baseline(suite_dir)
    own_ref = None
    if making_baseline:
        from .suite import baseline_pois
        grader = baseline_pois(task)
        own_ref = own_reference_pois(kind, task, results, scored_recs)
        rec = {s: {p: v for p, v in (e or {}).items() if p not in grader} for s, e in (ref_pois or {}).items()}
        stale = sorted(s for s, e in own_ref.items() if rec.get(s) and _numbers_differ(rec[s], e))
        if stale:
            print(f"note: closure.json's per-dataset baselines differ from the ones this closure run is scored on, on "
                  f"{len(stale)} datasets ({', '.join(stale[:5])}{', ...' if len(stale) > 5 else ''}); other runs read "
                  f"closure.json's until `sbh closure --phase score` records these")
        ref_pois = {s: {**e, **{p: v for p, v in ((ref_pois or {}).get(s) or {}).items() if p in grader}}
                    for s, e in own_ref.items()}
        closure_sigma, closure_passed = {}, True
    v2 = score_v2(scored_recs, truths, task, closure_sigma or sigma_ref, ref_pois, kind.case_status(inp))
    if v2 and not closure_passed:
        v2["ref_loss_source"] += " (closure not passed — rescore after it does)"
    v2_cases = v2.pop("cases") if v2 else None
    if v2 and measured:
        kind.mark_provisional(v2, v2_cases, inp)
    leak = leakage(scored_recs, truths, task)
    yld = kind.declared_yield(run_dir, scored_recs, truths, task)
    v3 = None
    st = {}
    if v2:
        lost = [] if score_rule(task) == SHIPPED_RULE else [r["seed"] for r in records if r.get("placeholder")]
        t_recs, t_inp = records, inp
        if lost:
            t_recs = [r for r in records if not r.get("placeholder")]
            t_inp = {**inp, "records": [r for r in inp["records"] if not r.get("placeholder")]}
        st = kind.score_terms(run_dir, t_recs, truths, task, t_inp, v2, leak, yld)
        terms = list(st["terms"]) + list(task.extra_terms(run_dir, t_recs, truths, t_inp)
                                         if hasattr(task, "extra_terms") else [])
        pending = list(st["pending"]) + ([f"{len(lost)} datasets without a record.json, left out until rerun "
                                          f"(infrastructure): {', '.join(lost[:8])}{', ...' if len(lost) > 8 else ''}"]
                                         if lost else [])
        ref_record = load_reference_record(suite_dir)
        v3 = score_v3(v2, terms, len(t_recs), comp=comp, comp_gate=compute_gate_params(task, ref_record),
                      comp_gate_source=compute_gate_source(task, ref_record),
                      pending=pending, extra=st["extra"])
        if score_rule(task) != SHIPPED_RULE:
            v3["fallbacks"] = fallback_block(st.get("fallbacks"), v2_cases)
        yardstick_weight(v3, yard)
    cluster_bias = cluster_bias_tests(records, truths, task)
    from .tripwire import tripwire
    trip = tripwire(scored_recs, getattr(task, "tripwire_spec", None), closure_sigma or sigma_ref, ref_pois)
    if measured:
        if trip is not None:
            trip["input"] = inp["source"]
        trip_rep = tripwire(records, getattr(task, "tripwire_spec", None), closure_sigma or sigma_ref, ref_pois)
    bgf = kind.background_flags(run_dir, records, task)

    def pub(sc):
        return {"s_phys": sc.s_phys if kind.reports_s_phys else None, "recovery": sc.recovery_rate, "fdr": sc.false_discovery_rate,
                "n_null": sc.n_null_seeds, "crosstalk": sc.crosstalk_flags,
                "pois": {p: asdict(m) for p, m in sc.per_poi.items()}}

    fdr = power = None
    n_null = 0
    disc = {}
    if task.result_type == "search":
        rep = discovery_rates(records, truths, task, lambda r: bool(record_ok(r) and r["result"].get("discovery")))
        fdr, power, n_null = rep["fdr"], rep["power"], rep["n_null"]
        md = kind.measured_discovery(inp) if measured else None
        if md is not None:
            mod = discovery_rates(records, truths, task, lambda r: bool(record_ok(r) and md["claim"](r)))
            n_z = sum(1 for r in records if record_ok(r) and md["has"](r))
            disc = {"fdr_reported": fdr, "discovery_power_reported": power, "n_z_obs_model": n_z}
            if n_z:
                fdr, power = mod["fdr"], mod["power"]
            disc["discovery_source"] = "model" if n_z else "reported (model Z_obs pending)"

    public = {
        "label": label, "suite": manifest["suite"], "task": task.name,
        "n_seeds": len(records), "n_completed": sum(record_ok(r) for r in records),
        "sigma_ref": sigma_ref, "sigma_ref_source": src,
        **({"mu_rule": inp["mu_rule"]} if inp.get("mu_rule") else {}),
        **({"mu_baseline_id": inp.get("mu_baseline_id")} if inp.get("mu_rule") else {}),
        "wall_s_mean": float(np.mean([r["wall_s"] for r in records if r.get("wall_s") is not None] or [0.0])),
        "isolation": records[0].get("isolation", {}),
        "scored_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **pub(suite), "v2": v2, "v3": v3, "cluster_bias": cluster_bias,
        "calibration_response": {"leakage": leak, "declared_yield": yld,
                                 **({"source": inp["source"], **st.get("calibration_response", {}),
                                     "leakage_reported": leakage(records, truths, task)} if measured else {})},
        "compute": comp,
        "fdr": fdr, "n_null": n_null, "discovery_power": power, **disc, "nuisances": nuis,
        "tripwire": trip, **({"tripwire_reported": trip_rep} if measured else {}),
        **({"standard_backgrounds": bgf} if bgf is not None else {}),
        "attackers_for_scale": {k: pub(v) for k, v in att.items()},
        "per_seed_status": [{"seed": r["seed"], "completed": record_ok(r), "wall_s": r.get("wall_s"),
                             "errors": r.get("result_errors", []),
                             **({"missing": True} if r.get("placeholder") else {})} for r in records],
    }
    private = {"label": label, "v2_cases": v2_cases, "reference_pois": own_ref, **{k: None for k in PRIVATE_TERM_KEYS},
               **st.get("private", {}), "per_seed": [
        {"seed": r.seed, "truth": truths[r.seed], "scoring_truth": r.truth, "completed": r.completed,
         "pois": {p: asdict(v) for p, v in r.pois.items()},
         "pulls": {p: ((v.value - r.truth[p]) / v.sigma() if (p in r.truth and v.sigma() > 0) else None)
                   for p, v in r.pois.items()},
         "discovery": r.discovery, "significance_obs": r.significance_obs} for r in results]}
    write_scores(run_dir, suite_dir, task, public, private, records, st.get("gof_cases"))
    return public


def write_scores(run_dir: Path, suite_dir: Path, task, public: dict, private: dict, records: List[dict],
                 gof_cases: Optional[List[dict]]) -> None:
    """Write a scored run's summary, scores and scoreboard, then its report card (every kind's scoring path).  The
    run's band (v3 "band", sbh.band.run_band) is resampled from its band inputs (private "band_inputs", every sealed
    dataset clustered by Task.cluster_key; none under the shipped rule).  The summary's `relative` is the score
    relative to the reference's (relative_score): the closure's own run against itself, every other run against the
    suite's closure record.  The per-dataset lists that could tell which datasets carry a POI's case are then moved to
    the private scores (public_split)."""
    from .suite import CLOSURE_LABEL
    if public.get("v3") and score_rule(task) != SHIPPED_RULE:
        from . import band
        try:
            man = read_json_maybe_shielded(Path(suite_dir) / "manifest.json")
            truths = {s["seed"]: s["truth"] for s in man["seeds"]}
            clusters = {r["seed"]: cluster_of(task, truths.get(r["seed"]) or {}, r["seed"]) for r in records}
            private["band_inputs"] = band.inputs(public, private, records, clusters,
                                                 getattr(task, "cluster_key", "profile") or "dataset")
            public["v3"]["band"] = band.run_band(private["band_inputs"])
        except Exception as e:  # noqa: BLE001
            import sys
            why = f"not computed: {type(e).__name__}: {e}"
            print(f"WARNING: score band of {run_dir} {why}", file=sys.stderr)
            private["band_inputs"] = None
            public["v3"]["band"] = {"rule": band.BAND_RULE, "s_v3": None, "delta_total": None, "terms": {},
                                    "why": why[:300]}
    ref = reference_record(public, private) if public.get("label") == CLOSURE_LABEL else load_reference_record(suite_dir)
    public["relative"] = relative_score(public, private, ref)
    if score_rule(task) != SHIPPED_RULE:
        public_split(public, private)
    (run_dir / "public").mkdir(exist_ok=True)
    (run_dir / "private").mkdir(exist_ok=True)
    from .runner import sandbox_mode
    if sandbox_mode() == "setpriv":
        (run_dir / "private").chmod(0o700)
    write_text_atomic(run_dir / "public" / "summary.json", json.dumps(public, indent=1, default=_default))
    write_text_atomic(run_dir / "private" / "scores.json", json.dumps(private, indent=1, default=_default))
    write_text_atomic(run_dir / "public" / "scoreboard.md", format_tables([public]))
    try:
        from .report_card import write as write_card
        write_card(run_dir, suite_dir, task, public=json.loads(json.dumps(public, default=_default)),
                   private=json.loads(json.dumps(private, default=_default)), records=records,
                   gof_cases=gof_cases, written_by="score")
    except ModuleNotFoundError as e:
        if e.name != f"{__package__}.report_card":
            import sys
            print(f"WARNING: report card not written for {run_dir}: {type(e).__name__}: {e}", file=sys.stderr)
    except Exception as e:
        import sys
        print(f"WARNING: report card not written for {run_dir}: {type(e).__name__}: {e}", file=sys.stderr)


def _f(x, fmt="{:.2f}"):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return fmt.format(x)


def format_tables(summaries: List[dict], with_attackers: bool = False) -> str:
    """The scoreboard (public/scoreboard.md).  A run shown unranked (unranked_reason) prints "unranked (its reason)" in
    place of its percentage and its S, and "—" for its Delta_phys and S_phys (both leave the reproduced cases out, so
    they would read better than the run is); a provisional run (v3 "provisional_reasons", not failed) is marked so
    beside its percentage and S, its reasons listed under the table."""
    L = ["| system | % of reference (±1σ) | S | Δ_phys (±1σ) | S_phys | recovery | FDR (mu=0) | disc. power | completed "
         "| wall/seed |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    not_final = []
    for s in summaries:
        v2, v3, rel = s.get("v2"), s.get("v3"), s.get("relative")
        unr = unranked_reason(v3)
        prov = list((v3 or {}).get("provisional_reasons") or []) if v3 and not v3.get("failed") else []
        mark = " (provisional)" if prov and not unr else ""
        if prov:
            not_final.append((s["label"], "unranked" if unr else "provisional", prov))
        rg = (rel or {}).get("range") if (rel or {}).get("band") else None
        pct = (f"**{rel['percent']:.0f} %**" + (f" ({100 * rg[0]:.0f}–{100 * rg[1]:.0f})" if rg else "") + mark
               if rel else "—")
        v2c = (f"{v3['s_v3']:.2f}" + (f" (failed: {v3['failed']['n_datasets']} datasets)" if v3.get("failed") else "")
               + mark if v3 else ("pending" if v2 else "—"))
        if unr:
            pct, v2c = f"unranked ({unr})", "unranked"
        pb = (((v3 or {}).get("band") or {}).get("terms") or {}).get("phys")
        v2d = ("—" if not v2 or v2.get("delta_mean") is None else
               f"{v2['delta_mean']:+.2f} ({pb[0]:+.2f} to {pb[1]:+.2f})" if pb else
               f"{v2['delta_mean']:+.2f} ± {_f(v2.get('delta_se'))}")
        phys = "—" if s["s_phys"] is None else f"{s['s_phys']:.2f}"
        if unr:
            v2d = phys = "—"
        L.append(f"| {s['label']} | {pct} | {v2c} | {v2d} | {phys} | {s['recovery']:.2f} | {_f(s['fdr'])} | {_f(s.get('discovery_power'))} "
                 f"| {s['n_completed']}/{s['n_seeds']} | {s['wall_s_mean']:.1f}s |")
    if with_attackers and summaries:
        for k, v in summaries[0]["attackers_for_scale"].items():
            L.append(f"| _{k}_ (in-memory) | — | — | — | {_f(v['s_phys'])} | {v['recovery']:.2f} | — | — | — | — |")
    if not_final:
        L += ["", "| system | not final | why |", "|---|---|---|"]
        for label, how, why in not_final:
            L.append(f"| {label} | {how} | {'; '.join(str(w) for w in why)} |")
    rows: List[str] = []

    def poi_rows(label, pois):
        for p, m in pois.items():
            sens = ("PASS" if m["sensitivity_pass"] else "FAIL") + ("" if m["truth_varies"] else " (fixed)")
            cal = "PASS" if (m["calibration_pass"] and m["bias_pass"]) else "FAIL"
            rows.append(f"| {label} | {p} | {m['n']} | {_f(m['slope'], '{:+.2f}')} ± {_f(m['slope_err'])} "
                        f"| {_f(m['bias'], '{:+.2f}')} | {_f(m['pull_width'])} | {_f(m['precision'])} "
                        f"| {sens} | {cal} | {_f(m['score'])} |")

    for s in summaries:
        poi_rows(s["label"], s["pois"])
    if with_attackers and summaries:
        for k, v in summaries[0]["attackers_for_scale"].items():
            poi_rows(f"_{k}_", v["pois"])
    from .kinds import scoreboard_tables
    kind_rows = scoreboard_tables(summaries)
    if rows or not kind_rows:
        L += ["", "| system | POI | n | slope ± err | bias | pull width | precision | sensitivity | calibration | score |",
              "|---|---|---|---|---|---|---|---|---|---|"] + rows
    nuis = [(s["label"], s.get("nuisances", {})) for s in summaries if s.get("nuisances")]
    if nuis:
        L += ["", "| system | calibration nuisance | n | slope ± err | bias | pull width | truth |",
              "|---|---|---|---|---|---|---|"]
        for label, nn in nuis:
            for n, m in nn.items():
                L.append(f"| {label} | {n} | {m['n']} | {_f(m['slope'], '{:+.2f}')} ± {_f(m['slope_err'])} "
                         f"| {_f(m['bias'], '{:+.2f}')} | {_f(m['pull_width'])} "
                         f"| {'varies' if m['truth_varies'] else 'fixed'} |")
    L += kind_rows
    fbs = [(s["label"], (s.get("v3") or {}).get("fallbacks")) for s in summaries
           if isinstance((s.get("v3") or {}).get("fallbacks"), dict)]
    if fbs:
        L += ["", "| system | fallback taken (counted) | datasets | which |", "|---|---|---|---|"]
        for label, fb in fbs:
            if not fb:
                L.append(f"| {label} | none | 0 | — |")
            for k, e in fb.items():
                ss = list(e.get("seeds") or [])
                L.append(f"| {label} | {k} | {e.get('n', len(ss))} | "
                         f"{', '.join(ss[:8]) or '—'}{', ...' if len(ss) > 8 else ''} |")
    return "\n".join(L) + "\n"
