"""The closure comparisons of a corrections run (the corrections checked together) and the Z-peak one.

The common core (Comparison, compare, fallback losses, the candidate cache): a comparison folds the reported
corrections into the simulation's candidates of an evaluator-fixed selection (Task.peak_spec: the evaluator's, not
the participant's) and compares the result with the data's per category, shape only (the prediction normalised to
the data's count there).  Per bin the uncertainty is the prediction's (the reported uncertainty model, propagated
source by source) and both samples' statistics; the loss is the per-point log loss, mean over a category's bins,
then over categories (over a stream's categories, then over streams, where a comparison has several).  A correction
not reported, or whose measurement fails a check, enters at the request's no-answer forecast (the spec's
"fallback": nominal, the fallback width its band), so the loss is continuous in what is reported.  The task names
its comparison (CorrectionsTask.closure_comparison, comparison.py); every comparison is scored alike (Delta_peak,
scoring.py).

The Z-peak comparison (lep_calib): the reported corrections applied to the simulation's Z -> ll candidates -- scale
and resolutions forward folded (refit.py's model), each leg's identification scale factor as a weight -- against the
data's dilepton mass, per category in the pair's pT and rapidity.  The candidates -- per event the opposite-charge
pair of identified objects closest to m_Z -- depend on the dataset only and are cached per dataset.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...budget import alarm_held
from . import measurement as M
from . import refit as R

M_Z = 91.1876
N_REPLICAS_PEAK = 1
PEAK_RULE = ("Z-peak closure (log loss of the corrected simulation's peak against the data's; an event's "
             "copies one unit, in the data's variance floor and the category threshold too)")
RULE, LABEL, NOUN, PENDING = PEAK_RULE, "Z-peak closure", "Z-peak", "Z peak"
RESOLUTIONS = {"res_pt": "pt", "res_eta": "eta", "res_phi": "phi"}
MIN_UNITS = 50



def _file_candidates(job) -> Tuple[str, Optional[dict]]:
    """One file's candidates (a pool unit): {pt1, eta1, phi1, charge1, pt2, ..., genWeight}, per event the
    opposite-charge pair of selected objects in the window closest to m_Z."""
    import awkward as ak
    import uproot
    path, sample, coll, branch, thr, spec = job
    names = [f"{coll}_{k}" for k in ("pt", "eta", "phi", "charge")] + [branch] + (["genWeight"] if sample == "sim"
                                                                                  else [])
    with uproot.open(path) as f:
        a = f["Events"].arrays(names, library="ak")
    v = a[branch]
    passes = (v >= thr) if thr is not None else ak.values_astype(v, bool)
    lep = ak.zip({"pt": ak.values_astype(a[f"{coll}_pt"], np.float64),
                  "eta": ak.values_astype(a[f"{coll}_eta"], np.float64),
                  "phi": ak.values_astype(a[f"{coll}_phi"], np.float64), "q": a[f"{coll}_charge"]})
    lep = lep[passes & (lep.pt > spec["raw_pt_min"]) & (abs(lep.eta) < spec["abs_eta_max"] + 0.1)]
    pairs = ak.combinations(lep, 2, fields=["a", "b"])
    m = np.sqrt(np.maximum(2.0 * pairs.a.pt * pairs.b.pt * (np.cosh(pairs.a.eta - pairs.b.eta)
                                                          - np.cos(pairs.a.phi - pairs.b.phi)), 0.0))
    ok = (pairs.a.q * pairs.b.q < 0) & (m > spec["window"][0]) & (m < spec["window"][1])
    pairs, m = pairs[ok], m[ok]
    has = ak.num(m) > 0
    best = pairs[has][ak.argmin(abs(m[has] - M_Z), axis=1, keepdims=True)]
    out = {f"{k}{j}": ak.to_numpy(ak.flatten(best[r][k])).astype(np.float64)
           for j, r in ((1, "a"), (2, "b")) for k in ("pt", "eta", "phi")}
    out.update({f"charge{j}": ak.to_numpy(ak.flatten(best[r]["q"])).astype(np.int8) for j, r in ((1, "a"), (2, "b"))})
    if sample == "sim":
        out["genWeight"] = ak.to_numpy(a["genWeight"])[ak.to_numpy(has)].astype(np.float64)
    return path, out


def candidates(dataset_dir, spec: dict, flavour: str, cache_dir: Optional[Path] = None) -> Dict[str, dict]:
    """{"data": arrays, "sim": arrays} of one flavour's candidates (cached); sim "weight" = genWeight x cross section
    x luminosity / sum of generator weights."""
    dataset_dir = Path(dataset_dir)
    branch, thr = M.selection_branch(spec["selection"], flavour)
    coll = M.COLLECTIONS[flavour]
    key = hashlib.sha256(json.dumps({"f": flavour, "sel": spec["selection"], "w": spec["window"],
                                     "pt": spec["raw_pt_min"], "eta": spec["abs_eta_max"]},
                                    sort_keys=True).encode()).hexdigest()[:12]
    cache = Path(cache_dir) / f"peak_{flavour}_{key}.npz" if cache_dir else None
    got = load_cached(cache)
    if got is not None:
        return got
    xs = json.loads((dataset_dir / "mc" / "cross_sections.json").read_text())["cross_sections_pb"]
    lumi_fb = json.loads((dataset_dir / "lumi.json").read_text())["lumi_fb"]
    jobs, scale = [], {}
    for p in sorted((dataset_dir / "data").glob("*.root")):
        jobs.append((str(p), "data", coll, branch, thr, spec))
    for proc in sorted(xs):
        meta = json.loads((dataset_dir / "mc" / proc / "meta.json").read_text())
        for p in sorted((dataset_dir / "mc" / proc).glob("*.root")):
            jobs.append((str(p), "sim", coll, branch, thr, spec))
            scale[str(p)] = xs[proc] * 1000.0 * lumi_fb / meta["genEventSumw"]
    parts: Dict[str, List[dict]] = {"data": [], "sim": []}
    got = dict(M.pool_imap(_file_candidates, jobs, M.READ_JOBS, "forkserver"))
    for path, smp, *_ in jobs:
        o = got[path]
        if smp == "sim":
            o["weight"] = o.pop("genWeight") * scale[path]
        parts[smp].append(o)
    out = {smp: {k: np.concatenate([o[k] for o in ps]) for k in ps[0]} for smp, ps in parts.items() if ps}
    save_cached(cache, out)
    return out


def load_cached(cache: Optional[Path]) -> Optional[Dict[str, dict]]:
    """A dataset's candidates {"data": arrays, "sim": arrays} from their cache file, or None."""
    if cache is None or not cache.is_file():
        return None
    with np.load(cache, allow_pickle=False) as z:
        out: Dict[str, dict] = {"data": {}, "sim": {}}
        for k in z.files:
            smp, f = k.split(".", 1)
            out[smp][f] = z[k]
    return out


def save_cached(cache: Optional[Path], out: Dict[str, dict]) -> None:
    """Write candidates to their cache file (atomically; a post-stage budget never leaves a part-written one)."""
    if cache is None:
        return
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(f".tmp{np.random.default_rng().integers(1 << 30)}.npz")
    with alarm_held():
        np.savez(tmp, **{f"{smp}.{k}": v for smp, o in out.items() for k, v in o.items()})
        tmp.replace(cache)


def _objects(c: dict) -> Dict[str, dict]:
    return {"first": {"pt": c["pt1"], "eta": c["eta1"], "phi": c["phi1"], "charge": c["charge1"]},
            "second": {"pt": c["pt2"], "eta": c["eta2"], "phi": c["phi2"], "charge": c["charge2"]}}



class Comparison:
    """A comparison's prediction, as compare() reads it.  A subclass sets, besides what __init__ does:
      used        the corrections it folds, in order (their uncertainty components: _components),
      y, kap      the data's counts per (category, bin) and its variance per count per category (1 without copies),
      nc, nb      categories and bins per category; streams [{"name", "variable", "edges", "cats": slice}] (one
                  histogram variable per stream, nb bins each),
      sim_unit    each simulated candidate's statistical unit, z the smearing replicas (one array of normals each),
      _cached     after the nominal template: (effects, [(candidates, weight multipliers, cells), ...]),
    and defines template(variation, params, shift, cset) -> (nc, nb), nominal_weights() (the simulated candidates'
    weights in the nominal template) and _point_objects() (objects a correction is evaluated at, for its components'
    consistency)."""
    min_units = MIN_UNITS

    def __init__(self, spec: dict, cset: dict, result: Any, request: dict, flavour: str, seed: int,
                 fallback: Optional[Dict[str, Tuple[float, Any]]] = None):
        from . import outputs as C
        self.C, self.spec, self.cset, self.result, self.flavour, self.seed = C, spec, cset, result, flavour, seed
        self.fallback = dict(fallback or {})
        self.band_only: set = set()
        self.entries = C.requested(request)

    def _fallback_value(self, name: str, o: dict, shift: Optional[Dict[str, float]]) -> np.ndarray:
        """A fallback correction at objects `o`: its forecast's nominal moved by `shift` widths, in the output space
        (a resolution: the signed square root of the extra variance)."""
        nom, width = self.fallback[name]
        v = float(nom) + (shift or {}).get(name, 0.0) * float(width)
        return np.full(len(o["pt"]), math.copysign(math.sqrt(abs(v)), v) if name in RESOLUTIONS else v)

    def _value(self, name: str, o: dict, variation: Optional[str], params: Optional[dict] = None,
               shift: Optional[Dict[str, float]] = None, cset: Optional[dict] = None) -> np.ndarray:
        """A reported correction at objects `o` ({pt, eta, phi, charge}; clamped to its domain), at a variation
        (nominal where it has none) or with formula parameters replaced; None if not reported or not evaluating.  A
        fallback correction is _fallback_value.  `cset`: a correction set in place of the reported one."""
        if name in self.fallback:
            return self._fallback_value(name, o, shift)
        C = self.C
        corr, _ = C.find(self.cset if cset is None else cset, name)
        if corr is None:
            return None
        dom = (self.entries.get(name) or {}).get("domain", {}).get(self.flavour)
        if dom is None:
            return None
        eta_in = float(dom["abs_eta_max"]) * (1 - 1e-6)
        pts = {"pt": np.maximum(o["pt"], float(dom["pt_min"]) * (1 + 1e-6) + 1e-6),
               "eta": np.clip(o["eta"], -eta_in, eta_in), "phi": o["phi"], "charge": o["charge"].astype(np.int64)}
        has_var = "variation" in C._input_names(corr)
        var = "nominal" if has_var else None
        if variation is not None and has_var and variation in C.variation_keys(corr.get("data")) \
                and (name not in self.band_only or variation in ("up", "down")):
            var = variation
        work = corr
        if params:
            import copy
            work = copy.deepcopy(corr)
            slots = C.param_slots(work["data"], nominal=has_var)
            for i, v in params.items():
                node, k = slots[i]
                node["parameters"][k] = float(v)
        try:
            return C._run(C.evaluator(work), work, pts, self.flavour, var)
        except C.CorrectionError:
            return None

    def points(self, name: str, n: int = 2000) -> Dict[str, np.ndarray]:
        """A sample of the simulated objects (_point_objects), clamped to a correction's domain (its consistency)."""
        dom = (self.entries.get(name) or {}).get("domain", {}).get(self.flavour) or {"pt_min": 0.0, "abs_eta_max": 1e9}
        o = self._point_objects()
        k = np.linspace(0, len(o["pt"]) - 1, min(n, len(o["pt"]))).astype(np.int64)
        eta_in = float(dom["abs_eta_max"]) * (1 - 1e-6)
        return {"pt": np.maximum(o["pt"][k], float(dom["pt_min"]) * (1 + 1e-6) + 1e-6),
                "eta": np.clip(o["eta"][k], -eta_in, eta_in), "phi": o["phi"][k],
                "charge": o["charge"][k].astype(np.int64)}

    def normalised(self, t: np.ndarray) -> np.ndarray:
        tot = t.sum(axis=1, keepdims=True)
        return t * np.where(tot > 0, self.y.sum(axis=1, keepdims=True) / np.where(tot > 0, tot, 1.0), 0.0)

    def data_variance(self, dc: np.ndarray, unit: np.ndarray) -> np.ndarray:
        """The data's variance per count per category from each data candidate's cell `dc` and statistical unit:
        1, more where a unit holds more than one count in a cell (an event held more than once)."""
        inc = dc >= 0
        kap = np.ones(self.nc)
        if not np.array_equal(unit, np.arange(len(unit))):
            y2 = R._unit_squares(dc[inc], np.ones(int(inc.sum())), unit[inc], self.nc * self.nb)
            yw = self.y.sum(axis=1)
            kap = np.where(yw > 0, y2.reshape(self.nc, self.nb).sum(axis=1) / np.where(yw > 0, yw, 1.0), 1.0)
        return kap


class _Peak(Comparison):
    """The corrected simulation's peak per category at a variation of the reported uncertainty model: candidates
    folded with scale and resolutions (absent ones nominal), both legs' identification scale factor as a weight;
    ``fallback`` ({name: (nominal, width)}) at the no-answer forecast."""

    def __init__(self, spec: dict, cand: Dict[str, dict], cset: dict, result: Any, request: dict, flavour: str,
                 seed: int, fallback: Optional[Dict[str, Tuple[float, float]]] = None):
        super().__init__(spec, cset, result, request, flavour, seed, fallback)
        self.used = ("scale", *RESOLUTIONS, spec["id"])
        self.sim, self.data = _objects(cand["sim"]), _objects(cand["data"])
        self.w = cand["sim"]["weight"]
        self.n = len(self.w)
        self.hist = {"variable": "mass", "edges": np.asarray(spec["mass"], dtype=float),
                     "categories": [(v, np.asarray(e, dtype=float)) for v, e in spec["categories"].items()]}
        self.sel = {"pt_min": spec["pt_min"], "abs_eta_max": spec["abs_eta_max"]}
        self.nb = len(self.hist["edges"]) - 1
        self.nc = M._n_bins(self.hist["categories"])
        self.streams = [{"name": None, "variable": self.hist["variable"], "edges": self.hist["edges"],
                         "cats": slice(None)}]
        x = _variables(self.data)
        dc = M.histogram_cells(self.hist, self.sel, x, len(x["mass"]))
        self.y = np.bincount(dc[dc >= 0], minlength=self.nc * self.nb).astype(float).reshape(self.nc, self.nb)
        self.kap = self.data_variance(dc, R.copies(self.data))
        self.sim_unit = R.copies(self.sim)
        self.z = [sign * np.random.default_rng([seed, 17, r]).standard_normal((6, self.n))
                  for r in range(N_REPLICAS_PEAK) for sign in (1.0, -1.0)]

    def _point_objects(self) -> dict:
        return self.sim["first"]

    def template(self, variation: Optional[str] = None, params: Optional[Dict[str, dict]] = None,
                 shift: Optional[Dict[str, float]] = None, cset: Optional[dict] = None) -> np.ndarray:
        """The prediction per (category, bin) at a variation, with formula parameters replaced ({name: {slot: value}}),
        fallback corrections shifted ({name: widths}) or another correction set.  Thread-safe once the nominal one is
        made (_cells)."""
        params = params or {}
        eff = {}
        for role in ("first", "second"):
            e = {"scale": np.zeros(self.n), "pt": np.zeros(self.n), "eta": np.zeros(self.n), "phi": np.zeros(self.n)}
            for name, key in (("scale", "scale"), *RESOLUTIONS.items()):
                v = self._value(name, self.sim[role], variation, params.get(name), shift, cset)
                if v is not None:
                    e[key] = v if key == "scale" else v * np.abs(v)
            eff[role] = e
        sf = np.ones(self.n)
        for role in ("first", "second"):
            v = self._value(self.spec["id"], self.sim[role], variation, params.get(self.spec["id"]), shift, cset)
            if v is not None:
                sf = sf * v
        w = self.w * sf
        t = np.zeros(self.nc * self.nb)
        for idx, mult, c in self._cells(eff):
            ok = c >= 0
            t += np.bincount(c[ok], (w[idx] * mult)[ok], len(t))
        return (t / len(self.z)).reshape(self.nc, self.nb)

    def nominal_weights(self) -> np.ndarray:
        """The candidates' weights in the nominal template: genWeight-based times both legs' identification factor."""
        sf = np.ones(self.n)
        for role in ("first", "second"):
            v = self._value(self.spec["id"], self.sim[role], None)
            if v is not None:
                sf = sf * v
        return self.w * sf

    def _cells(self, eff) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """(candidates, weight multipliers, cells) per replica and negative-variance term; the first call's are kept,
        so a variation that only reweights (the identification's) does not fold again."""
        cached = self.__dict__.get("_cached")
        if cached is not None and all(np.array_equal(eff[r][c], cached[0][r][c]) for r in eff for c in eff[r]):
            return cached[1]
        neg = np.zeros(self.n, dtype=bool)
        for role in ("first", "second"):
            for c in ("pt", "eta", "phi"):
                neg |= eff[role][c] < 0
        out = []
        for z in self.z:
            for idx, k, mult in R.negative_groups(self.n, neg, np.ones(self.n)):
                if not len(idx):
                    continue
                o = _fold(self.sim, eff, z, idx, k)
                out.append((idx, mult, M.histogram_cells(self.hist, self.sel, _variables(o), len(idx))))
        if cached is None:
            self._cached = (eff, out)
        return out


def _fold(sim, eff, z, idx, k: float) -> Dict[str, dict]:
    out = {}
    for j, role in enumerate(("first", "second")):
        o, e = sim[role], eff[role]
        wd = R.smearing_widths(e, idx, k)
        out[role] = {"pt": o["pt"][idx] * (1.0 + e["scale"][idx]) * (1.0 + wd["pt"] * z[3 * j][idx]),
                     "eta": o["eta"][idx] + wd["eta"] * z[3 * j + 1][idx],
                     "phi": o["phi"][idx] + wd["phi"] * z[3 * j + 2][idx], "charge": o["charge"][idx]}
    return out


def _variables(o: Dict[str, dict]) -> Dict[str, np.ndarray]:
    """The pair variables the peak's selection and categories use, and the pair's |rapidity| (massless objects)."""
    x = M.pair_variables(o["first"], o["second"], {"mass", "pt_pair"})
    a, b = o["first"], o["second"]
    e = a["pt"] * np.cosh(a["eta"]) + b["pt"] * np.cosh(b["eta"])
    pz = a["pt"] * np.sinh(a["eta"]) + b["pt"] * np.sinh(b["eta"])
    with np.errstate(divide="ignore", invalid="ignore"):
        x["abs_y_pair"] = np.abs(0.5 * np.log(np.maximum(e + pz, 1e-300) / np.maximum(e - pz, 1e-300)))
    return x



def _components(pk: Comparison) -> List[Tuple[str, Optional[str], Optional[Dict[str, dict]]]]:
    """The reported uncertainty model's independent components, as (label, variation stem, parameter shifts):
    - each named source, varied in every correction having it (fully correlated);
    - a correction's band without sources;
    - the eigen-directions of a correction's summed parameter covariances (same variance in the linear response as
      each on its own; a joint covariance by its block of the correction's own parameters, cross-correlations not
      propagated);
    - a fallback correction's band.
    Components are used where they agree with the band within the contract's tolerance at a sample of candidates,
    else the band alone (the contract: "or only the band is read")."""
    C = pk.C
    names = [n for n in pk.used if n not in pk.fallback]
    sources, banded, comps = set(), [], []
    for name in names:
        corr, _ = C.find(pk.cset, name)
        if corr is None:
            continue
        spec, problems = C.uncertainty_spec(corr, pk.result, pk.cset)
        if problems:
            continue
        if spec["band"]:
            agree = False
            if spec["sources"] or spec["covariances"] or spec["joint"]:
                try:
                    agree = C.components(corr, pk.result, pk.points(name), pk.flavour, pk.cset,
                                         signed=name in RESOLUTIONS)["consistent"]
                except C.CorrectionError:
                    agree = False
            if not agree:
                banded.append(name)
                pk.band_only.add(name)
                continue
        sources |= set(spec["sources"])
        mats = [np.asarray(m, dtype=float) for _, m in list(spec["covariances"].items()) + list(spec["joint"])]
        if mats:
            cov = sum(mats)
            lam, vec = np.linalg.eigh((cov + cov.T) / 2.0)
            p0 = [float(n["parameters"][k]) for n, k in C.param_slots(corr["data"],
                                                                      nominal="variation" in C._input_names(corr))]
            for i in np.flatnonzero(lam > 1e-30 * max(float(lam.max()), 1e-300)):
                d = np.sqrt(lam[i]) * vec[:, i]
                comps.append((f"{name}:covariance:{i}", None,
                              {"+": {name: {j: p0[j] + d[j] for j in range(len(p0))}},
                               "-": {name: {j: p0[j] - d[j] for j in range(len(p0))}}}))
    comps += [(f"source:{s}", s, None) for s in sorted(sources)]
    comps += [(f"{n}:band", "", {"only": n}) for n in banded]
    comps += [(f"{n}:fallback", None, {"shift": n}) for n in sorted(pk.fallback)]
    return comps


def compare(pk: Comparison) -> dict:
    """A comparison (Comparison) against the data: per category and bin data counts, the normalised prediction and
    its uncertainty (components in quadrature, simulation statistics), and per category the log loss
    mean[ln s + z^2 / 2], s = hypot(prediction uncertainty, simulation's, data's sqrt(kappa max(prediction, kappa))),
    kappa the category's variance per count (1 unless events are held more than once: an event's copies are one
    unit).  The loss is the mean over categories with >= pk.min_units units of data -- with several streams, the
    mean over the streams of that mean over their categories ("per_stream").  The page's curve sums a stream's
    categories (the first stream's; "curves" each stream's where there are several): data and its error (kappa x
    counts), prediction and band (each component's shift summed over categories first: they move together), pull
    against both."""
    t = pk.template()
    nominal = pk.normalised(t)

    def ends(component):
        label, stem, shifts = component
        if stem is not None and shifts is None:
            return pk.template(f"up_{stem}"), pk.template(f"down_{stem}")
        if shifts is not None and "only" in shifts:
            return _only(pk, shifts["only"], "up"), _only(pk, shifts["only"], "down")
        if shifts is not None and "shift" in shifts:
            return pk.template(shift={shifts["shift"]: 1.0}), pk.template(shift={shifts["shift"]: -1.0})
        return pk.template(params=shifts["+"]), pk.template(params=shifts["-"])

    comps = _components(pk)
    streams = pk.streams
    var = np.zeros_like(nominal)
    inc_var = [np.zeros(nominal.shape[1]) for _ in streams]
    labels, shifts = [], []
    for (label, _, _), (hi, lo) in zip(comps, R.in_threads(ends, comps)):
        shift = 0.5 * (pk.normalised(hi) - pk.normalised(lo))
        var += shift ** 2
        for k, st in enumerate(streams):
            inc_var[k] += shift[st["cats"]].sum(axis=0) ** 2
        labels.append(label)
        shifts.append(shift)
    tot = t.sum(axis=1, keepdims=True)
    f = np.where(tot > 0, pk.y.sum(axis=1, keepdims=True) / np.where(tot > 0, tot, 1.0), 0.0)
    w2 = _template_w2(pk) * f * f
    kap = pk.kap[:, None]
    dvar = kap * np.maximum(nominal, kap)
    s = np.sqrt(var + w2 + dvar)
    z = (nominal - pk.y) / s
    used = pk.y.sum(axis=1) / pk.kap >= pk.min_units
    cell = (pk.y > 0) | (nominal > 0)
    per_cat = [float(np.mean((np.log(s[c]) + 0.5 * z[c] ** 2)[cell[c]])) if used[c] and cell[c].any() else None
               for c in range(pk.nc)]
    losses = [v for v in per_cat if v is not None]

    def curve(k):
        st = streams[k]
        sl = st["cats"]
        inc_y, inc_p, band2 = pk.y[sl].sum(axis=0), nominal[sl].sum(axis=0), inc_var[k] + w2[sl].sum(axis=0)
        return {"variable": st["variable"], "edges": [float(e) for e in st["edges"]],
                "data": inc_y.tolist(), "data_err": np.sqrt((kap[sl] * pk.y[sl]).sum(axis=0)).tolist(),
                "prediction": inc_p.tolist(), "band": np.sqrt(band2).tolist(),
                "pull": ((inc_y - inc_p) / np.sqrt(band2 + dvar[sl].sum(axis=0))).tolist()}
    if len(streams) == 1:
        loss = float(np.mean(losses)) if losses else None
    else:
        per_stream = {}
        for st in streams:
            ls = [v for v in per_cat[st["cats"]] if v is not None]
            per_stream[st["name"]] = float(np.mean(ls)) if ls else None
        got = [v for v in per_stream.values() if v is not None]
        loss = float(np.mean(got)) if got else None
    out = {"loss": loss, "per_category": per_cat,
           "n_categories": len(losses), "components": labels,
           "source_fit": _source_fit(labels, shifts, pk.y - nominal, w2 + dvar, used[:, None] & cell),
           "curve": curve(0)}
    if len(streams) > 1:
        out.update(per_stream=per_stream, curves={st["name"]: curve(k) for k, st in enumerate(streams)})
    return out


def closure(spec: dict, cand: Dict[str, dict], cset: dict, result: Any, request: dict, flavour: str,
            seed: int, fallback: Optional[Dict[str, Tuple[float, float]]] = None) -> dict:
    """The Z-peak comparison of one flavour (compare); ``fallback`` ({name: (nominal, width)}) enters at its
    forecast."""
    return compare(_Peak(spec, cand, cset, result, request, flavour, seed, fallback))


def _source_fit(labels: List[str], shifts: List[np.ndarray], resid: np.ndarray, var: np.ndarray,
                sel: np.ndarray) -> Optional[dict]:
    """Linear fit of the components as unit-Gaussian-constrained nuisances over the loss's cells (display only: the
    page's pulls): the prediction moves by theta_k x each component's one-sigma shift, cells weighted by simulation
    and data variance.  {"labels", "pull": theta-hat, "constraint": its post-fit sd, "chi2_prefit", "chi2_postfit"
    (constraint terms included), "n_cells"}; None without a component or a cell."""
    if not shifts or not sel.any():
        return None
    a = np.stack([x[sel] for x in shifts], axis=1)
    w, r = 1.0 / var[sel], resid[sel]
    cov = np.linalg.inv(a.T @ (a * w[:, None]) + np.eye(len(shifts)))
    th = cov @ (a.T @ (w * r))
    return {"labels": list(labels), "pull": [float(x) for x in th],
            "constraint": [float(x) for x in np.sqrt(np.maximum(np.diag(cov), 0.0))],
            "chi2_prefit": float(np.sum(w * r * r)),
            "chi2_postfit": float(np.sum(w * (r - a @ th) ** 2) + th @ th), "n_cells": int(sel.sum())}


def cached_fallback_loss(closure_fn, rule: str, prefix: str, spec: dict, cand: Dict[str, dict], request: dict,
                         flavour: str, seed: int, cache_dir: Optional[Path] = None) -> Optional[float]:
    """A comparison's loss with every correction at its no-answer forecast (the cap's reference): closure_fn on no
    corrections, every one in spec["fallback"]; depends on the dataset alone, cached beside its candidates under the
    comparison's rule."""
    key = hashlib.sha256(json.dumps({"spec": spec, "seed": seed, "rule": rule}, sort_keys=True,
                                    default=float).encode()).hexdigest()[:12]
    cache = Path(cache_dir) / f"{prefix}_{flavour}_{key}.json" if cache_dir else None
    if cache is not None and cache.is_file():
        return json.loads(cache.read_text())["loss"]
    loss = closure_fn(spec, cand, {"schema_version": 2, "corrections": []}, None, request, flavour, seed,
                      spec["fallback"])["loss"]
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(f".tmp{np.random.default_rng().integers(1 << 30)}")
        tmp.write_text(json.dumps({"loss": loss}))
        tmp.replace(cache)
    return loss


def fallback_loss(spec: dict, cand: Dict[str, dict], request: dict, flavour: str, seed: int,
                  cache_dir: Optional[Path] = None) -> Optional[float]:
    """The Z-peak loss with every correction at its no-answer forecast (cached_fallback_loss)."""
    return cached_fallback_loss(closure, PEAK_RULE, "peak_fallback", spec, cand, request, flavour, seed, cache_dir)


def _only(pk: Comparison, name: str, which: str) -> np.ndarray:
    """The template with one correction at its band's end and the others nominal."""
    import copy
    corr, _ = pk.C.find(pk.cset, name)
    others = [c for c in pk.cset["corrections"] if c.get("name") != name]
    return pk.template(cset={**pk.cset, "corrections": others + [_pin_variation(copy.deepcopy(corr), which)]})


def _pin_variation(corr: dict, which: str) -> dict:
    """The correction with its "nominal" variation's value replaced by the `which` ("up" / "down") one, wherever its
    data has a category on the variation input."""
    stack = [corr.get("data")]
    while stack:
        n = stack.pop()
        if isinstance(n, dict):
            if n.get("nodetype") == "category" and n.get("input") == "variation":
                items = {it.get("key"): it for it in n.get("content") or [] if isinstance(it, dict)}
                if "nominal" in items and which in items:
                    items["nominal"]["value"] = items[which]["value"]
            stack.extend(v for v in n.values() if isinstance(v, (dict, list)))
        elif isinstance(n, list):
            stack.extend(n)
    return corr


def _template_w2(pk: Comparison) -> np.ndarray:
    """The nominal template's simulation variance per (category, bin): per simulated event unit the square of its
    whole share (replicas and negative-variance terms together, weights in the nominal template)."""
    w = pk.nominal_weights()
    size = pk.nc * pk.nb
    keys, ws = [], []
    for idx, mult, c in pk._cached[1]:
        ok = c >= 0
        keys.append(pk.sim_unit[idx[ok]] * size + c[ok])
        ws.append(w[idx][ok] * mult[ok] / len(pk.z))
    k, wk = (np.concatenate(keys), np.concatenate(ws)) if keys else (np.zeros(0, dtype=np.int64), np.zeros(0))
    if not len(k):
        return np.zeros((pk.nc, pk.nb))
    order = np.argsort(k, kind="stable")
    k, wk = k[order], wk[order]
    starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    return np.bincount(k[starts] % size, np.add.reduceat(wk, starts) ** 2, size).reshape(pk.nc, pk.nb)
