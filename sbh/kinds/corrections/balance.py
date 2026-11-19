"""The jet balance closure of a corrections run (comparison.py "balance") -- EVALUATOR-ONLY
(sbh/harbor_pack.py EVALUATOR_ONLY): the reported jet scale and pT resolution applied to the simulation's jets,
against the data's pT balance in two streams, on the common core of peak.py (compare: per bin ln s + z^2 / 2, shape
only, the reported uncertainty model source by source).

- Each jet's correction is evaluated at its raw (pT, eta, phi), pT clamped to the domain's pt_min as everywhere in
  the kind, charge +1 / -1 alternately (no jet correction takes it); pT -> pT (1 + scale) (1 + sqrt(V) z), V the
  signed extra variance width x |width|, z fixed standard normals (an antithetic pair of replicas per event, seeded
  by the dataset's index), a pT folded below 0 set to 0.  A negative V is continued through zero to second order
  as the refit and the Z peak do (refit.negative_groups: 3 T(0) - 3 T(|V|) + T(2 |V|)), so the prediction is smooth
  across V = 0 and has no spurious minimum below it.  The event's jets are then ordered again by pT and selected.
- Z(mumu)+jet: two opposite-charge muons of the evaluator's identification and isolation (Muon_pfRelIso04_all; no
  tracker isolation) closest to m_Z in the window, a single-muon path fired; the leading jet back to back with the
  Z, the second jet's pT over pT(Z) below the activity cut; the balance pT(j1) / pT(Z) in bins of pT(Z) x the
  leading jet's eta region.
- Dijet: the two leading jets back to back, the third's pT over pT_avg below its cut; one category per pT_avg bin
  and probe eta region, each bin read through one single-jet path, the one whose threshold the bin's lower edge is
  on the plateau of (its bit post-prescale in the data, unprescaled in the simulation: within a category the
  prescale is a constant, which the shape-only comparison drops); a tag jet in the central region, the other the
  probe, both orderings when both are central (one event, one statistical unit); the asymmetry
  (pT_probe - pT_tag) / (pT_probe + pT_tag).
- Jets are those with |eta| below the spec's maximum and Delta R >= its cone from every muon and electron above its
  pT (the evaluation population's lepton cleaning); an event keeps its K_JETS leading ones.  The selection's jet
  pT floor applies to the folded pT (an analysis' jets after the correction).
- The loss: the mean over the two streams of the mean over their categories with enough data (peak.MIN_UNITS).

The candidates (per event the jets, the Z, the jet paths' bits, the weight) depend on the dataset only and are
cached per dataset.  The spec (Task.peak_spec("jet", request), built by spec()) holds the selection, the binning and
the forecasts.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import measurement as M
from . import peak as P
from . import refit as R

M_Z = P.M_Z
K_JETS = 4
N_REPLICAS = 1
RULE = ("jet balance closure (log loss of the corrected simulation's Z+jet balance and dijet asymmetry against the "
        "data's, shape only per category; the mean over the two streams)")
LABEL, NOUN, PENDING = "balance closure", "balance", "balance"
RESOLUTIONS = {"res_pt": "pt"}



def _dphi(a, b):
    return np.abs(np.remainder(a - b + np.pi, 2.0 * np.pi) - np.pi)


def _bins(v: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Each value's bin of `edges` (numpy.histogram's convention, measurement.bin_index), -1 outside or not finite."""
    v = np.asarray(v, dtype=float)
    return M.bin_index([("v", edges)], {"v": np.where(np.isfinite(v), v, -np.inf)}, len(v))


def _file_candidates(job) -> Tuple[str, dict]:
    """One file's candidates (a pool unit): per preselected event the K_JETS leading clean jets (pt, eta, phi; 0
    padded), the Z (has_z, z_pt, z_phi), the jet paths' bits and, in the simulation, genWeight."""
    import awkward as ak
    import uproot
    path, sample, spec = job
    mu, cl, paths = spec["muon"], spec["clean"], [p for _, p in spec["dijet"]["paths"]]
    mu_f = [f"Muon_{k}" for k in ("pt", "eta", "phi", "charge", "dxy", "dz", "pfRelIso04_all", mu["id"])]
    names = (["Jet_pt", "Jet_eta", "Jet_phi"] + mu_f + ["Electron_pt", "Electron_eta", "Electron_phi"]
             + list(spec["muon_paths"]) + paths + (["genWeight"] if sample == "sim" else []))
    with uproot.open(path) as f:
        a = f["Events"].arrays(names, library="ak")
    n = len(a)
    f64 = (lambda x: ak.values_astype(x, np.float64))
    leps = ak.concatenate([ak.zip({"eta": f64(a[f"{c}_eta"]), "phi": f64(a[f"{c}_phi"])})[a[f"{c}_pt"]
                                                                                       > cl["lepton_pt_min"]]
                           for c in ("Muon", "Electron")], axis=1)
    jets = ak.zip({"pt": f64(a["Jet_pt"]), "eta": f64(a["Jet_eta"]), "phi": f64(a["Jet_phi"])})
    jets = jets[abs(jets.eta) < spec["jet_abs_eta_max"]]
    pairs = ak.cartesian({"j": jets, "l": leps}, axis=1, nested=True)
    dr2 = (pairs.j.eta - pairs.l.eta) ** 2 + _dphi(pairs.j.phi, pairs.l.phi) ** 2
    clean = ak.fill_none(ak.min(dr2, axis=2), np.inf) >= cl["dr"] ** 2
    jets = jets[clean]
    jets = jets[ak.argsort(jets.pt, axis=1, ascending=False, stable=True)]
    jpt, jeta, jphi = (ak.to_numpy(ak.fill_none(ak.pad_none(jets[k], K_JETS, axis=1, clip=True), 0.0))
                       .astype(np.float64).reshape(n, K_JETS) for k in ("pt", "eta", "phi"))
    m = ak.zip({"pt": f64(a["Muon_pt"]), "eta": f64(a["Muon_eta"]), "phi": f64(a["Muon_phi"]),
                "q": a["Muon_charge"]})
    ok = ((a["Muon_pt"] > mu["pt_min"]) & (abs(a["Muon_eta"]) < mu["abs_eta_max"])
          & ak.values_astype(a[f"Muon_{mu['id']}"], bool) & (a["Muon_pfRelIso04_all"] < mu["iso_max"])
          & (abs(a["Muon_dxy"]) < mu["dxy_max"]) & (abs(a["Muon_dz"]) < mu["dz_max"]))
    pr = ak.combinations(m[ok], 2, fields=["a", "b"])
    mass = np.sqrt(np.maximum(2.0 * pr.a.pt * pr.b.pt * (np.cosh(pr.a.eta - pr.b.eta) - np.cos(pr.a.phi - pr.b.phi)),
                              0.0))
    good = (pr.a.q * pr.b.q < 0) & (mass > spec["mass_window"][0]) & (mass < spec["mass_window"][1]) \
        & (np.maximum(pr.a.pt, pr.b.pt) > mu["lead_pt_min"])
    pr, mass = pr[good], mass[good]
    best = ak.firsts(pr[ak.argmin(abs(mass - M_Z), axis=1, keepdims=True)])
    px = ak.fill_none(best.a.pt * np.cos(best.a.phi) + best.b.pt * np.cos(best.b.phi), 0.0)
    py = ak.fill_none(best.a.pt * np.sin(best.a.phi) + best.b.pt * np.sin(best.b.phi), 0.0)
    trig = np.zeros(n, dtype=bool)
    for p in spec["muon_paths"]:
        trig |= ak.to_numpy(a[p]).astype(bool)
    has_z = ak.to_numpy(~ak.is_none(best)) & trig
    z_pt, z_phi = np.hypot(ak.to_numpy(px), ak.to_numpy(py)), np.arctan2(ak.to_numpy(py), ak.to_numpy(px))
    bits = np.zeros(n, dtype=np.int64)
    for k, p in enumerate(paths):
        bits |= ak.to_numpy(a[p]).astype(np.int64) << k
    njet = (jpt > 0).sum(axis=1)
    reach = spec["preselect"] * spec["dijet"]["pt_avg"][0]
    keep = (has_z & (njet >= 1)) | ((njet >= 2) & (bits != 0) & (0.5 * (jpt[:, 0] + jpt[:, 1]) >= reach))
    out = {"jet_pt": jpt[keep], "jet_eta": jeta[keep], "jet_phi": jphi[keep], "has_z": has_z[keep],
           "z_pt": np.where(has_z, z_pt, 0.0)[keep], "z_phi": np.where(has_z, z_phi, 0.0)[keep], "bits": bits[keep]}
    if sample == "sim":
        out["genWeight"] = ak.to_numpy(a["genWeight"]).astype(np.float64)[keep]
    return path, out


def selection_key(spec: dict) -> str:
    """The candidates' cache key: the spec without what does not select them (the forecasts, the corrections
    folded)."""
    sel = {k: v for k, v in spec.items() if k not in ("fallback", "used")}
    return hashlib.sha256(json.dumps(sel, sort_keys=True, default=float).encode()).hexdigest()[:12]


def candidates(dataset_dir, spec: dict, flavour: str, cache_dir: Optional[Path] = None,
               jobs: int = M.READ_JOBS) -> Dict[str, dict]:
    """{"data": arrays, "sim": arrays} of the jet candidates (cached); sim "weight" = genWeight x cross section x
    luminosity / sum of generator weights (its scale drops out per category)."""
    dataset_dir = Path(dataset_dir)
    cache = Path(cache_dir) / f"balance_{flavour}_{selection_key(spec)}.npz" if cache_dir else None
    got = P.load_cached(cache)
    if got is not None:
        return got
    xs = json.loads((dataset_dir / "mc" / "cross_sections.json").read_text())["cross_sections_pb"]
    lumi_fb = json.loads((dataset_dir / "lumi.json").read_text())["lumi_fb"]
    work, scale = [], {}
    for p in sorted((dataset_dir / "data").glob("*.root")):
        work.append((str(p), "data", spec))
    for proc in sorted(xs):
        meta = json.loads((dataset_dir / "mc" / proc / "meta.json").read_text())
        for p in sorted((dataset_dir / "mc" / proc).glob("*.root")):
            work.append((str(p), "sim", spec))
            scale[str(p)] = xs[proc] * 1000.0 * lumi_fb / meta["genEventSumw"]
    parts: Dict[str, List[dict]] = {"data": [], "sim": []}
    got = dict(M.pool_imap(_file_candidates, work, jobs, "forkserver"))
    for path, smp, _ in work:
        o = got[path]
        if smp == "sim":
            o["weight"] = o.pop("genWeight") * scale[path]
        parts[smp].append(o)
    out = {smp: {k: np.concatenate([o[k] for o in ps]) for k in ps[0]} for smp, ps in parts.items() if ps}
    P.save_cached(cache, out)
    return out



class _Balance(P.Comparison):
    """The corrected simulation's balance per category at a variation of the reported uncertainty model: every
    event's jets folded with the scale and pT resolution (absent ones nominal), then selected; ``fallback``
    ({name: (nominal, width or {region: width})}) at the no-answer forecast, a region's width at its jets."""

    def __init__(self, spec: dict, cand: Dict[str, dict], cset: dict, result: Any, request: dict, flavour: str,
                 seed: int, fallback: Optional[Dict[str, Tuple[float, Any]]] = None):
        super().__init__(spec, cset, result, request, flavour, seed, fallback)
        self.used = tuple(spec.get("used") or ("scale", "res_pt"))
        self.sim, self.data = cand["sim"], cand["data"]
        self.w = np.asarray(self.sim["weight"], dtype=float)
        self.n = len(self.w)
        zj, dj = spec["zjet"], spec["dijet"]
        self.regions = np.asarray(spec["regions"], dtype=float)
        self.nr = len(self.regions) - 1
        e_bal, e_asy = np.asarray(zj["balance"], dtype=float), np.asarray(dj["asymmetry"], dtype=float)
        if len(e_bal) != len(e_asy):
            raise ValueError("balance: the two streams' histograms need the same number of bins")
        self.nb = len(e_bal) - 1
        self.nz = (len(zj["pt_z"]) - 1) * self.nr
        self.nc = self.nz + (len(dj["pt_avg"]) - 1) * self.nr
        self.streams = [{"name": "zjet", "variable": "balance", "edges": e_bal, "cats": slice(0, self.nz)},
                        {"name": "dijet", "variable": "asymmetry", "edges": e_asy, "cats": slice(self.nz, self.nc)}]
        self.edges = {"pt_z": np.asarray(zj["pt_z"], dtype=float), "pt_avg": np.asarray(dj["pt_avg"], dtype=float),
                      "balance": e_bal, "asymmetry": e_asy}
        self.valid = self.sim["jet_pt"] > 0
        flat = np.flatnonzero(self.valid.ravel())
        self.flat = flat
        self.jets = {"pt": self.sim["jet_pt"].ravel()[flat], "eta": self.sim["jet_eta"].ravel()[flat],
                     "phi": self.sim["jet_phi"].ravel()[flat], "charge": np.where(flat % 2 == 0, 1, -1)}
        rows, dc = self._entries(self.data["jet_pt"], self.data, np.arange(len(self.data["jet_pt"])))
        self.y = np.bincount(dc, minlength=self.nc * self.nb).astype(float).reshape(self.nc, self.nb)
        self.kap = self.data_variance(dc, rows)
        self.sim_unit = np.arange(self.n, dtype=np.int64)
        self.z = [sign * np.random.default_rng([seed, 23, r]).standard_normal((self.n, K_JETS))
                  for r in range(N_REPLICAS) for sign in (1.0, -1.0)]

    def _point_objects(self) -> dict:
        return self.jets

    def _region(self, eta) -> np.ndarray:
        return np.clip(np.searchsorted(self.regions, np.abs(eta), side="right") - 1, 0, self.nr - 1)

    def _fallback_value(self, name: str, o: dict, shift: Optional[Dict[str, float]]) -> np.ndarray:
        nom, width = self.fallback[name]
        if not isinstance(width, dict):
            return super()._fallback_value(name, o, shift)
        w = np.asarray([float(width[r]) for r in self.spec["region_names"]])[self._region(o["eta"])]
        v = float(nom) + (shift or {}).get(name, 0.0) * w
        return np.sign(v) * np.sqrt(np.abs(v)) if name in RESOLUTIONS else v

    def _grid(self, v: np.ndarray) -> np.ndarray:
        """Values at the flattened jets back on the (event, jet) grid, 0 at the padding."""
        g = np.zeros(self.n * K_JETS)
        g[self.flat] = v
        return g.reshape(self.n, K_JETS)

    def template(self, variation: Optional[str] = None, params: Optional[Dict[str, dict]] = None,
                 shift: Optional[Dict[str, float]] = None, cset: Optional[dict] = None) -> np.ndarray:
        """The prediction per (category, bin) at a variation, with formula parameters replaced, fallback corrections
        shifted or another correction set (peak._Peak.template's arguments).  Thread-safe once the nominal one is
        made (_cells)."""
        params = params or {}
        eff = {"scale": np.zeros((self.n, K_JETS)), "pt": np.zeros((self.n, K_JETS))}
        for name, key in (("scale", "scale"), *RESOLUTIONS.items()):
            if name not in self.used:
                continue
            v = self._value(name, self.jets, variation, params.get(name), shift, cset)
            if v is not None:
                eff[key] = self._grid(v if key == "scale" else v * np.abs(v))
        t = np.zeros(self.nc * self.nb)
        for idx, mult, c in self._cells(eff):
            t += np.bincount(c, self.w[idx] * mult, len(t))
        return (t / len(self.z)).reshape(self.nc, self.nb)

    def nominal_weights(self) -> np.ndarray:
        return self.w

    def _cells(self, eff) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """(events, weight multipliers, cells) per replica and negative-variance term, one per selected entry; the
        first call's are kept."""
        cached = self.__dict__.get("_cached")
        if cached is not None and all(np.array_equal(eff[k], cached[0][k]) for k in eff):
            return cached[1]
        neg = (eff["pt"] < 0).any(axis=1)
        zero = np.zeros_like(eff["pt"])
        out = []
        for z in self.z:
            for idx, k, mult in R.negative_groups(self.n, neg, np.ones(self.n)):
                if not len(idx):
                    continue
                wd = R.smearing_widths({"pt": eff["pt"], "eta": zero, "phi": zero}, idx, k)["pt"]
                pt = np.maximum(self.sim["jet_pt"][idx] * (1.0 + eff["scale"][idx]) * (1.0 + wd * z[idx]), 0.0)
                rows, c = self._entries(pt, self.sim, idx)
                out.append((idx[rows], mult[rows], c))
        if cached is None:
            self._cached = (eff, out)
        return out

    def _entries(self, pt: np.ndarray, ev: dict, idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(rows of `idx`, cells) of the selected entries of events ev[idx] whose jets have pT `pt` (rows x K_JETS):
        the jets ordered by pT again, then the Z+jet and dijet selections."""
        sp, zj, dj = self.spec, self.spec["zjet"], self.spec["dijet"]
        order = np.argsort(-pt, axis=1, kind="stable")
        pt = np.take_along_axis(pt, order, axis=1)
        eta = np.take_along_axis(ev["jet_eta"][idx], order, axis=1)
        phi = np.take_along_axis(ev["jet_phi"][idx], order, axis=1)
        live = pt > sp["jet_pt_min"]
        p1, p2, p3 = (np.where(live[:, j], pt[:, j], 0.0) for j in range(3))
        nb = self.nb
        rows_out, cells_out = [], []
        zpt, zphi = ev["z_pt"][idx], ev["z_phi"][idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            ok = ev["has_z"][idx] & live[:, 0] & (_dphi(phi[:, 0], zphi) > zj["dphi_min"]) \
                & (p2 < zj["alpha_max"] * zpt)
            bal = np.where(zpt > 0, p1 / np.where(zpt > 0, zpt, 1.0), -1.0)
        ib, kb = _bins(zpt, self.edges["pt_z"]), _bins(bal, self.edges["balance"])
        ok &= (ib >= 0) & (kb >= 0)
        r = np.flatnonzero(ok)
        rows_out.append(r)
        cells_out.append(((ib[r] * self.nr + self._region(eta[r, 0])) * nb + kb[r]).astype(np.int64))
        avg = 0.5 * (p1 + p2)
        ia = _bins(avg, self.edges["pt_avg"])
        fired = ((ev["bits"][idx] >> np.maximum(ia, 0)) & 1).astype(bool)
        base = live[:, 1] & (ia >= 0) & fired & (_dphi(phi[:, 0], phi[:, 1]) > dj["dphi_min"]) \
            & (p3 < dj["alpha_max"] * avg)
        central = dj["tag_abs_eta_max"]
        for tag, probe in ((0, 1), (1, 0)):
            okd = base & (np.abs(eta[:, tag]) < central)
            pp, pt_ = pt[:, probe], pt[:, tag]
            with np.errstate(divide="ignore", invalid="ignore"):
                asy = (pp - pt_) / (pp + pt_)
            ka = _bins(asy, self.edges["asymmetry"])
            okd &= ka >= 0
            r = np.flatnonzero(okd)
            rows_out.append(r)
            cells_out.append(((self.nz + ia[r] * self.nr + self._region(eta[r, probe])) * nb + ka[r]).astype(np.int64))
        return np.concatenate(rows_out), np.concatenate(cells_out)


def closure(spec: dict, cand: Dict[str, dict], cset: dict, result: Any, request: dict, flavour: str,
            seed: int, fallback: Optional[Dict[str, Tuple[float, Any]]] = None) -> dict:
    """The balance comparison (peak.compare): "per_stream" the Z+jet and dijet losses, "curves" each stream's
    page curve; ``fallback`` ({name: (nominal, width or {region: width})}) enters at its forecast."""
    return P.compare(_Balance(spec, cand, cset, result, request, flavour, seed, fallback))


def fallback_loss(spec: dict, cand: Dict[str, dict], request: dict, flavour: str, seed: int,
                  cache_dir: Optional[Path] = None) -> Optional[float]:
    """The balance loss with every correction at its no-answer forecast (peak.cached_fallback_loss)."""
    return P.cached_fallback_loss(closure, RULE, "balance_fallback", spec, cand, request, flavour, seed, cache_dir)


def spec(selection: Dict[str, Any], regions: Dict[str, Dict[str, Any]], menu, plateau: float, request: Dict[str, Any],
         fallback: Dict[str, Tuple[float, Any]], flavour: str = "jet") -> Optional[Dict[str, Any]]:
    """A task's balance spec for a request: its `selection` (muon, muon_paths, mass_window, clean, jet_abs_eta_max,
    jet_pt_min, preselect, zjet, dijet), its eta `regions` in order ({name: {"abs_eta": [lo, hi]}}), its single-jet
    `menu` (thresholds; path HLT_SingleJet<T>) with dijet pT_avg bins from `plateau` x each threshold (path_edges), and
    the request's corrections it folds (scale and res_pt of `flavour`) with their forecasts (`fallback`: {name:
    (nominal, width or {region: width})}).  None when the request has neither."""
    import copy
    entries = (request or {}).get("corrections") or {}
    used = [n for n in ("scale", *RESOLUTIONS) if flavour in ((entries.get(n) or {}).get("domain") or {})]
    if not used:
        return None
    out = copy.deepcopy(selection)
    names = list(regions)
    out["regions"] = [regions[names[0]]["abs_eta"][0]] + [regions[r]["abs_eta"][1] for r in names]
    out["region_names"] = names
    out["dijet"]["paths"] = [[t, f"HLT_SingleJet{t}"] for t in menu]
    out["dijet"]["pt_avg"] = path_edges(menu, plateau)
    out["used"] = used
    out["fallback"] = {n: fallback[n] for n in used}
    return out


def path_edges(thresholds, plateau: float) -> List[float]:
    """The dijet pT_avg bins of a single-jet menu: bin k from plateau x T_k (where path k is on its plateau for the
    leading jet, whose pT is at least pT_avg) to the next one's, the last open to 10 TeV."""
    return [float(round(plateau * t)) for t in thresholds] + [1.0e4]

