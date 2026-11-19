"""The declared measurement behind a requested correction (the contract's MEASUREMENT.json section): read
MEASUREMENT.json and the pair lists of MEASUREMENT.npz, check their structure and the pairs against the dataset
(membership, uniqueness, pass flags, simulation weights), and recount.  check() is `sbh check`'s entry; the
evaluation runs the same checks.

Task-independent: the dataset is read as NanoAOD (TrigObj for trigger legs), the request entry defines a pass; no
truth is known here.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ...budget import BudgetExceeded, alarm_held, budget_left, budget_message
from ...contract import open_output, read_output

FILE = "MEASUREMENT.json"
NPZ = "MEASUREMENT.npz"
MAX_JSON = 20_000_000
MAX_NPZ = 1_000_000_000
MAX_NPZ_UNPACKED = 4_000_000_000
SAMPLES = ("data", "sim")
PAIR_FIELDS = ("file", "entry", "first", "second")
COLLECTIONS = {"muon": "Muon", "electron": "Electron", "tau": "Tau", "photon": "Photon", "jet": "Jet"}
OBJECT_VARIABLES = ("pt", "eta", "abs_eta", "phi", "charge")
PAIR_VARIABLES = ("mass", "pt_pair", "delta_eta", "delta_phi", "abs_eta_max", "abs_eta_min", "pt_max", "pt_min") \
    + tuple(f"{r}_{v}" for r in ("first", "second") for v in OBJECT_VARIABLES)
SIM_REL_TOL = 1e-6
AMBIGUOUS_REL = 1e-5


class MeasurementError(ValueError):
    """A measurement that does not satisfy the contract; the message says why (participant-readable)."""


def _first_line(e: BaseException) -> str:
    s = str(e)
    return s.splitlines()[0][:200] if s else type(e).__name__



def read(out_dir) -> Tuple[Optional[dict], Optional[Dict[str, np.ndarray]], Optional[str]]:
    """(MEASUREMENT.json, MEASUREMENT.npz arrays, None) or (None, None, why): missing, not regular files, over the
    size caps (refused before reading; npz members by declared size), not JSON / npz, or holding pickled objects."""
    import zipfile
    out_dir = Path(out_dir)
    try:
        spec = json.loads(read_output(out_dir / FILE, max_bytes=MAX_JSON))
    except FileNotFoundError:
        return None, None, f"{FILE} not produced"
    except (OSError, ValueError, RecursionError) as e:
        return None, None, f"{FILE} unreadable: {_first_line(e)}"
    if not isinstance(spec, dict):
        return None, None, f"{FILE} is not a JSON object"
    try:
        fh = open_output(out_dir / NPZ)
    except FileNotFoundError:
        return None, None, f"{NPZ} not produced"
    except (OSError, ValueError) as e:
        return None, None, f"{NPZ} unreadable: {_first_line(e)}"
    with fh:
        import os
        if os.fstat(fh.fileno()).st_size > MAX_NPZ:
            return None, None, f"{NPZ} is larger than {MAX_NPZ / 1e9:.0f} GB"
        try:
            with zipfile.ZipFile(fh) as zf:
                unpacked = sum(i.file_size for i in zf.infolist())
            if unpacked > MAX_NPZ_UNPACKED:
                return None, None, f"{NPZ} unpacks to {unpacked / 1e9:.1f} GB, above {MAX_NPZ_UNPACKED / 1e9:.0f} GB"
            fh.seek(0)
            with np.load(fh, allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
        except Exception as e:  # noqa: BLE001
            return None, None, f"{NPZ} unreadable: {_first_line(e)}"
    return spec, arrays, None



def _edges(e: Any) -> Optional[np.ndarray]:
    """Finite, strictly increasing edges (at least two), or None."""
    if not (isinstance(e, list) and len(e) >= 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                                         for x in e)):
        return None
    a = np.asarray(e, dtype=float)
    return a if np.all(np.isfinite(a)) and np.all(np.diff(a) > 0) else None


def _binning(b: Any, allowed=None, empty: bool = False) -> Tuple[Optional[List[Tuple[str, np.ndarray]]], Optional[str]]:
    if not isinstance(b, dict) or not (b or empty):
        return None, "names no binning variables"
    out = []
    for var, e in b.items():
        if not (isinstance(var, str) and re.fullmatch(r"[A-Za-z0-9_]+", var)):
            return None, f"has a binning variable {var!r} that is not a plain name"
        if allowed is not None and var not in allowed:
            return None, f"bins in {var!r}, which is not a pair variable ({', '.join(allowed)})"
        a = _edges(e)
        if a is None:
            return None, f"has edges for {var!r} that are not at least two finite increasing numbers"
        out.append((var, a))
    return out, None


def _n_bins(binning: List[Tuple[str, np.ndarray]]) -> int:
    return int(np.prod([len(e) - 1 for _, e in binning]))


def _counts(x: Any, n: int, integer: bool) -> Optional[np.ndarray]:
    if not (isinstance(x, list) and len(x) == n and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                                         for v in x)):
        return None
    a = np.asarray(x, dtype=float)
    if not (np.all(np.isfinite(a)) and np.all(a >= 0)) or (integer and not np.all(a == np.round(a))):
        return None
    return a


def _canonical(name: str) -> Optional[str]:
    """A pair list's file name as a relative POSIX path without '.' parts, or None (absolute, escaping its directory,
    not a ROOT file)."""
    if not (isinstance(name, str) and name.endswith(".root")) or name.startswith("/"):
        return None
    parts = Path(name).parts
    return None if ".." in parts or not parts else "/".join(parts)


def _list_arrays(arrays: Dict[str, np.ndarray], lst: str, sample: str, fields) -> Tuple[Optional[dict], Optional[str]]:
    """One sample's arrays of a pair list (fields of one length and the contract's types; file table
    `<list>.<sample>.files` of distinct names, `file` indexing it), or why not."""
    out, n = {}, None
    files_key = f"{lst}.{sample}.files"
    table = arrays.get(files_key)
    if table is None:
        return None, f"{NPZ} has no array {files_key!r}"
    if table.ndim != 1 or table.dtype.kind != "U":
        return None, f"{files_key} is not a one-dimensional array of strings"
    names = [_canonical(str(x)) for x in table]
    if any(x is None for x in names):
        return None, f"{files_key} has a name that is not a relative path of a .root file"
    if len(set(names)) != len(names):
        return None, f"{files_key} names a file twice"
    for f in fields:
        key = f"{lst}.{sample}.{f}"
        if key not in arrays:
            return None, f"{NPZ} has no array {key!r}"
        a = arrays[key]
        if a.ndim != 1:
            return None, f"{key} is not one-dimensional"
        if n is None:
            n = len(a)
        elif len(a) != n:
            return None, f"{key} has {len(a)} elements, the list's other arrays {n}"
        kind = ("int" if f in PAIR_FIELDS else "bool" if f.startswith("pass_") else "real")
        ok = {"int": a.dtype.kind in "iu", "bool": a.dtype.kind == "b",
              "real": a.dtype.kind in "fiu" and bool(np.all(np.isfinite(a.astype(float))))}[kind]
        if not ok:
            return None, f"{key} is not of the contract's type ({kind}{', finite' if kind == 'real' else ''})"
        out[f] = a
    if not n:
        return None, f"{lst} has no {sample} pairs"
    if np.any(out["file"] < 0) or np.any(out["file"] >= len(names)):
        return None, f"{lst}.{sample}.file has an index outside {files_key}"
    out["files"] = np.asarray(names)
    return out, None


def declared_id(spec: Any, name: str, flavour: str) -> Optional[str]:
    """The measurement id mapped to (name, flavour), or None (undeclared or malformed)."""
    mapping = spec.get("corrections") if isinstance(spec, dict) else None
    per = mapping.get(name) if isinstance(mapping, dict) else None
    mid = per.get(flavour) if isinstance(per, dict) else None
    return mid if isinstance(mid, str) else None


def undeclared(spec: Any, name: str, flavour: str) -> bool:
    """Whether a well-formed MEASUREMENT.json leaves (name, flavour) out of "corrections" (an optional declaration
    not made).  A malformed file or entry is a failing declaration, not one left out."""
    if not (isinstance(spec, dict) and isinstance(spec.get("measurements"), dict)
            and isinstance(spec.get("corrections"), dict)):
        return False
    per = spec["corrections"].get(name)
    return per is None or (isinstance(per, dict) and per.get(flavour) is None)


def parse(spec: Any, arrays: Dict[str, np.ndarray], request: Any) -> Tuple[Dict[str, dict], Dict[Tuple[str, str], str]]:
    """({id: parsed measurement}, {(name, flavour): why its measurement fails the structure}) over every requested
    (correction, flavour).  Parsed efficiency: {"type", "list", "binning", "data": {"pass", "fail"}, "sim": {...},
    "pairs": {sample: arrays}}; lineshape: {"type", "list", "corrections", "selection", "histograms": [{"variable",
    "edges", "categories", "data"}], "pairs", "flavour"}."""
    from .outputs import requested
    entries = requested(request)
    problems: Dict[Tuple[str, str], str] = {}
    parsed: Dict[str, dict] = {}
    measurements = spec.get("measurements") if isinstance(spec, dict) else None
    mapping = spec.get("corrections") if isinstance(spec, dict) else None
    if not (isinstance(measurements, dict) and isinstance(mapping, dict)):
        why = f'{FILE} is not an object with "measurements" and "corrections" objects'
        return {}, {(n, fl): why for n, e in entries.items() for fl in ((e or {}).get("domain") or {})}
    bad: Dict[str, str] = {}
    for mid, m in measurements.items():
        why, pm = _parse_one(mid, m, arrays)
        if why:
            bad[mid] = why
        else:
            parsed[mid] = pm
    users: Dict[str, List[Tuple[str, str]]] = {}
    for name, per in mapping.items():
        for flavour, mid in (per.items() if isinstance(per, dict) else []):
            if isinstance(mid, str):
                users.setdefault(mid, []).append((name, flavour))
    for mid, pm in parsed.items():
        if pm["type"] == "lineshape":
            fls = sorted({fl for _, fl in users.get(mid, [])})
            pm["flavour"] = fls[0] if len(fls) == 1 else None
    for name, entry in sorted(entries.items()):
        for flavour in sorted(((entry or {}).get("domain") or {})):
            mid = declared_id(spec, name, flavour)
            if mid is None:
                problems[(name, flavour)] = (f"{FILE} maps no measurement to {name} for {flavour}"
                                             if undeclared(spec, name, flavour) else
                                             f"{FILE}'s mapping of {name} for {flavour} is not a measurement id")
            elif mid in bad:
                problems[(name, flavour)] = f"its measurement {mid!r} {bad[mid]}"
            elif mid not in parsed:
                problems[(name, flavour)] = f"its measurement {mid!r} is not in {FILE}"
            else:
                why = _serves(parsed[mid], name, entry, flavour, users[mid], entries)
                if why:
                    problems[(name, flavour)] = f"its measurement {mid!r} {why}"
    return parsed, problems


def _histogram(h: Any) -> Tuple[Optional[str], Optional[dict]]:
    if not isinstance(h, dict):
        return "is not an object", None
    var = h.get("variable")
    if var not in PAIR_VARIABLES:
        return f'has a "variable" that is not a pair variable ({", ".join(PAIR_VARIABLES)})', None
    edges = _edges(h.get("edges"))
    if edges is None:
        return 'has no "edges" (at least two finite increasing numbers)', None
    cats, why = _binning(h.get("categories", {}), allowed=PAIR_VARIABLES, empty=True)
    if why:
        return f"categories {why}", None
    nc, nb = _n_bins(cats), len(edges) - 1
    data = h.get("data")
    rows = [_counts(r, nb, True) for r in data] if isinstance(data, list) and len(data) == nc else None
    if rows is None or any(r is None for r in rows):
        return f'has no {nc} x {nb} non-negative integer "data" counts (per category, per bin)', None
    return None, {"variable": var, "edges": edges, "categories": cats, "data": np.asarray(rows, dtype=float)}


def _parse_one(mid: str, m: Any, arrays: Dict[str, np.ndarray]) -> Tuple[Optional[str], Optional[dict]]:
    if not isinstance(m, dict):
        return "is not an object", None
    t, lst = m.get("type"), m.get("pairs")
    if not (isinstance(lst, str) and re.fullmatch(r"[A-Za-z0-9_]+", lst)):
        return 'has no "pairs" list name (letters, digits, underscores)', None
    if t == "efficiency":
        binning, why = _binning(m.get("bins"))
        if why:
            return why, None
        n = _n_bins(binning)
        d, s = m.get("data") or {}, m.get("simulation") or {}
        if not (isinstance(d, dict) and isinstance(s, dict)):
            return 'has no "data" and "simulation" objects', None
        counts = {"data": {k: _counts(d.get(k), n, True) for k in ("pass", "fail")},
                  "sim": {k: _counts(s.get(k), n, False) for k in ("pass", "fail", "pass_w2", "fail_w2")}}
        for sample, cs in counts.items():
            for k, a in cs.items():
                if a is None:
                    return (f"has no {n} {'non-negative integer counts' if sample == 'data' else 'non-negative sums'} "
                            f"for {'data' if sample == 'data' else 'simulation'} {k!r} (one per bin)"), None
        fields = PAIR_FIELDS + tuple(f"x_{v}" for v, _ in binning) + (f"pass_{mid}",)
        pairs = {}
        for sample in SAMPLES:
            arr, why = _list_arrays(arrays, lst, sample, fields + (("weight",) if sample == "sim" else ()))
            if why:
                return why, None
            pairs[sample] = arr
        return None, {"type": t, "list": lst, "binning": binning, "data": counts["data"], "sim": counts["sim"],
                      "pairs": pairs}
    if t == "lineshape":
        sel = m.get("selection")
        if not (isinstance(sel, dict) and all(isinstance(sel.get(k), (int, float)) and not isinstance(sel.get(k), bool)
                                              and math.isfinite(sel[k]) for k in ("pt_min", "abs_eta_max"))):
            return 'has no "selection" with numbers "pt_min" and "abs_eta_max"', None
        corr = m.get("corrections")
        if not (isinstance(corr, list) and corr and all(isinstance(c, str) for c in corr)
                and len(set(corr)) == len(corr)):
            return 'lists no corrections (distinct names) in "corrections"', None
        hs = m.get("histograms")
        if not (isinstance(hs, list) and hs):
            return 'has no "histograms" list', None
        hists = []
        for i, h in enumerate(hs):
            why, ph = _histogram(h)
            if why:
                return f"histogram {i} {why}", None
            hists.append(ph)
        pairs = {}
        for sample in SAMPLES:
            arr, why = _list_arrays(arrays, lst, sample, PAIR_FIELDS + (("weight",) if sample == "sim" else ()))
            if why:
                return why, None
            pairs[sample] = arr
        return None, {"type": t, "list": lst, "corrections": corr,
                      "selection": {k: float(sel[k]) for k in ("pt_min", "abs_eta_max")}, "histograms": hists,
                      "pairs": pairs}
    return 'has a "type" other than "efficiency" or "lineshape"', None


def _serves(pm: dict, name: str, entry: dict, flavour: str, users: List[Tuple[str, str]],
            entries: Dict[str, dict]) -> Optional[str]:
    """Why the parsed measurement cannot be the one of (name, flavour), or None."""
    kind = (entry or {}).get("kind")
    if kind in ("id", "trigger"):
        if pm["type"] != "efficiency":
            return f"is a {pm['type']} measurement; an {kind} correction's is an efficiency"
        return None if users == [(name, flavour)] else "serves more than one correction"
    if kind in ("scale", "resolution"):
        if pm["type"] != "lineshape":
            return f"is an {pm['type']} measurement; a {kind} correction's is a lineshape"
        if pm.get("flavour") is None:
            return "serves corrections of more than one object type"
        if name not in pm["corrections"]:
            return f"does not list {name} in its corrections"
        for c in pm["corrections"]:
            e = entries.get(c) or {}
            if e.get("kind") not in ("scale", "resolution") or flavour not in (e.get("domain") or {}):
                return f"lists {c!r}, which is not a requested scale or resolution correction for {flavour}"
        return None
    return f"serves a correction of kind {kind!r}, which takes no measurement"



def _interval(v: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Each value's interval index of edges e (-1 below, len(e) - 1 above; an inner edge belongs to the interval
    above): searchsorted, or a direct estimate for (log-)uniform edges corrected against the edges to equal it."""
    n_int = len(e) - 1
    d = np.diff(e)
    k = None
    if n_int >= 8 and np.allclose(d, d[0], rtol=1e-9, atol=0):
        k = np.floor((v - e[0]) / d[0])
    elif n_int >= 8 and e[0] > 0 and np.allclose(np.diff(np.log(e)), np.log(e[1] / e[0]), rtol=1e-9, atol=0):
        with np.errstate(divide="ignore", invalid="ignore"):
            k = np.floor(np.log(np.maximum(v, 1e-300) / e[0]) / np.log(e[1] / e[0]))
    if k is None:
        return np.searchsorted(e, v, side="right") - 1
    k = np.clip(np.nan_to_num(k, nan=-1.0, posinf=n_int, neginf=-1.0), -1, n_int).astype(np.int64)
    kc = np.clip(k, 0, n_int - 1)
    k = np.where((k >= 0) & (k < n_int) & (v < e[kc]), k - 1, k)
    kc = np.clip(k, 0, n_int - 1)
    k = np.where((k >= 0) & (k < n_int) & (v >= e[kc + 1]), k + 1, k)
    k = np.where(v < e[0], -1, np.where(v >= e[-1], n_int, k))
    return k


def bin_index(binning: List[Tuple[str, np.ndarray]], x: Dict[str, np.ndarray], n: int) -> np.ndarray:
    """Each of n pairs' bin (last variable fastest; 0 for none), -1 outside; numpy.histogram's edge convention."""
    idx = np.zeros(n, dtype=np.int64)
    inside = np.ones(n, dtype=bool)
    for var, e in binning:
        v = np.asarray(x[var], dtype=float)
        k = _interval(v, e)
        k = np.where(v == e[-1], len(e) - 2, k)
        inside &= (k >= 0) & (k < len(e) - 1)
        idx = idx * (len(e) - 1) + np.clip(k, 0, len(e) - 2)
    return np.where(inside, idx, -1)


def unique_pairs(arr: dict) -> Optional[str]:
    """Why a sample's pairs are not distinct pairs of distinct objects, or None."""
    if np.any(arr["first"] == arr["second"]):
        return "lists a pair of an object with itself"
    key = np.stack([arr["file"].astype(np.int64), arr["entry"].astype(np.int64), arr["first"].astype(np.int64),
                    arr["second"].astype(np.int64)], axis=1)
    if len(key) and len(np.unique(key, axis=0)) != len(key):
        return "lists a pair twice"
    return None


def recount_efficiency(pm: dict, mid: str) -> Optional[str]:
    """Why the declared counts of an efficiency measurement are not those of its pairs' declared values, or None."""
    n = _n_bins(pm["binning"])
    for sample in SAMPLES:
        arr = pm["pairs"][sample]
        b = bin_index(pm["binning"], {v: arr[f"x_{v}"] for v, _ in pm["binning"]}, len(arr["entry"]))
        ok, p = b >= 0, arr[f"pass_{mid}"].astype(bool)
        w = arr["weight"].astype(float) if sample == "sim" else np.ones(len(b))
        got = {"pass": np.bincount(b[ok & p], w[ok & p], n), "fail": np.bincount(b[ok & ~p], w[ok & ~p], n)}
        if sample == "sim":
            got.update(pass_w2=np.bincount(b[ok & p], w[ok & p] ** 2, n),
                       fail_w2=np.bincount(b[ok & ~p], w[ok & ~p] ** 2, n))
        declared = pm["data"] if sample == "data" else pm["sim"]
        for k, a in got.items():
            if sample == "data" and not np.array_equal(a, declared[k]):
                first = int(np.argmax(a != declared[k]))
                return f"its data {k} counts are not those of its pairs (first bin off: {first})"
            if sample == "sim" and not np.allclose(a, declared[k], rtol=SIM_REL_TOL,
                                                   atol=1e-9 * max(1.0, float(np.abs(a).sum()))):
                return f"its simulation {k} sums are not those of its pairs"
    return None


def pair_variables(a: dict, b: dict, needed=None) -> Dict[str, np.ndarray]:
    """The contract's pair variables of objects a (first) and b (second), each {"pt", "eta", "phi", "charge"}:
    massless invariant mass, pair pT, |delta eta|, |delta phi| in [0, pi], larger and smaller |eta| and pT, each
    object's own (``needed``: those plus what the selection uses)."""
    if needed is not None:
        return _pair_variables_some(a, b, needed)
    dphi = np.abs(np.remainder(a["phi"] - b["phi"] + np.pi, 2.0 * np.pi) - np.pi)
    deta = a["eta"] - b["eta"]
    m2 = 2.0 * a["pt"] * b["pt"] * (np.cosh(deta) - np.cos(dphi))
    px = a["pt"] * np.cos(a["phi"]) + b["pt"] * np.cos(b["phi"])
    py = a["pt"] * np.sin(a["phi"]) + b["pt"] * np.sin(b["phi"])
    ea, eb = np.abs(a["eta"]), np.abs(b["eta"])
    out = {"mass": np.sqrt(np.maximum(m2, 0.0)), "pt_pair": np.hypot(px, py), "delta_eta": np.abs(deta),
           "delta_phi": dphi, "abs_eta_max": np.maximum(ea, eb), "abs_eta_min": np.minimum(ea, eb),
           "pt_max": np.maximum(a["pt"], b["pt"]), "pt_min": np.minimum(a["pt"], b["pt"])}
    for role, o in (("first", a), ("second", b)):
        out.update({f"{role}_pt": o["pt"], f"{role}_eta": o["eta"], f"{role}_abs_eta": np.abs(o["eta"]),
                    f"{role}_phi": o["phi"], f"{role}_charge": np.asarray(o["charge"], dtype=float)})
    return out


def _pair_variables_some(a: dict, b: dict, needed) -> Dict[str, np.ndarray]:
    """pair_variables, computing only the needed variables and the objects' own pT and |eta| (the selection's)."""
    out: Dict[str, np.ndarray] = {}
    for role, o in (("first", a), ("second", b)):
        out[f"{role}_pt"], out[f"{role}_abs_eta"] = o["pt"], np.abs(o["eta"])
        for v in ("eta", "phi"):
            if f"{role}_{v}" in needed:
                out[f"{role}_{v}"] = o[v]
        if f"{role}_charge" in needed:
            out[f"{role}_charge"] = np.asarray(o["charge"], dtype=float)
    if needed & {"mass", "delta_phi"}:
        dphi = np.abs(np.remainder(a["phi"] - b["phi"] + np.pi, 2.0 * np.pi) - np.pi)
        out["delta_phi"] = dphi
        if "mass" in needed:
            out["mass"] = np.sqrt(np.maximum(2.0 * a["pt"] * b["pt"] * (np.cosh(a["eta"] - b["eta"]) - np.cos(dphi)),
                                             0.0))
    if "delta_eta" in needed:
        out["delta_eta"] = np.abs(a["eta"] - b["eta"])
    if "pt_pair" in needed:
        out["pt_pair"] = np.hypot(a["pt"] * np.cos(a["phi"]) + b["pt"] * np.cos(b["phi"]),
                                  a["pt"] * np.sin(a["phi"]) + b["pt"] * np.sin(b["phi"]))
    for v, f in (("abs_eta_max", np.maximum), ("abs_eta_min", np.minimum)):
        if v in needed:
            out[v] = f(out["first_abs_eta"], out["second_abs_eta"])
    for v, f in (("pt_max", np.maximum), ("pt_min", np.minimum)):
        if v in needed:
            out[v] = f(out["first_pt"], out["second_pt"])
    return out


def _selected(sel: dict, x: Dict[str, np.ndarray]) -> np.ndarray:
    """Both objects pass the selection: pT > pt_min, |eta| < abs_eta_max."""
    return ((x["first_pt"] > sel["pt_min"]) & (x["second_pt"] > sel["pt_min"])
            & (x["first_abs_eta"] < sel["abs_eta_max"]) & (x["second_abs_eta"] < sel["abs_eta_max"]))


def histogram_cells(h: dict, sel: dict, x: Dict[str, np.ndarray], n: int) -> np.ndarray:
    """Each pair's cell (category x bin, bin fastest), -1 unless both objects pass the selection and it lies in a
    category and the edges."""
    cat = bin_index(h["categories"], x, n)
    b = bin_index([(h["variable"], h["edges"])], x, n)
    ok = _selected(sel, x) & (cat >= 0) & (b >= 0)
    return np.where(ok, cat * (len(h["edges"]) - 1) + b, -1)


def _near(v: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Whether each value lies within AMBIGUOUS_REL (relative) of one of the edges."""
    k = np.searchsorted(edges, v)
    d = np.minimum(np.abs(v - edges[np.clip(k - 1, 0, len(edges) - 1)]),
                   np.abs(v - edges[np.clip(k, 0, len(edges) - 1)]))
    return d <= AMBIGUOUS_REL * (np.abs(v) + 1e-3)


def _ambiguous(h: dict, sel: dict, x: Dict[str, np.ndarray]) -> np.ndarray:
    """Pairs whose cell can change with a relative AMBIGUOUS_REL change of a compared value (selection pT / |eta|,
    category variables, the variable)."""
    out = np.zeros(len(x["mass"]), dtype=bool)
    for role in ("first", "second"):
        out |= (_near(x[f"{role}_pt"], np.array([sel["pt_min"]]))
                | _near(x[f"{role}_abs_eta"], np.array([sel["abs_eta_max"]])))
    for var, e in list(h["categories"]) + [(h["variable"], h["edges"])]:
        out |= _near(x[var], e)
    return out


def _alternatives(h: dict, sel: dict, x: Dict[str, np.ndarray], idx: np.ndarray) -> np.ndarray:
    """Cells (-1: not counted) pairs idx take with one compared value moved by 2 AMBIGUOUS_REL either way, shape
    (alternatives, len(idx))."""
    variables = {v for v, _ in h["categories"]} | {h["variable"]} \
        | {f"{r}_{k}" for r in ("first", "second") for k in ("pt", "abs_eta")}
    sub = {k: np.asarray(v)[idx] for k, v in x.items()}
    out = []
    for var in sorted(variables):
        for sign in (-1.0, 1.0):
            y = dict(sub)
            y[var] = sub[var] * (1.0 + sign * 2.0 * AMBIGUOUS_REL) + sign * 2.0 * AMBIGUOUS_REL * 1e-3
            out.append(histogram_cells(h, sel, y, len(idx)))
    return np.asarray(out) if out else np.zeros((0, len(idx)), dtype=np.int64)


def recount_lineshape(pm: dict, data: dict) -> Tuple[Optional[List[np.ndarray]], Optional[str]]:
    """(each histogram's cell of every data pair, None) or (None, why the declared counts are not the pairs'): the
    recount uses the reconstructed objects in double precision; it may differ only through pairs within AMBIGUOUS_REL
    of an edge or threshold -- per cell by at most the ambiguous pairs that recount or could count there, over the
    histogram by at most twice their number."""
    x = pair_variables(data["first"], data["second"])
    n = len(x["mass"])
    cells = []
    for i, h in enumerate(pm["histograms"]):
        c = histogram_cells(h, pm["selection"], x, n)
        size = h["data"].size
        got = np.bincount(c[c >= 0], minlength=size).astype(float)
        diff = np.abs(got - h["data"].ravel())
        amb = np.flatnonzero(_ambiguous(h, pm["selection"], x))
        slack = np.zeros(size)
        if len(amb):
            alt = _alternatives(h, pm["selection"], x, amb)
            own = c[amb]
            np.add.at(slack, own[own >= 0], 1.0)
            moved = (alt != own[None, :]) & (alt >= 0)
            for j in range(len(amb)):
                for cell in set(alt[moved[:, j], j].tolist()):
                    slack[cell] += 1.0
        bad = diff > slack
        if np.any(bad) or float(diff.sum()) > 2 * len(amb):
            first = int(np.argmax(bad)) if np.any(bad) else int(np.argmax(diff))
            nb = len(h["edges"]) - 1
            return None, (f"histogram {i}'s data counts are not those of its pairs ({diff.sum():.0f} counts off; "
                          f"first: category {first // nb}, bin {first % nb})")
        cells.append(c)
    return cells, None



_SELECTION = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*(?:>=\s*(-?\d+))?\s*$")


def selection_branch(selection: str, flavour: str) -> Tuple[str, Optional[int]]:
    """(branch, threshold) of a request's selection, "<branch>" (boolean) or "<branch> >= <n>" (integer), in the
    flavour's collection.  Raises MeasurementError."""
    m = _SELECTION.match(selection or "")
    coll = COLLECTIONS.get(flavour)
    if not m or coll is None or not m.group(1).startswith(coll + "_"):
        raise MeasurementError(f"selection {selection!r} is not a {coll or flavour} branch (or '<branch> >= <n>')")
    return m.group(1), (int(m.group(2)) if m.group(2) is not None else None)


def _pick(values, counts: np.ndarray, entries: np.ndarray, index: np.ndarray) -> np.ndarray:
    """values[entry][index] for jagged per-event values (awkward), vectorized through the flat offsets."""
    import awkward as ak
    flat = ak.to_numpy(ak.flatten(values))
    off = np.concatenate([[0], np.cumsum(counts)])
    return flat[off[entries] + index]


CPUS = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
READ_JOBS = min(8, CPUS)


def pool_imap(fn: Callable, items, jobs: int, context: str, log: Optional[Callable[[str], None]] = None):
    """fn(item) for every item, yielded in completion order from up to `jobs` processes of the multiprocessing
    `context` ("fork" inherits the caller's state, "forkserver" pickles fn and item); fn's result must not depend on
    the process.
    - A worker dying (an out-of-memory kill) breaks the pool, where multiprocessing.Pool would hang: the unfinished
      items rerun in half as many workers, finally here one at a time, and `log` (else stderr) says so.
    - fn's exception or the caller's (BudgetExceeded, stopping early) is raised: nothing more starts and running
      workers are killed (a budget is never waited out).
    - A worker killed while sending can still wedge the reader: a budget, else the job's limit, is the backstop.
    - Start, waits and shutdown are alarm_held: a budget rings between them, never inside the pool's locks."""
    import sys
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    from concurrent.futures.process import BrokenProcessPool
    from multiprocessing import get_context
    say = log or (lambda m: print(m, file=sys.stderr, flush=True))
    pending = dict(enumerate(items))
    workers = min(jobs, len(pending))
    while workers > 1 and pending:
        futs: Dict[Any, int] = {}
        try:
            ex = ProcessPoolExecutor(workers, mp_context=get_context(context))
            try:
                try:
                    with alarm_held():
                        futs = {ex.submit(fn, item): i for i, item in pending.items()}
                    todo = set(futs)
                    while todo:
                        left = budget_left()
                        if left is not None and left <= 0:
                            raise BudgetExceeded(budget_message() or "the stage ran out of its budget")
                        with alarm_held():
                            done, todo = wait(todo, timeout=left, return_when=FIRST_COMPLETED)
                        while done:
                            f = done.pop()
                            i = futs.pop(f)
                            out = f.result()
                            del f
                            del pending[i]
                            yield out
                            del out
                except BrokenProcessPool:
                    raise
                except BaseException:
                    with alarm_held():
                        ex.kill_workers()
                    raise
            finally:
                with alarm_held():
                    ex.shutdown(wait=True)
        except BrokenProcessPool:
            while futs:
                f, i = futs.popitem()
                if f.done() and not f.cancelled() and f.exception() is None:
                    del pending[i]
                    yield f.result()
            workers = min(workers // 2, len(pending))
            if pending:
                say(f"a worker process was killed (out of memory?): {len(pending)} item(s) run again "
                    + (f"in {workers} processes" if workers > 1 else "here, one at a time"))
    for i in sorted(pending):
        left = budget_left()
        if left is not None and left <= 0:
            raise BudgetExceeded(budget_message())
        yield fn(pending.pop(i))


def _leg_flags(start: np.ndarray, count: np.ndarray, cols: Dict[str, np.ndarray], eta: np.ndarray, phi: np.ndarray,
               leg: dict) -> Tuple[np.ndarray, np.ndarray]:
    """(probe has the trigger leg's object, that is within AMBIGUOUS_REL of the cone or online threshold): a TrigObj
    with TrigObj_id == trigobj_id, delta R < delta_r_max, bit filter_bit of TrigObj_filterBits set and
    TrigObj_pt >= online_pt_min (the contract's request.json section)."""
    n = len(start)
    total = int(count.sum())
    if not total:
        return np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    rep = np.repeat(np.arange(n), count)
    first = np.cumsum(count) - count
    obj = start[rep] + (np.arange(total) - first[rep])
    tid, bit = int(leg["trigobj_id"]), int(leg["filter_bit"])
    ptmin, dr2max = float(leg["online_pt_min"]), float(leg["delta_r_max"]) ** 2
    dphi = np.remainder(cols["phi"][obj].astype(float) - phi[rep] + np.pi, 2.0 * np.pi) - np.pi
    dr2 = (cols["eta"][obj].astype(float) - eta[rep]) ** 2 + dphi ** 2
    pt = cols["pt"][obj].astype(float)
    cand = (cols["id"][obj] == tid) & (((cols["filterBits"][obj].astype(np.int64) >> bit) & 1) == 1)
    ok = cand & (pt >= ptmin) & (dr2 < dr2max)
    edge = cand & ((np.abs(dr2 - dr2max) <= 2.0 * AMBIGUOUS_REL * dr2max)
                   | (np.abs(pt - ptmin) <= AMBIGUOUS_REL * ptmin))
    return np.bincount(rep[ok], minlength=n) > 0, np.bincount(rep[edge], minlength=n) > 0


def _read_file(job) -> Tuple[str, Dict[str, Any]]:
    """One file's part of every list with pairs in it (a pool unit): {list: (positions, objects) or why not}.  A
    message quotes the submission's values only, never the dataset's (a sealed file's entry count stays sealed;
    sbh/eventlist.py's rule)."""
    import awkward as ak
    import uproot
    path, name, sample, lists = job
    out: Dict[str, Any] = {}
    with uproot.open(path) as f:
        if "Events" not in f:
            return name, {lst: f"names {name!r}, which has no Events tree" for lst in lists}
        t = f["Events"]
        n_ent = t.num_entries
        for lst, w in lists.items():
            if w["entry"].min() < 0 or w["entry"].max() >= n_ent:
                bad = w["entry"][(w["entry"] < 0) | (w["entry"] >= n_ent)]
                out[lst] = f"lists entry {int(bad[0])} of {name!r}, which is not an entry of its Events tree"
        good = {k: v for k, v in lists.items() if k not in out}
        if not good:
            return name, out
        want = set()
        for w in good.values():
            c = w["coll"]
            want |= {f"n{c}", f"{c}_pt", f"{c}_eta", f"{c}_phi", f"{c}_charge", *w["branches"]}
            if w["legs"]:
                want |= {"nTrigObj", "TrigObj_id", "TrigObj_pt", "TrigObj_eta", "TrigObj_phi", "TrigObj_filterBits"}
        if sample == "sim":
            want.add("genWeight")
        missing = sorted(b for b in want if b not in t.keys())
        if missing:
            return name, {**out, **{lst: f"needs branches {missing} that {name} does not have" for lst in good}}
        lo = min(int(w["entry"].min()) for w in good.values())
        hi = max(int(w["entry"].max()) for w in good.values()) + 1
        a = t.arrays(sorted(want), entry_start=lo, entry_stop=hi, library="ak")
    flat = {b: ak.to_numpy(ak.flatten(a[b])) if a[b].ndim > 1 else ak.to_numpy(a[b]) for b in a.fields}
    for lst, w in good.items():
        c = w["coll"]
        counts = flat[f"n{c}"].astype(np.int64)
        off = np.concatenate([[0], np.cumsum(counts)])
        local = w["entry"].astype(np.int64) - lo
        objs: Dict[str, Any] = {}
        for role in ("first", "second"):
            idx = w[role].astype(np.int64)
            if np.any(idx < 0) or np.any(idx >= counts[local]):
                out[lst] = f"lists an object index outside its event's {c} collection ({name})"
                break
            g = off[local] + idx
            objs[role] = {k: flat[f"{c}_{k}"][g] for k in ("pt", "eta", "phi", "charge")}
            objs[role].update({b: flat[b][g] for b in w["branches"]})
        if lst in out:
            continue
        if sample == "sim":
            objs["genWeight"] = flat["genWeight"][local].astype(float)
        if w["legs"]:
            tc = flat["nTrigObj"].astype(np.int64)
            toff = np.concatenate([[0], np.cumsum(tc)])
            cols = {k: flat[f"TrigObj_{k}"] for k in ("id", "pt", "eta", "phi", "filterBits")}
            p = objs["second"]
            objs["legs"] = {mid: _leg_flags(toff[local], tc[local], cols, p["eta"].astype(float),
                                            p["phi"].astype(float), leg) for mid, leg in w["legs"].items()}
        out[lst] = (w["pos"], objs)
    return name, out


def read_lists(dataset_dir, sample: str, wants: Dict[str, dict], jobs: int = READ_JOBS
               ) -> Tuple[Dict[str, dict], Dict[str, str]]:
    """Both objects of every pair of several lists of one sample, each dataset file read once (in `jobs` processes).
    wants = {list: {"pairs", "flavour", "branches": object branches needed, "legs": {measurement id: request entry}}}.
    Returns ({list: {role: {pt, eta, phi, charge, branches...}, "genWeight", "process", "legs": {id: (passes,
    ambiguous)}}}, {list: why not the dataset's: unknown file, entry outside the tree, object index outside the
    collection, branch missing})."""
    base = Path(dataset_dir) / ("data" if sample == "data" else "mc")
    whys: Dict[str, str] = {}
    per_file: Dict[str, Dict[str, dict]] = {}
    for lst, w in wants.items():
        coll = COLLECTIONS.get(w["flavour"])
        if coll is None:
            whys[lst] = f"object type {w['flavour']!r} has no collection"
            continue
        arr = w["pairs"]
        names = [str(x) for x in arr["files"]]
        used = np.unique(arr["file"])
        gone = [names[fi] for fi in used if not (base / names[fi]).is_file()]
        if gone:
            whys[lst] = (f"names {gone[0]!r}, which is not a {'data' if sample == 'data' else 'simulation'} "
                         "file of the dataset")
            continue
        order = np.argsort(arr["file"], kind="stable")
        bounds = np.searchsorted(arr["file"][order], np.append(used, used[-1] + 1 if len(used) else 0))
        for k, fi in enumerate(used):
            pos = order[bounds[k]:bounds[k + 1]]
            per_file.setdefault(names[fi], {})[lst] = {
                "coll": coll, "branches": tuple(w["branches"]), "legs": dict(w["legs"]), "pos": pos,
                "entry": arr["entry"][pos], "first": arr["first"][pos], "second": arr["second"][pos]}
    out: Dict[str, dict] = {lst: {"first": {}, "second": {}, "legs": {}} for lst in wants if lst not in whys}
    for lst in out:
        n = len(wants[lst]["pairs"]["entry"])
        out[lst].update(genWeight=np.zeros(n), process=np.empty(n, dtype=object))
    jobs_list = [(str(base / name), name, sample, lists) for name, lists in sorted(per_file.items())]
    if jobs_list:
        for name, parts in pool_imap(_read_file, jobs_list, jobs, "forkserver"):
            for lst, got in parts.items():
                if lst in whys:
                    continue
                if isinstance(got, str):
                    whys[lst] = got
                    continue
                pos, objs = got
                o, n = out[lst], len(out[lst]["genWeight"])
                for r in ("first", "second"):
                    for k, v in objs[r].items():
                        if k not in o[r]:
                            o[r][k] = np.zeros(n, dtype=v.dtype)
                        o[r][k][pos] = v
                if sample == "sim":
                    o["genWeight"][pos] = objs["genWeight"]
                    o["process"][pos] = Path(name).parts[0] if len(Path(name).parts) > 1 else ""
                for mid, (passes, amb) in objs.get("legs", {}).items():
                    if mid not in o["legs"]:
                        o["legs"][mid] = (np.zeros(n, dtype=bool), np.zeros(n, dtype=bool))
                    o["legs"][mid][0][pos], o["legs"][mid][1][pos] = passes, amb
    return {k: v for k, v in out.items() if k not in whys}, whys


def read_objects(dataset_dir, sample: str, pairs: dict, flavour: str, branches=(), legs: Optional[dict] = None
                 ) -> Tuple[Optional[dict], Optional[str]]:
    """read_lists for one list: (its objects, None) or (None, why)."""
    objs, whys = read_lists(dataset_dir, sample, {"_": {"pairs": pairs, "flavour": flavour, "branches": branches,
                                                         "legs": legs or {}}})
    return (None, whys["_"]) if "_" in whys else (objs["_"], None)


def weights_problem(arr: dict, objs: dict) -> Optional[str]:
    """Why simulation weights are not positively proportional to genWeight within each mc/ process, or None."""
    w = arr["weight"].astype(float)
    for proc in sorted(set(objs["process"])):
        m = objs["process"] == proc
        g, wp = objs["genWeight"][m], w[m]
        nz = g != 0
        if np.any(wp[~nz] != 0):
            return f"its simulation weights are not zero where genWeight is ({proc or 'mc/'})"
        r = wp[nz] / g[nz]
        if len(r) and not (r[0] > 0 and np.allclose(r, r[0], rtol=1e-6, atol=0)):
            return f"its simulation weights are not proportional to genWeight within {proc or 'mc/'}"
    return None


def pair_wants(parsed: Dict[str, dict], users: Dict[str, List[Tuple[str, str]]], entries: Dict[str, dict]
               ) -> Tuple[Dict[str, dict], Dict[str, str]]:
    """What reading each pair list needs (flavour, selection branches, trigger legs), and {list: why} for a list
    serving two flavours."""
    wants: Dict[str, dict] = {}
    whys: Dict[str, str] = {}
    for mid, keys in sorted(users.items()):
        pm = parsed[mid]
        flavour = keys[0][1] if pm["type"] == "efficiency" else pm["flavour"]
        w = wants.setdefault(pm["list"], {"flavour": flavour, "branches": set(), "legs": {}, "pairs": pm["pairs"]})
        if w["flavour"] != flavour:
            whys[pm["list"]] = f"pair list {pm['list']!r} serves measurements of more than one object type"
        if pm["type"] == "efficiency":
            entry = entries[keys[0][0]]
            sel = entry.get("selection") if entry.get("kind") == "id" else entry.get("probe_selection")
            try:
                w["branches"].add(selection_branch(sel, flavour)[0])
            except MeasurementError:
                pass
            if entry.get("kind") == "trigger":
                w["legs"][mid] = entry
    for w in wants.values():
        w["branches"] = tuple(sorted(w["branches"]))
    return wants, whys


def read_measurement_pairs(dataset_dir, parsed: Dict[str, dict], users: Dict[str, List[Tuple[str, str]]],
                           entries: Dict[str, dict], log=None) -> Tuple[Dict[str, Dict[str, dict]], Dict[str, str]]:
    """({sample: {list: objects}}, {list: why it fails}): pair uniqueness, then the pairs against the dataset."""
    wants, whys = pair_wants(parsed, users, entries)
    objs: Dict[str, Dict[str, dict]] = {}
    for sample in SAMPLES:
        t0 = time.time()
        todo = {}
        for lst, w in wants.items():
            if lst in whys:
                continue
            arr = next(pm["pairs"][sample] for pm in parsed.values() if pm["list"] == lst)
            why = unique_pairs(arr)
            if why:
                whys[lst] = f"its {sample} pairs: {why}"
                continue
            todo[lst] = {**w, "pairs": arr}
        got, bad = read_lists(dataset_dir, sample, todo)
        whys.update({lst: f"its {sample} pairs: {why}" for lst, why in bad.items()})
        objs[sample] = got
        if log:
            log(f"read the {sample} objects of {len(todo)} pair lists ({time.time() - t0:.0f} s)")
    return objs, whys


def check_pairs_efficiency(pm: dict, mid: str, entry: dict, flavour: str, objs: Dict[str, dict]) -> Optional[str]:
    """Why an efficiency measurement's pairs are not the dataset's, or None: pass flags (the selection; for a trigger
    leg the probe selection and the leg's object, ambiguous probes either way) and simulation weights."""
    kind = entry.get("kind")
    sel_text = entry.get("selection") if kind == "id" else entry.get("probe_selection")
    try:
        branch, thr = selection_branch(sel_text, flavour)
    except MeasurementError as e:
        return str(e)
    for sample in SAMPLES:
        arr, o = pm["pairs"][sample], objs[sample][pm["list"]]
        v = o["second"][branch]
        passes_sel = (v >= thr) if thr is not None else (v != 0)
        declared = arr[f"pass_{mid}"].astype(bool)
        if kind == "id":
            truth, amb = passes_sel, np.zeros(len(declared), dtype=bool)
        else:
            if not np.all(passes_sel):
                return f"its {sample} pairs have probes that do not pass the probe selection {sel_text!r}"
            truth, amb = o["legs"][mid]
        off = (declared != truth) & ~amb
        if np.any(off):
            n_off = int(np.count_nonzero(off))
            return f"its {sample} pass flags are not the dataset's ({n_off} of {len(truth)} differ)"
        if sample == "sim":
            why = weights_problem(arr, o)
            if why:
                return why
    return None


def lineshape_objects(pm: dict, objs: Dict[str, dict]) -> Optional[str]:
    """Why a lineshape measurement's simulated pairs' weights are not the contract's, or None."""
    return weights_problem(pm["pairs"]["sim"], objs["sim"][pm["list"]])


def _objects_float(o: dict) -> dict:
    """A list's objects with the kinematics as float64 (they are read in the dataset's types)."""
    return {**o, **{r: {k: (v.astype(float) if k in ("pt", "eta", "phi") else v) for k, v in o[r].items()}
                     for r in ("first", "second")}}


def check(out_dir, dataset_dir, log: Optional[Callable[[str], None]] = None) -> Dict[str, Optional[str]]:
    """`sbh check`'s measurement check: {"<name>|<flavour>": None, or why its measurement fails} -- structure,
    recount, pairs against the dataset."""
    from . import outputs as C
    log = log or (lambda m: None)
    out_dir, dataset_dir = Path(out_dir), Path(dataset_dir)
    request = json.loads((dataset_dir / "request.json").read_text())
    entries = C.requested(request)
    wanted = [(n, fl) for n, e in sorted(entries.items()) for fl in sorted(((e or {}).get("domain") or {}))]
    spec, arrays, why = read(out_dir)
    if why:
        return {f"{n}|{fl}": why for n, fl in wanted}
    parsed, problems = parse(spec, arrays, request)
    res: Dict[str, Optional[str]] = {f"{n}|{fl}": problems.get((n, fl)) for n, fl in wanted}
    users: Dict[str, List[Tuple[str, str]]] = {}
    for key in wanted:
        if key not in problems:
            users.setdefault(spec["corrections"][key[0]][key[1]], []).append(key)
    objs, list_whys = read_measurement_pairs(dataset_dir, parsed, users, entries, log)
    for mid, keys in sorted(users.items()):
        pm = parsed[mid]
        why = list_whys.get(pm["list"])
        if not why and pm["type"] == "efficiency":
            (name, flavour), = keys
            why = recount_efficiency(pm, mid) or check_pairs_efficiency(pm, mid, entries[name], flavour, objs)
        elif not why:
            why = lineshape_objects(pm, objs) or recount_lineshape(pm, _objects_float(objs["data"][pm["list"]]))[1]
        log(f"measurement {mid} ({pm['type']}): {'OK' if not why else why}")
        for key in keys:
            res[f"{key[0]}|{key[1]}"] = f"its measurement {mid!r} {why}" if why else None
    return res
