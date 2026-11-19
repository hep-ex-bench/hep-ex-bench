"""The corrections kind's outputs (the contract's corrections.json and request.json sections): read, check and
evaluate a submission's correctionlib CorrectionSet and its RESULT.json parameter covariances.

Participant-facing (`sbh check`) and evaluator alike: the contract's rules, no truth.  Each requested correction is
checked and evaluated on its own, through correctionlib's schema (pydantic, which also turns integer bin edges into
the floats the loader requires) into a one-correction CorrectionSet, so a problem affects that correction only.
Without correctionlib (imported lazily) the checks are structural and the evaluation is NOT VALIDATED HERE (the
evaluator never waives it, check.py).
"""
from __future__ import annotations

import copy
import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...contract import read_output

FILE = "corrections.json"
MAX_BYTES = 20_000_000
MAX_PARAMS = 500
GRADER_CORRECTIONLIB = "2.9.0"
INPUTS = {"pt": "real", "eta": "real", "abs_eta": "real", "phi": "real", "charge": "int", "flavour": "string",
          "variation": "string"}
SIGNED_VARIANCE_KINDS = ("resolution",)
SOURCE_CLASSES = ("stat", "mcstat", "bkg", "tag", "model", "theory", "other")
_LABEL = r"[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*"
TOTAL_TOLERANCE = 0.10
NOT_VALIDATED_NOTE = ("correctionlib is not importable here, so the corrections were checked for structure only and "
                      "their evaluation was NOT VALIDATED HERE (every reported correction is evaluated with "
                      f"correctionlib {GRADER_CORRECTIONLIB})")


class CorrectionError(ValueError):
    """A requested correction that does not satisfy the contract; the message says why (participant-readable)."""


def import_correctionlib():
    """correctionlib, or None."""
    try:
        import correctionlib
        import correctionlib.schemav2  # noqa: F401
        return correctionlib
    except ImportError:
        return None


def _first_line(e: Exception) -> str:
    return (str(e).strip().splitlines() or [type(e).__name__])[0][:240]



def read_set(path) -> Tuple[Optional[dict], Optional[str]]:
    """(the CorrectionSet JSON, None), or (None, why the run fails): missing, not a regular file, over the size cap
    (refused before reading), not JSON, or not an object with "schema_version": 2 and a "corrections" list."""
    try:
        raw = read_output(path, max_bytes=MAX_BYTES)
    except FileNotFoundError:
        return None, f"{FILE} not produced"
    except (OSError, ValueError) as e:
        return None, f"{FILE} unreadable: {e}"
    try:
        cset = json.loads(raw)
    except ValueError as e:
        return None, f"{FILE} is not JSON: {_first_line(e)}"
    except RecursionError:
        return None, f"{FILE} is nested too deeply to be read"
    if not (isinstance(cset, dict) and cset.get("schema_version") == 2 and isinstance(cset.get("corrections"), list)):
        return None, f'{FILE} is not a JSON object with "schema_version": 2 and a "corrections" list'
    return cset, None


def result_problems(result: Any) -> List[str]:
    """The RESULT.json gate: an object whose optional "covariances" is an object (name -> matrix or {source: matrix})
    and optional "joint_covariances" a list; an invalid entry is its corrections' problem (uncertainty_spec)."""
    if not isinstance(result, dict):
        return ["RESULT.json is not a JSON object"]
    cov = result.get("covariances")
    if cov is not None and not isinstance(cov, dict):
        return ['"covariances" must be an object of correction name -> matrix (or -> {source: matrix})']
    joint = result.get("joint_covariances")
    if joint is not None and not isinstance(joint, list):
        return ['"joint_covariances" must be a list of {"source", "corrections", "matrix"} entries']
    return []


def requested(request: Any) -> Dict[str, Dict[str, Any]]:
    """The dataset's request.json entries by correction name."""
    entries = (request or {}).get("corrections") if isinstance(request, dict) else None
    return dict(entries) if isinstance(entries, dict) else {}


def find(cset: dict, name: str) -> Tuple[Optional[dict], Optional[str]]:
    """(the one correction under a requested name, None) or (None, the problem)."""
    hits = [c for c in cset.get("corrections") or [] if isinstance(c, dict) and c.get("name") == name]
    if not hits:
        return None, "missing"
    if len(hits) > 1:
        return None, f"{len(hits)} entries under this name"
    return hits[0], None



def param_slots(node, nominal: bool = False) -> List[Tuple[dict, int]]:
    """(node, index) of every formula / formularef parameter, in the contract's depth-first order: (multi)binning
    content then flow; category values then default; transform rule then content.  ``nominal``: a category on the
    variation input contributes only its "nominal" value (else its default) -- what a covariance covers.  Any JSON is
    walked iteratively (no nesting depth breaks it); other nodes hold no parameters, which count only as a list."""
    out: List[Tuple[dict, int]] = []
    stack = [node]
    while stack:
        n = stack.pop()
        if not isinstance(n, dict):
            continue
        t = n.get("nodetype")
        content = n.get("content") if isinstance(n.get("content"), list) else []
        if t in ("formula", "formularef"):
            ps = n.get("parameters")
            out.extend((n, i) for i in range(len(ps) if isinstance(ps, list) else 0))
            continue
        if t in ("binning", "multibinning"):
            kids = content + [n.get("flow")]
        elif t == "category" and nominal and n.get("input") == "variation":
            hit = [item.get("value") for item in content if isinstance(item, dict) and item.get("key") == "nominal"]
            kids = hit[:1] or [n.get("default")]
        elif t == "category":
            kids = [item.get("value") for item in content if isinstance(item, dict)] + [n.get("default")]
        elif t == "transform":
            kids = [n.get("rule"), n.get("content")]
        else:
            kids = []
        stack.extend(reversed(kids))
    return out


def numeric_slots(corr: dict, nominal: bool = False) -> List[Tuple[Any, Any]]:
    """(container, key) of every free number of a correction's data in param_slots' order and ``nominal`` rule:
    formula parameters (as a list only), (multi)binning contents and flows, numeric category values and defaults,
    and the data itself when a constant."""
    out: List[Tuple[Any, Any]] = []
    stack: List[Tuple[Any, Any]] = [(corr, "data")]
    while stack:
        cont, key = stack.pop()
        n = cont[key]
        if isinstance(n, (int, float)) and not isinstance(n, bool):
            out.append((cont, key))
            continue
        if not isinstance(n, dict):
            continue
        t = n.get("nodetype")
        content = n.get("content") if isinstance(n.get("content"), list) else []
        if t in ("formula", "formularef"):
            ps = n.get("parameters")
            out.extend((ps, i) for i in range(len(ps) if isinstance(ps, list) else 0)
                       if isinstance(ps[i], (int, float)) and not isinstance(ps[i], bool))
            continue
        kids: List[Tuple[Any, Any]] = []
        if t in ("binning", "multibinning"):
            kids = [(content, i) for i in range(len(content))] + ([(n, "flow")] if "flow" in n else [])
        elif t == "category" and nominal and n.get("input") == "variation":
            hit = [it for it in content if isinstance(it, dict) and it.get("key") == "nominal"]
            kids = [(hit[0], "value")] if hit else ([(n, "default")] if "default" in n else [])
        elif t == "category":
            kids = [(it, "value") for it in content if isinstance(it, dict)] + ([(n, "default")] if "default" in n
                                                                                 else [])
        elif t == "transform":
            kids = [(n, "rule"), (n, "content")]
        stack.extend(reversed(kids))
    return out


LEAF_NODES = ("formula", "formularef", "hashprng")


def leaves(corr: dict) -> Optional[List[Tuple[Any, Any]]]:
    """(container, key) of every leaf (formula, formularef, hashprng node or number) in depth-first order through
    (multi)binning contents and flow and category values and default; None with a transform (its rule moves every
    leaf's input below it) or any other node.  Each point is routed to one leaf by the inputs alone, so only that
    leaf's numbers move its value: the refit evaluates a number's leaf alone, at the points it reaches
    (refit._Form)."""
    out: List[Tuple[Any, Any]] = []
    stack: List[Tuple[Any, Any]] = [(corr, "data")]
    while stack:
        cont, key = stack.pop()
        n = cont[key]
        if isinstance(n, (int, float)) and not isinstance(n, bool):
            out.append((cont, key))
            continue
        if not isinstance(n, dict):
            return None
        t = n.get("nodetype")
        if t in LEAF_NODES:
            out.append((cont, key))
            continue
        content = n.get("content")
        if not isinstance(content, list):
            return None
        if t in ("binning", "multibinning"):
            flow = n.get("flow")
            kids = [(content, i) for i in range(len(content))] + (
                [(n, "flow")] if isinstance(flow, (dict, int, float)) and not isinstance(flow, bool) else [])
        elif t == "category" and all(isinstance(it, dict) for it in content):
            kids = [(it, "value") for it in content] + ([(n, "default")] if n.get("default") is not None else [])
        else:
            return None
        stack.extend(reversed(kids))
    return out


def _edge_list(e) -> Optional[List[float]]:
    """A binning's edges as numbers (a uniform binning's expanded), or None when they are not readable."""
    if isinstance(e, dict) and {"n", "low", "high"} <= set(e):
        try:
            n, lo, hi = int(e["n"]), float(e["low"]), float(e["high"])
        except (TypeError, ValueError):
            return None
        return [lo + (hi - lo) * k / n for k in range(n + 1)] if n > 0 else None
    if isinstance(e, list):
        try:
            return [float(x) for x in e]
        except (TypeError, ValueError):
            return None
    return None


def _interval(name: str, edges: Optional[List[float]], k: int) -> str:
    if edges is None or k + 1 >= len(edges):
        return f"{name} bin {k}"
    return f"{name} [{edges[k]:g}, {edges[k + 1]:g})"


def numeric_slot_labels(corr: dict, nominal: bool = False) -> List[str]:
    """numeric_slots' numbers named for a reader: "p<i>", (multi)binning content by its intervals ("pt [20, 30) ·
    abs_eta [0, 0.9)"), flow by input, category value by input and key, the data "value"; nesting joined by " / "."""
    def join(a, b):
        return f"{a} / {b}" if a and b else (a or b)
    out: List[str] = []
    stack: List[Tuple[Any, Any, str]] = [(corr, "data", "")]
    while stack:
        cont, key, path = stack.pop()
        n = cont[key]
        if isinstance(n, (int, float)) and not isinstance(n, bool):
            out.append(path or "value")
            continue
        if not isinstance(n, dict):
            continue
        t = n.get("nodetype")
        content = n.get("content") if isinstance(n.get("content"), list) else []
        if t in ("formula", "formularef"):
            ps = n.get("parameters")
            out.extend(join(path, f"p{i}") for i in range(len(ps) if isinstance(ps, list) else 0)
                       if isinstance(ps[i], (int, float)) and not isinstance(ps[i], bool))
            continue
        kids: List[Tuple[Any, Any, str]] = []
        if t == "binning":
            inp, edges = str(n.get("input", "x")), _edge_list(n.get("edges"))
            kids = [(content, i, join(path, _interval(inp, edges, i))) for i in range(len(content))] + (
                [(n, "flow", join(path, f"{inp} flow"))] if "flow" in n else [])
        elif t == "multibinning":
            inps = [str(x) for x in n.get("inputs") or []]
            edges = [_edge_list(x) for x in n.get("edges") or []]
            dims = [len(e) - 1 if e else 0 for e in edges]

            def label(i):
                if not dims or not all(dims) or len(inps) != len(dims):
                    return f"bin {i}"
                idx, r = [], i
                for d in reversed(dims):
                    idx.append(r % d)
                    r //= d
                return " · ".join(_interval(inps[k], edges[k], j) for k, j in enumerate(reversed(idx)))
            kids = [(content, i, join(path, label(i))) for i in range(len(content))] + (
                [(n, "flow", join(path, "flow"))] if "flow" in n else [])
        elif t == "category" and nominal and n.get("input") == "variation":
            hit = [it for it in content if isinstance(it, dict) and it.get("key") == "nominal"]
            kids = [(hit[0], "value", path)] if hit else ([(n, "default", path)] if "default" in n else [])
        elif t == "category":
            inp = str(n.get("input", "x"))
            kids = [(it, "value", join(path, f"{inp} {it.get('key')}")) for it in content if isinstance(it, dict)] + (
                [(n, "default", join(path, f"{inp} default"))] if "default" in n else [])
        elif t == "transform":
            kids = [(n, "rule", join(path, "rule")), (n, "content", path)]
        stack.extend(reversed(kids))
    return out


def variation_keys(node) -> set:
    """The values of every category on the variation input in a data tree."""
    keys, stack = set(), [node]
    while stack:
        n = stack.pop()
        if isinstance(n, dict):
            if n.get("nodetype") == "category" and n.get("input") == "variation":
                keys |= {item.get("key") for item in n.get("content") or [] if isinstance(item, dict)}
            stack.extend(v for v in n.values() if isinstance(v, (dict, list)))
        elif isinstance(n, list):
            stack.extend(n)
    return {k for k in keys if isinstance(k, str)}


def source_class(name: Any) -> Optional[str]:
    """The class of an uncertainty source's name (`<class>` or `<class>_<label>`), or None outside the rules."""
    if isinstance(name, str):
        for c in SOURCE_CLASSES:
            if name == c or re.fullmatch(rf"{c}_{_LABEL}", name):
                return c
    return None


def _input_names(corr: dict) -> List[str]:
    inputs = corr.get("inputs")
    return [v.get("name") for v in inputs if isinstance(v, dict)] if isinstance(inputs, list) else []


def _real(x) -> bool:
    """A JSON number (never a boolean) that a float can hold."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return False
    try:
        float(x)
    except OverflowError:
        return False
    return True


def _matrix_problem(m: Any, p: int) -> Optional[str]:
    """Why m is not the symmetric, positive semi-definite p x p matrix of numbers a covariance must be, or None."""
    if not 1 <= p <= MAX_PARAMS:
        return f"would cover {p} formula parameters; a covariance covers 1 to {MAX_PARAMS}"
    if not (isinstance(m, list) and len(m) == p and all(isinstance(r, list) and len(r) == p for r in m)):
        return f"is not a {p} x {p} matrix for its {p} formula parameters"
    if not all(_real(x) for row in m for x in row):
        return "has entries that are not numbers"
    c = np.asarray(m, dtype=float)
    scale = max(float(np.abs(c).max()) if c.size else 0.0, 1e-300)
    if not np.all(np.isfinite(c)):
        return "has non-finite entries"
    if not np.allclose(c, c.T, rtol=1e-6, atol=1e-12 * scale):
        return "is not symmetric"
    if float(np.linalg.eigvalsh(0.5 * (c + c.T)).min()) < -1e-9 * scale:
        return "is not positive semi-definite"
    return None


def _n_params(corr: dict) -> int:
    """The number of parameters a covariance of the correction covers (its nominal ones with a variation input)."""
    return len(param_slots(corr.get("data"), nominal="variation" in _input_names(corr)))


def _joint_block(blk: Any, corr: dict, cset: Optional[dict]) -> Tuple[Optional[str], Optional[tuple]]:
    """(None, (source, the block for corr's own parameters)) or (why, None): a joint covariance names a source class,
    lists distinct corrections present once in the set, and covers their parameters concatenated in listed order."""
    if not isinstance(blk, dict):
        return "is not an object", None
    src, listed = blk.get("source"), blk.get("corrections")
    if source_class(src) is None:
        return f"has source {src!r}, not a class of {list(SOURCE_CLASSES)} (or <class>_<label>)", None
    if not (isinstance(listed, list) and all(isinstance(x, str) for x in listed) and len(set(listed)) == len(listed)):
        return "does not list its corrections as distinct names", None
    if cset is None:
        return "cannot be checked without the correction set", None
    sizes = []
    for nm in listed:
        c, why = find(cset, nm)
        if c is None:
            return f"lists {nm!r}, which is {why}", None
        sizes.append(_n_params(c))
    why = _matrix_problem(blk.get("matrix"), sum(sizes))
    if why:
        return f"matrix {why}", None
    i = listed.index(corr.get("name"))
    off, k = sum(sizes[:i]), sizes[i]
    return None, (src, np.asarray(blk["matrix"], dtype=float)[off:off + k, off:off + k])


def uncertainty_spec(corr: dict, result: Any, cset: Optional[dict] = None) -> Tuple[dict, List[str]]:
    """(components, problems): {"band": has the total band, "sources": varied (up_/down_<source>), "covariances":
    {source: matrix}, "joint": [(source, own block)]}.  A source appears in at most one component; a matrix without
    one is stat."""
    spec: Dict[str, Any] = {"band": False, "sources": [], "covariances": {}, "joint": []}
    problems, seen = [], []
    if "variation" in _input_names(corr):
        keys = variation_keys(corr.get("data"))
        if "nominal" not in keys:
            problems.append('its variation input has no "nominal" value')
        if ("up" in keys) != ("down" in keys):
            problems.append('has one of "up" and "down" without the other')
        spec["band"] = "up" in keys and "down" in keys
        ups = {k[3:] for k in keys if k.startswith("up_")}
        downs = {k[5:] for k in keys if k.startswith("down_")}
        problems += [f'has "up_{x}" or "down_{x}" without the other' for x in sorted(ups ^ downs)]
        for x in sorted(ups & downs):
            if source_class(x) is None:
                problems.append(f"varies source {x!r}, not a class of {list(SOURCE_CLASSES)} (or <class>_<label>)")
            else:
                spec["sources"].append(x)
                seen.append(x)
        other = sorted(keys - {"nominal", "up", "down"} - {f"up_{x}" for x in ups} - {f"down_{x}" for x in downs})
        if other:
            problems.append(f"its variation input takes values outside the contract's: {other}")
    cov = result.get("covariances") if isinstance(result, dict) else None
    entry = cov.get(corr.get("name")) if isinstance(cov, dict) else None
    if entry is not None:
        p = _n_params(corr)
        for src, m in (entry.items() if isinstance(entry, dict) else [("stat", entry)]):
            why = (None if source_class(src) else f"names source {src!r}, not a class of {list(SOURCE_CLASSES)}") \
                or _matrix_problem(m, p)
            if why:
                problems.append(f"its covariance ({src}) {why}")
            else:
                spec["covariances"][src] = m
                seen.append(src)
    joint = result.get("joint_covariances") if isinstance(result, dict) else None
    for i, blk in enumerate(joint if isinstance(joint, list) else []):
        if isinstance(blk, dict) and isinstance(blk.get("corrections"), list) \
                and corr.get("name") in blk["corrections"]:
            why, block = _joint_block(blk, corr, cset)
            if why:
                problems.append(f"joint covariance {i} {why}")
            else:
                spec["joint"].append(block)
                seen.append(block[0])
    dup = sorted({x for x in seen if seen.count(x) > 1})
    if dup:
        problems.append(f"names a source in more than one of its components: {dup}")
    if not (spec["band"] or spec["sources"] or spec["covariances"] or spec["joint"]):
        problems.append("states no uncertainty component (no variation, no covariance in RESULT.json)")
    return spec, problems


def structure_problems(corr: dict, result: Any, cset: Optional[dict] = None) -> List[str]:
    """Structural rules (no correctionlib needed): input set and types, uncertainty components."""
    inputs = corr.get("inputs")
    if not isinstance(inputs, list) or not all(isinstance(v, dict) for v in inputs):
        return ["has no inputs list"]
    problems = []
    for v in inputs:
        n, t = v.get("name"), v.get("type")
        if not isinstance(n, str) or n not in INPUTS:
            problems.append(f"input {n!r} is outside the contract's set {sorted(INPUTS)}")
        elif t != INPUTS[n]:
            problems.append(f"input {n!r} has type {t!r}, not {INPUTS[n]!r}")
    names = [n for n in _input_names(corr) if isinstance(n, str)]
    if len(set(names)) != len(names):
        problems.append("declares an input twice")
    return problems + uncertainty_spec(corr, result, cset)[1]



def normalized(corr: dict) -> dict:
    """The correction as correctionlib's schema writes it (schema types, unset fields left out).  Raises
    CorrectionError."""
    if import_correctionlib() is None:
        raise CorrectionError(NOT_VALIDATED_NOTE)
    from correctionlib import schemav2
    try:
        model = schemav2.Correction.model_validate(corr)
    except Exception as e:  # noqa: BLE001
        raise CorrectionError(f"is not a valid correctionlib Correction: {_first_line(e)}") from None
    return json.loads(model.model_dump_json(exclude_unset=True))


def evaluator(corr: dict, checked: bool = True):
    """correctionlib's evaluator of one correction through normalized(), or with ``checked`` False loaded as is: for
    a normalized correction whose numbers alone changed, or one built of its nodes (the refit, where the check costs
    more than loading: 1.9 s for a 5.8 MB correction).  Raises CorrectionError."""
    cl = import_correctionlib()
    if cl is None:
        raise CorrectionError(NOT_VALIDATED_NOTE)
    if checked:
        corr = normalized(corr)
    try:
        return cl.CorrectionSet.from_string(json.dumps({"schema_version": 2, "corrections": [corr]}))[corr["name"]]
    except Exception as e:  # noqa: BLE001
        raise CorrectionError(f"cannot be loaded by correctionlib: {_first_line(e)}") from None


def _arguments(corr: dict, points: Dict[str, np.ndarray], flavour: str, variation: Optional[str]) -> list:
    """The correction's inputs at the points, in its order (an int input other than charge: refit._Form.moved's
    leaf selector)."""
    args = []
    inputs = corr.get("inputs")
    for v in (inputs if isinstance(inputs, list) else []):
        if not isinstance(v, dict):
            continue
        n = v.get("name")
        if n == "flavour":
            args.append(flavour)
        elif n == "variation":
            args.append(variation)
        elif n == "abs_eta":
            args.append(np.abs(np.asarray(points["eta"], dtype=float)))
        elif n == "charge" or v.get("type") == "int":
            args.append(np.asarray(points[n], dtype=np.int64))
        else:
            args.append(np.asarray(points[n], dtype=float))
    return args


def _run(ev, corr, points, flavour, variation=None) -> np.ndarray:
    n = len(points["pt"])
    try:
        out = np.asarray(ev.evaluate(*_arguments(corr, points, flavour, variation)), dtype=float)
    except Exception as e:  # noqa: BLE001
        what = f" for variation {variation!r}" if variation else ""
        raise CorrectionError(f"does not evaluate for {flavour}{what}: {_first_line(e)}") from None
    out = np.array(np.broadcast_to(out, (n,)), dtype=float)
    bad = int(np.count_nonzero(~np.isfinite(out)))
    if bad:
        raise CorrectionError(f"evaluates to a non-finite value at {bad} of {n} points ({flavour})")
    return out


def _gradient(corr: dict, points: Dict[str, np.ndarray], flavour: str, variation: Optional[str],
              sigma: np.ndarray) -> np.ndarray:
    """d correction / d parameter at the points by central differences over the covariance's parameters: steps of
    1e-3 sigma, at least the parameter's float resolution, at most 1 % of its size (no overflow).  Raises
    CorrectionError."""
    try:
        work = copy.deepcopy(corr)
    except RecursionError:
        raise CorrectionError("is nested too deeply to be evaluated") from None
    slots = param_slots(work["data"], nominal=variation is not None)
    grad = np.zeros((len(slots), len(points["pt"])))
    for j, (node, i) in enumerate(slots):
        if not sigma[j] > 0.0:
            continue
        p0 = float(node["parameters"][i])
        h = min(max(1e-3 * float(sigma[j]), 1e-9 * abs(p0)), 1e-2 * (abs(p0) + 1.0))
        node["parameters"][i] = p0 + h
        hi = _run(evaluator(work), work, points, flavour, variation)
        node["parameters"][i] = p0 - h
        lo = _run(evaluator(work), work, points, flavour, variation)
        node["parameters"][i] = p0
        grad[j] = (hi - lo) / (2.0 * h)
    return grad


def components(corr: dict, result: Any, points: Dict[str, np.ndarray], flavour: str, cset: Optional[dict] = None,
               signed: bool = False) -> Dict[str, Any]:
    """A correction at the points under the contract's uncertainty model: {"value", "lo", "hi" (band ends, or
    value -/+ total), "total", "parts": {source: 1 sigma >= 0}, "shifts": {varied source: signed shift},
    "consistent": parts' quadrature sum within TOTAL_TOLERANCE of the band (True without one)}.  ``signed`` (a
    resolution): total, parts, shifts and consistent in the comparison space w|w|; value and band ends as evaluated.
    Raises CorrectionError (structure first)."""
    problems = structure_problems(corr, result, cset)
    if problems:
        raise CorrectionError("; ".join(problems))
    spec, _ = uncertainty_spec(corr, result, cset)
    ev = evaluator(corr)
    var = "nominal" if "variation" in _input_names(corr) else None
    f = _run(ev, corr, points, flavour, var)

    def space(x):
        return x * np.abs(x) if signed else x
    parts: Dict[str, np.ndarray] = {}
    as_evaluated: Dict[str, np.ndarray] = {}
    shifts: Dict[str, np.ndarray] = {}
    for src in spec["sources"]:
        up, down = (_run(ev, corr, points, flavour, f"{v}_{src}") for v in ("up", "down"))
        shifts[src] = 0.5 * (space(up) - space(down))
        parts[src] = np.abs(shifts[src])
        as_evaluated[src] = 0.5 * np.abs(up - down)
    mats = [(src, np.asarray(m, dtype=float)) for src, m in spec["covariances"].items()] + list(spec["joint"])
    if mats:
        sigma = np.sqrt(np.maximum(np.max([np.diag(c) for _, c in mats], axis=0), 0.0))
        grad = _gradient(corr, points, flavour, var, sigma)
        for src, c in mats:
            as_evaluated[src] = np.sqrt(np.maximum(np.einsum("ik,ij,jk->k", grad, c, grad), 0.0))
            parts[src] = as_evaluated[src] * 2.0 * np.abs(f) if signed else as_evaluated[src]
    if not all(np.all(np.isfinite(v)) for v in parts.values()):
        raise CorrectionError(f"its propagated uncertainty is not finite ({flavour})")
    quad = np.sqrt(np.sum([v ** 2 for v in parts.values()], axis=0)) if parts else None
    if spec["band"]:
        up, down = (_run(ev, corr, points, flavour, v) for v in ("up", "down"))
        lo, hi = np.minimum(up, down), np.maximum(up, down)
        total = 0.5 * np.abs(space(hi) - space(lo))
        consistent = quad is None or bool(np.all(np.abs(quad - total) <= TOTAL_TOLERANCE * total
                                                 + 1e-12 * np.maximum(np.abs(space(f)), 1e-300)))
    else:
        q = np.sqrt(np.sum([v ** 2 for v in as_evaluated.values()], axis=0)) if as_evaluated else np.zeros_like(f)
        lo, hi, total, consistent = f - q, f + q, quad, True
    return {"value": f, "lo": lo, "hi": hi, "total": total, "parts": parts, "shifts": shifts, "consistent": consistent}


def evaluate(corr: dict, result: Any, points: Dict[str, np.ndarray], flavour: str, cset: Optional[dict] = None
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(value, lo, hi) of components().  Raises CorrectionError."""
    c = components(corr, result, points, flavour, cset)
    return c["value"], c["lo"], c["hi"]



def domain_grid(domain: Dict[str, float], n_pt: int = 24, n_eta: int = 25) -> Dict[str, np.ndarray]:
    """Points spanning a correction's domain for one flavour (the self-check): pT from just above pt_min to 1e4 GeV,
    |eta| up to just below abs_eta_max, every phi, both charges."""
    pt_min, eta_max = float(domain["pt_min"]), float(domain["abs_eta_max"])
    pt = np.geomspace(pt_min * (1.0 + 1e-6) + 1e-6, 1e4, n_pt)
    eta = np.linspace(-eta_max, eta_max, n_eta) * (1.0 - 1e-6)
    phi = np.linspace(-math.pi, math.pi, 9)
    g = np.meshgrid(pt, eta, phi, np.array([-1, 1]), indexing="ij")
    return {"pt": g[0].ravel(), "eta": g[1].ravel(), "phi": g[2].ravel(), "charge": g[3].ravel().astype(np.int64)}


def signed_variance(f: np.ndarray, lo: np.ndarray, hi: np.ndarray, flavour: str) -> Tuple[np.ndarray, np.ndarray]:
    """A resolution's comparison: value and band ends mapped through value x |value|, (V, sigma_V = half the mapped
    ends' distance).  Raises CorrectionError on overflow (a width beyond ~1e154)."""
    with np.errstate(over="ignore", invalid="ignore"):
        v, v_lo, v_hi = f * np.abs(f), lo * np.abs(lo), hi * np.abs(hi)
        sigma = 0.5 * np.abs(v_hi - v_lo)
    if not (np.all(np.isfinite(v)) and np.all(np.isfinite(sigma))):
        raise CorrectionError(f"its extra variance, value x |value|, or that of its uncertainty band is not finite "
                              f"({flavour})")
    return v, sigma


def flavour_problems(corr: dict, result: Any, domains: Any, signed: bool = False, cset: Optional[dict] = None
                     ) -> Dict[str, List[str]]:
    """{flavour: problems} per requested domain (empty: satisfies the contract there, else counts as not reported):
    structure, then with correctionlib evaluation, value and uncertainty (``signed``: also the extra variance) on
    the domain grid."""
    structure = structure_problems(corr, result, cset)
    out = {}
    for flavour, dom in sorted((domains or {}).items()):
        problems = list(structure)
        if not problems and import_correctionlib() is not None:
            try:
                f, lo, hi = evaluate(corr, result, domain_grid(dom), flavour, cset)
                if signed:
                    signed_variance(f, lo, hi, flavour)
            except CorrectionError as e:
                problems.append(str(e))
        out[flavour] = problems
    return out


def check_request(cset: dict, result: Any, request: Any) -> Tuple[Dict[str, List[str]], List[str]]:
    """({name: problems} per requested correction, notes): flavour_problems; without correctionlib structural only,
    with NOT_VALIDATED_NOTE."""
    have = import_correctionlib() is not None
    out: Dict[str, List[str]] = {}
    for name, entry in sorted(requested(request).items()):
        corr, why = find(cset, name)
        if corr is None:
            out[name] = [why]
            continue
        per = flavour_problems(corr, result, (entry or {}).get("domain"),
                               signed=(entry or {}).get("kind") in SIGNED_VARIANCE_KINDS, cset=cset)
        structure = structure_problems(corr, result, cset)
        out[name] = structure or [p for f in sorted(per) for p in per[f]]
    return out, ([] if have else [NOT_VALIDATED_NOTE])


def grid_values(cset: dict, result: Any, request: Any) -> Dict[str, np.ndarray]:
    """{"name:flavour": (value, lo, hi) on the domain grid} for the determinism check; unevaluable ones left out."""
    out = {}
    for name, entry in sorted(requested(request).items()):
        corr, _ = find(cset, name)
        if corr is None:
            continue
        for flavour, dom in sorted(((entry or {}).get("domain") or {}).items()):
            try:
                out[f"{name}:{flavour}"] = np.stack(evaluate(corr, result, domain_grid(dom), flavour, cset))
            except CorrectionError:
                continue
    return out
