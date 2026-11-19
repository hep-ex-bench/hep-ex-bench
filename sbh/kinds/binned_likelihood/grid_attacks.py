"""Grid-content attackers: pure JSON post-processing of a finished output directory (MODEL.json + MODEL_GRID/).

Each attack maps an honest out/ to the same out/ with one lever pulled on the delivered mass grid, a lever the
marginal mu rule (METRICS.md, EVAL_CONTRACT section 3) must catch or price: signal templates scaled in part of the
grid (wing_inflator, grid_far, grid_allbut), narrowed at a fixed yield (template_sharpen), one template behind every
mass label (constant_grid), a grid that does not cover the range (end_cut, local_gridder, grid_less), a dead point
(dead_point), and declared POI bounds that exclude values (bound_mover, hi_b_clamp).  Nothing here fits anything: "the minimum" is MODEL.json's own mass (the
contract puts MODEL.json at the grid's in-range global minimum) and "own Z_max" is RESULT.json's significance_obs,
which is all a post-processor of the reference's outputs knows.

Stdlib only, on purpose: sbh/attackgen.py copies this file into a generated attacker as grid_attack.py, and the
attacker's run.sh runs it on the output directory inside the sandbox after the reference pipeline, where sbh is not
importable.  The evaluator side imports it as a module (apply / apply_copy).

    python grid_attack.py <attack> <out_dir> [key=value ...]      e.g.  wing_inflator /out c=3

A report of what changed is written to <out_dir>/GRID_ATTACK.json.
"""
from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

REPORT = "GRID_ATTACK.json"
RANGE = (110.0, 140.0)
FAMILIES = {
    "wing_inflator": "scales the signal templates of the grid cells outside 117-134 GeV by c (a tilt of the units "
                     "of mu towards the insensitive wings)",
    "grid_far": "scales the signal templates of the grid cells more than 3 GeV from MODEL.json's mass by c",
    "grid_allbut": "scales the signal templates of every grid cell but MODEL.json's own by c",
    "constant_grid": "every MODEL_GRID file holds the template of one mass behind its own label",
    "end_cut": "drops the grid cells below 116 GeV (the insensitive end of the range)",
    "local_gridder": "keeps only the grid cells within 3 GeV of MODEL.json's mass",
    "grid_less": "delivers no MODEL_GRID at all",
    "dead_point": "one grid cell's signal template scaled by 1e-5 (a point that can see nothing)",
    "bound_mover": "declares the POI lower bound lo_b = 0.2 in MODEL.json and every grid cell (mu = 0 impossible)",
    "hi_b_clamp": "declares the POI upper bound hi_b = 0.001 wherever its own significance is below 3.5 "
                  "(a null pin by bounds, conditional on its own data)",
    "template_sharpen": "narrows the signal template of every grid cell but MODEL.json's own at a fixed yield (each "
                        "bin raised to the power p and renormalised): more sensitivity per cell, and next to the "
                        "best fit neighbours that see less of a fluctuation",
}


def poi_name(spec):
    """The measurement's POI (the first measurement's config, pyhf's default), "mu" when absent."""
    ms = spec.get("measurements") or []
    cfg = ms[0].get("config") if ms and isinstance(ms[0], dict) else None
    return (cfg or {}).get("poi") or "mu"


def is_poi_sample(sample, poi):
    return any(m.get("type") == "normfactor" and m.get("name") == poi for m in sample.get("modifiers") or [])


def poi_samples(spec):
    """[(channel name, sample)] of every POI-scaled sample."""
    poi = poi_name(spec)
    return [(c.get("name"), s) for c in spec.get("channels") or [] for s in c.get("samples") or []
            if is_poi_sample(s, poi)]


def signal_yield(spec):
    """S: the summed nominal of every POI-scaled sample at mu = 1, all channels (INTERFACE yield_mu1)."""
    return float(sum(sum(float(x) for x in s.get("data") or []) for _, s in poi_samples(spec)))


def signal_centroid(spec):
    """The signal template's mass centroid (GeV) from _meta.binning edges over the channels with a POI-scaled
    sample whose binning is given; None without binning or signal."""
    binning = (spec.get("_meta") or {}).get("binning") or {}
    num = den = 0.0
    for ch, s in poi_samples(spec):
        edges = (binning.get(ch) or {}).get("edges")
        data = s.get("data") or []
        if not edges or len(edges) != len(data) + 1:
            continue
        for lo, hi, y in zip(edges[:-1], edges[1:], data):
            num += float(y) * 0.5 * (float(lo) + float(hi))
            den += float(y)
    return num / den if den > 0 else None


def poi_bounds(spec):
    """[lo_b, hi_b] of the POI from the measurement's parameters; pyhf's normfactor default [0, 10] when unset."""
    poi = poi_name(spec)
    for p in _measurement_pars(spec):
        if p.get("name") == poi and p.get("bounds"):
            return [float(p["bounds"][0][0]), float(p["bounds"][0][1])]
    return [0.0, 10.0]


def _measurement_pars(spec):
    ms = spec.get("measurements") or []
    cfg = ms[0].setdefault("config", {}) if ms and isinstance(ms[0], dict) else {}
    return cfg.setdefault("parameters", []) if isinstance(cfg, dict) else []


def _scale_sample(sample, c):
    """Scale one sample's nominal and its absolute-valued modifier data by c (histosys hi/lo templates, staterror
    and shapesys absolute uncertainties); ratio-valued modifiers (normsys, normfactor, lumi) are left alone."""
    sample["data"] = [float(x) * c for x in sample.get("data") or []]
    for m in sample.get("modifiers") or []:
        if m.get("type") == "histosys" and isinstance(m.get("data"), dict):
            for k in ("hi_data", "lo_data"):
                if m["data"].get(k) is not None:
                    m["data"][k] = [float(x) * c for x in m["data"][k]]
        elif m.get("type") in ("staterror", "shapesys") and isinstance(m.get("data"), list):
            m["data"] = [float(x) * c for x in m["data"]]


def scale_signal(spec, c):
    """Scale every POI-scaled sample of `spec` by c in place; returns how many samples it touched."""
    n = 0
    for _, s in poi_samples(spec):
        _scale_sample(s, c)
        n += 1
    return n


def set_poi_bounds(spec, lo=None, hi=None):
    """Set the POI's declared bounds (one or both ends) in the measurement, adding the parameter entry when the
    workspace relies on pyhf's defaults; the init is clamped into the new bounds so the workspace still fits."""
    poi = poi_name(spec)
    pars = _measurement_pars(spec)
    p = next((p for p in pars if p.get("name") == poi), None)
    if p is None:
        p = {"name": poi, "bounds": [[0.0, 10.0]], "inits": [1.0]}
        pars.append(p)
    b = p.setdefault("bounds", [[0.0, 10.0]])
    b[0] = [float(lo) if lo is not None else float(b[0][0]), float(hi) if hi is not None else float(b[0][1])]
    inits = p.get("inits") or [1.0]
    p["inits"] = [min(max(float(inits[0]), b[0][0]), b[0][1])]
    return b[0]


class OutDir:
    """A finished out/: MODEL.json, MODEL_GRID/mH_<m>.json, RESULT.json, read lazily, written compactly."""

    def __init__(self, path):
        self.path = Path(path)
        self.changed, self.removed, self.notes = [], [], []
        self._model = None

    def read(self, rel):
        return json.loads((self.path / rel).read_text())

    def write(self, rel, spec):
        (self.path / rel).write_text(json.dumps(spec) + "\n", encoding="utf-8")
        self.changed.append(str(rel))

    @property
    def model(self):
        if self._model is None:
            self._model = self.read("MODEL.json")
        return self._model

    def grid_files(self):
        """Sorted [(mass, relative path)] of MODEL_GRID/mH_<m>.json; the single-file MODEL_GRID.json form is not an
        output these attackers know how to edit."""
        gd = self.path / "MODEL_GRID"
        out = []
        if gd.is_dir():
            for f in sorted(gd.glob("mH_*.json")):
                try:
                    out.append((float(f.stem[3:]), f"MODEL_GRID/{f.name}"))
                except ValueError:
                    continue
        if not out and (self.path / "MODEL_GRID.json").exists():
            raise SystemExit("grid attack: MODEL_GRID.json (single-file grid) is not supported; a MODEL_GRID/ "
                             "directory is")
        return sorted(out)

    def model_mass(self):
        """MODEL.json's mass: _meta.mH_fixed_GeV, else RESULT.json's mH."""
        m = (self.model.get("_meta") or {}).get("mH_fixed_GeV")
        if m is None:
            try:
                res = self.read("RESULT.json")
                m = next(p.get("value") for p in res.get("pois") or [] if p.get("name") == "mH")
            except (OSError, StopIteration, ValueError):
                raise SystemExit("grid attack: MODEL.json has no _meta.mH_fixed_GeV and RESULT.json no mH")
        return float(m)

    def own_index(self, files):
        """Index of the grid file nearest MODEL.json's mass (the cell the attackers never touch, so the grid stays
        MODEL.json's model at that point and the link holds); None without a grid."""
        if not files:
            return None
        m0 = self.model_mass()
        return min(range(len(files)), key=lambda i: abs(files[i][0] - m0))

    def own_significance(self):
        try:
            z = self.read("RESULT.json").get("significance_obs")
        except OSError:
            return None
        return float(z) if z is not None else None


def _scale_cells(out, c, keep):
    """Scale the signal templates of every grid cell for which keep(mass, index) is False; MODEL.json and its own
    cell are never touched (the attack is on the marginal's weights, not on MODEL.json's fit)."""
    files = out.grid_files()
    own = out.own_index(files)
    n = 0
    for i, (m, rel) in enumerate(files):
        if i == own or keep(m, i):
            continue
        spec = out.read(rel)
        scale_signal(spec, c)
        out.write(rel, spec)
        n += 1
    out.notes.append(f"scaled {n} of {len(files)} grid cells by {c:g}; own cell "
                     f"{files[own][0] if own is not None else None} untouched")


def _sharpen(values, p):
    """A template raised to the power p bin by bin and renormalised to its own sum (same yield, same peak, narrower
    for p > 1); an all-zero template is returned as it is."""
    a = [max(float(x), 0.0) for x in values or []]
    tot = sum(float(x) for x in values or [])
    b = [x ** p for x in a]
    sb = sum(b)
    return [x * tot / sb for x in b] if sb > 0 else [float(x) for x in values or []]


def sharpen_signal(spec, p):
    """Sharpen every POI-scaled sample's nominal and histosys templates of `spec` in place (_sharpen); returns how
    many samples it touched."""
    n = 0
    for _, s in poi_samples(spec):
        s["data"] = _sharpen(s.get("data"), p)
        for m in s.get("modifiers") or []:
            if m.get("type") == "histosys" and isinstance(m.get("data"), dict):
                for k in ("hi_data", "lo_data"):
                    if m["data"].get(k) is not None:
                        m["data"][k] = _sharpen(m["data"][k], p)
        n += 1
    return n


def wing_inflator(out, c=3.0, lo=117.0, hi=134.0):
    """Signal templates x c on every cell outside [lo, hi] GeV."""
    _scale_cells(out, float(c), lambda m, i: float(lo) <= m <= float(hi))


def grid_far(out, c=3.0, dist=3.0):
    """Signal templates x c on every cell more than `dist` GeV from MODEL.json's mass."""
    m0 = out.model_mass()
    _scale_cells(out, float(c), lambda m, i: abs(m - m0) <= float(dist))


def grid_allbut(out, c=1.2):
    """Signal templates x c on every cell but MODEL.json's own."""
    _scale_cells(out, float(c), lambda m, i: False)


def constant_grid(out, mass=None):
    """Every grid file carries the channels (hence the signal template) of one mass: MODEL.json's own by default,
    else the grid file nearest `mass`.  The label (file name, _meta.mH_fixed_GeV) keeps claiming the file's mass."""
    files = out.grid_files()
    if not files:
        out.notes.append("no MODEL_GRID: nothing to replace")
        return
    if mass is None:
        src, what = out.model, f"MODEL.json ({out.model_mass():g} GeV)"
    else:
        j = min(range(len(files)), key=lambda i: abs(files[i][0] - float(mass)))
        src, what = out.read(files[j][1]), f"{files[j][1]} ({files[j][0]:g} GeV)"
    channels = src.get("channels") or []
    for m, rel in files:
        spec = out.read(rel)
        spec["channels"] = copy.deepcopy(channels)
        if isinstance(spec.get("_meta"), dict) and "mH_fixed_GeV" in spec["_meta"]:
            spec["_meta"]["mH_fixed_GeV"] = m
        out.write(rel, spec)
    out.notes.append(f"{len(files)} grid files now hold the template of {what}")


def _drop_cells(out, drop):
    files = out.grid_files()
    own = out.own_index(files)
    for i, (m, rel) in enumerate(files):
        if i != own and drop(m):
            (out.path / rel).unlink()
            out.removed.append(rel)
    out.notes.append(f"removed {len(out.removed)} of {len(files)} grid cells")


def end_cut(out, cut=116.0):
    """The grid stops at `cut` GeV: every cell below it goes (MODEL.json's own cell stays)."""
    _drop_cells(out, lambda m: m < float(cut) - 1e-9)


def local_gridder(out, half=3.0):
    """Only the cells within +-half GeV of MODEL.json's mass stay."""
    m0 = out.model_mass()
    _drop_cells(out, lambda m: abs(m - m0) > float(half) + 1e-9)


def grid_less(out):
    """No MODEL_GRID at all."""
    for name in ("MODEL_GRID", "MODEL_GRID.json"):
        p = out.path / name
        if p.is_dir():
            shutil.rmtree(p)
            out.removed.append(name + "/")
        elif p.exists():
            p.unlink()
            out.removed.append(name)
    out.notes.append("MODEL_GRID removed" if out.removed else "no MODEL_GRID to remove")


def dead_point(out, mass=None, factor=1e-5, lo=RANGE[0], hi=RANGE[1]):
    """One grid cell's signal template x factor: the cell nearest `mass`, else the in-range cell farthest from
    MODEL.json's mass (never its own)."""
    files = out.grid_files()
    own = out.own_index(files)
    cands = [i for i in range(len(files)) if i != own and float(lo) <= files[i][0] <= float(hi)]
    if not cands:
        out.notes.append("no grid cell to kill")
        return
    m0 = out.model_mass()
    j = (min(cands, key=lambda i: abs(files[i][0] - float(mass))) if mass is not None
         else max(cands, key=lambda i: abs(files[i][0] - m0)))
    spec = out.read(files[j][1])
    scale_signal(spec, float(factor))
    out.write(files[j][1], spec)
    out.notes.append(f"cell {files[j][0]:g} GeV scaled by {float(factor):g}")


def _rebound(out, lo=None, hi=None):
    for rel in ["MODEL.json"] + [rel for _, rel in out.grid_files()]:
        spec = out.model if rel == "MODEL.json" else out.read(rel)
        b = set_poi_bounds(spec, lo=lo, hi=hi)
        out.write(rel, spec)
    out.notes.append(f"POI bounds now {b} in MODEL.json and {len(out.changed) - 1} grid cells")


def bound_mover(out, lo=0.2):
    """lo_b = `lo` on the POI everywhere: the model declares mu below it impossible."""
    _rebound(out, lo=float(lo))


def hi_b_clamp(out, hi=0.001, zmax=3.5):
    """hi_b = `hi` on the POI everywhere, but only when RESULT.json's significance_obs is below `zmax` (the
    attacker pins the null where its own data look like one); otherwise the outputs stay honest."""
    z = out.own_significance()
    if z is None or z >= float(zmax):
        out.notes.append(f"not applied: own significance {z} is not below {float(zmax):g}")
        return
    _rebound(out, hi=float(hi))
    out.notes.append(f"applied: own significance {z:.2f} < {float(zmax):g}")


def template_sharpen(out, p=2.0, dist=0.0):
    """Every cell farther than `dist` GeV from MODEL.json's mass (MODEL.json's own cell never) gets its signal
    templates sharpened to the power p at a fixed yield: the per-cell sensitivity / resolution lever the yield checks
    cannot see (the template-shape checks must)."""
    files = out.grid_files()
    own = out.own_index(files)
    m0 = files[own][0] if own is not None else out.model_mass()
    n = 0
    for i, (m, rel) in enumerate(files):
        if i == own or (m0 is not None and abs(m - m0) <= float(dist)):
            continue
        spec = out.read(rel)
        sharpen_signal(spec, float(p))
        out.write(rel, spec)
        n += 1
    out.notes.append(f"sharpened {n} of {len(files)} grid cells to the power {float(p):g}; own cell {m0} untouched")


ATTACKS = {f.__name__: f for f in (wing_inflator, grid_far, grid_allbut, constant_grid, end_cut, local_gridder,
                                   grid_less, dead_point, bound_mover, hi_b_clamp, template_sharpen)}
assert set(ATTACKS) == set(FAMILIES)


def apply(attack, out_dir, **params):
    """Run `attack` in place on out_dir; returns and writes the report."""
    if attack not in ATTACKS:
        raise SystemExit(f"grid attack: unknown attack {attack!r}; one of {', '.join(sorted(ATTACKS))}")
    out = OutDir(out_dir)
    try:
        ATTACKS[attack](out, **params)
    except TypeError as e:
        raise SystemExit(f"grid attack {attack}: bad parameters {params}: {e}")
    report = {"attack": attack, "params": params, "changed": out.changed, "removed": out.removed,
              "notes": out.notes}
    (out.path / REPORT).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return report


def apply_copy(attack, src, dst, **params):
    """Copy src (an honest out/) to dst and run `attack` on the copy; the original is never touched."""
    shutil.copytree(src, dst)
    return apply(attack, dst, **params)


def parse_params(tokens):
    """key=value tokens to {key: float|str}."""
    params = {}
    for t in tokens:
        if "=" not in t:
            raise SystemExit(f"grid attack: parameter {t!r} is not key=value")
        k, v = t.split("=", 1)
        try:
            params[k] = float(v)
        except ValueError:
            params[k] = v
    return params


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2:
        print(f"usage: grid_attack.py <attack> <out_dir> [key=value ...]; attacks: {', '.join(sorted(ATTACKS))}",
              file=sys.stderr)
        return 2
    report = apply(argv[0], argv[1], **parse_params(argv[2:]))
    print(f"GRID ATTACK {report['attack']} {report['params']}: " + "; ".join(report["notes"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
