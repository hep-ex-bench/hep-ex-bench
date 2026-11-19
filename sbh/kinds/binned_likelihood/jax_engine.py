"""pyhf-JSON -> JAX: the model validator's fast engine (SBH_TOY_ENGINE=jax; ``evermore`` is an alias).

The declared MODEL.json workspace is rebuilt as a differentiable JAX likelihood (the "twin"), fitted by iminuit with
the exact JAX gradient; per-fit uncertainties come from the exact autodiff Hessian, not HESSE.  Needs only jax.

The effect algebra (evermore's OffsetAndScale: scale * (hist + sum offsets)) is evaluated with all modifier
applications stacked into arrays rather than a per-modifier graph, so compile stays ~1 s for any workspace size.
Workspace numbers are graph constants by default; built ``shareable`` they are jit arguments, so structure-identical
workspaces (a mass grid) share one compiled graph (``build(..., share=)``).

Supported modifiers: normfactor, normsys, histosys, staterror, shapesys, lumi; anything else makes ``build`` return
(None, reason) and the caller stays on pyhf.  Interpolations are pyhf's defaults (normsys code4, histosys code4p);
``build`` re-checks expected rates against pyhf at two points on every construction.

Zero/negative yields follow CMS combine (v11.1.0, ``autoMCStats`` datacards), not pyhf's xlogy semantics:

  per sample : max(nominal + sum of histosys deltas, CROP) before the multiplicative factors (CMSHistFunc.cc:612)
  per bin    : max(sum over samples, CROP)                                               (CMSHistErrorPropagator.cc:185)
  NLL        : sum(nu - n log nu) + constraints; nu >= CROP > 0 everywhere, so a filled bin at the floor pays
               -n ln(1e-9) (finite, zero gradient) and no point is unevaluable

Where the floor is inactive the twin equals pyhf.  An optimum on a floor kink can leave Migrad's EDM above goal; such
a fit is accepted (``kink``) when the EDM is within KINK_EDM_FACTOR of the goal.  Fits report ``n_floor``,
``n_sample_floor`` and ``min_rate`` (the unfloored lowest total rate).

Toys (``toy_pdf``) sample the floored rates; ``q_saturated`` is combine's GoodnessOfFit --algo saturated (constraints
in both terms).  ``build(..., floor=False)`` gives pyhf's own likelihood for agreement checks.

MC statistics as combine's analytic ``autoMCStats`` (Barlow-Beeston-lite, CMSHistErrorPropagator): a staterror gamma
with a Gaussian constraint, alone in its bin, is not given to Migrad.  With nu = A + gamma B in its bin (B the rate it
multiplies), d NLL / d gamma = 0 is the quadratic B g^2 + (s^2 B^2 + A - a B) g + s^2 B (A - n) - a A = 0, whose larger
root is the minimum (clipped to the bounds); Migrad fits the other parameters on that profiled NLL.  The minimum is
the full fit's; the POI's sigma comes from the profiled Hessian (the Schur complement, so the same number) and a
gamma's from 1/h_gg + J S^-1 J^T (J = d gamma-hat / d theta).  A fit that fixes such a gamma, a bin with two gammas,
and shapesys stay on the full fit; SBH_BB_LITE=0 turns the analytic gammas off.  Impacts skip MC-statistics
parameters (staterror, shapesys), as combine's impact workflow does.

POI scaling (``build(..., poi_scaling=)``, {channel: {sample: entry}}): a sample's rate scales with a monomial in the
POIs instead of their product, entry {"factor": c, "powers": {P: p}} or {"factor": c, "min_of": [{P: p}, ...]}
(exactly one of the two; c finite and non-zero, either sign; p in model.POI_POWERS; the one parser, shared with
the participant-side gate, is model.poi_scaling_entries).  Its rate is max(nominal + histosys, CROP) x c x (the
monomial prod P^p, or the least of the monomials) x its other modifiers.  The sample carries a normfactor
named P for exactly each P with a non-zero power in its entry (the markers: pyhf creates the parameters and bounds);
the twin replaces the markers' product by the scaling.  p = 1 is P itself; another non-zero power needs P > 0: at
P <= 0 a positive power is 0 and a monomial with a negative power is +inf, so it loses a min_of but leaves a lone
monomial infinite (the POI's bounds must exclude 0 there).  Every branch stays finite, so gradients are finite at
P = 0.  A gamma on a sample with c < 0 stays on the full fit (the rate it multiplies can be negative).

Parameter order, names, inits, bounds, fixed mask and constraint metadata come from pyhf ``model.config``.

Threads: XLA:CPU sizes its pools from the CPU affinity at backend start, so parallel callers must call
``pin_single_cpu`` (sbh/kinds/binned_likelihood/cpu_pin.py, re-exported here) before the first JAX computation.
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import xlogy as jxlogy
from iminuit import Minuit
from scipy.special import gammaln

from sbh.kinds.binned_likelihood.cpu_pin import pin_single_cpu  # noqa: F401
from sbh.kinds.binned_likelihood.model import ENGINE_MODIFIERS, constraint_width_problem, poi_scaling_entries

jax.config.update("jax_enable_x64", True)

SUPPORTED = set(ENGINE_MODIFIERS)
CROP = 1e-9
KINK_RATE = 1e-6
KINK_EDM_FACTOR = 10.0
MIGRAD_NCALL = 10000
BB_LITE = os.environ.get("SBH_BB_LITE", "1") != "0"
BB_TINY = 1e-12
SCALING_PROBES = ((0.6, 0.8), (1.7, 1.25))


class SpecError(ValueError):
    """The spec is one the twin does not read: the model's refusal, raised where the spec is read."""


def _code4_coeffs(hi, lo):
    """The 6 coefficients of pyhf normsys code4 (alpha0 = 1): the 6th-order polynomial matching hi^a / lo^-a in
    value and first two derivatives at a = +-1."""
    pu, pd = float(hi), float(lo)
    lu, ld = np.log(pu), np.log(pd)
    pul, pdl = pu * lu, -pd * ld
    pul2, pdl2 = pul * lu, -pdl * ld
    s0, a0 = (pu + pd) / 2, (pu - pd) / 2
    s1, a1 = (pul + pdl) / 2, (pul - pdl) / 2
    s2, a2 = (pul2 + pdl2) / 2, (pul2 - pdl2) / 2
    return ((15 * a0 - 7 * s1 + a2) / 8,
            (-24 + 24 * s0 - 9 * a1 + s2) / 8,
            (-5 * a0 + 5 * s1 - a2) / 4,
            (12 - 12 * s0 + 7 * a1 - s2) / 4,
            (3 * a0 - 3 * s1 + a2) / 8,
            (-8 + 8 * s0 - 5 * a1 + s2) / 8)


def _padded_row_index(pairs, n_rows, sentinel):
    """[R, M] matrix: row r lists the application indices that multiply it, padded with ``sentinel`` (an appended
    identity).  Gather-and-prod over it replaces scatter-multiply, whose JAX gradient needs unique indices."""
    rows, apps = pairs
    per = [[] for _ in range(n_rows)]
    for r, k in zip(rows, apps):
        per[r].append(k)
    m = max((len(p) for p in per), default=0) or 1
    out = np.full((n_rows, m), sentinel, int)
    for r, p in enumerate(per):
        out[r, :len(p)] = p
    return out


class _Compiled:
    """Jitted value/gradient/Hessian of one workspace STRUCTURE.

    ``baked``: the workspace's numbers as graph constants (fastest per call).  Without it they are an argument, so
    structure-identical workspaces reuse the compile (~10-20 % slower per call)."""

    def __init__(self, st, baked=None):
        self.st = st
        self.args = baked is None
        if self.args:
            self.exp = jax.jit(self._exp)
            self.exp_raw = jax.jit(lambda x, D: self._exp(x, D, crop=False))
            self.diag = jax.jit(self._diag)
            self.nll = jax.jit(self._nll)
            self.grad = jax.jit(jax.grad(self._nll))
            self.hess = jax.jit(jax.hessian(self._nll))
        else:
            D = baked
            nll = lambda x, m, a: self._nll(x, m, a, D)  # noqa: E731
            self.exp = jax.jit(lambda x: self._exp(x, D))
            self.exp_raw = jax.jit(lambda x: self._exp(x, D, crop=False))
            self.diag = jax.jit(lambda x: self._diag(x, D))
            self.nll = jax.jit(nll)
            self.grad = jax.jit(jax.grad(nll))
            self.hess = jax.jit(jax.hessian(nll))
        if st["pg"] is not None:
            for name, f in (("gam", self._gam), ("nll_prof", self._nll_prof), ("grad_prof", jax.grad(self._nll_prof)),
                            ("hess_red", self._hess_red), ("dgam", self._dgam), ("hgg", self._hgg)):
                setattr(self, name, jax.jit(f) if self.args else jax.jit(lambda x, m, a, f=f: f(x, m, a, D)))

    def _pre(self, x, D):
        """Per (channel, sample) row: nominal + histosys code4p offsets, i.e. the rate before the multiplicative
        factors (combine's CMSHistFunc output, which it floors)."""
        st = self.st
        nom = D["nom"]
        rows = nom
        if st["hs"] is not None:
            r, p = st["hs"]
            up, dn = D["hs_up"], D["hs_dn"]
            a = x[p][:, None]
            base = nom[r]
            du, dd = up - base, base - dn
            s_, a_ = 0.5 * (du + dd), 0.0625 * (du - dd)
            inner = a * (s_ + a * a_ * (15.0 + a * a * (-10.0 + 3.0 * a * a)))
            d = jnp.where(a >= 1.0, du * a, jnp.where(a <= -1.0, dd * a, inner))
            rows = rows.at[r].add(d)
        return rows

    def _exp(self, x, D, crop=None):
        """Expected main-bin rates.  ``crop`` (default: the twin's ``floor``) applies combine's row and total floors;
        without it, pyhf's unfloored rates."""
        crop = self.st["floor"] if crop is None else crop
        tot = self._total(x, D, crop)
        return jnp.maximum(tot, CROP) if crop else tot

    def _diag(self, x, D):
        """(pre-factor rows, totals before the total floor), row floor applied: floor-occupancy diagnostics."""
        return self._pre(x, D), self._total(x, D, True)

    def _total(self, x, D, crop):
        return self._bins(self._factored(x, D, crop) * self._gamma_prod(x, D))

    def _bins(self, rows):
        """Per-row rates -> the main bins (summed per channel, channels concatenated)."""
        st = self.st
        ch = jax.ops.segment_sum(rows, st["chan_of_row"], num_segments=st["n_chan"])
        return jnp.concatenate([ch[i, :n] for i, n in enumerate(st["nbins"])])

    def _gamma_prod(self, x, D, skip_profiled=False):
        """Per (row, bin) product of the staterror / shapesys gammas; ``skip_profiled``: the analytic ones as 1."""
        st = self.st
        if st["gm"] is None:
            return 1.0
        gm, gidx = st["gm"]
        g = jnp.where(gm >= 0, x[jnp.clip(gm, 0, None)], 1.0)
        if skip_profiled:
            g = jnp.where(st["pg"]["app_mask"], 1.0, g)
        g_ext = jnp.concatenate([g, jnp.ones((1, D["nom"].shape[1]))])
        return jnp.prod(g_ext[gidx], axis=1)

    def _ab(self, x, D):
        """Per main bin of an analytic gamma: (A, B) with nu = A + gamma B -- B the rate the gamma multiplies, A the
        rest, both at x's other parameters (per-process floor applied, total floor not: nu >= CROP is assumed)."""
        pg = self.st["pg"]
        rows = self._factored(x, D, self.st["floor"]) * self._gamma_prod(x, D, skip_profiled=True)
        A = self._bins(jnp.where(pg["row_mask"], 0.0, rows))
        B = self._bins(jnp.where(pg["row_mask"], rows, 0.0))
        return A[pg["bin"]], B[pg["bin"]]

    def _gam(self, x, main, aux, D):
        """x with every analytic gamma at its conditional minimum given the other parameters (module docstring)."""
        pg = self.st["pg"]
        A, B = self._ab(x, D)
        n, a0, s2 = main[pg["bin"]], aux[pg["aux"]], D["pg_sig"] ** 2
        live = B > BB_TINY
        Bs = jnp.where(live, B, 1.0)
        qb = s2 * Bs * Bs + A - a0 * Bs
        qc = s2 * Bs * (A - n) - a0 * A
        sq = jnp.sqrt(jnp.maximum(qb * qb - 4.0 * Bs * qc, 1e-300))
        pos = qb >= 0
        root = jnp.where(pos, -2.0 * qc / jnp.where(pos, qb + sq, 1.0), (sq - qb) / (2.0 * Bs))
        g = jnp.clip(jnp.where(live, root, a0), D["pg_lo"], D["pg_hi"])
        return x.at[pg["par"]].set(g)

    def _nll_prof(self, x, main, aux, D):
        """The NLL with the analytic gammas profiled out (x's own gamma entries are ignored)."""
        return self._nll(self._gam(x, main, aux, D), main, aux, D)

    def _hess_red(self, x, main, aux, D):
        """Hessian of the profiled NLL over the other parameters (st["pg"]["other"] order)."""
        other = self.st["pg"]["other"]
        return jax.hessian(lambda t: self._nll_prof(x.at[other].set(t), main, aux, D))(x[other])

    def _dgam(self, x, main, aux, D):
        """d gamma-hat / d (other parameters): [analytic gammas, other parameters]."""
        pg = self.st["pg"]
        return jax.jacfwd(lambda t: self._gam(x.at[pg["other"]].set(t), main, aux, D)[pg["par"]])(x[pg["other"]])

    def _hgg(self, x, main, aux, D):
        """d^2 NLL / d gamma^2 per analytic gamma at x (its block of the full Hessian is diagonal)."""
        pg = self.st["pg"]
        _A, B = self._ab(x, D)
        nu = self._exp(x, D)[pg["bin"]]
        return main[pg["bin"]] * (B / nu) ** 2 + 1.0 / D["pg_sig"] ** 2

    def _poi_scale(self, x, D):
        """Per scaled row: c x its monomial, or the least of its monomials (module docstring)."""
        pw = D["ps_pow"]
        v = x[self.st["ps"][1]]
        pos = v > 0
        s = jnp.where(pos, v, 1.0)
        t = jnp.where(pw == 1.0, v, jnp.where(pos, s ** pw, jnp.where(pw > 0, 0.0, 1.0)))
        mono = jnp.where(jnp.any((pw < 0) & ~pos, axis=2), jnp.inf, jnp.prod(t, axis=2))
        return D["ps_c"] * jnp.min(mono, axis=1)

    def _factored(self, x, D, crop):
        """Per-row rates before the gammas: the pre-factor rows (per-process floor if ``crop``) times the
        normfactor / lumi / normsys factors and the POI scaling."""
        st = self.st
        rows = self._pre(x, D)
        if crop:
            rows = jnp.maximum(rows, CROP)
        parts = []
        if st["sf"] is not None:
            parts.append(x[st["sf"]])
        if st["ns"] is not None:
            p = st["ns"]
            hi, lo, c = D["ns_hi"], D["ns_lo"], D["ns_c"]
            al = x[p]
            inner = 1.0 + al * (c[:, 0] + al * (c[:, 1] + al * (c[:, 2] + al * (c[:, 3] + al * (c[:, 4] + al * c[:, 5])))))
            parts.append(jnp.where(al >= 1.0, hi ** al, jnp.where(al <= -1.0, lo ** (-al), inner)))
        if st["ps"] is not None:
            parts.append(self._poi_scale(x, D))
        if parts:
            f = jnp.concatenate(parts + [jnp.ones(1)])
            rows = rows * jnp.prod(f[st["fidx"]], axis=1)[:, None]
        return rows

    def _nll(self, x, main, aux, D):
        nu = self._exp(x, D)
        if self.st["floor"]:
            val = jnp.sum(nu - main * jnp.log(nu))
        else:
            val = jnp.sum(nu - jxlogy(main, nu))
        for (kind, (p0, p1), (a0, a1)), c in zip(self.st["cons"], D["cons"]):
            th, a = x[p0:p1], aux[a0:a1]
            if kind == "normal":
                val = val + 0.5 * jnp.sum(((th - a) / c) ** 2)
            else:
                rate = jnp.clip(th * c, 1e-12, None)
                val = val + jnp.sum(rate - a * jnp.log(rate))
        return val


def _same(a, b):
    """Structural equality over nested tuples / numpy arrays / scalars."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
                and a.shape == b.shape and np.array_equal(a, b))
    if isinstance(a, (tuple, list)) or isinstance(b, (tuple, list)):
        return (isinstance(a, (tuple, list)) and isinstance(b, (tuple, list))
                and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b)))
    if isinstance(a, dict) or isinstance(b, dict):
        return (isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys()
                and all(_same(a[k], b[k]) for k in a))
    return a == b


class JaxPyhfModel:
    """The pyhf workspace as a differentiable JAX model on pyhf's flat parameter vector (same order, names, bounds,
    inits and fixed mask as ``model.config``).

    Per sample: (nominal + histosys deltas) x normfactor/lumi/normsys/gamma factors and its POI scaling, summed per
    channel, with combine's floor (module docstring).  nll(x, main, aux) equals pyhf's -logpdf up to an x-independent
    constant (``full_nll_const``) wherever the floor is inactive and no sample is POI-scaled.
    """

    def __init__(self, spec, model, share: "JaxPyhfModel | None" = None, shareable: bool = False,
                 floor: bool = True, poi_scaling=None, finite_nll: bool = True):
        cfg = model.config
        self._pyhf_model = model
        self.floor = bool(floor)
        self.finite_nll = bool(finite_nll)
        self.poi_index = cfg.poi_index
        self.n_main = cfg.nmaindata
        self.par_names = list(cfg.par_names)
        self.npars = len(self.par_names)
        self.init = np.asarray(cfg.suggested_init(), float)
        self.bounds = [tuple(map(float, b)) for b in cfg.suggested_bounds()]
        self.fixed = np.asarray(cfg.suggested_fixed(), bool)
        slices = {name: cfg.par_map[name]["slice"] for name in cfg.par_order}
        self.constraints, cons_layout, cons_data, pos = [], [], [], 0
        for name in cfg.auxdata_order:
            ps = cfg.param_set(name)
            s, n = slices[name], ps.n_parameters
            if ps.pdf_type == "normal":
                c = np.asarray(getattr(ps, "sigmas", None) or [1.0] * n, float)
                kind = "normal"
            elif ps.pdf_type == "poisson":
                c = np.asarray(ps.auxdata, float)
                if not np.all(np.isfinite(c)) or np.any(c < 0):
                    raise SpecError(f"non-finite Poisson constraint factors for {name}")
                kind = "poisson"
            else:
                raise SpecError(f"unsupported constraint pdf {ps.pdf_type} for {name}")
            layout = (kind, (s.start, s.stop), (pos, pos + n))
            cons_layout.append(layout)
            cons_data.append(c)
            self.constraints.append(layout + (c,))
            pos += n
        self.n_aux = pos
        by_name = {c["name"]: c for c in spec["channels"]}
        nbins = [len(by_name[c]["samples"][0]["data"]) for c in cfg.channels]
        B = max(nbins)
        rows, chan_of_row = [], []
        sf_row, sf_par = [], []
        ns_row, ns_par, ns_hi, ns_lo, ns_c = [], [], [], [], []
        hs_row, hs_par, hs_up, hs_dn = [], [], [], []
        gm_row, gm_idx = [], []
        ps_row, ps_ent = [], []
        self.poi_scaling = poi_scaling_entries(poi_scaling or {}, spec)
        offs = np.cumsum([0] + nbins)
        stat_bin, bin_gammas = {}, {}
        for ci, cname in enumerate(cfg.channels):
            nb = nbins[ci]
            for s in by_name[cname]["samples"]:
                if len(s["data"]) != nb:
                    raise SpecError(f"ragged sample data in channel {cname}")
                r = len(rows)
                rows.append(np.pad(np.asarray(s["data"], float), (0, B - nb)))
                chan_of_row.append(ci)
                ent = self.poi_scaling.get((cname, s["name"]))
                if ent is not None:
                    ps_row.append(r)
                    ps_ent.append(ent)
                for m in s.get("modifiers", []):
                    typ, sl = m["type"], slices[m["name"]]
                    if typ == "normfactor" and ent is not None and m["name"] in ent[2]:
                        continue
                    if typ in ("normfactor", "lumi"):
                        sf_row.append(r)
                        sf_par.append(sl.start)
                    elif typ == "normsys":
                        hi, lo = float(m["data"]["hi"]), float(m["data"]["lo"])
                        if hi <= 0 or lo <= 0:
                            raise SpecError(f"non-positive normsys kappa on {m['name']}")
                        ns_row.append(r)
                        ns_par.append(sl.start)
                        ns_hi.append(hi)
                        ns_lo.append(lo)
                        ns_c.append(_code4_coeffs(hi, lo))
                    elif typ == "histosys":
                        if len(m["data"]["hi_data"]) != nb or len(m["data"]["lo_data"]) != nb:
                            raise SpecError(f"ragged histosys data on {m['name']}")
                        hs_row.append(r)
                        hs_par.append(sl.start)
                        hs_up.append(np.pad(np.asarray(m["data"]["hi_data"], float), (0, B - nb)))
                        hs_dn.append(np.pad(np.asarray(m["data"]["lo_data"], float), (0, B - nb)))
                    elif typ in ("staterror", "shapesys"):
                        if sl.stop - sl.start != nb:
                            raise SpecError(f"{typ} {m['name']} width != channel bins")
                        gm_row.append(r)
                        gm_idx.append(np.pad(np.arange(sl.start, sl.stop), (0, B - nb),
                                             constant_values=-1))
                        for b in range(nb):
                            bin_gammas.setdefault(int(offs[ci] + b), set()).add(sl.start + b)
                            if typ == "staterror":
                                stat_bin[sl.start + b] = int(offs[ci] + b)
                    else:
                        raise SpecError(f"unsupported modifier type {typ}")
        n_rows = len(rows)
        ia = lambda v: np.asarray(v, int)  # noqa: E731
        napps = len(sf_par) + len(ns_par) + len(ps_row)
        self.ps_index = {P: slices[P].start for P in sorted({P for _c, _m, mk in ps_ent for P in mk})}
        st = {"nbins": tuple(nbins), "n_chan": len(cfg.channels), "floor": self.floor,
              "chan_of_row": ia(chan_of_row),
              "fidx": _padded_row_index([sf_row + ns_row + ps_row, list(range(napps))], n_rows, napps),
              "sf": ia(sf_par) if sf_par else None,
              "ns": ia(ns_par) if ns_par else None,
              "ps": (ia(ps_row), ia(list(self.ps_index.values()))) if ps_row else None,
              "hs": (ia(hs_row), ia(hs_par)) if hs_row else None,
              "gm": ((np.stack(gm_idx),
                      _padded_row_index([gm_row, list(range(len(gm_row)))], n_rows, len(gm_row)))
                     if gm_row else None),
              "cons": tuple(cons_layout)}
        D = {"nom": np.stack(rows), "cons": tuple(cons_data)}
        if hs_row:
            D["hs_up"], D["hs_dn"] = np.stack(hs_up), np.stack(hs_dn)
        if ns_par:
            D["ns_hi"], D["ns_lo"], D["ns_c"] = (np.asarray(ns_hi, float), np.asarray(ns_lo, float),
                                                 np.asarray(ns_c, float))
        if ps_row:
            k = max(len(m) for _c, m, _mk in ps_ent)
            D["ps_c"] = np.asarray([c for c, _m, _mk in ps_ent], float)
            D["ps_pow"] = np.asarray([[[float(mono.get(P, 0.0)) for P in self.ps_index]
                                       for mono in m + m[:1] * (k - len(m))] for _c, m, _mk in ps_ent], float)
        self._mcstat = np.zeros(self.npars, bool)
        for name, typ in cfg.modifiers:
            if typ in ("staterror", "shapesys"):
                self._mcstat[slices[name]] = True
        sig_of, aux_of = {}, {}
        for kind, (p0, p1), (a0, _a1), c in self.constraints:
            if kind == "normal":
                for j in range(p1 - p0):
                    sig_of[p0 + j], aux_of[p0 + j] = float(c[j]), a0 + j
        neg_rows = {r for r, e in zip(ps_row, ps_ent) if e[0] < 0}
        neg = {p for r, idx in zip(gm_row, gm_idx) if r in neg_rows for p in idx if p >= 0}
        pg = sorted(p for p, b in stat_bin.items()
                    if BB_LITE and sig_of.get(p, 0.0) > 0 and not self.fixed[p] and len(bin_gammas[b]) == 1
                    and p not in neg)
        st["pg"] = None
        if pg:
            app_mask = np.isin(np.stack(gm_idx), pg)
            row_mask = np.zeros((n_rows, B), bool)
            for k, r in enumerate(gm_row):
                row_mask[r] |= app_mask[k]
            st["pg"] = {"par": ia(pg), "bin": ia([stat_bin[p] for p in pg]), "aux": ia([aux_of[p] for p in pg]),
                        "app_mask": app_mask, "row_mask": row_mask,
                        "other": np.setdiff1d(np.arange(self.npars), pg)}
            D["pg_sig"] = np.asarray([sig_of[p] for p in pg], float)
            D["pg_lo"] = np.asarray([self.bounds[p][0] for p in pg], float)
            D["pg_hi"] = np.asarray([self.bounds[p][1] for p in pg], float)
        self._pg = None if st["pg"] is None else st["pg"]["par"]
        self._pg_other = None if st["pg"] is None else st["pg"]["other"]
        self.n_analytic_gammas = len(pg)
        self._st = st
        self._row_valid = np.arange(B)[None, :] < np.asarray([nbins[c] for c in chan_of_row])[:, None]
        self._shapes = {k: (tuple(np.shape(v)) if k != "cons" else tuple(np.shape(c) for c in v))
                        for k, v in D.items()}
        self.data = {k: (jnp.asarray(v) if k != "cons" else tuple(jnp.asarray(c) for c in v))
                     for k, v in D.items()}
        self.shared = bool(share is not None and share._fn.args and self.same_structure(share))
        if self.shared:
            self._fn = share._fn
        else:
            self._fn = _Compiled(st, baked=None if (shareable or share is not None) else self.data)
        self._args = (self.data,) if self._fn.args else ()

    def same_structure(self, other: "JaxPyhfModel") -> bool:
        """True when ``other``'s compiled graph computes this workspace (numbers may differ)."""
        return self._shapes == other._shapes and _same(self._st, other._st)

    def expected_main(self, x):
        """Combine-floored expected main-bin rates."""
        return self._fn.exp(jnp.asarray(x), *self._args)

    def expected_main_unfloored(self, x):
        """pyhf's unfloored rates at x (can be negative)."""
        return self._fn.exp_raw(jnp.asarray(x), *self._args)

    def floor_state(self, x):
        """Floor occupancy at x: n_floor (bins at the floor), n_sample_floor (sample bins with negative pre-factor
        rate, where pyhf would go negative), kink (some rate within KINK_RATE of zero), min_rate (unfloored lowest
        total)."""
        pre, tot = (np.asarray(v, float) for v in self._fn.diag(jnp.asarray(x), *self._args))
        low = (pre < CROP) | ~self._row_valid
        st = self._st
        chan_all = [np.all(low[st["chan_of_row"] == c], axis=0)[:nb] for c, nb in enumerate(st["nbins"])]
        at_floor = np.concatenate(chan_all) | (tot < CROP)
        pre = pre[self._row_valid]
        raw = np.asarray(self.expected_main_unfloored(x), float)
        return {"n_floor": int(np.sum(at_floor)),
                "n_sample_floor": int(np.sum(pre < 0.0)),
                "kink": bool(np.any((pre < KINK_RATE) & (pre != 0.0)) or np.any(tot < KINK_RATE)),
                "min_rate": float(raw.min())}

    def nll(self, x, main, aux):
        return self._fn.nll(jnp.asarray(x), main, aux, *self._args)

    def grad(self, x, main, aux):
        return self._fn.grad(jnp.asarray(x), main, aux, *self._args)

    def hess(self, x, main, aux):
        return self._fn.hess(jnp.asarray(x), main, aux, *self._args)

    def full_nll_const(self, main, aux) -> float:
        """pyhf -logpdf(x) - nll(x) (x-independent where the floor is inactive): lnGamma terms of main and Poisson-aux
        counts plus Gaussian log-norms."""
        c = float(np.sum(gammaln(np.asarray(main, float) + 1.0)))
        aux = np.asarray(aux, float)
        for kind, _p, (a0, a1), cc in self.constraints:
            if kind == "normal":
                c += float(np.sum(np.log(np.asarray(cc)) + 0.5 * np.log(2 * np.pi)))
            else:
                c += float(np.sum(gammaln(aux[a0:a1] + 1.0)))
        return c

    def toy_pdf(self, x):
        """Frequentist toy pdf at x (combine --toysFrequentist): pyhf's ``model.make_pdf(x)`` with the main Poisson
        rates replaced by the floored ones, so draws match pyhf's draw for draw where the floor is inactive."""
        import pyhf
        from pyhf import probability as prob
        model = self._pyhf_model
        tl = pyhf.tensorlib
        parts = [prob.Independent(prob.Poisson(tl.astensor(np.asarray(self.expected_main(x), float))))]
        cons = model.constraint_model.make_pdf(tl.astensor([float(v) for v in x]))
        if cons:
            parts.append(cons)
        return prob.Simultaneous(parts, model.fullpdf_tv, model.batch_size)

    def nll_saturated(self, main, aux) -> float:
        """Saturated-model NLL in the twin's convention: main bins at their counts, constraints at their minimum."""
        from scipy.special import xlogy
        m = np.asarray(main, float)
        v = float(np.sum(m - xlogy(m, m)))
        aux = np.asarray(aux, float)
        for kind, _p, (a0, a1), _c in self.constraints:
            if kind == "poisson":
                a = aux[a0:a1]
                rate = np.clip(a, 1e-12, None)
                v += float(np.sum(rate - a * np.log(rate)))
        return v

    def q_saturated(self, x, main, aux) -> float:
        """combine's GoodnessOfFit --algo saturated at fitted x: 2 (NLL(x) - NLL_sat), constraints in both terms
        (GoodnessOfFit.cc:122-235), so the Gaussian pulls are part of the statistic."""
        return 2.0 * (float(self.nll(x, jnp.asarray(main, float), jnp.asarray(aux, float)))
                      - self.nll_saturated(main, aux))

    def _active(self, x, fixed):
        """Free non-POI parameters within 1e-4 (hi - lo) of a bound."""
        active = np.zeros(self.npars, bool)
        for i in np.flatnonzero(~fixed):
            if i == self.poi_index:
                continue
            lo, hi = self.bounds[i]
            d = 1e-4 * (hi - lo) if np.isfinite(hi - lo) else 1e-4 * max(abs(x[i]), 1.0)
            active[i] = (x[i] - lo < d) or (hi - x[i] < d)
        return active

    def _profiles(self, fixed) -> bool:
        """True when a fit with this fixed mask runs on the analytic gammas (none of them fixed by the caller)."""
        return self._pg is not None and not np.asarray(fixed, bool)[self._pg].any()

    def _migrad(self, x0, main, aux, fixed, tol, profiled=False, bounds=None):
        """One start: Migrad, again if not converged, then Simplex + Migrad.  ``profiled``: on the NLL with the
        analytic gammas profiled out, which Migrad then holds fixed.  ``bounds``: per-parameter limits in place of
        the model's (a conditional fit at a POI value the declared bounds exclude)."""
        nll_f, grad_f = ((self._fn.nll_prof, self._fn.grad_prof) if profiled else (self._fn.nll, self._fn.grad))

        def fcn(x):
            return float(nll_f(jnp.asarray(x), main, aux, *self._args))

        def gr(x):
            return np.asarray(grad_f(jnp.asarray(x), main, aux, *self._args))

        if profiled:
            fixed = np.asarray(fixed, bool).copy()
            fixed[self._pg] = True
        m = Minuit(fcn, x0, grad=gr, name=self.par_names)
        m.errordef = Minuit.LIKELIHOOD
        for i, b in enumerate(self.bounds if bounds is None else bounds):
            m.limits[i] = b
            if fixed[i]:
                m.fixed[i] = True
        m.strategy = 0
        m.tol = tol
        m.migrad(ncall=MIGRAD_NCALL)
        if not m.valid:
            m.migrad(ncall=MIGRAD_NCALL)
        if not m.valid:
            m.simplex(ncall=20000)
            m.migrad(ncall=MIGRAD_NCALL)
        return m

    def fit(self, main, aux, init=None, tol=0.1, want_sigma=True, fixed=None, starts=(), bounds=None):
        """Migrad from ``init`` (default: model init); if not converged, from each of ``starts``, then the model init.
        Fixed parameters keep ``init``'s values; candidates are clipped into bounds (``bounds`` overrides the model's,
        e.g. the POI bound widened to min(lo_b, 0) for a conditional fit at mu = 0 the declared bounds exclude).  A fit
        failing only on EDM is accepted at a floor kink within KINK_EDM_FACTOR (``kink`` True).  A fit is valid only at
        a finite NLL (``finite_nll``; a likelihood that is not a number at its minimum, e.g. from rates beyond the
        float range, is no fit), in both branches.  Returns x, nll, valid, sigma, kink, the ``floor_state`` fields and
        start (candidate index; the lowest-NLL attempt when none converged)."""
        main = jnp.asarray(main, float)
        aux = jnp.asarray(aux, float)
        fixed = self.fixed if fixed is None else np.asarray(fixed, bool)
        bnds = self.bounds if bounds is None else [tuple(map(float, b)) for b in bounds]
        lo = np.array([b[0] for b in bnds])
        hi = np.array([b[1] for b in bnds])
        first = np.clip(np.asarray(self.init if init is None else init, float), lo, hi)
        if fixed.all():
            fs = self.floor_state(first)
            kink = fs.pop("kink")
            nll = float(self.nll(first, main, aux))
            return {"x": first, "nll": nll, "valid": bool(np.isfinite(nll) or not self.finite_nll), "kink": False,
                    **fs, "start": 0, "sigma": None, "all_fixed": True, "at_kink": bool(kink)}
        best, tried = None, []
        profiled = self._profiles(fixed)
        for k, cand in enumerate([first, *starts, self.init]):
            c = np.clip(np.asarray(cand, float), lo, hi)
            c[fixed] = first[fixed]
            if any(np.array_equal(c, t) for t in tried):
                continue
            tried.append(c)
            m = self._migrad(c, main, aux, fixed, tol, profiled, bounds=None if bounds is None else bnds)
            x = np.array(m.values, float)
            if profiled:
                x = np.asarray(self._fn.gam(jnp.asarray(x), main, aux, *self._args), float)
            nll = float(m.fval)
            fs = self.floor_state(x)
            at_kink = fs.pop("kink")
            fm = m.fmin
            kink = bool(not m.valid and at_kink and fm.is_above_max_edm and not fm.has_reached_call_limit
                        and np.isfinite(nll) and fm.edm <= KINK_EDM_FACTOR * 0.002 * tol * m.errordef)
            valid = (bool(m.valid) or kink) and bool(np.isfinite(nll) or not self.finite_nll)
            r = {"x": x, "nll": nll, "valid": valid, "kink": kink, **fs, "start": k}
            if r["valid"]:
                best = r
                break
            if best is None or (np.isfinite(nll) and nll < best["nll"]):
                best = r
        best["sigma"] = self._sigma(best["x"], main, aux, fixed) if (want_sigma and best["valid"]) else None
        return best

    def _sigma(self, x, main, aux, fixed=None):
        """Per-parameter uncertainties from the exact autodiff Hessian at the fitted point.  Exact, not finite
        differences: fits often end at a bound or floor kink, where an FD step gives indefinite curvature.  Requires
        positive definiteness (Cholesky); None otherwise."""
        x = np.asarray(x, float)
        fixed = self.fixed if fixed is None else np.asarray(fixed, bool)
        if self._profiles(fixed):
            return self._sigma_profiled(x, jnp.asarray(main, float), jnp.asarray(aux, float), fixed)
        active = self._active(x, fixed)
        free = np.flatnonzero(~fixed & ~active)
        h_full = np.asarray(self.hess(x, main, aux))
        hess = h_full[np.ix_(free, free)]
        hess = 0.5 * (hess + hess.T)
        try:
            np.linalg.cholesky(hess)
            cov = np.linalg.inv(hess)
        except np.linalg.LinAlgError:
            return None
        d = np.diag(cov)
        if not np.all(np.isfinite(d)) or np.any(d <= 0):
            return None
        out = np.zeros(self.npars)
        out[active] = np.nan
        out[np.asarray(free)] = np.sqrt(d)
        return out

    def _sigma_profiled(self, x, main, aux, fixed):
        """``_sigma`` on the analytic gammas: the other parameters from the profiled NLL's Hessian S (the Schur
        complement of the gamma block, so the full fit's numbers), each gamma from 1/h_gg + J S^-1 J^T.  Same active
        set and conventions as ``_sigma``."""
        args = (jnp.asarray(x), main, aux, *self._args)
        active = self._active(x, fixed)
        other = self._pg_other
        sel = np.flatnonzero(~fixed[other] & ~active[other])
        hess = np.asarray(self._fn.hess_red(*args))[np.ix_(sel, sel)]
        hess = 0.5 * (hess + hess.T)
        try:
            np.linalg.cholesky(hess)
            cov = np.linalg.inv(hess)
        except np.linalg.LinAlgError:
            return None
        d = np.diag(cov)
        if not np.all(np.isfinite(d)) or np.any(d <= 0):
            return None
        J = np.asarray(self._fn.dgam(*args))[:, sel]
        var_g = 1.0 / np.asarray(self._fn.hgg(*args)) + np.einsum("ij,jk,ik->i", J, cov, J)
        out = np.zeros(self.npars)
        out[other[sel]] = np.sqrt(d)
        out[self._pg] = np.sqrt(np.where(var_g > 0, var_g, np.nan))
        out[active] = np.nan
        return out

    def postfit_impacts(self, main, aux, bestfit, uncertainty):
        """cabinetry.fit.ranking's postfit impacts on the twin: each non-POI parameter fixed at bestfit +- uncertainty,
        everything else refitted from init; impact = mu(refit) - mu_hat.  Zero shifts are 0 without a fit.  Unlike
        cabinetry, a shift beyond a bound is evaluated at the bound (``clipped``) and a failed refit loses only its
        own entry (``failed``).  Returns ({label: {"up", "down"}}, info)."""
        import time
        t0 = time.perf_counter()
        poi = self.poi_index
        bestfit = np.asarray(bestfit, float)
        unc = np.where(self.fixed, 0.0, np.asarray(uncertainty, float))
        lo = np.array([b[0] for b in self.bounds])
        hi = np.array([b[1] for b in self.bounds])
        impacts, info = {}, {"engine": "jax", "n_refits": 0, "n_invalid": 0, "n_floor": 0,
                             "n_kink": 0, "n_start_fallback": 0, "clipped": [], "failed": [],
                             "n_mcstat_skipped": int(self._mcstat.sum())}
        for i, lbl in enumerate(self.par_names):
            if i == poi or self._mcstat[i]:
                continue
            out = []
            for tag, v in (("up", bestfit[i] + unc[i]), ("down", bestfit[i] - unc[i])):
                if v == bestfit[i]:
                    out.append(0.0)
                    continue
                if not np.isfinite(v):
                    info["failed"].append(f"{lbl}:{tag}")
                    out.append(None)
                    continue
                vc = float(np.clip(v, lo[i], hi[i]))
                if vc != v:
                    info["clipped"].append(f"{lbl}:{tag}")
                x0 = self.init.copy()
                x0[i] = vc
                fx = self.fixed.copy()
                fx[i] = True
                alt = bestfit.copy()
                alt[i] = vc
                r = self.fit(main, aux, init=x0, fixed=fx, want_sigma=False, starts=(alt,))
                info["n_refits"] += 1
                info["n_start_fallback"] += int(r["start"] > 0)
                if not r["valid"]:
                    info["n_invalid"] += 1
                    info["failed"].append(f"{lbl}:{tag}")
                    out.append(None)
                    continue
                info["n_floor"] += int(r["n_floor"] > 0)
                info["n_kink"] += int(r["kink"])
                out.append(float(r["x"][poi] - bestfit[poi]))
            impacts[lbl] = {"up": out[0], "down": out[1]}
        info["wall_s"] = round(time.perf_counter() - t0, 2)
        return impacts, info

    def q0(self, main, aux):
        """pyhf q0 on the twin: clip(2 NLL(mu=0) - 2 NLL(free), 0), 0 when mu_hat < 0; both fits from the model init.
        Returns (q0, info)."""
        poi = self.poi_index
        free = self.fit(main, aux, want_sigma=False)
        x0 = self.init.copy()
        x0[poi] = 0.0
        fx = self.fixed.copy()
        fx[poi] = True
        cond = self.fit(main, aux, init=x0, fixed=fx, want_sigma=False)
        info = {"valid": bool(free["valid"] and cond["valid"]),
                "n_floor": int(free["n_floor"] > 0) + int(cond["n_floor"] > 0),
                "n_kink": int(free["kink"]) + int(cond["kink"]),
                "mu_hat": float(free["x"][poi])}
        if not info["valid"]:
            return None, info
        q = max(2.0 * (cond["nll"] - free["nll"]), 0.0)
        if free["x"][poi] < 0:
            q = 0.0
        return float(q), info




FLOOR = True


def build(spec, model, share: JaxPyhfModel | None = None, shareable: bool = False,
          floor: bool | None = None, poi_scaling=None, shipped: bool = False):
    """The JAX twin of a pyhf model, or (None, reason) when the spec uses unsupported modifiers, a parameter's init
    lies outside its bounds (init_outside_bounds: pyhf's fit refuses such a model, so the twin does too), a constraint
    width is not a positive finite number (model.constraint_width_problem: pyhf's fit fails there), its POI scaling
    (module docstring) is outside the grammar, or it fails the expected-rates cross-check against pyhf.
    ``shareable`` takes workspace numbers as arguments; ``share`` reuses a structure-identical shareable twin's graph
    (``twin.shared`` reports it).  ``floor`` defaults to FLOOR.  ``shipped`` (a task kept on the rule it shipped
    with): the twin as that grader built it, which refused an init outside its bounds only on a free parameter, where
    its cross-check point raised (probe_point, total False), checked no constraint width and counted a fit at an NLL
    that is not a number as any other (``finite_nll`` False)."""
    kinds = {m["type"] for c in spec.get("channels", []) for s in c.get("samples", [])
             for m in s.get("modifiers", [])}
    extra = kinds - SUPPORTED
    if extra:
        return None, f"unsupported modifiers: {sorted(extra)}"
    if not shipped:
        why = init_outside_bounds(model.config) or constraint_width_problem(model.config)
        if why:
            return None, why
    try:
        em = JaxPyhfModel(spec, model, share=share, shareable=shareable,
                          floor=FLOOR if floor is None else floor, poi_scaling=poi_scaling, finite_nll=not shipped)
        why = pyhf_mismatch(model, em, total=not shipped)
    except Exception as e:  # noqa: BLE001
        return None, f"build failed: {type(e).__name__}: {e}"
    return (None, why) if why else (em, None)


def init_outside_bounds(cfg):
    """Why a model cannot be fitted from its declared start (pyhf.infer.mle's own rule, fixed parameters included): the
    first parameter whose init does not lie within its bounds (bounds that exclude every value, or an init that is
    not a number, among them), as "init outside its bounds: <name> init <v>, bounds [<lo>, <hi>]"; None when every
    init lies within its bounds."""
    inits, bounds = cfg.suggested_init(), cfg.suggested_bounds()
    for name in cfg.par_order:
        sl = cfg.par_slice(name)
        for j, i in enumerate(range(sl.start, sl.stop)):
            v, (lo, hi) = inits[i], bounds[i]
            if not (lo <= v <= hi):
                label = name if sl.stop - sl.start == 1 else f"{name}[{j}]"
                return f"init outside its bounds: {label} init {float(v):g}, bounds [{float(lo):g}, {float(hi):g}]"
    return None


def pyhf_mismatch(model, em, gross: bool = False, total: bool = True):
    """Why the twin's unfloored rates disagree with pyhf's at the init and a shifted point (probe_point, ``total`` as
    there; the floor differs from pyhf by design), the scaled POIs at SCALING_PROBES there; None when they agree.  The
    tolerance is relative to each bin's rate; ``gross``: to its gross rate, the sum of its samples' |rates| (samples
    of either sign that cancel leave a net rate below the rounding of their sum)."""
    points = [em.init.copy(), probe_point(em, 1, total)]
    for x, (v, r) in zip(points, SCALING_PROBES):
        for k, i in enumerate(em.ps_index.values()):
            x[i] = v * r ** k
    for x in points:
        ours = np.asarray(em.expected_main_unfloored(x))
        if gross:
            per = _pyhf_sample_rates(model, em, x)
            theirs, scale = per.sum(axis=0), np.maximum(np.abs(per).sum(axis=0), 1.0)
        else:
            theirs = _pyhf_rates(model, em, x)
            scale = np.maximum(np.abs(theirs), 1.0)
        if not np.all(np.abs(ours - theirs) <= 1e-8 * scale):
            return f"expected rates disagree with pyhf (max rel {np.max(np.abs(ours - theirs) / scale):.2e})"
    return None


def _pyhf_sample_rates(model, em, x):
    """pyhf's expected main rates at x per sample ([sample, bin]); a POI-scaled sample's is pyhf's with its markers
    at 1 times its scaling at x."""
    import pyhf
    tl = pyhf.tensorlib

    def by_sample(v):
        return np.asarray(tl.tolist(model.main_model.expected_data(tl.astensor(list(map(float, v))),
                                                                   return_by_sample=True)))
    cfg = model.config
    rates = by_sample(x)
    off = np.cumsum([0] + [cfg.channel_nbins[c] for c in cfg.channels])
    for (ch, name), (c, monos, markers) in em.poi_scaling.items():
        i, ci = cfg.samples.index(name), cfg.channels.index(ch)
        b = slice(off[ci], off[ci + 1])
        unit = np.array(x, float)
        unit[[em.ps_index[P] for P in markers]] = 1.0
        f = c * min(np.prod([x[em.ps_index[P]] ** p for P, p in m.items() if p != 0]) for m in monos)
        rates[i, b] = by_sample(unit)[i, b] * f
    return rates


def _pyhf_rates(model, em, x):
    """pyhf's expected main rates at x, a POI-scaled sample's with its scaling (_pyhf_sample_rates)."""
    import pyhf
    tl = pyhf.tensorlib
    if not em.poi_scaling:
        return np.asarray(tl.tolist(model.expected_actualdata(tl.astensor(list(map(float, x))))))
    return _pyhf_sample_rates(model, em, x).sum(axis=0)


def probe_point(em, seed, total: bool = True):
    """A deterministic in-bounds point away from the init: each free parameter moved by up to 0.3 x its distance to
    the nearer bound (at most 0.3).  Total: an init outside its bounds is first taken to the nearer bound, and a step
    that is not a non-negative finite number is 0 (the draw still made, so the points after it are the same).
    ``total`` False: the shipped grader's point, where a free parameter's init outside its bounds raises
    (numpy's "high - low < 0"), which build reads as the twin not built."""
    rng = np.random.default_rng(seed)
    x = em.init.copy()
    for i, (lo, hi) in enumerate(em.bounds):
        if em.fixed[i]:
            continue
        if not total:
            step = 0.3 * min(x[i] - lo, hi - x[i], 1.0)
            x[i] = x[i] + rng.uniform(-step, step)
            continue
        xi = min(max(x[i], lo), hi) if lo <= hi else x[i]
        step = 0.3 * min(xi - lo, hi - xi, 1.0)
        step = step if np.isfinite(step) and step > 0 else 0.0
        x[i] = xi + rng.uniform(-step, step)
    return x


def pyhf_nll(model, x, data):
    """-log L through pyhf itself."""
    import pyhf
    return -float(np.asarray(pyhf.tensorlib.tolist(
        model.logpdf(pyhf.tensorlib.astensor(list(map(float, x))), pyhf.tensorlib.astensor(data))))[0])
