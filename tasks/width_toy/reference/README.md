Reference pipeline for `width_toy` (numpy only, the base runtime): MC templates B_qq, B_gg, S, SBI and the on-shell
peak; binned Poisson likelihood in xi and mu_on with the qq -> ZZ normalisation and the luminosity constrained and
profiled at every point of a (mu_on, xi) grid; each quantity's value = its profile's refined minimum, its interval
= the q <= 1 piece holding it (xi: every piece).  Writes RESULT.json, MODEL.json (the same likelihood as a pyhf
workspace in positive templates with `_meta.poi_scaling`) and EVENTS.json.  Used for closure (sigma_ref, per-seed
baselines, the model verification's cap).
