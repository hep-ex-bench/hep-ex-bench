Reference pipeline for `peak_calib`: the muon scale and pT resolution fitted
to the data's Z peak by folding the simulation (soft mass bins, Gauss-Newton,
stat and mcstat covariance from the events' influences), the `Muon_goodId`
scale factor by tag and probe, fitted linearly in |η| with its parameter
covariance per source.  numpy, uproot and awkward only (the base
environment).  Used for closure validation and as sigma_ref; its
`--attack-*` switches make the generated attackers.
