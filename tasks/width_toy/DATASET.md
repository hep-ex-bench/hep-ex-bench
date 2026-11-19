# DATASET — `width_toy` (toy stand-in for the off-shell H → ZZ → 4ℓ width measurement)

## Definitions

ξ = Γ_H/Γ_SM, the Higgs width in units of its Standard-Model value.  Off-shell
(m4ℓ well above 125 GeV) the gg → H* → ZZ yield scales as μ_on·ξ, its
interference with the gg → ZZ continuum as √(μ_on·ξ), and the on-shell peak
at 125 GeV as μ_on (the on-shell signal strength).  The qq → ZZ continuum does
not interfere with the Higgs.

## Layout (identical for every seed)

    data/events.npy          reconstructed m4l [GeV] in [100, 1000) — the pseudo-data, one event per row
                             (an event's entry index is its row)
    mc/B_qq.npy              qq -> ZZ continuum (unit-weight events)
    mc/B_gg.npy              gg -> ZZ continuum, without the Higgs
    mc/S.npy                 gg -> H* -> ZZ alone, at xi = 1
    mc/SBI_xi1.npy           gg -> ZZ with the Higgs (signal + continuum + interference), at xi = 1
    mc/H_onshell.npy         on-shell H -> ZZ at m_H = 125 GeV
    mc/cross_sections.json   expected events in the full dataset for each MC sample at its generated point
                             (xi = 1, mu_on = 1, nominal normalisations); each MC event carries weight
                             expected_events / number of events in its file
    lumi.json                {"lumi_fb": 10.0, "lumi_unc_rel": 0.025}: the integrated luminosity and its
                             relative uncertainty

The MC is identical for every seed and carries no detector effects beyond
those of the data.  No MC is provided at any ξ ≠ 1.  The qq → ZZ normalisation
of the data may differ from the nominal one, and the on-shell signal strength
of a dataset is not given.  The public development seed is generated at
ξ = 1 and μ_on = 1 with nominal normalisations.
