# DATASET — `h4l_width` input description

Real reconstructed collision-like events (simulation under 2016 CMS
conditions: CMS Open Data samples and private samples through the same
detector simulation) in NanoAOD format, slimmed to the collections
`Muon`, `Electron`, `FsrPhoton` and `TrigObj`, the `HLT_*` decisions, the
`PV` and `MET` summaries, and the event IDs (`run`, `luminosityBlock`,
`event`; in the data they label events uniquely within a dataset, MC
keeps its own).  The private MC samples also carry a pruned generator
record, `GenPart`; the data carry no generator information.

## Layout (the same for every dataset your submission runs on)

    data/part_*.root                # the dataset: slim NanoAOD, `Events` tree
    mc/<process>/*.root + meta.json # simulation, one directory per process
    mc/cross_sections.json          # cross sections used by the simulation
    lumi.json                       # {"lumi_fb": ..., "lumi_unc_rel": ...} integrated luminosity
                                    #   and its relative uncertainty
    meta.json                       # file count, event count, HLT paths kept, the trigger menu

The layout is shared, the size is not: each evaluation dataset carries
about twice the integrated luminosity of the public development dataset,
and every dataset's `lumi.json` gives its own value.

The production preselection (≥ 2 same-flavour leptons — nMuon ≥ 2 or
nElectron ≥ 2) is applied to data and MC alike; each MC process's
`meta.json` records its entry bookkeeping relative to that preselection.

## The trigger menu

The data were recorded by the lepton-multiplicity menu that the data's
`meta.json` documents under `hlt_menu`: every data event was recorded by at
least one of its paths.  A path fires on at least `min_leptons` menu
leptons (a menu lepton passes every cut under `lepton`), its stored bit is
post-prescale, and the prescales are nested — an event with exactly n menu
leptons was recorded with probability 1 / the prescale of the highest path
it qualifies for.  Every menu lepton is seen online (the menu is fully
efficient on menu leptons).  MC carries the three bits unprescaled.  The
other stored `HLT_*` paths are the production's decisions, not part of the
recording menu.

## The simulation

All simulation uses the nominal detector response and the Standard Model
Higgs width; no simulation at any other width is provided.

Higgs-width convention.  m_H = 125 GeV and Γ_H = ξ · Γ_SM with
Γ_SM = 4.097 MeV.  The Higgs couplings scale with the width such that
every on-shell H → ZZ rate keeps its Standard Model value ("on-shell
fixed"); the electroweak samples' H → bb̄, below, is the exception.
This convention defines the simulated samples; it fixes no dataset's
on-shell rate.  The `xi1` in a sample name means ξ = 1.

Normalisation.  The effective cross sections in `mc/cross_sections.json`
include every generator-level factor listed below.

**gg → (H\*) → ZZ → 4ℓ** (ℓ = e, μ), loop-induced, leading order in
MadGraph5_aMC@NLO 3.6.7 (interference model `loop_sm_modif`; G_μ scheme,
m_t = 172.5 GeV, m_b = 4.18 GeV, PDF4LHC15_nlo_100,
μ_R = μ_F = m4ℓ/2, fixed widths), showered with Pythia8 (CP5).  The width
enters through the top- and bottom-quark Yukawa couplings × √ξ and
Γ_H = ξ Γ_SM in the propagator.  Normalisation: σ_LO × K(m4ℓ), with K the
NNLO/LO gg→H K-factor as a function of the four-lepton mass, times 1.10,
the same K for every component; it is applied by accept–reject on the
generated events (the events are unweighted and their m4ℓ distribution
already carries K).  Below 73 GeV, K is held at its 73 GeV value.

| process | content | generator-level m4ℓ |
| --- | --- | --- |
| `GGZZ_S` | \|A_H\|², the Higgs-mediated amplitude squared | > 130 GeV |
| `GGZZ_B_high` | \|A_B\|², the continuum (no Higgs) | > 130 GeV |
| `GGZZ_B_low` | \|A_B\|², the continuum (no Higgs) | 70–130 GeV |
| `GGZZ_SBI_xi1` | \|A_H + A_B\|² at ξ = 1, interference included | > 130 GeV |

**Electroweak qq → 4ℓ jj** (ℓ = e, μ), tree level, purely electroweak
(no QCD vertex), MadGraph5_aMC@NLO 3.6.7 with the same inputs, showered
with Pythia8 (CP5, dipole recoil).  Every such diagram enters: vector-boson
fusion and scattering, and associated production, where the two quarks
come from the decay of a vector boson or of a Higgs boson.  The width
enters through every hWW and hZZ coupling × ξ^(1/4) and Γ_H = ξ Γ_SM in
the propagator; the other Higgs couplings are not modified.
Normalisation: σ_LO × 1.0766 for every component.  The components are
defined by the hVV vertices of the diagrams: S = the squared amplitude of
the diagrams with two hVV vertices (the Higgs-mediated amplitude), B = the
squared amplitude of the diagrams without an hVV vertex, S+B+I = the
complete squared amplitude.

| process | content | generator-level m4ℓ |
| --- | --- | --- |
| `EWZZ_S` | S; it includes qq → ZH with the Higgs on shell, H → ZZ\* → ℓℓ jj, the other lepton pair from the associated Z | > 130 GeV |
| `EWZZ_B_high` | B | > 130 GeV |
| `EWZZ_B_low` | B | 70–130 GeV |
| `EWZZ_SBI_xi1` | S + B + I at ξ = 1; it includes the Higgs-strahlung qq̄ → Z\* → Z\*H with H → bb̄ on shell, the four leptons from the Z\* pair | > 130 GeV |

Generator cuts of both families: lepton pT > 3 GeV and |η| < 2.7, every
same-flavour opposite-sign lepton pair m_ℓℓ > 2.5 GeV; for the electroweak
family also quark pT > 15 GeV, |η| < 6.5, ΔR > 0.3, m_qq > 30 GeV.  The
generator-level m4ℓ is the invariant mass of the four hard-process
leptons.

**On-shell Higgs, m_H = 125 GeV**, leading-order Pythia8, H → ZZ\* → 4ℓ
(ℓ = e, μ).  The Higgs lineshape is truncated to 124.75–125.25 GeV, and
the lighter Z is generated with a mass above 10 GeV, while the
normalisation is that of the full H → 4ℓ branching ratio.

| process | content |
| --- | --- |
| `GluGluToHToZZ_M125` | gluon fusion |
| `VBF_HToZZ_M125` | vector-boson fusion |
| `VHToZZ_M125` | WH and ZH at Pythia8's leading-order ratio; every Z of the event, the associated one included, decays to ee or μμ |

The samples split the generator-level four-lepton mass at 130 GeV
without overlap: below it the continuum is in `*_B_low` and the Higgs in
the on-shell samples; above it the S, B and S+B+I samples.  The Higgs
signal and interference below 130 GeV outside the truncated on-shell
lineshape are in no sample.

**Other processes**, the official CMS Open Data NanoAODv9 samples (2016):
`ZZTo4L` (qq̄ → ZZ → 4ℓ), `DYJetsToLL` (Drell–Yan, the inclusive
madgraphMLM sample) and `TTBar` (tt̄ → 2ℓ2ν).  The MC of each is a part of
its official sample; its `meta.json` counts that part (`n_selected`,
`n_preselection` and the generator-weight sum), so the normalisation gives
the whole sample's yield.

The data and the MC templates are disjoint: no simulated event appears in
both, and no event of the development dataset appears in an evaluation
dataset.  The Drell–Yan in the data is drawn from the inclusive DY sample
together with jet- and HT-binned samples of the same generator, so the data
follow the inclusive sample's cross section and kinematics; the MC keeps
the inclusive sample.  Where a generator bin (a number of partons and an
HT range) holds fewer simulated events than the luminosity implies, the
data's `meta.json` lists it under `known_shortfalls`: the recorded events
of that process, path and bin are a random subset of what the cross
section implies, so a data/MC comparison in a region dominated by them
shows the data short by the stated fraction, while the simulation is
normalised to the full yield.

`GenPart` (the private samples only; `pt`, `eta`, `phi`, `mass`, `pdgId`,
`status`, `statusFlags`, `genPartIdxMother`) keeps the quarks, leptons,
gluons and W, Z and Higgs bosons flagged `isHardProcess`, plus the last
copy of every lepton flagged `fromHardProcess`.  `genPartIdxMother` points
to the nearest kept ancestor (−1 if none).  The MC also carries
`genWeight` per event and the `Runs` tree.

## The development dataset

The single public dataset is SM-like: a Standard Model Higgs boson at
m_H = 125 GeV, ξ = 1, with on-shell signal strength μ_on = 1.
