# TASK CARD — `peak_calib` (toy calibration from a Z → μμ peak)

## Physics statement

A pseudo-dataset of Z → μμ events comes with its simulation.  The data's
muons differ from the simulated ones by a momentum scale, an extra pT
resolution and the efficiency of an identification selection.  Measure
these corrections from the Z peak and report them as functions with
uncertainties, as `request.json` asks (EVAL_CONTRACT.md §3–§4).

## Dataset layout (identical for every dataset)

    data/events.root            the pseudo-data: ~40 000 events
    mc/zmumu/events.root        the simulation: 150 000 events, the same in every dataset
    mc/zmumu/meta.json          {"genEventSumw": f}: the simulation's sum of generator weights
    mc/cross_sections.json      {"cross_sections_pb": {"zmumu": f}}
    lumi.json                   {"lumi_fb": f}: the data's integrated luminosity
    request.json                the requested corrections (EVAL_CONTRACT.md §4)

Each ROOT file holds one `Events` tree with NanoAOD-style branches, one
entry per event:

| branch | type | content |
|---|---|---|
| `nMuon` | int32 | number of reconstructed muons (pT > 5 GeV, \|η\| < 2.5) |
| `Muon_pt`, `Muon_eta`, `Muon_phi` | float32[nMuon] | their kinematics, pT descending |
| `Muon_charge` | int32[nMuon] | ±1 |
| `Muon_goodId` | bool[nMuon] | whether the muon passes the identification |
| `genWeight` | float32 | generator weight (simulation only) |

A simulated event's weight is `genWeight` × cross section × luminosity /
`genEventSumw`.  Every event is a Z → μμ decay (no background).

The public development dataset ships no `truth.json`; sealed truths are
never shipped.

## Sealed parameters (drawn per dataset)

| parameter | meaning | prior |
|---|---|---|
| `scale` | the data's momentum scale relative to the simulation, minus 1 | uniform −0.004 – 0.004 |
| `res_pt` | extra relative pT resolution of the data | uniform 0 – 0.015 |
| identification | data/simulation efficiency ratio of `Muon_goodId`, sf0 + sf1 \|η\| | sf0 uniform 0.95 – 1, sf1 uniform −0.03 – 0 |

The scale and the resolution are the same for every muon; the scale factor
depends on |η| only.  The simulation and the request are the same in every
dataset.
