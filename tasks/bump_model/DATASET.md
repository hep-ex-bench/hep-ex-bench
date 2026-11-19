# DATASET — `bump_model` input description

A toy pseudo-dataset of reconstructed lepton-pair masses: a smoothly falling
background that may contain a narrow resonance.  Everything is synthetic.

## Layout (the same for every dataset your submission runs on)

    data/part_0000.root      the data: tree `Events`, one branch `mass`, the
                             reconstructed pair mass in GeV, kept in [100, 160)
    control/events.npy       a control sample: "Z → ℓℓ" pair masses in [75, 105) GeV
    mc/signal_mH110.npy … mc/signal_mH140.npy
                             signal simulation at 110, 115, …, 140 GeV
                             (20 000 pair masses each, simulated resolution, no detector response)
    mc/background.npy        background simulation (100 000 pair masses, no detector response)
    mc/control_mc.npy        control-sample simulation (no detector response)
    mc/cross_sections.json   the expected signal yield at mu = 1 as a function of the
                             resonance mass, the nominal background expectation and the
                             control sample's description
    lumi.json                {"lumi_fb": ...}

Every dataset carries the same integrated luminosity.

## Detector response

The lepton momentum scale of the data is not the simulated one: every
reconstructed pair mass in the data is a fixed unknown factor times its
simulated value.  The control sample carries the same factor, so it can be
measured there and transferred to the data.

## The development dataset

The public dataset has a resonance at mH = 125 GeV with signal strength
mu = 1, the nominal background normalisation and the nominal detector
response.
