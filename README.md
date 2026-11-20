# HEP-Ex-Bench

**HEP-Ex-Bench** benchmarks AI agents that carry out experimental high-energy physics measurements.  It evaluates an
agent's analysis quantitatively with blind analysis and injection tests, the checks at the heart of an expert review.
The agent develops a data analysis pipeline on a public dataset.  That single pipeline is then evaluated on sealed
worlds: private datasets generated programmatically, with injected signals and with simulation settings varied within
realistic detector uncertainties.  This tests both the accuracy and the robustness of the workflow.  The benchmarks
are built from CMS open data.

The pipeline is a directory with `run.sh <dataset> <out>` that writes `RESULT.json`.  It is frozen, run offline on
every sealed world and scored against the injected truths relative to an expert reference analysis
(`docs/METRICS.md`).  The sealed worlds and their truths are not public.

## Benchmark

| Benchmark | Task | Output |
|---|---|---|
| HiggsLepMass (`h4l`) | $\mu$ and $m_\mathrm{H}$ from $\mathrm{H} \to \mathrm{ZZ}^* \to 4\ell$, detector calibrations included | POIs and a binned likelihood |
| HiggsWidth (`h4l_width`) | the Higgs width ratio $\Gamma_\mathrm{H}/\Gamma_\mathrm{SM}$ and the on-shell signal strength from $\mathrm{H} \to \mathrm{ZZ} \to 4\ell$ | intervals and a binned likelihood whose rates scale with them |
| LepCalib (`lep_calib`) | lepton calibrations from $\mathrm{Z} \to \ell\ell$ | correctionlib corrections |

### Toys

Small tasks shipped whole, the worked examples for adding a task (`docs/ADDING_A_TASK.md`).

| Toy | Task | Output |
|---|---|---|
| `gauss_peak` | a Gaussian peak | POIs |
| `bump_model` | a resonance on a falling background | POIs and a binned likelihood |
| `peak_calib` | a muon scale, resolution and efficiency from a Z peak | correctionlib corrections |
| `width_toy` | a width ratio from an off-shell mass spectrum | intervals and a binned likelihood whose rates scale with them |

For details see our paper:
> *HEP-Ex-Bench: Evaluating Agentic Performance in Physics Analyses with Sealed Worlds*  
> A. Novák, Y. Zhao, D. Garcia, E. A. Moreno, J. Krupa, V. Bresó-Pla, S. Bright-Thonney, L. Heinrich, M. Kagan,
> P. Harris

## Usage

### Requirements

Linux x86-64, [pixi](https://pixi.sh) (it installs each environment on first use), and unprivileged user namespaces
for the bubblewrap sandbox in which submissions and agent sessions run (`pixi run sandbox` shows what the runner can
use).  Commands run from the repository's root unless a step says otherwise:

```bash
pixi run check     # consistency checks
pixi run smoke     # the checks plus the test batteries and the toys' suite builds, closures and stress
```

### Run an agent on a benchmark

1. Download the benchmark's public dataset (its data and its simulation) from
   [Hugging Face](https://huggingface.co/hep-ex-bench) and build the participant bundle from it (the dataset, the
   task's documents and the self-check tool).  `<benchmark>` is a name in the Benchmark column of the first
   table above and `<task>` the id beside it, for example `HiggsLepMass` and `h4l`:

   ```bash
   pixi exec --spec huggingface_hub hf download hep-ex-bench/<benchmark> --repo-type dataset --revision v0.2 \
       --local-dir work/data/<benchmark>
   pixi run python -m sbh pack --task <task> --public-dataset work/data/<benchmark> --out work/dist
   pixi install -e agentbase    # the submission's runtime (lep_calib: pixi install -e lepcalib)
   ```

2. Make a sandbox for the session.  Its prompt is `tasks/<task>/PROMPT.md`, or another tier's file from
   `tasks/<task>/prompts/` given as the second argument:

   ```bash
   SBH_BUNDLE=work/dist/<task>_public_participant scripts/fleet/make_h4l_sandbox.sh my_session [<prompt file>]
   ```

3. Run the agent with your own Claude Code, Codex or opencode login (the scripts serve every task):

   ```bash
   scripts/fleet/run_claude_h4l.sh   my_session <model>
   scripts/fleet/run_codex_h4l.sh    my_session [<model>]
   scripts/fleet/run_opencode_h4l.sh my_session <provider/model>
   ```

   The agent works confined in `work/agents/my_session/` for `AGENT_TIMEOUT` seconds (default 7200) and leaves its
   pipeline in `my_analysis/` and its log in `work/agents/my_session.<harness>.log`.  The sandbox and the login options:
   `docs/ISOLATION.md`.

4. Check the pipeline as the evaluation runs it, on the public dataset:

   ```bash
   cd work/agents/my_session
   pixi run --manifest-path ../../../pixi.toml python -m sbh check --task <task> --submission my_analysis \
       --dataset datasets/public/seed_000 --determinism
   ```

## Directory structure

```
sbh/            the harness: suites, the sandboxed runner, scoring, participant bundles
tasks/          one directory per task: the toys whole, the benchmarks' participant material
docs/           the score (METRICS.md), adding a task, isolation, model validation
checks/         the consistency checks and the test batteries
scripts/fleet/  the agent session scripts
pixi.toml       the environments and tasks
```

## License

The code is under Apache-2.0 (`LICENSE`, `NOTICE`), the public datasets under CC BY 4.0 (the `LICENSE` of each).
