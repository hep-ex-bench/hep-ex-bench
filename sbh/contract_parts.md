<!-- The core parts of every task's EVAL_CONTRACT.md: the submission form, outputs, hygiene, runtime and budgets.
A template (the unified tasks/EVAL_CONTRACT.in.md, or a task's own EVAL_CONTRACT.in.md) includes them as
{{core:<part>}}, {{hep:<part>}}, {{kind:<part>}} (the task's own kind's) or {{kind.<name>:<part>}} (a named kind's);
{{value:<name>}} is filled from the task (Task.contract_values; the unified contract's own names win there).  Each
part is the exact text between its opening marker line and its closing marker (`python -m sbh.contract render`
writes tasks/<task>/EVAL_CONTRACT.md). -->

<!-- part: intro -->
This file is part of the input given to the analysis agent.  Everything below
is checked mechanically.
<!-- /part -->

<!-- part: submission -->
A submission is a directory whose root contains an executable `run.sh`.  The
harness freezes the directory into an immutable snapshot and executes, from a
clean unpack:

    ./run.sh <dataset_dir> <output_dir>

- `<dataset_dir>` has exactly the layout of the development dataset
  (DATASET.md).<!-- /part -->

<!-- part: outputs -->
- `<output_dir>` exists and is empty; write `RESULT.json` there (GATE),
  and the other files your results take (the sections below name them).
  Output files are read only as regular files: a symlink there is never
  followed, and a symlinked output file counts as malformed.
- Exit code 0 signals success.  Non-zero exit, timeout, or a missing or
  malformed GATE file marks the run failed.
- Repeated execution on the same dataset must be deterministic (fix your
  random seeds; keep parallel reductions in a fixed order).  This is
  checked by the self-check only (`--determinism` compares the reported
  values between two runs); the evaluator runs each dataset once.
<!-- /part -->

<!-- part: hygiene -->
Hygiene, checked before freezing: no data files (`.npy`, `.root`, `.parquet`,
…), no committed outputs, no file named `truth.json`, `manifest.json` or
`suite_truth.json` (the evaluator's own names), no symlinks, ≤ 5 MB per file,
≤ 50 MB total.
{{value:vendoring}}
<!-- /part -->

<!-- part: runtime -->
Runtime: your submission runs in its own pixi environment.
The base environment is Python 3.14 with `numpy`, `uproot`, `awkward` and
`vector`.{{value:runtime}}  To use anything else — anything conda-forge or PyPI carries
(pixi resolves both: `pixi add` / `pixi add --pypi`) — put a `pixi.toml`
**and its `pixi.lock`**
at the submission root (create them with `pixi init` / `pixi add` inside
your submission directory); the harness materializes the environment from
your lockfile before the runs.  A manifest without a lockfile is a hygiene
violation; an environment that fails to build within 1500 s, or whose
installed size exceeds 4 GB, fails every run.  The runs themselves have
no network; the pipeline may read only `<dataset_dir>` and its own directory.
<!-- /part -->

<!-- part: budgets -->
Budgets: each evaluation dataset is run alone, on {{value:cpus}} (enforced as the
process's CPU affinity), with a {{value:memory_gib}} GiB memory cap (the run's total resident
memory, including the files it writes to `/tmp` — where `$TMPDIR` points —
which is held in memory; exceeding it kills the run) and {{value:wall_article}} {{value:wall_s}} s wall
clock (the run is killed at the limit); within it, faster is better.
<!-- /part -->

<!-- part: selfcheck_memory -->
The self-check runs on the
development dataset with the evaluation's wall clock, CPU count and network
isolation, prints its limits and reports the run's wall time and peak
resident memory (the whole process tree and its `/tmp` files) against the
{{value:memory_gib}} GiB cap: it fails when the peak exceeds the cap (the evaluation would
kill that run) and warns above half of it<!-- /part -->

<!-- part: selfcheck_memory_end -->
; it measures memory but does not stop the run at the
cap.
<!-- /part -->
