# Isolation — what is sealed from whom, and how

Each boundary below names what enforces it and how strong it is.  The code
is the source of truth: `sbh/snapshot.py` (freeze), `sbh/runner.py`
(execution), `sbh/contract.py` (declared results).

## The map

```
                     what the other side must never see
participant/agent ──────────────────────────────────────► evaluator
  bundle: public seed,        │   suites/<s>/truth/, manifest.json,
  MC templates, docs,         │   sealed/ datasets' truths, injection
  self-check                  │   records, reference scores, designs/
```

Submitters see only aggregates; a dataset's truth is the answer key.

## Boundary 1 — freeze (what enters a submission)

`sbh/snapshot.py` (inside every evaluation command) snapshots a submission and
rejects contraband before anything runs (`hygiene`):

- forbidden data-like suffixes and the reserved file names (a committed
  output, the evaluator's `truth.json` / `manifest.json` /
  `suite_truth.json`); no symlinks;
- 5 MB per file, 50 MB total;
- no installed packages (a `*.dist-info` / `*.egg-info` tree or a
  `site-packages` directory anywhere): dependencies go in the submission's
  `pixi.toml` + `pixi.lock`, which the harness materializes before the runs.

Exception: a task whose verifier runs offline on a fixed runtime and cannot
build an environment names a vendor directory (`Task.vendor_dir`, 4 GB
budget): `pylib/` for the toy `gauss_peak`.
The suffix and reserved-name checks still apply there, except for data-like files
byte-identical (size + sha256) to the same relative path in the freezing
interpreter's own site-packages (`pip install --target` re-vendors
numpy/scipy data files; an unchanged copy of the evaluator's install is not
agent-authored).  A dist-info RECORD match is not accepted (agent-writable).

## Boundary 2 — the runner (what a pipeline can reach while running)

`sbh/runner.py` runs `run.sh <dataset> <out>` per seed from a clean unpack,
with wall/memory limits and a scrubbed environment.  Modes, best first; the
mode is recorded per run (`pixi run sandbox` shows the host's):

| mode | mechanism | network | sealed suite |
|---|---|---|---|
| bwrap | `bwrap --unshare-all --unshare-user --disable-userns`; the run's reaper (`sbh/reaper.py`) is PID 1; ro-binds of env prefixes + dataset only. | blocked | **not mounted** |
| setpriv | in a verifier container: drop to `nobody`; truth root-owned 700 | container's (none) | unreadable |
| unshare | `unshare -rn` user+net namespace | blocked | filesystem-visible (weaker) |
| none | bare subprocess | host | **NOT isolated** (recorded, dev only) |

`pixi run smoke` exercises the toy task's attackers (generated, `sbh/attackgen.py`).

Reading a run's output (all modes): `out/` belongs to the submission, so
`RESULT.json`, `MODEL.json` and anything else score/verify read from `out/`
go through `sbh.contract.read_output` — regular, singly-linked files only,
opened `O_NOFOLLOW | O_NONBLOCK`.  Under setpriv the grader is root over a
nobody sandbox on one filesystem; a followed symlink could copy the answer
key into `runs/<seed>/record.json` for a later seed to read.  So setpriv also
closes the grader's work tree to nobody:

- `runs/<seed>/` 0711 (a run reaches its own `repo/`, `out/`, `home/`, cannot
  list or read the rest);
- `record.json`, stdout/stderr 0600; `private/`, `snapshot/` 0700.

Tests: `attacker_symlink` (smoke).

A leftover process cannot outlive its run: the reaper (root, outside setpriv) kills every descendant
2 s after run.sh exits.

## Boundary 3 — the declared model (what a submission must prove)

`sbh/contract.py` and `tasks/<t>/EVAL_CONTRACT.md` make the analysis declare
its statistical model (MODEL.json, nuisance log with grammar-checked names).
The refit of the declared model (the score from the fit) and the
calibration/leakage responses make "a number without the analysis behind it"
a losing strategy.  This isolates claims from performance, not files.

## Boundary 4 — dev sessions (weak by construction)

Dev-harness sessions (`work/agents/<sandbox>`) run as the evaluator's Unix
user inside `scripts/fleet/dev_sandbox.sh` (bubblewrap).
Real participants never run this way; they get boundaries 1–3.

**Filesystem.**  The script header lists the intended semantics.

- Hidden roots (`SBH_DEV_HIDE`) are empty and read-only (writes fail with
  EROFS rather than vanishing into a tmpfs).
- `$HOME` is an ephemeral tmpfs.
- nproc reports the job's CPU allocation: `OMP_NUM_THREADS` and
  `OMP_THREAD_LIMIT` are unset.
- Each sandbox has its own `~/.cache` (`<sandbox>/.cache`, never a link) and
  pixi cache (`PIXI_CACHE_DIR=<sandbox>/.cache/rattler`).  The operator's
  `~/.cache` holds MCP logs and gh's cache, and a shared read-write package
  cache would let one agent poison another's installs.
- The task's base env is bound read-only at `.pixi/envs/<base_env>` so the
  bundle's `sbh check` uses the evaluation runtime.

**Environment.**  No operator credential or evaluator path reaches the agent:

- It unsets `SBH_*`; `HF_*`, `*TOKEN*`, `*SECRET*`, `*PASSWORD*` (except
  harness auth `CLAUDE_CODE_OAUTH_TOKEN`/`ANTHROPIC_AUTH_TOKEN`);
  `CLAUDECODE`, `CLAUDE_*` (except the sandbox's `CLAUDE_CONFIG_DIR`),
  `AI_AGENT`, `ANTHROPIC_*`, `SSH_*`, `TMUX*` (a launching session's
  `CLAUDE_CODE_EFFORT_LEVEL` would otherwise set the agent's effort); and
  any path variable pointing into a hidden root outside the binds (it would
  resolve to an empty read-only mask).
- Variables are removed before bwrap starts: bwrap's `--unsetenv` would leave
  them in its init's `/proc/1/environ`.

**Claude login.**  `scripts/fleet/run_claude_h4l.sh` authenticates a
session with a long-lived token when `SBH_CLAUDE_TOKEN_FILE` names one
(`claude setup-token`; an owner-only file with one line), else with a copy
of the running user's login.  A token sits in the session environment:
Claude Code keeps it out of its commands' environment, but the claude
process's and bwrap init's environments are readable via
`/proc/<pid>/environ`, so an agent that looks can find it.  A copied login
is refused unless it outlives the session (a refresh would log out every
other copy), and a token session refuses a sandbox that still holds a copy.

**Run record.**  `scripts/fleet/agent_run_record.py start` before a session
and `end` after it keep `work/agents/<sandbox>.run.json`: harness, model,
effort, harness version, exit code; for claude the auth method, and why a
session was refused (the session scripts add the version and the refusals
to a record that exists).

## Boundary 5 — the data itself

Generation knowledge is kept out of the shipped bytes: per-seed truths live
only in the sealed `truth.json`.
Participant-facing text has single sources (`DATASET.md` = input mechanics
only; prompts never restate the contract).
