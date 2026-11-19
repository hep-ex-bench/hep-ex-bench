"""The dev session's run record, work/agents/<sandbox>.run.json, and the checks the session scripts run before a
session starts.

    python scripts/fleet/agent_run_record.py start <sandbox> --harness H --model M [--effort E] [--prompt FILE] [--bundle DIR]
                                       [--budget S] [--suite DIR] [--unless-open]
    python scripts/fleet/agent_run_record.py left <sandbox> --budget S [--harness H --model M]
    python scripts/fleet/agent_run_record.py resume <sandbox> --budget S [--harness H --model M]
    python scripts/fleet/agent_run_record.py session-id <sandbox>
    python scripts/fleet/agent_run_record.py version <sandbox> --harness H [--auth MODE] [--value VERSION]
    python scripts/fleet/agent_run_record.py login-check <sandbox> --budget S [--config DIR]
    python scripts/fleet/agent_run_record.py token-check [<sandbox>] --file PATH [--emit]
    python scripts/fleet/agent_run_record.py refuse <sandbox> --reason TEXT
    python scripts/fleet/agent_run_record.py variant-check <sandbox> --model M --level L   < `opencode models <provider> --verbose`
    python scripts/fleet/agent_run_record.py vibe-config <sandbox> --model M --template TOML --out TOML
    python scripts/fleet/agent_run_record.py end <sandbox> --exit RC
    python scripts/fleet/agent_run_record.py --selftest

start: harness, model, effort (null: harness default), prompt, bundle, the suite the session is evaluated on, whole
budget, host, batch job; --unless-open: not over a record whose last segment is still open (exit 4).  end: the harness's exit code (last "<harness> exit=<code>" log line; 124 = budget kill), the
caller's, and what the harness's own record says ran (AS_RUN: claude init event, codex exec header, opencode session
store, Vibe export).  A session has one or more `segments`; `resume` opens the next and prints the budget left (whole budget less
previous segments' walls, gaps not counted); `left` only tells; `session-id` prints the session to resume (a harness
RESUMES names).  Each run_*_h4l.sh adds the harness version (`version`, plus --auth for claude) to an existing record.

Refusals (exit 3, reason on stderr and in the record):
  login-check    a copied claude login whose access token expires before budget + LOGIN_MARGIN_S: a refresh by any
                 holder of a copy rotates the shared refresh token and logs out everyone else.
  token-check    SBH_CLAUDE_TOKEN_FILE unless an owner-only regular file with one token line; --emit prints it.
  variant-check  an opencode effort variant the model's listing does not define (opencode would silently run
                 without it); records the variant definition.
  vibe-config    a Vibe model the config template does not define (Vibe would run its default model); writes
                 the session's config with the model pinned and records the model's definition.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import socket
import sqlite3
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

AGENTS = Path(__file__).resolve().parents[2] / "work" / "agents"
EFFORT_HARNESSES = ("claude", "codex", "opencode")
VERSION_ARGS = {h: [h, "--version"] for h in ("claude", "codex", "opencode", "omp", "agy", "vibe")}
LOGIN_MARGIN_S = 900
TOKEN_LINE = re.compile(r"[A-Za-z0-9_-]{20,1024}")
TOKEN_FILE_MAX = 65536
RESUME_MIN_S = 600
RESUMES = {"fleet": ("claude", "codex")}


def path_of(sandbox: str) -> Path:
    return AGENTS / f"{sandbox}.run.json"


def load(sandbox: str):
    try:
        return json.loads(path_of(sandbox).read_text())
    except (OSError, ValueError):
        return None


def _write(sandbox: str, rec: dict) -> None:
    p = path_of(sandbox)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    os.replace(tmp, p)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def harness_version(harness: str):
    """First line of `<harness> --version` on the caller's PATH, or None."""
    cmd = VERSION_ARGS.get(harness)
    if not cmd:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = ((out.stdout or "").strip() or (out.stderr or "").strip()).splitlines()
    return lines[0] if out.returncode == 0 and lines else None


def harness_exit(sandbox: str, offset: int = 0):
    """Last "<harness> exit=<code>" in the sandbox's logs at or after `offset` (this segment's start), or None."""
    for log in sorted(AGENTS.glob(f"{sandbox}.*.log")):
        try:
            with open(log, "rb") as fh:
                fh.seek(max(offset, log.stat().st_size - 4096))
                m = re.findall(rb"\b[a-z]+ exit=(\d+)", fh.read())
        except OSError:
            continue
        if m:
            return int(m[-1])
    return None


def log_size(sandbox: str, harness: Optional[str]) -> int:
    """Size of <sandbox>.<harness>.log (0 if none): the next segment's offset."""
    try:
        return (AGENTS / f"{sandbox}.{harness}.log").stat().st_size if harness else 0
    except OSError:
        return 0


def _iso(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t))


def _epoch(s: str) -> float:
    from datetime import datetime
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S%z").timestamp()


def segments(rec: dict) -> list:
    """Segments oldest first; a pre-segments record yields one from its top-level fields."""
    if rec.get("segments"):
        return rec["segments"]
    seg = {k: rec[k] for k in ("started", "ended", "exit_code", "launcher_exit", "host", "slurm_job_id") if k in rec}
    return [seg] if seg.get("started") else []


def used_s(rec: dict) -> float:
    """Wall time of finished segments, gaps not counted."""
    return sum(_epoch(s["ended"]) - _epoch(s["started"]) for s in segments(rec) if s.get("started") and s.get("ended"))


def refuse(sandbox, reason: str) -> int:
    """Return 3; reason to stderr (stdout may carry a token) and into the run record if any."""
    print(f"refused: {reason}", file=sys.stderr)
    rec = load(sandbox) if sandbox else None
    if rec is not None:
        rec["refused"] = reason
        _write(sandbox, rec)
    return 3


def token_check(path: str, sandbox=None, emit: bool = False) -> int:
    """0 if `path` is an owner-only regular UTF-8 file with exactly one non-blank line matching TOKEN_LINE (printed
    with `emit`), else `refuse` without ever printing the token.  Non-blocking open: a FIFO is refused."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
    except OSError as e:
        return refuse(sandbox, f"the token file {path!r} cannot be opened ({e.strerror})")
    why, data = None, b""
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            why = "is not a regular file"
        elif st.st_mode & 0o077:
            why = f"is open to group or others (mode {stat.S_IMODE(st.st_mode):o}): chmod 600 it"
        else:
            while len(data) <= TOKEN_FILE_MAX:
                chunk = os.read(fd, TOKEN_FILE_MAX + 1 - len(data))
                if not chunk:
                    break
                data += chunk
    except OSError as e:
        why = f"cannot be read ({e.strerror})"
    finally:
        os.close(fd)
    if why:
        return refuse(sandbox, f"the token file {path!r} {why}")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = ""
    lines = [line.strip(" \t") for line in text.replace("\r\n", "\n").split("\n") if line.strip(" \t")]
    if len(data) > TOKEN_FILE_MAX or len(lines) != 1 or not TOKEN_LINE.fullmatch(lines[0]):
        return refuse(sandbox, f"the token file {path!r} does not hold one token line (claude setup-token)")
    if emit:
        print(lines[0])
    return 0


def login_check(sandbox: str, budget_s: float, config=None) -> int:
    """0 if the login in `config` (else CLAUDE_CONFIG_DIR, else <sandbox>/.claude_cfg) outlives budget +
    LOGIN_MARGIN_S, else `refuse` (prints only the expiry).  Called before the login is copied."""
    cfg = Path(config or os.environ.get("CLAUDE_CONFIG_DIR") or AGENTS / sandbox / ".claude_cfg")
    try:
        exp = float(json.loads((cfg / ".credentials.json").read_text())["claudeAiOauth"]["expiresAt"]) / 1000
    except (OSError, ValueError, KeyError, TypeError):
        return refuse(sandbox, f"no readable login in {cfg}")
    rec = load(sandbox)
    if rec is not None:
        rec["login_expires"] = _iso(exp)
        _write(sandbox, rec)
    need = time.time() + budget_s + LOGIN_MARGIN_S
    if exp < need:
        return refuse(sandbox, f"the access token of the login in {cfg} expires at {_iso(exp)}, before this "
                               f"session's budget ends (+{LOGIN_MARGIN_S} s: {_iso(need)}); the session would refresh "
                               "it, logging out every other holder of a copy: use a long-lived token "
                               "(SBH_CLAUDE_TOKEN_FILE, from `claude setup-token`)")
    return 0


def claude_as_run(log: Path):
    """From the init events (one per segment): the first's model, version, permission mode and key source, and the
    last's session id (what `claude --resume` continues); None without one."""
    first = sid = None
    try:
        with open(log, errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                if ev.get("type") == "system" and ev.get("subtype") == "init":
                    first = first or {k: ev.get(k) for k in ("model", "claude_code_version", "permissionMode",
                                                             "apiKeySource")}
                    sid = ev.get("session_id") or sid
    except OSError:
        return None
    return dict(first, session_id=sid) if first else None


CODEX_HEADER = {"model": "model", "reasoning effort": "reasoning_effort", "approval": "approval", "sandbox": "sandbox",
                "session id": "session_id"}


def codex_as_run(log: Path):
    """Parse the first `codex exec` header ("OpenAI Codex v<version>", then "<key>: <value>" lines between "--------"
    markers) into CODEX_HEADER fields (its session id: what `codex exec resume` continues); None without one."""
    out, state = None, "version"
    try:
        with open(log, errors="replace") as f:
            for line in f:
                line = line.rstrip("\r\n")
                if state == "version":
                    m = re.match(r"OpenAI Codex v(\S+)", line)
                    if m:
                        out, state = {"codex_version": m.group(1)}, "open"
                elif state == "open":
                    if line.startswith("--------"):
                        state = "keys"
                elif line.startswith("--------"):
                    break
                else:
                    key, sep, val = line.partition(": ")
                    if not sep:
                        break
                    if key in CODEX_HEADER:
                        out[CODEX_HEADER[key]] = val.strip()
    except OSError:
        return None
    return out


def variant_check(sandbox: str, listing: str, model: str, level: str) -> int:
    """0 if the `opencode models <provider> --verbose` listing defines variant `level` for `model` (recorded as
    `variant`), else refuse()."""
    parts = re.split(r"^(\S+/\S+)\n", listing, flags=re.M)
    models = dict(zip(parts[1::2], parts[2::2]))
    if not models:
        return refuse(sandbox, f"opencode listed no models for {model} (opencode models failed or printed nothing)")
    if model not in models:
        return refuse(sandbox, f"opencode lists no model {model} (opencode models --verbose)")
    try:
        variants = json.loads(models[model]).get("variants")
    except (ValueError, AttributeError):
        variants = None
    if not isinstance(variants, dict) or not isinstance(variants.get(level), dict):
        known = (", ".join(sorted(variants)) or "none") if isinstance(variants, dict) else "none readable"
        return refuse(sandbox, f"opencode defines no variant {level!r} for {model} (its variants: {known})")
    rec = load(sandbox)
    if rec is not None:
        rec["variant"] = {"name": level, "definition": variants[level]}
        _write(sandbox, rec)
    return 0


def opencode_as_run(db: Path):
    """Model and variant (as reasoning_effort; distinct values ","-joined) and session id of the top-level session
    (first without a parent: not a subagent's) in opencode's store; None without one."""
    try:
        con = sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)
        try:
            row = con.execute("select id from session where parent_id is null order by time_created limit 1").fetchone()
            data = con.execute("select data from message where session_id = ? order by time_created",
                               (row[0],)).fetchall() if row else []
        finally:
            con.close()
    except sqlite3.Error:
        return None
    if not row:
        return None
    models, variants = [], []
    for (d,) in data:
        try:
            m = json.loads(d)
        except (TypeError, ValueError):
            continue
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        model = "/".join(m[k] for k in ("providerID", "modelID") if isinstance(m.get(k), str) and m[k])
        variant = m.get("variant") if isinstance(m.get("variant"), str) else None
        for seen, v in ((models, model), (variants, variant)):
            if v and v not in seen:
                seen.append(v)
    out = {"model": ",".join(models) or None, "session_id": row[0]}
    if variants:
        out["reasoning_effort"] = ",".join(variants)
    return out


def vibe_config(sandbox: str, model: str, template: Path, out: Path) -> int:
    """Write Vibe's config for the session to `out`: `template` with `active_model` set to `model` and
    `allowed_models` to its API name alone, so no other model can run in the session; the definition is recorded as
    `vibe_model`.  Refused: a model the template does not define (Vibe would run its
    default model) or a template that sets either key itself.  `out` is replaced, never written through (an
    earlier agent in the sandbox could leave a link there)."""
    import tomllib
    try:
        text = template.read_text()
        cfg = tomllib.loads(text)
    except (OSError, ValueError) as e:
        return refuse(sandbox, f"the Vibe config {template} cannot be read ({e})")
    models = cfg.get("models") if isinstance(cfg.get("models"), dict) else {}
    pinned = [k for k in ("active_model", "allowed_models") if k in cfg]
    if pinned:
        return refuse(sandbox, f"{template} sets {', '.join(pinned)} itself: the session's model goes there")
    if not isinstance(models.get(model), dict) or not isinstance(models[model].get("name"), str):
        return refuse(sandbox, f"{template} defines no Vibe model {model!r} with an API name (its models: "
                               f"{', '.join(sorted(models)) or 'none'})")
    fd, tmp = tempfile.mkstemp(dir=out.parent, prefix=f".{out.name}.")
    with os.fdopen(fd, "w") as fh:
        fh.write(f"active_model = {json.dumps(model)}\nallowed_models = [{json.dumps(models[model]['name'])}]\n\n{text}")
    os.replace(tmp, out)
    rec = load(sandbox)
    if rec is not None:
        rec["vibe_model"] = dict(models[model], alias=model)
        _write(sandbox, rec)
    return 0


def vibe_as_run(export: Path):
    """Model (API name and alias), thinking level (as reasoning_effort), version, session id, outcome and stop reason
    from Vibe's export (`vibe -p --output-dir` writes it at every exit); None without one."""
    try:
        d = json.loads(export.read_text()) if export.is_file() else None
    except (OSError, ValueError):
        return None
    if not isinstance(d, dict):
        return None
    cfg = d.get("config") if isinstance(d.get("config"), dict) else {}
    alias = cfg.get("active_model") or cfg.get("routed_default_model") or None
    m = (cfg.get("models") or {}).get(alias) if isinstance(cfg.get("models"), dict) else None
    m = m if isinstance(m, dict) else {}
    return {"model": m.get("name"), "alias": alias, "reasoning_effort": m.get("thinking"),
            "vibe_version": d.get("vibe_version"), "session_id": d.get("session_id"),
            "outcome": d.get("outcome"), "stop_reason": d.get("stop_reason")}


AS_RUN = {"claude": (lambda sb: AGENTS / f"{sb}.claude.log", claude_as_run),
          "codex": (lambda sb: AGENTS / f"{sb}.codex.log", codex_as_run),
          "opencode": (lambda sb: AGENTS / sb / ".xdg_data" / "opencode" / "opencode.db", opencode_as_run),
          "vibe": (lambda sb: AGENTS / sb / ".vibe_home" / "export" / "export.json", vibe_as_run)}


def resume(sandbox: str, budget_s: float, harness=None, model=None, write: bool = True):
    """Open the next segment (only tell if not `write`): (budget left, wall used) in seconds, or `refuse` (no record,
    harness/model mismatch, a segment open, < RESUME_MIN_S left).  Whole budget: the record's, else `budget_s`."""
    rec = load(sandbox)
    segs = segments(rec) if rec else []
    say = (lambda why: refuse(sandbox, why)) if write else (lambda why: refuse(None, why))
    if not segs:
        return say("no run record with a finished segment to resume")
    for k, v in (("harness", harness), ("model", model)):
        if v and rec.get(k) != v:
            return say(f"its record's {k} is {rec.get(k)!r}, not {v!r}")
    if not segs[-1].get("ended"):
        return say("its last segment has not ended (still running, or its job died without the end step: close it "
                   "with `agent_run_record.py end <sandbox> --exit <code> --ended <when its job ended>`)")
    if segs[-1].get("exit_code") == 0:
        return say("its last segment ended by itself (exit 0): nothing interrupted it")
    whole = float(rec.get("budget_s") or budget_s)
    used = used_s(rec)
    left = int(whole - used)
    if left < RESUME_MIN_S:
        return say(f"only {left} s of its {int(whole)} s budget left (resumed only with {RESUME_MIN_S} s)")
    if write:
        rec["segments"] = segs + [{"started": _now(), "budget_s": left, "host": socket.getfqdn(),
                                   "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                                   "log_offset": log_size(sandbox, rec.get("harness"))}]
        rec["budget_s"] = whole
        rec.pop("refused", None)
        _write(sandbox, rec)
    return left, int(used)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    if argv is None:
        argv = sys.argv[1:]
    if argv == ["--selftest"]:
        return selftest()
    ap.add_argument("step", choices=("start", "left", "resume", "session-id", "version", "login-check",
                                     "token-check", "refuse", "variant-check", "vibe-config", "end"))
    ap.add_argument("sandbox", nargs="?", help="the session's sandbox (token-check: optional, for its refusal)")
    ap.add_argument("--harness")
    ap.add_argument("--model")
    ap.add_argument("--effort", default="")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--bundle", default="")
    ap.add_argument("--suite", default="", help="start: the suite the session's submission is evaluated on")
    ap.add_argument("--auth", default="", help="version: how the session authenticates (claude: token | login-copy)")
    ap.add_argument("--value", default=None, help="version: the version that ran (a harness installed elsewhere, e.g. "
                                                  "in a container), instead of asking the one on PATH")
    ap.add_argument("--budget", type=float, default=None,
                    help="start, resume: the session's whole budget [s]; login-check: the budget it must outlive")
    ap.add_argument("--config", default=None, help="login-check: the Claude config holding the login to check")
    ap.add_argument("--reason", default="", help="refuse: why the session was not started")
    ap.add_argument("--level", default="", help="variant-check: the opencode variant (the session's effort)")
    ap.add_argument("--template", default=None, help="vibe-config: the Vibe config template")
    ap.add_argument("--out", default=None, help="vibe-config: the session's Vibe config to write")
    ap.add_argument("--file", default=None, help="token-check: the token file (SBH_CLAUDE_TOKEN_FILE)")
    ap.add_argument("--emit", action="store_true", help="token-check: print the token when the file is usable")
    ap.add_argument("--exit", type=int, default=None)
    ap.add_argument("--unless-open", action="store_true", help="start: not over a record with an open segment")
    ap.add_argument("--ended", default=None,
                    help="end: when the segment ended (%%Y-%%m-%%dT%%H:%%M:%%S%%z; default now): to close a segment "
                         "whose job died without the end step")
    a = ap.parse_args(argv)
    if a.step != "token-check" and not a.sandbox:
        ap.error(f"{a.step} needs a sandbox")
    if a.step in ("login-check", "left", "resume") and a.budget is None:
        ap.error(f"{a.step} needs --budget")
    if a.step == "token-check" and a.file is None:
        ap.error("token-check needs --file")
    if a.step == "refuse" and not a.reason:
        ap.error("refuse needs --reason")
    if a.step == "variant-check" and not (a.model and a.level):
        ap.error("variant-check needs --model and --level")
    if a.step == "vibe-config" and not (a.model and a.template and a.out):
        ap.error("vibe-config needs --model, --template and --out")
    if a.step == "start":
        old = segments(load(a.sandbox) or {})
        if a.unless_open and old and not old[-1].get("ended"):
            return 4
        seg = {"started": _now(), "host": socket.getfqdn(), "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
        rec = {"sandbox": a.sandbox, "harness": a.harness, "model": a.model, "effort": a.effort or None,
               "prompt": a.prompt or "PROMPT.md (the bundle's, P1)", "bundle": a.bundle or None,
               "suite": str(Path(a.suite).absolute()) if a.suite else None,
               "budget_s": a.budget, **seg, "segments": [dict(seg, budget_s=a.budget)]}
    elif a.step in ("left", "resume"):
        got = resume(a.sandbox, a.budget, a.harness, a.model, write=a.step == "resume")
        if not isinstance(got, tuple):
            return got
        print(*got)
        return 0
    elif a.step == "session-id":
        rec = load(a.sandbox) or {}
        h, runner = rec.get("harness") or "claude", rec.get("runner") or "fleet"
        if h not in RESUMES.get(runner, ()):
            return refuse(a.sandbox, f"a {h} session cannot be resumed under the {runner} runner "
                                     f"({', '.join(RESUMES.get(runner, ())) or 'none'} only)")
        source, parse = AS_RUN[h]
        sid = (rec.get("as_run") or {}).get("session_id") or (parse(source(a.sandbox)) or {}).get("session_id")
        if not sid:
            return refuse(a.sandbox, f"no {h} session id in its record or log")
        print(sid)
        return 0
    elif a.step == "login-check":
        return login_check(a.sandbox, a.budget, a.config)
    elif a.step == "token-check":
        return token_check(a.file, a.sandbox, a.emit)
    elif a.step == "refuse":
        return refuse(a.sandbox, a.reason)
    elif a.step == "variant-check":
        return variant_check(a.sandbox, sys.stdin.read(), a.model, a.level)
    elif a.step == "vibe-config":
        return vibe_config(a.sandbox, a.model, Path(a.template), Path(a.out))
    elif a.step == "version":
        rec = load(a.sandbox)
        if rec is None:
            return 0
        rec["harness_version"] = a.value if a.value is not None else harness_version(a.harness)
        if a.auth:
            rec["auth"] = a.auth
    else:
        rec = load(a.sandbox) or {"sandbox": a.sandbox}
        segs = segments(rec)
        if a.ended is not None:
            try:
                _epoch(a.ended)
            except ValueError:
                ap.error(f"--ended {a.ended!r} is no %Y-%m-%dT%H:%M:%S%z time")
        offset = int(segs[-1].get("log_offset") or 0) if segs and not segs[-1].get("ended") else 0
        end = {"ended": a.ended or _now(), "exit_code": harness_exit(a.sandbox, offset), "launcher_exit": a.exit}
        if segs and not segs[-1].get("ended"):
            segs[-1].update(end)
            rec["segments"] = segs
        rec.update(end)
        if rec.get("harness") in AS_RUN:
            source, parse = AS_RUN[rec["harness"]]
            rec["as_run"] = parse(source(a.sandbox))
    _write(a.sandbox, rec)
    return 0


def selftest() -> int:
    """Every subcommand through main() on a scratch agents directory: start/version/end and the as-run parsers,
    resume and segment accounting (claude and codex), refuse, token-check on usable and many unusable token files
    (never leaking the token), and login-check by config source.  Runs in smoke."""
    global AGENTS
    bad, keep, keep_cfg = [], AGENTS, os.environ.pop("CLAUDE_CONFIG_DIR", None)

    def run(*argv):
        with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err:
            try:
                rc = main(list(argv))
            except SystemExit as e:
                rc = e.code
        return rc, out.getvalue(), err.getvalue()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            AGENTS = Path(tmp)
            run("version", "sb0", "--harness", "claude", "--auth", "token")
            bad += [] if not path_of("sb0").exists() else ["version without start wrote a record"]
            run("start", "sb1", "--harness", "claude", "--model", "m", "--effort", "high")
            run("version", "sb1", "--harness", "nosuch-harness", "--auth", "token")
            init = {"type": "system", "subtype": "init", "model": "m", "claude_code_version": "9",
                    "permissionMode": "bypassPermissions", "apiKeySource": "none"}
            (AGENTS / "sb1.claude.log").write_text(json.dumps(init) + "\nnoise\nclaude exit=124\n")
            run("end", "sb1", "--exit", "0")
            r = load("sb1") or {}
            want = {"effort": "high", "auth": "token", "harness_version": None, "exit_code": 124, "launcher_exit": 0}
            bad += [f"sb1 {k}: {r.get(k)!r}, want {v!r}" for k, v in want.items() if r.get(k) != v]
            bad += [] if (r.get("as_run") or {}).get("model") == "m" else ["sb1: no init event"]
            bad += [] if "runner" not in r else [f"sb1: a session without --runner records runner {r['runner']!r}"]
            run("start", "sbx", "--harness", "codex", "--model", "gpt-x", "--effort", "high")
            (AGENTS / "sbx.codex.log").write_text(
                "Reading additional input from stdin...\nOpenAI Codex v0.160.1\n--------\nworkdir: /w\nmodel: gpt-x\n"
                "provider: openai\napproval: never\nsandbox: danger-full-access\nreasoning effort: high\n"
                "reasoning summaries: none\nsession id: 01a1\n--------\nuser\nmodel: not the header\ncodex exit=0\n")
            run("end", "sbx", "--exit", "0")
            r = load("sbx") or {}
            want = {"codex_version": "0.160.1", "model": "gpt-x", "reasoning_effort": "high", "approval": "never",
                    "sandbox": "danger-full-access", "session_id": "01a1"}
            bad += [] if (r.get("as_run"), r.get("exit_code")) == (want, 0) else [
                f"sbx (codex): as_run {r.get('as_run')}, exit {r.get('exit_code')}"]
            (AGENTS / "sby.codex.log").write_text("OpenAI Codex v1\n--------\nmodel: gpt-x\nuser\nmodel: not it\n")
            bad += [] if codex_as_run(AGENTS / "sby.codex.log") == {"codex_version": "1", "model": "gpt-x"} else [
                f"a codex header without its closing marker: {codex_as_run(AGENTS / 'sby.codex.log')}"]
            rc, out, err = run("session-id", "sbx")
            bad += [] if (rc, out.strip()) == (0, "01a1") else [f"session-id of a codex record: rc {rc}, {out!r}"]
            _write("sbz", {"sandbox": "sbz", "harness": "codex", "model": "gpt-x"})
            (AGENTS / "sbz.codex.log").write_text("OpenAI Codex v1\n--------\nmodel: gpt-x\n--------\ncodex exit=1\n")
            rc, out, err = run("session-id", "sbz")
            bad += [] if rc == 3 and not out and "no codex session id" in err else [
                f"session-id of a codex session without one: rc {rc}, {err!r}"]
            run("start", "sbo", "--harness", "opencode", "--model", "prov/m", "--effort", "max")
            store = AGENTS / "sbo" / ".xdg_data" / "opencode"
            store.mkdir(parents=True)
            con = sqlite3.connect(store / "opencode.db")
            con.execute("create table session (id text, parent_id text, time_created integer, model text)")
            con.execute("create table message (id text, session_id text, time_created integer, data text)")
            con.executemany("insert into session values (?, ?, ?, ?)", [("ses_sub", "ses_run", 0, None),
                                                                         ("ses_run", None, 1, None)])
            msgs = [("ses_run", {"role": "user", "model": {"providerID": "prov", "modelID": "m", "variant": "max"}}),
                    ("ses_run", {"role": "assistant", "providerID": "prov", "modelID": "m", "variant": "max"}),
                    ("ses_sub", {"role": "assistant", "providerID": "prov", "modelID": "small", "variant": "low"}),
                    ("ses_run", {"role": "assistant", "providerID": "prov", "modelID": "m", "variant": "max"})]
            con.executemany("insert into message values (?, ?, ?, ?)",
                            [(f"msg{i}", sid, i, json.dumps(d)) for i, (sid, d) in enumerate(msgs)])
            con.commit()
            con.close()
            (AGENTS / "sbo.opencode.log").write_text("> build · m\nopencode exit=0\n")
            run("end", "sbo", "--exit", "0")
            r = load("sbo") or {}
            want = {"model": "prov/m", "session_id": "ses_run", "reasoning_effort": "max"}
            bad += [] if (r.get("as_run"), r.get("exit_code"), r.get("effort")) == (want, 0, "max") else [
                f"sbo (opencode): as_run {r.get('as_run')}, exit {r.get('exit_code')}, effort {r.get('effort')}"]
            listing = ('INFO noise line\nprov/m\n' + json.dumps({"id": "m", "variants": {
                "max": {"chat_template_kwargs": {"reasoning_effort": "max"}}, "low": {}}}, indent=2)
                + '\nprov/k\n' + json.dumps({"id": "k", "variants": {}}, indent=2) + '\n')
            for args, text, want_rc, word in (
                    (("--model", "prov/m", "--level", "max"), listing, 0, None),
                    (("--model", "prov/k", "--level", "max"), listing, 3, "no variant 'max' for prov/k"),
                    (("--model", "prov/x", "--level", "max"), listing, 3, "lists no model prov/x"),
                    (("--model", "prov/m", "--level", "max"), "", 3, "listed no models"),
                    (("--model", "prov/m", "--level", "max"), listing[:60], 3, "no variant 'max'")):
                stdin = sys.stdin
                sys.stdin = io.StringIO(text)
                try:
                    rc, _, err = run("variant-check", "sbo", *args)
                finally:
                    sys.stdin = stdin
                r = load("sbo") or {}
                ok = rc == want_rc and (word is None or word in err)
                if want_rc == 0:
                    ok = ok and (r.get("variant") or {}).get("definition") == {"chat_template_kwargs":
                                                                                {"reasoning_effort": "max"}}
                bad += [] if ok else [f"variant-check {args} on a {len(text)}-character listing: rc {rc}, {err!r}"]
            con = sqlite3.connect(store / "opencode.db")
            con.executemany("insert into message values (?, ?, ?, ?)",
                            [("msg9", "ses_run", 9, None), ("msg10", "ses_run", 10, "not json"),
                             ("msg11", "ses_run", 11, json.dumps({"role": "assistant", "providerID": "prov",
                                                                  "modelID": "m", "variant": 3})),
                             ("msg12", "ses_run", 12, json.dumps({"role": "assistant", "providerID": "prov",
                                                                  "modelID": "m", "variant": "low"}))])
            con.commit()
            con.close()
            got = opencode_as_run(store / "opencode.db")
            bad += [] if got == {"model": "prov/m", "session_id": "ses_run", "reasoning_effort": "max,low"} else [
                f"opencode store with odd messages and two variants: {got}"]
            child = AGENTS / "sbc" / "opencode.db"
            child.parent.mkdir(parents=True)
            con = sqlite3.connect(child)
            con.execute("create table session (id text, parent_id text, time_created integer, model text)")
            con.execute("create table message (id text, session_id text, time_created integer, data text)")
            con.execute("insert into session values ('ses_sub', 'ses_gone', 1, null)")
            con.commit()
            con.close()
            bad += [] if opencode_as_run(child) is None else ["an opencode store with only a subagent's session"]
            run("start", "sbn", "--harness", "opencode", "--model", "prov/m")
            (AGENTS / "sbn.opencode.log").write_text("opencode exit=1\n")
            run("end", "sbn", "--exit", "0")
            r = load("sbn") or {}
            bad += [] if (r.get("as_run"), r.get("exit_code"), r.get("effort")) == (None, 1, None) else [
                f"sbn (opencode, no store): as_run {r.get('as_run')}, exit {r.get('exit_code')}"]
            rc, out, err = run("session-id", "sbn")
            bad += [] if rc == 3 and not out and "opencode session cannot be resumed" in err else [
                f"session-id of an opencode record: rc {rc}, {err!r}"]
            run("start", "sbv", "--harness", "vibe", "--model", "big")
            home = AGENTS / "sbv" / ".vibe_home"
            (home / "export").mkdir(parents=True)
            tpl = AGENTS / "vibe_config.toml"
            tpl.write_text('enable_telemetry = false\n\n[models."big"]\nname = "big-2610"\nprovider = "mistral"\n'
                           'thinking = "high"\n')
            (AGENTS / "elsewhere.toml").write_text("kept\n")
            (home / "config.toml").symlink_to(AGENTS / "elsewhere.toml")
            for model, template, want_rc, word in (("small", tpl, 3, "no Vibe model 'small'"),
                                                   ("big", AGENTS / "missing.toml", 3, "cannot be read"),
                                                   ("big", tpl, 0, None)):
                rc, _, err = run("vibe-config", "sbv", "--model", model, "--template", str(template),
                                 "--out", str(home / "config.toml"))
                bad += [] if rc == want_rc and (word is None or word in err) else [
                    f"vibe-config {model} from {template.name}: rc {rc}, {err.strip()}"]
            import tomllib
            got = tomllib.loads((home / "config.toml").read_text())
            bad += [] if (got.get("active_model"), got.get("allowed_models"), (AGENTS / "elsewhere.toml").read_text()
                          ) == ("big", ["big-2610"], "kept\n") else [
                f"vibe-config wrote active_model {got.get('active_model')!r}, allowed_models "
                f"{got.get('allowed_models')!r} or wrote through a link"]
            (AGENTS / "pinned.toml").write_text('active_model = "big"\n' + tpl.read_text())
            rc, _, err = run("vibe-config", "sbv", "--model", "big", "--template", str(AGENTS / "pinned.toml"),
                             "--out", str(home / "config.toml"))
            bad += [] if rc == 3 and "sets active_model" in err else [f"vibe-config, template pinned: rc {rc}"]
            (AGENTS / "allowed.toml").write_text('allowed_models = ["x"]\n' + tpl.read_text())
            rc, _, err = run("vibe-config", "sbv", "--model", "big", "--template", str(AGENTS / "allowed.toml"),
                             "--out", str(home / "config.toml"))
            bad += [] if rc == 3 and "sets allowed_models" in err else [f"vibe-config, template allowlist: rc {rc}"]
            run("start", "sbv", "--harness", "vibe", "--model", "big")
            run("vibe-config", "sbv", "--model", "big", "--template", str(tpl), "--out", str(home / "config.toml"))
            (home / "export" / "export.json").write_text(json.dumps({
                "outcome": "stopped", "stop_reason": "time_limit", "session_id": "s1", "vibe_version": "2.0",
                "config": {"active_model": "big", "models": {"big": {"name": "big-2610", "thinking": "high"}}}}))
            (AGENTS / "sbv.vibe.log").write_text('{"role": "user"}\nvibe exit=124\n')
            run("end", "sbv", "--exit", "0")
            r = load("sbv") or {}
            want = {"model": "big-2610", "alias": "big", "reasoning_effort": "high", "vibe_version": "2.0",
                    "session_id": "s1", "outcome": "stopped", "stop_reason": "time_limit"}
            bad += [] if (r.get("as_run"), r.get("exit_code"), (r.get("vibe_model") or {}).get("name")) == (
                want, 124, "big-2610") else [
                f"sbv (vibe): as_run {r.get('as_run')}, exit {r.get('exit_code')}, vibe_model {r.get('vibe_model')}"]
            rc, out, err = run("refuse", "sb1", "--reason", "a copied login in the config")
            r = load("sb1") or {}
            ok = (rc, r.get("refused"), out) == (3, "a copied login in the config", "") and "copied" in err
            bad += [] if ok else [f"refuse: rc {rc}, refused {r.get('refused')!r}, stdout {out!r}"]
            rc, _, _ = run("login-check", "sb1")
            bad += [] if rc == 2 else [f"login-check without --budget: rc {rc}, want 2 (usage)"]
            tok = "sk-ant-oat01-abcDEF_123-xyz"
            os.mkfifo(AGENTS / "token_fifo", 0o600)
            (AGENTS / "token_dir").mkdir(mode=0o700)
            (AGENTS / "token_link").symlink_to(AGENTS / "token_loose")
            r_end, w_end = os.pipe()
            os.write(w_end, f"{tok}\n".encode())
            os.close(w_end)
            for name, content, mode, emitted in (
                    ("valid", f"{tok}\n", 0o600, tok), ("crlf", f"\n  {tok}\r\n\n", 0o400, tok),
                    ("loose", f"{tok}\n", 0o640, None), ("group_write", f"{tok}\n", 0o620, None),
                    ("two_lines", "sk-ant-oat01-abcdefghij\nDEF_1234567890abcdefghij\n", 0o600, None),
                    ("export", f"export CLAUDE_CODE_OAUTH_TOKEN={tok}\n", 0o600, None), ("empty", "", 0o600, None),
                    ("bom", f"\ufeff{tok}\n", 0o600, None), ("lone_cr", f"{tok}\rmore\n", 0o600, None),
                    ("short", "sk-ant\n", 0o600, None), ("long", "a" * 2000 + "\n", 0o600, None),
                    ("huge", f"{tok}\n" + "\n" * TOKEN_FILE_MAX, 0o600, None),
                    ("non_utf8", b"\xff" + tok.encode(), 0o600, None), ("missing", None, None, None),
                    ("fifo", None, None, None), ("dir", None, None, None), ("link", None, None, None),
                    ("pipe", None, None, None)):
                f = Path(f"/proc/self/fd/{r_end}") if name == "pipe" else AGENTS / f"token_{name}"
                if content is not None:
                    (f.write_bytes if isinstance(content, bytes) else f.write_text)(content)
                    f.chmod(mode)
                rc, out, err = run("token-check", "--file", str(f), "--emit")
                ok = (rc, out.strip()) == ((0, emitted) if emitted else (3, "")) and tok not in err
                bad += [] if ok else [f"token-check {name}: rc {rc}, emitted {out.strip() == tok}, {err.strip()}"]
            os.close(r_end)
            run("start", "sbt", "--harness", "claude", "--model", "m")
            run("token-check", "sbt", "--file", str(AGENTS / "token_loose"))
            bad += [] if "group or others" in (load("sbt") or {}).get("refused", "") else [
                "token-check: the refusal not noted in the session's record"]
            now = time.time()
            run("start", "sr", "--harness", "claude", "--model", "m", "--budget", "1000")
            r = load("sr")
            r["segments"][0].update(started=_iso(now - 400))
            _write("sr", r)
            (AGENTS / "sr.claude.log").write_text(json.dumps(dict(init, session_id="sid1")) + "\nclaude exit=1\n")
            run("end", "sr", "--exit", "0")
            r = load("sr")
            r["segments"][0].update(ended=_iso(now - 100))
            _write("sr", r)
            rc, out, _ = run("resume", "sr", "--budget", "5000")
            r = load("sr")
            bad += [] if (rc, out.split()) == (0, ["700", "300"]) and len(r["segments"]) == 2 and \
                r["segments"][1].get("budget_s") == 700 and not r["segments"][1].get("ended") else [
                f"resume: rc {rc}, printed {out!r}, segments {r.get('segments')}"]
            rc, _, err = run("resume", "sr", "--budget", "1000")
            bad += [] if rc == 3 and "not ended" in err else [f"resume of an open segment: rc {rc}"]
            with open(AGENTS / "sr.claude.log", "a") as fh:
                fh.write(json.dumps(dict(init, session_id="sid2")) + "\nclaude exit=124\n")
            run("end", "sr", "--exit", "0")
            r = load("sr")
            rc, out, _ = run("session-id", "sr")
            bad += [] if (r["segments"][1].get("exit_code"), r.get("exit_code"), out.strip(),
                          (r.get("as_run") or {}).get("model")) == (124, 124, "sid2", "m") else [
                f"end of the resumed segment / session-id: {r.get('segments')}, {out!r}"]
            hdr = "OpenAI Codex v{}\n--------\nmodel: gpt-x\nreasoning effort: high\nsession id: 01c1\n--------\nuser\n"
            run("start", "sc", "--harness", "codex", "--model", "gpt-x", "--budget", "1000")
            r = load("sc")
            r["segments"][0].update(started=_iso(now - 400))
            _write("sc", r)
            (AGENTS / "sc.codex.log").write_text("Reading additional input from stdin...\n" + hdr.format("0.167.2")
                                                 + "codex exit=124\n")
            run("end", "sc", "--exit", "0")
            r = load("sc")
            r["segments"][0].update(ended=_iso(now - 100))
            _write("sc", r)
            rc, _, err = run("left", "sc", "--budget", "1000", "--harness", "claude", "--model", "gpt-x")
            bad += [] if rc == 3 and "harness" in err else [f"left of a codex session as claude's: rc {rc}"]
            rc, out, _ = run("resume", "sc", "--budget", "5000", "--harness", "codex", "--model", "gpt-x")
            r = load("sc")
            bad += [] if (rc, out.split(), r["segments"][1].get("log_offset")) == (
                0, ["700", "300"], (AGENTS / "sc.codex.log").stat().st_size) else [
                f"resume of a codex session: rc {rc}, printed {out!r}, segments {r.get('segments')}"]
            with open(AGENTS / "sc.codex.log", "a") as fh:
                fh.write("codex resume: 300 s used, 700 s left\n" + hdr.format("0.168.0")
                         + "codex exit=1\n")
            run("end", "sc", "--exit", "1")
            r = load("sc")
            rc, out, _ = run("session-id", "sc")
            bad += [] if ([s.get("exit_code") for s in r["segments"]], r.get("exit_code"), out.strip(),
                          (r.get("as_run") or {}).get("codex_version")) == ([124, 1], 1, "01c1", "0.167.2") else [
                f"end of a resumed codex segment / session-id: {r.get('segments')}, {out!r}, {r.get('as_run')}"]
            _write("sl", {"sandbox": "sl", "harness": "claude", "model": "m", "started": _iso(now - 500),
                          "ended": _iso(now - 100)})
            before = path_of("sl").read_text()
            rc, out, _ = run("left", "sl", "--budget", "1000", "--harness", "claude", "--model", "m")
            bad += [] if (rc, out.split(), path_of("sl").read_text()) == (0, ["600", "400"], before) else [
                f"left: {rc} {out!r} (it must write nothing)"]
            rc, _, err = run("left", "sl", "--budget", "1000", "--model", "other")
            bad += [] if rc == 3 and "model" in err and "refused" not in (load("sl") or {}) else [
                f"left with another model: rc {rc}"]
            rc, out, _ = run("resume", "sl", "--budget", "1000")
            bad += [] if (rc, out.split()) == (0, ["600", "400"]) else [f"resume of a record without segments: {rc} {out!r}"]
            _write("sl", {"sandbox": "sl", "harness": "claude", "model": "m", "started": _iso(now - 500),
                          "ended": _iso(now - 100)})
            rc, _, err = run("resume", "sl", "--budget", "900")
            bad += [] if rc == 3 and "budget left" in err else [f"resume with too little left: rc {rc}"]
            _write("sq", {"sandbox": "sq", "harness": "claude", "model": "m", "budget_s": 5000, "segments": [
                {"started": _iso(now - 500), "ended": _iso(now - 400), "exit_code": 1}]})
            (AGENTS / "sq.claude.log").write_text(json.dumps(dict(init, session_id="sid1")) + "\nclaude exit=1\n")
            run("resume", "sq", "--budget", "5000")
            t_end = _iso(time.time() + 1)
            run("end", "sq", "--exit", "3", "--ended", t_end)
            r = load("sq")
            bad += [] if (r["segments"][1].get("exit_code"), r["segments"][1].get("ended"),
                          r["segments"][1].get("launcher_exit")) == (None, t_end, 3) else [
                f"a refused resumed segment's end: {r['segments'][1]}"]
            rc, _, err = run("end", "sq", "--exit", "0", "--ended", "yesterday")
            bad += [] if rc not in (0, None) else ["end with a malformed --ended time was accepted"]
            _write("s0", {"sandbox": "s0", "harness": "claude", "model": "m", "budget_s": 5000, "segments": [
                {"started": _iso(now - 500), "ended": _iso(now - 400), "exit_code": 0}]})
            rc, _, err = run("resume", "s0", "--budget", "5000")
            bad += [] if rc == 3 and "by itself" in err else [f"resume of a session that ended by itself: rc {rc}"]
            for name, exp, budget, rc_want, where in (
                    ("far", time.time() + 9 * 3600, 7.5 * 3600, 0, "sandbox"),
                    ("near", time.time() + 7.6 * 3600, 7.5 * 3600, 3, "sandbox"),
                    ("none", None, 60, 3, "sandbox"),
                    ("env", time.time() + 9 * 3600, 7.5 * 3600, 0, "env"),
                    ("opt", time.time() + 9 * 3600, 7.5 * 3600, 0, "option")):
                run("start", name, "--harness", "claude", "--model", "m")
                cfg = AGENTS / name / (".claude_cfg" if where == "sandbox" else ".elsewhere")
                cfg.mkdir(parents=True)
                if exp is not None:
                    (cfg / ".credentials.json").write_text(json.dumps({"claudeAiOauth": {"expiresAt": exp * 1000}}))
                args = ["login-check", name, "--budget", str(budget)]
                if where == "env":
                    os.environ["CLAUDE_CONFIG_DIR"] = str(cfg)
                if where == "option":
                    os.environ["CLAUDE_CONFIG_DIR"] = str(AGENTS / "nowhere")
                    args += ["--config", str(cfg)]
                rc, _, err = run(*args)
                os.environ.pop("CLAUDE_CONFIG_DIR", None)
                r = load(name) or {}
                bad += [] if rc == rc_want else [f"login-check {name}: {rc}, want {rc_want} ({err.strip()})"]
                bad += [] if (exp is None) == ("login_expires" not in r) else [
                    f"login-check {name}: login_expires {r.get('login_expires')!r}"]
                bad += [] if (rc_want == 3) == ("refused" in r) else [
                    f"login-check {name}: refused {r.get('refused')!r}"]
    finally:
        AGENTS = keep
        if keep_cfg is not None:
            os.environ["CLAUDE_CONFIG_DIR"] = keep_cfg
    for b in bad:
        print("  FAIL ", b)
    print("run record selftest:", "FAIL" if bad else "PASS")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
