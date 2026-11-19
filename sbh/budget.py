"""Wall-clock budget for a step of the evaluator's own work (e.g. kind post/verify stages): budget() raises
BudgetExceeded in the main thread once the seconds have passed.  SIGALRM-based: a running numpy/XLA call finishes
first; the job's time limit is the backstop."""
from __future__ import annotations

import contextlib
import os
import signal
import threading
import time
from typing import Optional


class BudgetExceeded(BaseException):
    """A step's budget ran out.  A BaseException so no `except Exception` on the way can swallow it and misname
    why the check failed."""


_BUDGET: Optional[dict] = None
_PRIOR_HANDLER = signal.SIG_DFL
_HOLD = 0
_DEFERRED: list = []


def _reset_in_child() -> None:
    """A forked worker inherits no budget, hold or deferred ring; SIGALRM's prior handler is restored."""
    global _BUDGET, _HOLD
    _BUDGET, _HOLD = None, 0
    _DEFERRED.clear()
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGALRM, _PRIOR_HANDLER)


os.register_at_fork(after_in_child=_reset_in_child)


def budget_left() -> Optional[float]:
    """Seconds left of the budget in force, or None."""
    return None if _BUDGET is None else _BUDGET["end"] - time.time()


def budget_message() -> Optional[str]:
    """What the budget in force says when it runs out, or None."""
    return None if _BUDGET is None else _BUDGET["what"]


@contextlib.contextmanager
def alarm_held():
    """Hold SIGALRM (main thread) while a pool forks/registers workers, waits or shuts down (pool_imap), so a ring
    lands after the hold -- when every worker is known to the pool -- never inside a fork callback (logging swallowed
    it) or a lock wait (left the lock broken).  Blocking the signal is not enough, since another thread can take it
    and Python still runs the handler in the main thread; so the handler defers a ring during the hold and it is
    raised at the end if the block completed (a block's own exception takes precedence).  The hold is counted before
    blocking and the signal unblocked before the hold ends, so no ring leaves the signal blocked."""
    global _HOLD
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    _HOLD += 1
    if _HOLD == 1:
        _DEFERRED.clear()
    deferred = None
    try:
        old = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
        try:
            yield
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, old)
    finally:
        if _HOLD == 1:
            deferred = _DEFERRED[-1] if _DEFERRED else None
            _DEFERRED.clear()
        _HOLD -= 1
    if deferred is not None:
        raise BudgetExceeded(deferred)


@contextlib.contextmanager
def budget(seconds: Optional[float], what: str = "the stage"):
    """Run the block under a wall-clock budget (None: none); after `seconds` BudgetExceeded is raised in the main
    thread (pool_imap's workers are then killed; inside alarm_held(), at the hold's end).  A block that swallowed the
    ring but completed past the deadline still ends with BudgetExceeded.  Off the main thread there is no signal, only
    pool_imap's deadline check and the end check.  On exit the alarm is cleared and an outer timer restored."""
    global _BUDGET, _PRIOR_HANDLER
    if seconds is None:
        yield
        return
    t0 = time.time()
    end = t0 + max(float(seconds), 1e-3)
    msg = f"{what} ran out of its {float(seconds):.0f} s budget" if seconds >= 10 else \
        f"{what} ran out of its {float(seconds):.2g} s budget"
    state = {"armed": True, "rang": False}

    def ring(signum, frame):
        if state["armed"]:
            state["armed"], state["rang"] = False, True
            if _HOLD:
                _DEFERRED.append(msg)
                return
            raise BudgetExceeded(msg)

    main = threading.current_thread() is threading.main_thread()
    outer, prior = _BUDGET, _PRIOR_HANDLER
    _BUDGET = {"end": end if outer is None else min(end, outer["end"]), "what": msg}
    if main:
        old = signal.signal(signal.SIGALRM, ring)
        _PRIOR_HANDLER = old if old is not None else signal.SIG_DFL
        old_timer = signal.setitimer(signal.ITIMER_REAL, max(float(seconds), 1e-3))
    try:
        yield
        if state["rang"] or time.time() >= end:
            state["armed"] = False
            raise BudgetExceeded(msg)
    finally:
        state["armed"] = False
        _BUDGET = outer
        if main:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old if old is not None else signal.SIG_DFL)
            _PRIOR_HANDLER = prior
            if old_timer[0] > 0:
                signal.setitimer(signal.ITIMER_REAL, max(old_timer[0] - (time.time() - t0), 1e-3), old_timer[1])
