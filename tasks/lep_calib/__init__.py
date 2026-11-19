"""Participant-side task metadata: exactly what `sbh check` needs.

The evaluator's full task module (generation, priors, injected-truth
machinery) is not part of the bundle.
"""
from sbh.task import Task


class _ParticipantTask(Task):
    name = 'lep_calib'
    kind = 'corrections'
    pois = []
    result_type = 'measurement'
    requires_model = False
    event_list_from = None
    nuisance_prefix = ''
    run_timeout_s = 1800
    run_memory_mb = 0
    worker_memory_mb = 16384
    run_cpus = 4
    working_points = {}
    base_env = 'lepcalib'
    measurement_required = False

    example_submission = 'attacker_constant_lepcalib'

TASK = _ParticipantTask()
