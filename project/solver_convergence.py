"""Versioned engineering stopping contract; standalone, without cloth imports."""
import math


def convergence_policy(precision):
    if precision not in ("f32", "f64"):
        raise ValueError("unknown convergence precision")
    floor = 2. * (2.**-23 if precision == "f32" else 2.**-52)
    return dict(schema="xpbd-convergence-policy-v1", omega=1.5, minimum=8,
                check_every=4, consecutive=2, maximum=96,
                requested_stretch=1e-7, requested_bend=1e-6,
                effective_stretch=max(1e-7, floor), effective_bend=max(1e-6, floor),
                precision_floor=floor)


def stop_reason(checks, policy):
    if len(checks) < 2 or checks[-1]["iteration"] < policy["minimum"]:
        return None
    def within(prefix):
        return all(c["stretch_residual"] <= policy[prefix+"_stretch"]
                   and c["bend_residual"] <= policy[prefix+"_bend"] for c in checks[-2:])
    if not within("effective"):
        return None
    return "converged" if within("requested") else "precision_floor"


def validate_convergence_record(value, precision, cap):
    p = convergence_policy(precision)
    if type(cap) is not int or not p["minimum"] <= cap <= p["maximum"]:
        raise ValueError("convergence cap must be an integer in 8..96")
    if not isinstance(value, dict) or set(value) != {"schema", "iterations", "stop_reason", "checks"}:
        raise ValueError("invalid convergence record fields")
    if value["schema"] != "xpbd-convergence-result-v1":
        raise ValueError("invalid convergence result schema")
    count, checks = value["iterations"], value["checks"]
    if type(count) is not int or not p["minimum"] <= count <= cap or not isinstance(checks, list):
        raise ValueError("invalid iteration count/check list")
    if count % p["check_every"] and count != cap:
        raise ValueError("off-grid checkpoint is only allowed at the iteration cap")
    expected = list(range(p["check_every"], count+1, p["check_every"]))
    if not expected or expected[-1] != count:
        expected.append(count)
    if len(checks) != len(expected):
        raise ValueError("missing convergence checkpoints")
    for i, (c, iteration) in enumerate(zip(checks, expected)):
        if not isinstance(c, dict) or set(c) != {"iteration", "stretch_residual", "bend_residual"}:
            raise ValueError("invalid convergence checkpoint")
        if type(c["iteration"]) is not int or c["iteration"] != iteration:
            raise ValueError("wrong convergence checkpoint iteration")
        for key in ("stretch_residual", "bend_residual"):
            v = c[key]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                raise ValueError("convergence residual must be finite nonnegative")
        if i < len(checks)-1 and stop_reason(checks[:i+1], p) is not None:
            raise ValueError("continued after declared stopping condition")
    reason = stop_reason(checks, p)
    if reason is None and count != cap:
        raise ValueError("unconverged solve stopped before its cap")
    if value["stop_reason"] != (reason or "iteration_cap"):
        raise ValueError("convergence status disagrees with residuals")
    return value


# ---------------------------------------------------------------------------
# Same-step-size high-iteration reference window (diagnostic).
# One window applies its load, predicts and clears multipliers once, then keeps
# projecting: a budget checkpoint at the first target or at `checkpoint` rounds,
# a fixed tail after the first target, and a hard budget that includes the tail.
# Optional reference diagnostics leave the primary convergence policy unchanged.

REFERENCE_POLICY_SCHEMA = "xpbd-convergence-reference-policy-v1"
REFERENCE_RESULT_SCHEMA = "xpbd-convergence-reference-v1"
REFERENCE_PRODUCTION = dict(budget=1536, tail=32, checkpoint=96)
_REFERENCE_PURPOSES = ("heavy-load-reference", "contract-test")
_REFERENCE_OUTCOMES = ("tail_complete", "tail_budget_exhausted", "target_not_reached")


def reference_policy(precision, *, budget=1536, tail=32, checkpoint=96, purpose="heavy-load-reference"):
    """Fixed production policy, or a reduced one explicitly marked for contract tests."""
    base = convergence_policy(precision)
    if precision != "f64":
        raise ValueError("the reference window is defined for f64 only")
    if purpose not in _REFERENCE_PURPOSES:
        raise ValueError("unknown reference policy purpose")
    values = dict(budget=budget, tail=tail, checkpoint=checkpoint)
    step = base["check_every"]
    for name, value in values.items():
        if type(value) is not int or value <= 0 or value % step:
            raise ValueError(f"reference {name} must be a positive multiple of the check interval")
    if purpose == "heavy-load-reference" and values != REFERENCE_PRODUCTION:
        raise ValueError("the production reference policy is fixed at budget 1536, tail 32, checkpoint 96")
    # The last two tail checks must both lie inside the tail; the checkpoint and the
    # last first-target round that still completes its tail must be legal stops.
    if (tail < base["consecutive"] * step or checkpoint < base["minimum"]
            or budget - tail < base["minimum"] or checkpoint > budget):
        raise ValueError("inconsistent reference budget/tail/checkpoint")
    policy = dict(base, schema=REFERENCE_POLICY_SCHEMA, purpose=purpose, base_schema=base["schema"], **values)
    policy.pop("maximum")           # the 96-round cap of the stopping contract is not this window's cap
    return policy


def validate_reference_policy(policy, precision):
    if not isinstance(policy, dict) or policy.get("schema") != REFERENCE_POLICY_SCHEMA:
        raise ValueError("invalid reference policy")
    expected = reference_policy(precision, budget=policy.get("budget"), tail=policy.get("tail"),
                                checkpoint=policy.get("checkpoint"), purpose=policy.get("purpose"))
    if policy != expected:
        raise ValueError("reference policy differs from its canonical form")
    return expected


def first_target_iteration(checks, policy):
    """Round of the first check at which the unchanged stopping condition holds, or None."""
    for count in range(1, len(checks) + 1):
        if stop_reason(checks[:count], policy) is not None:
            return checks[count - 1]["iteration"]
    return None


def reference_stop_iteration(first_target, policy):
    """Total projection rounds of a window whose first target round is `first_target`."""
    if first_target is not None and first_target <= policy["budget"] - policy["tail"]:
        return first_target + policy["tail"]
    return policy["budget"]


def checkpoint_convergence_record(value, policy):
    """The budget checkpoint written as a result of the unchanged stopping contract.

    For the production policy it must equal, field for field, what the existing
    cap-96 entry records when it stops at the same round from the same state.
    """
    iteration = value["checkpoint"]["iteration"]
    checks = [dict(c) for c in value["checks"] if c["iteration"] <= iteration]
    return dict(schema="xpbd-convergence-result-v1", iterations=iteration,
                stop_reason=stop_reason(checks, policy) or "iteration_cap", checks=checks)


def validate_reference_record(value, precision, policy):
    """Round accounting of one reference window; says nothing about its gates."""
    p = validate_reference_policy(policy, precision)
    fields = {"schema", "iterations", "checks", "first_target_iteration", "observed_lower_bound",
              "checkpoint", "tail", "outcome"}
    if not isinstance(value, dict) or set(value) != fields or value["schema"] != REFERENCE_RESULT_SCHEMA:
        raise ValueError("invalid reference record fields")
    count, checks, step = value["iterations"], value["checks"], p["check_every"]
    if type(count) is not int or not isinstance(checks, list) or count % step or not step <= count <= p["budget"]:
        raise ValueError("invalid reference iteration count/check list")
    if len(checks) != count // step:
        raise ValueError("missing or extra reference checkpoints")
    for index, c in enumerate(checks):
        if not isinstance(c, dict) or set(c) != {"iteration", "stretch_residual", "bend_residual"}:
            raise ValueError("invalid reference checkpoint")
        if type(c["iteration"]) is not int or c["iteration"] != step * (index + 1):
            raise ValueError("wrong reference checkpoint iteration")
        for key in ("stretch_residual", "bend_residual"):
            v = c[key]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                raise ValueError("reference residual must be finite nonnegative")
    first = first_target_iteration(checks, p)
    if value["first_target_iteration"] != first or (first is not None and type(value["first_target_iteration"]) is not int):
        raise ValueError("first target round disagrees with the residual history")
    if count != reference_stop_iteration(first, p):
        raise ValueError("reference window did not stop at its tail end or budget")
    if first is None:
        outcome, lower = "target_not_reached", p["budget"]
    elif first <= p["budget"] - p["tail"]:
        outcome, lower = "tail_complete", None
    else:
        outcome, lower = "tail_budget_exhausted", None
    if value["outcome"] != outcome or value["observed_lower_bound"] != lower or (
            lower is not None and type(value["observed_lower_bound"]) is not int):
        raise ValueError("reference outcome disagrees with the residual history")
    # Budget checkpoint: the earlier of the first target round and the checkpoint round.
    early = first is not None and first < p["checkpoint"]
    expected_checkpoint = dict(kind="cap96_early_stop" if early else "iteration96",
                               iteration=first if early else p["checkpoint"],
                               target_met=bool(early or first == p["checkpoint"]))
    if count < expected_checkpoint["iteration"]:
        raise ValueError("reference window ended before its budget checkpoint")
    if (not isinstance(value["checkpoint"], dict) or value["checkpoint"] != expected_checkpoint
            or type(value["checkpoint"]["target_met"]) is not bool or type(value["checkpoint"]["iteration"]) is not int):
        raise ValueError("budget checkpoint disagrees with the residual history")
    if outcome == "tail_complete":
        last_two = checks[-p["consecutive"]:]
        ok = all(c["stretch_residual"] <= p["requested_stretch"] and c["bend_residual"] <= p["requested_bend"]
                 for c in last_two)
        expected_tail = dict(start=first, end=first + p["tail"], completed=True, residual_ok=bool(ok))
    else:
        expected_tail = dict(start=first, end=None, completed=False, residual_ok=None)
    tail = value["tail"]
    if (not isinstance(tail, dict) or tail != expected_tail or type(tail["completed"]) is not bool
            or any(tail[key] is not None and type(tail[key]) is not int for key in ("start", "end"))
            or (tail["residual_ok"] is not None and type(tail["residual_ok"]) is not bool)):
        raise ValueError("tail record disagrees with the residual history")
    return value


def reference_record_from_checks(checks, policy):
    """Assemble the record of a finished window from its residual history alone."""
    first = first_target_iteration(checks, policy)
    count = checks[-1]["iteration"] if checks else 0
    early = first is not None and first < policy["checkpoint"]
    complete = first is not None and first <= policy["budget"] - policy["tail"]
    if complete:
        ok = all(c["stretch_residual"] <= policy["requested_stretch"]
                 and c["bend_residual"] <= policy["requested_bend"] for c in checks[-policy["consecutive"]:])
        tail = dict(start=first, end=first + policy["tail"], completed=True, residual_ok=bool(ok))
    else:
        tail = dict(start=first, end=None, completed=False, residual_ok=None)
    return dict(schema=REFERENCE_RESULT_SCHEMA, iterations=count, checks=checks,
                first_target_iteration=first,
                observed_lower_bound=policy["budget"] if first is None else None,
                checkpoint=dict(kind="cap96_early_stop" if early else "iteration96",
                                iteration=first if early else policy["checkpoint"],
                                target_met=bool(early or first == policy["checkpoint"])),
                tail=tail,
                outcome=("tail_complete" if complete else
                         "target_not_reached" if first is None else "tail_budget_exhausted"))
