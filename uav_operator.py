"""uav-operator Verifiers environment module."""

from __future__ import annotations

import verifiers as vf

ENV_ID = "uav-operator"


def load_environment(**kwargs: object) -> vf.Environment:
    """Load the UAV operator environment."""
    raise NotImplementedError(
        "uav-operator environment skeleton is not implemented yet. "
        "See PLAN.md Day 1 for the first end-to-end target."
    )
