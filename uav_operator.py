import verifiers as vf


def load_environment(**kwargs) -> vf.Environment:
    """
    Load this environment.

    v0 environments typically return vf.SingleTurnEnv, vf.ToolEnv, etc.
    For the v1 Taskset/Harness pattern: prime env init <name> --v1
    """
    raise NotImplementedError("Implement load_environment here.")
