"""Transitional shim: this process moved to `panoengine.decentralized.parameter_server`.

DUAL-USE, unlike the panoserve shims. This module is both:
  * an entry point — the control plane launches
    `python -m torchtitan.experiments.decentralized_rl.parameter_server` ON THE NODE
    (panofabric spec/coordination.py), so `__main__` must reach the real module;
  * a library — sibling modules here import its client classes.

The two halves are mutually EXCLUSIVE on purpose. Doing both unconditionally is a
bug: the star-import puts the real module in sys.modules, and runpy then executes
it a SECOND time under the name `__main__`, which duplicates module-level server
state and earns a RuntimeWarning about unpredictable behaviour. Under `-m` we want
only runpy; when imported we want only the re-export.

Delete once no live image and no stored run names this path.
"""

if __name__ == "__main__":
    import runpy

    runpy.run_module("panoengine.decentralized.parameter_server", run_name="__main__", alter_sys=True)
else:
    from panoengine.decentralized.parameter_server import *  # noqa: F401,F403
    from panoengine.decentralized.parameter_server import (  # noqa: F401  explicit: star skips _private
        HeLoCoRLClient,
        build_server,
        param_metadata,
    )
