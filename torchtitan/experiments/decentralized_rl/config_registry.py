"""Transitional shim: the RL presets moved to ``panoengine.train.rl.config_registry``.

``--module decentralized_rl`` is a stored-spec value that torchtitan's ConfigManager
resolves through its experiment registry (torchtitan/experiments/__init__.py), and it
resolves to THIS module. A star-import re-export answers it, because ConfigManager
reaches a preset with getattr(module, config_name).

The presets moved because they only COMPOSE public classes. The trainer classes they
name — actors, controller, replicas — stay here, because they subclass upstream's own
experimental PolicyTrainer and Controller. See FORK-DELTA.md.
"""

from panoengine.train.rl.config_registry import *  # noqa: F401,F403
