from __future__ import annotations

try:
    from openenv.core import EnvClient
except ImportError:  # Local smoke-test fallback; production uses openenv-core.
    class EnvClient:  # type: ignore[no-redef]
        def __class_getitem__(cls, _item):
            return cls

from social_engineer_arena.models import ArenaAction, ArenaObservation, ArenaState


class SocialEngineerArenaEnv(EnvClient[ArenaAction, ArenaObservation, ArenaState]):
    action_type = ArenaAction
    observation_type = ArenaObservation
    state_type = ArenaState
