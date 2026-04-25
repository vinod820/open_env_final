from __future__ import annotations

import os

from fastapi import FastAPI
try:
    from openenv.core.env_server import create_web_interface_app
except ImportError:
    create_web_interface_app = None

from social_engineer_arena.models import ArenaAction, ArenaObservation
from social_engineer_arena.server.environment import SocialEngineerArenaEnvironment


if os.getenv("ENABLE_WEB_INTERFACE", "true").lower() == "true" and create_web_interface_app:
    # openenv-core expects an environment class/factory, not an instantiated object.
    app = create_web_interface_app(SocialEngineerArenaEnvironment, ArenaAction, ArenaObservation)
else:
    env = SocialEngineerArenaEnvironment()
    app = FastAPI(title="SocialEngineerArena")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/reset")
    def reset() -> ArenaObservation:
        return env.reset()

    @app.post("/step")
    def step(action: ArenaAction) -> dict:
        observation, reward, done = env.step(action)
        return {"observation": observation, "reward": reward, "done": done}

    @app.get("/state")
    def state() -> dict:
        return env.state.model_dump()


def main() -> None:
    import uvicorn

    uvicorn.run(
        "social_engineer_arena.server.app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
    )


if __name__ == "__main__":
    main()
