from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib import error, request

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
try:
    from openenv.core.env_server import create_web_interface_app
except ImportError:
    create_web_interface_app = None

from social_engineer_arena.models import ArenaAction, ArenaObservation, ArenaState
from social_engineer_arena.server.environment import SocialEngineerArenaEnvironment


env = SocialEngineerArenaEnvironment()
TRAIN_LOG_LIMIT = 400
TRAIN_SCRIPT_ENV = "TRAIN_SCRIPT_PATH"
_train_lock = threading.Lock()
_train_thread: threading.Thread | None = None
_train_state: dict[str, Any] = {
    "status": "idle",
    "started_at": None,
    "finished_at": None,
    "exit_code": None,
    "pid": None,
    "command": None,
    "overrides": {},
    "logs": [],
    "error": "",
}


class TrainRequest(BaseModel):
    model_name: str | None = None
    output_repo: str | None = None
    output_dir: str | None = None
    scenarios_path: str | None = None
    data_multiplier: int | None = Field(default=None, ge=1)
    max_steps: int | None = Field(default=None, ge=1)
    learning_rate: float | None = Field(default=None, gt=0)
    grad_accum_steps: int | None = Field(default=None, ge=1)
    max_length: int | None = Field(default=None, ge=32)
    eval_strategy: str | None = None
    push_to_hub: bool | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_train_script() -> Path:
    configured = os.getenv(TRAIN_SCRIPT_ENV, "").strip()
    if configured:
        candidate = Path(configured)
        if not candidate.is_absolute():
            candidate = _repo_root() / candidate
        if candidate.exists():
            return candidate
    candidate = _repo_root() / "scripts" / "train_suggest_model.py"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        "Training script not found. Set TRAIN_SCRIPT_PATH or ensure scripts/train_suggest_model.py exists."
    )


def _append_train_log(line: str) -> None:
    with _train_lock:
        logs = _train_state.setdefault("logs", [])
        logs.append(line.rstrip("\n"))
        if len(logs) > TRAIN_LOG_LIMIT:
            del logs[: len(logs) - TRAIN_LOG_LIMIT]


def _serialize_overrides(payload: TrainRequest) -> dict[str, str]:
    env_overrides: dict[str, str] = {}
    if payload.model_name:
        env_overrides["MODEL_NAME"] = payload.model_name
    if payload.output_repo:
        env_overrides["OUTPUT_REPO"] = payload.output_repo
    if payload.output_dir:
        env_overrides["OUTPUT_DIR"] = payload.output_dir
    if payload.scenarios_path:
        env_overrides["SCENARIOS_PATH"] = payload.scenarios_path
    if payload.data_multiplier is not None:
        env_overrides["DATA_MULTIPLIER"] = str(payload.data_multiplier)
    if payload.max_steps is not None:
        env_overrides["MAX_STEPS"] = str(payload.max_steps)
    if payload.learning_rate is not None:
        env_overrides["LEARNING_RATE"] = str(payload.learning_rate)
    if payload.grad_accum_steps is not None:
        env_overrides["GRAD_ACCUM_STEPS"] = str(payload.grad_accum_steps)
    if payload.max_length is not None:
        env_overrides["MAX_LENGTH"] = str(payload.max_length)
    if payload.eval_strategy:
        env_overrides["EVAL_STRATEGY"] = payload.eval_strategy
    if payload.push_to_hub is not None:
        env_overrides["PUSH_TO_HUB"] = "1" if payload.push_to_hub else "0"
    return env_overrides


def _run_train_job(env_overrides: dict[str, str], command: list[str]) -> None:
    global _train_thread
    process: subprocess.Popen[str] | None = None
    try:
        child_env = os.environ.copy()
        child_env.update(env_overrides)
        process = subprocess.Popen(
            command,
            cwd=str(_repo_root()),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=child_env,
        )
        with _train_lock:
            _train_state["pid"] = process.pid
        _append_train_log(f"[train] started pid={process.pid}")
        if process.stdout is not None:
            for line in process.stdout:
                _append_train_log(line)
        exit_code = process.wait()
        with _train_lock:
            _train_state["status"] = "completed" if exit_code == 0 else "failed"
            _train_state["exit_code"] = exit_code
            _train_state["finished_at"] = datetime.now(UTC).isoformat()
            _train_state["pid"] = None
            if exit_code != 0 and not _train_state.get("error"):
                _train_state["error"] = f"Training process exited with code {exit_code}."
    except Exception as exc:
        with _train_lock:
            _train_state["status"] = "failed"
            _train_state["error"] = str(exc)
            _train_state["finished_at"] = datetime.now(UTC).isoformat()
            _train_state["pid"] = None
        _append_train_log(f"[train] error: {exc}")
    finally:
        if process is not None and process.poll() is None:
            process.kill()
        with _train_lock:
            _train_thread = None


def _train_status_snapshot() -> dict[str, Any]:
    with _train_lock:
        return {
            "status": _train_state.get("status"),
            "started_at": _train_state.get("started_at"),
            "finished_at": _train_state.get("finished_at"),
            "exit_code": _train_state.get("exit_code"),
            "pid": _train_state.get("pid"),
            "command": _train_state.get("command"),
            "overrides": dict(_train_state.get("overrides", {})),
            "error": _train_state.get("error"),
            "logs": list(_train_state.get("logs", [])),
        }


def _suggest_prompt(obs: ArenaObservation) -> str:
    history = "\n".join(obs.thread_history) if obs.thread_history else "(none)"
    return f"""You are assisting in SocialEngineerArena.
Role: {obs.role}
Persona: {obs.persona}
Organization: {obs.organization}
Policy Excerpt: {obs.policy_excerpt}
Conflicting Context: {obs.conflicting_context}
Thread Context: {obs.thread_context}
Thread History:
{history}
Incoming Message: {obs.incoming_message}
Task: {obs.task}

Return ONLY one JSON object and nothing else.
First character must be "{{" and last character must be "}}".
Required keys:
verdict, explanation, cues_found, response, safety_boundary.
Rules:
- verdict must be one of: safe, phishing, pretexting, unknown
- cues_found must be an array of short strings
- no markdown, no prose outside JSON, no code fences
"""


def _extract_generated_text(body: Any) -> str:
    """Normalize HF serverless / TGI / Inference Endpoint response shapes."""
    if body is None:
        return ""
    if isinstance(body, str):
        return body
    if isinstance(body, dict):
        if "generated_text" in body and isinstance(body["generated_text"], str):
            return body["generated_text"]
        if "text" in body and isinstance(body["text"], str):
            return body["text"]
        choices = body.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                msg = first.get("message") or first.get("delta") or {}
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    return msg["content"]
        data = body.get("data")
        if isinstance(data, list) and data:
            return _extract_generated_text(data[0])
    if isinstance(body, list):
        if not body:
            return ""
        first = body[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict) and "generated_text" in first:
            gt = first.get("generated_text")
            return gt if isinstance(gt, str) else ""
        if isinstance(first, list) and first:
            return _extract_generated_text(first[0])
    return ""


def _parse_action_json(text: str) -> ArenaAction:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Model returned empty output.")
    try:
        return ArenaAction.model_validate_json(raw)
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        raise ValueError("Model output did not contain a valid JSON object.")
    return ArenaAction.model_validate_json(match.group(0))


def _looks_like_prompt_echo(text: str) -> bool:
    lowered = text.lower()
    markers = [
        "return only one json object",
        "do not include markdown",
        "you are assisting in socialengineerarena",
        "required keys",
    ]
    return any(marker in lowered for marker in markers)


def _request_suggestion_once(
    *,
    target_url: str,
    headers: dict[str, str],
    prompt: str,
    max_new_tokens: int,
    suggest_timeout: int,
) -> Any:
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
            "do_sample": False,
            "return_full_text": False,
            "repetition_penalty": 1.05,
        },
    }
    req = request.Request(
        target_url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with request.urlopen(req, timeout=suggest_timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _variant_config(variant: str) -> tuple[str, str, str]:
    model_id = os.getenv("SUGGEST_MODEL_ID", "").strip()
    endpoint_url = os.getenv("HF_ENDPOINT_URL", "").strip()
    hf_token = os.getenv("HF_TOKEN", "").strip()
    return model_id, endpoint_url, hf_token


def _generate_suggestion(obs: ArenaObservation, variant: str = "primary") -> tuple[ArenaAction, str, str]:
    model_id, endpoint_url, hf_token = _variant_config(variant)
    max_new_tokens = int(os.getenv("SUGGEST_MAX_NEW_TOKENS", "180"))
    suggest_timeout = int(os.getenv("SUGGEST_TIMEOUT_SEC", "120"))
    if not model_id and not endpoint_url:
        raise RuntimeError("No suggestion model configured. Set SUGGEST_MODEL_ID or HF_ENDPOINT_URL.")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is required for model-backed suggestion.")

    target_url = endpoint_url or f"https://api-inference.huggingface.co/models/{model_id}"
    prompt = _suggest_prompt(obs)
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }
    if os.getenv("SUGGEST_WAIT_FOR_MODEL", "").strip().lower() in {"1", "true", "yes"}:
        headers["x-wait-for-model"] = "true"

    try:
        body = _request_suggestion_once(
            target_url=target_url,
            headers=headers,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            suggest_timeout=suggest_timeout,
        )
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Suggestion request failed ({exc.code}): {detail}") from exc
    except Exception as exc:
        raise RuntimeError(f"Suggestion request failed: {exc}") from exc

    generated = _extract_generated_text(body)
    if not generated.strip():
        raise RuntimeError(
            "Model returned no text. Raw response (truncated): "
            + json.dumps(body, default=str)[:800]
        )
    # Retry once with an even stricter prompt if model echoed instructions.
    if _looks_like_prompt_echo(generated):
        strict_prompt = prompt + "\nOutput only compact JSON. Begin with '{' now."
        try:
            retry_body = _request_suggestion_once(
                target_url=target_url,
                headers=headers,
                prompt=strict_prompt,
                max_new_tokens=max_new_tokens,
                suggest_timeout=suggest_timeout,
            )
            retry_generated = _extract_generated_text(retry_body)
            if retry_generated.strip():
                generated = retry_generated
        except Exception:
            pass

    action = _parse_action_json(generated)
    source = endpoint_url or model_id
    return action, source, generated


def attach_showcase_routes(app: FastAPI) -> None:
    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    def reset_handler() -> ArenaObservation:
        try:
            return env.reset()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    def step_handler(action: ArenaAction) -> dict:
        try:
            observation, reward, done = env.step(action)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "observation": observation,
            "reward": reward,
            "done": done,
        }

    def state_handler() -> ArenaState:
        return env.state

    def suggest_handler() -> dict:
        if not env.state.episode_id:
            raise RuntimeError("Call reset() before suggest().")
        if env.state.done:
            raise RuntimeError("Episode is complete. Call reset() to start a new episode.")
        scenario = env._scenario_by_id(env.state.scenario_id)
        obs = env._observation(scenario)
        action, source, raw = _generate_suggestion(obs)
        preview = raw.strip()[:1200]
        return {
            "action": action,
            "source": source,
            "raw_preview": preview,
        }

    def train_handler(payload: TrainRequest) -> dict[str, Any]:
        global _train_thread
        with _train_lock:
            if _train_state.get("status") == "running":
                raise RuntimeError("Training is already running. Check /train/status for progress.")
        script_path = _resolve_train_script()
        env_overrides = _serialize_overrides(payload)
        command = [sys.executable, str(script_path)]
        with _train_lock:
            _train_state.update(
                {
                    "status": "running",
                    "started_at": datetime.now(UTC).isoformat(),
                    "finished_at": None,
                    "exit_code": None,
                    "pid": None,
                    "command": command,
                    "overrides": env_overrides,
                    "logs": [],
                    "error": "",
                }
            )
        worker = threading.Thread(
            target=_run_train_job,
            args=(env_overrides, command),
            name="sea-train-worker",
            daemon=True,
        )
        _train_thread = worker
        worker.start()
        return _train_status_snapshot()

    @app.post("/reset")
    def reset() -> ArenaObservation:
        return reset_handler()

    @app.post("/step")
    def step(action: ArenaAction) -> dict:
        return step_handler(action)

    @app.get("/state")
    def state() -> ArenaState:
        return state_handler()

    @app.post("/suggest")
    def suggest() -> dict:
        try:
            return suggest_handler()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/train")
    def train(payload: TrainRequest = TrainRequest()) -> dict[str, Any]:
        try:
            return train_handler(payload)
        except (RuntimeError, FileNotFoundError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/train/status")
    def train_status() -> dict[str, Any]:
        return _train_status_snapshot()

    @app.post("/api/reset")
    def api_reset() -> ArenaObservation:
        return reset_handler()

    @app.post("/api/step")
    def api_step(action: ArenaAction) -> dict:
        return step_handler(action)

    @app.get("/api/state")
    def api_state() -> ArenaState:
        return state_handler()

    @app.post("/api/suggest")
    def api_suggest() -> dict:
        try:
            return suggest_handler()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/train")
    def api_train(payload: TrainRequest = TrainRequest()) -> dict[str, Any]:
        try:
            return train_handler(payload)
        except (RuntimeError, FileNotFoundError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/train/status")
    def api_train_status() -> dict[str, Any]:
        return _train_status_snapshot()

    @app.get("/arena", response_class=HTMLResponse)
    def arena_showcase() -> str:
        return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SocialEngineerArena</title>
  <style>
    :root { --bg0:#070a18; --bg1:#111636; --card:rgba(255,255,255,.08); --line:rgba(255,255,255,.16); --txt:#f4f7ff; --muted:#aab5d5; --accent:#7c6dff; --accent2:#00d8c2; --ok:#57e29d; --warn:#ffd166; --bad:#ff6b88; }
    * { box-sizing:border-box; margin:0; padding:0; }
    body { font-family:Inter,Segoe UI,system-ui,sans-serif; color:var(--txt); background:radial-gradient(1200px 700px at 12% -10%,#2e1f88 0%,transparent 50%),radial-gradient(900px 500px at 92% -15%,#004d5c 0%,transparent 50%),linear-gradient(135deg,var(--bg0),var(--bg1)); min-height:100vh; overflow-x:hidden; }
    .ambient { position:fixed; inset:0; z-index:0; pointer-events:none; } #particles { width:100%; height:100%; opacity:.6; }
    .wrap { position:relative; z-index:1; max-width:1320px; margin:18px auto; padding:0 14px 24px; }
    .topbar { display:grid; grid-template-columns:1.2fr .8fr; gap:12px; margin-bottom:12px; } .panel { background:var(--card); border:1px solid var(--line); backdrop-filter:blur(14px); border-radius:16px; box-shadow:0 12px 32px rgba(6,10,30,.35); padding:14px; }
    h1 { font-size:26px; letter-spacing:.3px; margin-bottom:8px; } .sub { color:var(--muted); margin-bottom:10px; }
    .row { display:flex; gap:8px; flex-wrap:wrap; align-items:center; } .badge { background:rgba(255,255,255,.08); border:1px solid var(--line); border-radius:999px; padding:7px 11px; font-size:13px; }
    .health-dot { display:inline-block; width:8px; height:8px; border-radius:999px; margin-right:6px; background:var(--warn); box-shadow:0 0 14px rgba(255,209,102,.7); animation:pulse 1.6s infinite; } @keyframes pulse { 0%{transform:scale(1);opacity:1;}70%{transform:scale(1.6);opacity:.25;}100%{transform:scale(1);opacity:1;} }
    .grid { display:grid; grid-template-columns:1.1fr .9fr; gap:12px; } .left-col,.right-col { display:grid; gap:12px; min-width:0; }
    .label { font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; margin-bottom:5px; }
    .box { background:rgba(0,0,0,.24); border:1px solid var(--line); border-radius:12px; padding:10px; min-height:100px; white-space:pre-wrap; line-height:1.48; }
    textarea,select,input,button { font:inherit; color:var(--txt); }
    textarea,select,input { width:100%; border:1px solid var(--line); background:rgba(0,0,0,.28); border-radius:12px; padding:10px 11px; outline:none; transition:border-color .2s ease, transform .2s ease, box-shadow .2s ease; }
    textarea:focus,select:focus,input:focus { border-color:#8aa0ff; box-shadow:0 0 0 3px rgba(124,109,255,.22); transform:translateY(-1px); }
    input.error,textarea.error,select.error { border-color:var(--bad); box-shadow:0 0 0 3px rgba(255,107,136,.22); }
    .form-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:10px; } .actions { display:flex; gap:9px; flex-wrap:wrap; margin-top:10px; }
    button { border:0; border-radius:12px; cursor:pointer; padding:10px 13px; font-weight:600; transition:transform .16s ease, box-shadow .2s ease, opacity .2s ease; } button:hover { transform:translateY(-2px); } button:disabled { opacity:.55; cursor:default; transform:none; }
    .primary { background:linear-gradient(90deg,var(--accent),#9763ff); box-shadow:0 8px 20px rgba(124,109,255,.33); } .secondary { background:linear-gradient(90deg,#0ba8bb,var(--accent2)); box-shadow:0 8px 20px rgba(0,216,194,.28); } .ghost { background:rgba(255,255,255,.08); border:1px solid var(--line); } .danger { background:linear-gradient(90deg,#c94067,#d64f7d); box-shadow:0 8px 20px rgba(198,64,103,.28); }
    .kpis { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:8px; margin-top:10px; } .kpi { background:rgba(255,255,255,.06); border:1px solid var(--line); border-radius:12px; padding:10px; } .kpi .k { color:var(--muted); font-size:12px; margin-bottom:4px; } .kpi .v { font-weight:700; font-size:18px; }
    .reward { font-size:32px; font-weight:800; letter-spacing:.3px; margin:3px 0; transition:color .2s ease, text-shadow .2s ease; } .reward.good { color:var(--ok); text-shadow:0 0 14px rgba(87,226,157,.45); } .reward.mid { color:var(--warn); text-shadow:0 0 14px rgba(255,209,102,.38); } .reward.bad { color:var(--bad); text-shadow:0 0 14px rgba(255,107,136,.38); } .small { color:var(--muted); font-size:12px; } .spark { height:100px; width:100%; margin-top:6px; }
    .timeline { max-height:280px; overflow:auto; display:grid; gap:8px; padding-right:3px; } .event { border:1px solid var(--line); background:rgba(255,255,255,.05); border-radius:10px; padding:8px 9px; font-size:13px; line-height:1.4; animation:rise .25s ease; } @keyframes rise { from { opacity:0; transform:translateY(6px);} to { opacity:1; transform:translateY(0);} } .event .meta { color:var(--muted); font-size:12px; margin-bottom:4px; }
    .error-summary { margin:8px 0 0; border:1px solid rgba(255,107,136,.45); background:rgba(201,64,103,.15); border-radius:10px; padding:8px 10px; font-size:13px; display:none; } .error-summary.show { display:block; }
    .inline-error { margin-top:4px; color:#ffb2c4; font-size:12px; min-height:16px; }
    .toast { position:fixed; right:12px; top:12px; background:rgba(15,19,38,.95); border:1px solid var(--line); border-radius:12px; padding:9px 12px; opacity:0; transform:translateY(-10px); transition:all .22s ease; z-index:15; } .toast.show { opacity:1; transform:translateY(0); }
    .doc-grid { display:grid; gap:8px; margin-top:8px; }
    .doc-item { border:1px solid var(--line); border-radius:10px; background:rgba(255,255,255,.04); padding:9px; }
    .doc-item code { font-size:12px; color:#cce0ff; }
    .feature-list { margin-left:16px; color:var(--muted); line-height:1.5; }
    .feature-list li { margin-bottom:6px; }
    .score-grid { display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:8px; margin-top:8px; }
    .score { border:1px solid var(--line); border-radius:10px; background:rgba(255,255,255,.04); padding:8px; }
    .score .k { color:var(--muted); font-size:12px; }
    .score .v { font-size:16px; font-weight:700; }
    @media (max-width:1120px) { .topbar,.grid { grid-template-columns:1fr; } .kpis { grid-template-columns:repeat(2,minmax(0,1fr)); } }
    @media (max-width:1000px) { .form-grid { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <div class="ambient"><canvas id="particles"></canvas></div>
  <div class="wrap">
    <div class="topbar">
      <div class="panel">
        <h1>SocialEngineerArena Control Center</h1>
        <div class="sub">Production-ready interface for running episodes, validating actions, and tracking rewards in real time.</div>
        <div class="row">
          <div class="badge"><span class="health-dot" id="healthDot"></span>API: <strong id="apiStatus">checking...</strong></div>
          <div class="badge">Role: <strong id="rolePill">-</strong></div>
          <div class="badge">Scenario: <strong id="scenarioPill">-</strong></div>
          <div class="badge">Episode: <strong id="episodePill">0</strong></div>
          <div class="badge">Step: <strong id="stepPill">0</strong></div>
        </div>
      </div>
      <div class="panel">
        <div class="label">Interaction Mode</div>
        <div class="form-grid">
          <div><select id="modeSelect" aria-label="Action mode"><option value="manual">Manual (you decide)</option><option value="assisted">Assisted (model suggestion)</option></select></div>
          <div><input id="boundaryPreset" value="Defensive analysis only." aria-label="Safety boundary preset" /></div>
        </div>
        <div class="label">Reward Meter</div>
        <div id="rewardValue" class="reward mid">0.000</div>
        <canvas id="spark" class="spark"></canvas>
        <div class="small">Live trend of submitted actions (session-local).</div>
      </div>
    </div>
    <div class="grid">
      <div class="left-col">
        <div class="panel">
          <div class="label">Incoming Message</div>
          <div id="incomingMessage" class="box">Click "Start Episode".</div>
          <div class="label" style="margin-top:10px;">Context</div>
          <div id="contextBox" class="box"></div>
          <div class="kpis">
            <div class="kpi"><div class="k">Episodes</div><div class="v" id="kpiEpisodes">0</div></div>
            <div class="kpi"><div class="k">Submitted</div><div class="v" id="kpiActions">0</div></div>
            <div class="kpi"><div class="k">Completed</div><div class="v" id="kpiDone">0</div></div>
            <div class="kpi"><div class="k">Avg Reward</div><div class="v" id="kpiAvg">0.000</div></div>
          </div>
        </div>
        <div class="panel"><div class="label">Episode Timeline</div><div id="timeline" class="timeline"></div></div>
      </div>
      <div class="right-col">
        <div class="panel">
          <div class="label">Action Editor</div>
          <div class="form-grid">
            <div><div class="label">Verdict</div><select id="verdict"><option value="unknown">unknown</option><option value="safe">safe</option><option value="phishing">phishing</option><option value="pretexting">pretexting</option></select><div id="verdictErr" class="inline-error" aria-live="polite"></div></div>
            <div><div class="label">Safety Boundary</div><input id="safetyBoundary" value="Defensive analysis only." /><div id="boundaryErr" class="inline-error" aria-live="polite"></div></div>
          </div>
          <div class="label">Explanation</div><textarea id="explanation" rows="4" placeholder="Give evidence-backed reasoning tied to the message and policy..."></textarea><div id="explanationErr" class="inline-error" aria-live="polite"></div>
          <div class="label">Cues Found (comma-separated)</div><input id="cuesFound" placeholder="urgency pressure, lookalike sender, credential solicitation" /><div id="cuesErr" class="inline-error" aria-live="polite"></div>
          <div class="label">Response</div><textarea id="response" rows="3" placeholder="Optional responder message or simulation text"></textarea>
          <div id="errorSummary" class="error-summary" role="alert"></div>
          <div class="actions"><button id="resetBtn" class="secondary">Start Episode</button><button id="suggestBtn" class="ghost">Suggest Action</button><button id="stepBtn" class="primary">Submit Action</button><button id="exportBtn" class="ghost">Export Trace JSONL</button><button id="clearBtn" class="danger">Clear Form</button></div>
        </div>
        <div class="panel"><div class="label">Result</div><div id="resultBox" class="box">No actions submitted yet.</div></div>
        <div class="panel">
          <div class="label">Evaluation Scoreboard</div>
          <div class="small">Live metrics from this session.</div>
          <div class="score-grid">
            <div class="score"><div class="k">Episodes Completed</div><div id="sbCompleted" class="v">0</div></div>
            <div class="score"><div class="k">Avg Final Reward</div><div id="sbAvgReward" class="v">0.000</div></div>
            <div class="score"><div class="k">Suggest JSON Parse Rate</div><div id="sbParseRate" class="v">0%</div></div>
            <div class="score"><div class="k">Suggest Avg Latency</div><div id="sbLatency" class="v">0 ms</div></div>
          </div>
        </div>
        <div class="panel">
          <div class="label">API Docs</div>
          <div class="small">In-app reference for integration, testing, and hackathon demos.</div>
          <div class="doc-grid">
            <div class="doc-item"><strong>POST /reset</strong><br><code>{}</code><br><span class="small">Starts a new episode and returns <code>ArenaObservation</code>.</span></div>
            <div class="doc-item"><strong>POST /step</strong><br><code>{"verdict":"safe","explanation":"...","cues_found":["..."],"response":"...","safety_boundary":"..."}</code><br><span class="small">Scores current turn and returns <code>{observation,reward,done}</code>.</span></div>
            <div class="doc-item"><strong>POST /suggest</strong><br><code>{}</code><br><span class="small">Gets model-backed action suggestion. Requires <code>HF_TOKEN</code> + model endpoint env.</span></div>
            <div class="doc-item"><strong>GET /state</strong><br><code>n/a</code><br><span class="small">Returns current <code>ArenaState</code> for UI sync/monitoring.</span></div>
            <div class="doc-item"><strong>GET /health</strong><br><code>n/a</code><br><span class="small">Basic service health probe. Returns <code>{"status":"ok"}</code>.</span></div>
          </div>
          <div class="label" style="margin-top:10px;">More Features You Can Implement</div>
          <ul class="feature-list">
            <li>Replay mode with per-turn diff of model suggestion vs submitted action.</li>
            <li>Leaderboard panel for average reward by model/version.</li>
            <li>Evaluation tab for train/test split metrics and confusion matrix.</li>
            <li>Prompt/version registry in UI to compare multiple suggest models live.</li>
            <li>Judge mode export: one-click bundle of episode traces + scores as JSONL.</li>
          </ul>
        </div>
      </div>
    </div>
  </div>
  <div id="toast" class="toast"></div>
  <script>
    const rewardHistory = []; const actionLog = []; let observation = null; let episodeCount = 0; let completedEpisodes = 0;
    const sessionMetrics = { suggestCalls: 0, parsedSuggest: 0, suggestLatencyMs: [], finalRewards: [] };
    const $ = (id) => document.getElementById(id); const rewardValueEl = $("rewardValue"); const resultBoxEl = $("resultBox");
    const modeSelectEl = $("modeSelect"); const verdictEl = $("verdict"); const explanationEl = $("explanation"); const cuesFoundEl = $("cuesFound"); const responseEl = $("response"); const safetyBoundaryEl = $("safetyBoundary"); const boundaryPresetEl = $("boundaryPreset");
    function toast(msg) { const t = $("toast"); t.textContent = msg; t.classList.add("show"); setTimeout(() => t.classList.remove("show"), 1500); }
    function setBusy(v) { for (const id of ["resetBtn","stepBtn","suggestBtn","clearBtn"]) $(id).disabled = v; }
    function syncActionControls() {
      const done = !!(observation && observation.done);
      const active = !!(observation && !observation.done);
      $("stepBtn").disabled = done;
      $("suggestBtn").disabled = done;
      $("resetBtn").disabled = active;
    }
    function updateKpis() { $("kpiEpisodes").textContent = String(episodeCount); $("kpiActions").textContent = String(actionLog.length); $("kpiDone").textContent = String(completedEpisodes); const avg = rewardHistory.length ? rewardHistory.reduce((a,b)=>a+b,0)/rewardHistory.length : 0; $("kpiAvg").textContent = avg.toFixed(3); updateScoreboard(); }
    function updateScoreboard() { $("sbCompleted").textContent = String(completedEpisodes); const avgFinal = sessionMetrics.finalRewards.length ? sessionMetrics.finalRewards.reduce((a,b)=>a+b,0)/sessionMetrics.finalRewards.length : 0; $("sbAvgReward").textContent = avgFinal.toFixed(3); const rate = sessionMetrics.suggestCalls ? (100 * sessionMetrics.parsedSuggest / sessionMetrics.suggestCalls) : 0; $("sbParseRate").textContent = rate.toFixed(1) + "%"; const lat = sessionMetrics.suggestLatencyMs.length ? (sessionMetrics.suggestLatencyMs.reduce((a,b)=>a+b,0)/sessionMetrics.suggestLatencyMs.length) : 0; $("sbLatency").textContent = Math.round(lat) + " ms"; }
    function setReward(v) { rewardValueEl.textContent = Number(v).toFixed(3); rewardValueEl.className = "reward " + (v >= 0.7 ? "good" : v >= 0.35 ? "mid" : "bad"); rewardHistory.push(v); if (rewardHistory.length > 60) rewardHistory.shift(); drawSparkline(); updateKpis(); }
    function drawSparkline() { const canvas = $("spark"); const ctx = canvas.getContext("2d"); const ratio = window.devicePixelRatio || 1; canvas.width = canvas.clientWidth * ratio; canvas.height = canvas.clientHeight * ratio; ctx.setTransform(ratio,0,0,ratio,0,0); const w = canvas.clientWidth; const h = canvas.clientHeight; ctx.clearRect(0,0,w,h); if (!rewardHistory.length) return; ctx.strokeStyle = "rgba(255,255,255,.18)"; ctx.lineWidth = 1; for (const y of [0.25,0.5,0.75]) { ctx.beginPath(); ctx.moveTo(0,h*y); ctx.lineTo(w,h*y); ctx.stroke(); } const step = w / Math.max(1, rewardHistory.length - 1); const grad = ctx.createLinearGradient(0,0,w,0); grad.addColorStop(0,"#00d8c2"); grad.addColorStop(1,"#7c6dff"); ctx.strokeStyle = grad; ctx.lineWidth = 2.4; ctx.beginPath(); rewardHistory.forEach((r,i) => { const x = i*step; const y = h - Math.max(0,Math.min(1,r))*h; if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y); }); ctx.stroke(); }
    function clearErrors() { $("errorSummary").classList.remove("show"); $("errorSummary").textContent = ""; for (const id of ["verdictErr","boundaryErr","explanationErr","cuesErr"]) $(id).textContent = ""; for (const id of ["verdict","safetyBoundary","explanation","cuesFound"]) $(id).classList.remove("error"); }
    function validateForm() { clearErrors(); const errors = []; const explanation = explanationEl.value.trim(); const boundary = safetyBoundaryEl.value.trim(); const cues = cuesFoundEl.value.split(",").map((x)=>x.trim()).filter(Boolean); if (!verdictEl.value) { errors.push("Verdict is required."); $("verdictErr").textContent = "Choose a verdict."; verdictEl.classList.add("error"); } if (explanation.length < 16) { errors.push("Explanation should be at least 16 characters."); $("explanationErr").textContent = "Add evidence-based reasoning."; explanationEl.classList.add("error"); } if (!boundary) { errors.push("Safety boundary cannot be empty."); $("boundaryErr").textContent = "Set a clear safety boundary."; safetyBoundaryEl.classList.add("error"); } if (cues.length === 0) { errors.push("Add at least one cue."); $("cuesErr").textContent = "Add at least one cue."; cuesFoundEl.classList.add("error"); } if (errors.length) { const summary = $("errorSummary"); summary.textContent = "Please fix: " + errors.join(" "); summary.classList.add("show"); return false; } return true; }
    async function suggestAction() { if (!observation) { toast("Start episode first"); return; } if (observation.done) { toast("Episode complete. Start a new episode."); syncActionControls(); return; } setBusy(true); const t0 = performance.now(); try { const res = await fetch("suggest", { method:"POST" }); const out = await res.json(); if (!res.ok) throw new Error((typeof out.detail === "string" ? out.detail : JSON.stringify(out.detail)) || "Suggest failed"); const action = out.action || {}; sessionMetrics.suggestCalls += 1; sessionMetrics.parsedSuggest += 1; sessionMetrics.suggestLatencyMs.push(performance.now() - t0); verdictEl.value = action.verdict || "unknown"; explanationEl.value = action.explanation || ""; cuesFoundEl.value = Array.isArray(action.cues_found) ? action.cues_found.join(", ") : ""; responseEl.value = action.response || ""; safetyBoundaryEl.value = action.safety_boundary || boundaryPresetEl.value.trim() || "Defensive analysis only."; const src = out.source || "(unknown)"; const raw = out.raw_preview || ""; resultBoxEl.textContent = ["Suggestion source: " + src, "", "Parsed action (JSON):", JSON.stringify(action, null, 2), "", "Raw model output (preview):", raw || "(empty)"].join("\\n"); toast("Suggestion loaded from model"); } catch (e) { sessionMetrics.suggestCalls += 1; sessionMetrics.suggestLatencyMs.push(performance.now() - t0); resultBoxEl.textContent = "Suggest error: " + e.message + "\\n\\nSet HF_ENDPOINT_URL + HF_TOKEN on the server (Space secrets), or run locally with the same env vars."; toast("Suggestion unavailable"); } finally { updateScoreboard(); setBusy(false); syncActionControls(); } }
    function pushEvent(title, details, reward, done) { const item = { at:new Date().toLocaleTimeString(), title, details, reward, done }; actionLog.unshift(item); const host = $("timeline"); host.innerHTML = actionLog.slice(0,50).map((e)=>( '<div class="event"><div class="meta">' + e.at + " • " + e.title + (e.done ? " • episode complete" : "") + '</div><div>' + e.details.replace(/</g,"&lt;").replace(/>/g,"&gt;") + '</div><div class="meta">reward: ' + Number(e.reward || 0).toFixed(3) + '</div></div>' )).join(""); updateKpis(); }
    async function checkHealth() { try { const res = await fetch("health"); if (!res.ok) throw new Error("bad"); $("apiStatus").textContent = "online"; $("healthDot").style.background = "var(--ok)"; $("healthDot").style.boxShadow = "0 0 14px rgba(87,226,157,.7)"; } catch (_e) { $("apiStatus").textContent = "offline"; $("healthDot").style.background = "var(--bad)"; $("healthDot").style.boxShadow = "0 0 14px rgba(255,107,136,.6)"; } }
    async function resetScenario() { setBusy(true); clearErrors(); try { const res = await fetch("reset", { method:"POST" }); const body = await res.json(); if (!res.ok) throw new Error(body.detail || "Reset failed"); observation = body; episodeCount += 1; $("rolePill").textContent = observation.role; $("scenarioPill").textContent = observation.scenario_id; $("episodePill").textContent = String(episodeCount); $("stepPill").textContent = String(observation.turn_index + 1) + "/" + String(observation.total_turns); $("incomingMessage").textContent = observation.incoming_message; $("contextBox").textContent = ["Channel: " + observation.channel, "Persona: " + observation.persona, "Organization: " + observation.organization, "Task: " + observation.task, "", "Policy:", observation.policy_excerpt || "(none)", "", "Thread Context:", observation.thread_context || "(none)"].join("\\n"); resultBoxEl.textContent = "Episode initialized. Submit one action to complete this episode."; safetyBoundaryEl.value = boundaryPresetEl.value.trim() || "Defensive analysis only."; pushEvent("Episode started", "Scenario " + observation.scenario_id + " loaded.", 0, false); toast("Episode loaded"); } catch (e) { resultBoxEl.textContent = "Error: " + e.message; } finally { setBusy(false); syncActionControls(); } }
    async function submitAction() { if (!observation) { toast("Start episode first"); return; } if (observation.done) { toast("Episode complete. Start a new episode."); syncActionControls(); return; } if (!validateForm()) return; setBusy(true); try { const payload = { verdict: verdictEl.value, explanation: explanationEl.value.trim(), cues_found: cuesFoundEl.value.split(",").map((x)=>x.trim()).filter(Boolean), response: responseEl.value.trim(), safety_boundary: safetyBoundaryEl.value.trim() }; const res = await fetch("step", { method:"POST", headers:{ "Content-Type":"application/json" }, body: JSON.stringify(payload) }); const out = await res.json(); if (!res.ok) throw new Error(out.detail || "Step failed"); observation = out.observation; $("stepPill").textContent = String(observation.turn_index + 1) + "/" + String(observation.total_turns); $("incomingMessage").textContent = observation.incoming_message; const reward = Number(out.reward || 0); setReward(reward); if (out.done) { completedEpisodes += 1; sessionMetrics.finalRewards.push(reward); } const breakdown = out?.observation?.reward_breakdown || {}; resultBoxEl.textContent = ["done: " + out.done, "reward: " + reward.toFixed(3), "", "breakdown:", JSON.stringify(breakdown, null, 2)].join("\\n"); pushEvent("Action submitted", "Verdict: " + payload.verdict + " | cues: " + payload.cues_found.slice(0,3).join(", "), reward, !!out.done); if (out.done) toast("Episode complete. Start next episode."); else toast("Turn scored"); } catch (e) { resultBoxEl.textContent = "Error: " + e.message; } finally { updateScoreboard(); setBusy(false); syncActionControls(); } }
    function exportTrace() { const lines = actionLog.map((e) => JSON.stringify(e)); const blob = new Blob([lines.join("\\n") + "\\n"], { type:"application/jsonl" }); const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "social_engineer_arena_trace.jsonl"; a.click(); URL.revokeObjectURL(a.href); toast("Trace exported"); }
    function clearForm() { explanationEl.value = ""; cuesFoundEl.value = ""; responseEl.value = ""; verdictEl.value = "unknown"; safetyBoundaryEl.value = boundaryPresetEl.value.trim() || "Defensive analysis only."; clearErrors(); toast("Form cleared"); }
    function persistPrefs() { localStorage.setItem("sea_mode", modeSelectEl.value); localStorage.setItem("sea_boundary", boundaryPresetEl.value); }
    function loadPrefs() { modeSelectEl.value = localStorage.getItem("sea_mode") || "manual"; boundaryPresetEl.value = localStorage.getItem("sea_boundary") || "Defensive analysis only."; safetyBoundaryEl.value = boundaryPresetEl.value; }
    function bootParticles() { const canvas = $("particles"); const ctx = canvas.getContext("2d"); let w = 0; let h = 0; const dots = Array.from({ length:42 }).map(() => ({ x:Math.random(), y:Math.random(), r:Math.random()*2.2+.8, vx:(Math.random()-0.5)*0.0005, vy:(Math.random()-0.5)*0.0005 })); function resize() { w = canvas.clientWidth = window.innerWidth; h = canvas.clientHeight = window.innerHeight; const dpr = window.devicePixelRatio || 1; canvas.width = w*dpr; canvas.height = h*dpr; ctx.setTransform(dpr,0,0,dpr,0,0); } function tick() { ctx.clearRect(0,0,w,h); for (const d of dots) { d.x += d.vx; d.y += d.vy; if (d.x < 0 || d.x > 1) d.vx *= -1; if (d.y < 0 || d.y > 1) d.vy *= -1; const x = d.x*w; const y = d.y*h; const g = ctx.createRadialGradient(x,y,0,x,y,d.r*8); g.addColorStop(0,"rgba(122,109,255,.35)"); g.addColorStop(1,"rgba(0,216,194,0)"); ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x,y,d.r*8,0,Math.PI*2); ctx.fill(); } requestAnimationFrame(tick); } resize(); window.addEventListener("resize", resize); requestAnimationFrame(tick); }
    $("resetBtn").addEventListener("click", resetScenario); $("stepBtn").addEventListener("click", submitAction); $("suggestBtn").addEventListener("click", suggestAction); $("exportBtn").addEventListener("click", exportTrace); $("clearBtn").addEventListener("click", clearForm); modeSelectEl.addEventListener("change", persistPrefs); boundaryPresetEl.addEventListener("change", () => { safetyBoundaryEl.value = boundaryPresetEl.value; persistPrefs(); }); window.addEventListener("resize", drawSparkline); window.addEventListener("keydown", (e) => { if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "enter") submitAction(); });
    loadPrefs(); bootParticles(); checkHealth(); setInterval(checkHealth, 10000); drawSparkline(); updateKpis(); syncActionControls(); resetScenario();
  </script>
</body>
</html>"""

    @app.get("/", response_class=HTMLResponse)
    def root_showcase() -> str:
        return arena_showcase()

    @app.get("/web", response_class=HTMLResponse)
    def web_showcase() -> str:
        return arena_showcase()


app = FastAPI(title="SocialEngineerArena")
attach_showcase_routes(app)

if os.getenv("ENABLE_WEB_INTERFACE", "true").lower() == "true" and create_web_interface_app:
    # Keep custom showcase as primary app and mount OpenEnv web UI separately.
    web_app = create_web_interface_app(SocialEngineerArenaEnvironment, ArenaAction, ArenaObservation)
    app.mount("/web", web_app)


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
