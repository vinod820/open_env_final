from __future__ import annotations

import json
import os
import re
from urllib import error, request

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
try:
    from openenv.core.env_server import create_web_interface_app
except ImportError:
    create_web_interface_app = None

from social_engineer_arena.models import ArenaAction, ArenaObservation, ArenaState
from social_engineer_arena.server.environment import SocialEngineerArenaEnvironment


env = SocialEngineerArenaEnvironment()


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

Return ONLY one JSON object with keys:
verdict, explanation, cues_found, response, safety_boundary.
"""


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


def _generate_suggestion(obs: ArenaObservation) -> tuple[ArenaAction, str]:
    model_id = os.getenv("SUGGEST_MODEL_ID", "").strip()
    endpoint_url = os.getenv("HF_ENDPOINT_URL", "").strip()
    hf_token = os.getenv("HF_TOKEN", "").strip()
    max_new_tokens = int(os.getenv("SUGGEST_MAX_NEW_TOKENS", "180"))
    if not model_id and not endpoint_url:
        raise RuntimeError(
            "No suggestion model configured. Set SUGGEST_MODEL_ID or HF_ENDPOINT_URL."
        )
    if not hf_token:
        raise RuntimeError("HF_TOKEN is required for model-backed suggestion.")

    target_url = endpoint_url or f"https://api-inference.huggingface.co/models/{model_id}"
    payload = {
        "inputs": _suggest_prompt(obs),
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.2,
            "return_full_text": False,
        },
    }
    req = request.Request(
        target_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {hf_token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=45) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Suggestion request failed ({exc.code}): {detail}") from exc
    except Exception as exc:
        raise RuntimeError(f"Suggestion request failed: {exc}") from exc

    generated = ""
    if isinstance(body, list) and body and isinstance(body[0], dict):
        generated = body[0].get("generated_text", "")
    elif isinstance(body, dict):
        generated = body.get("generated_text", "")
    action = _parse_action_json(generated)
    source = endpoint_url or model_id
    return action, source


def attach_showcase_routes(app: FastAPI) -> None:
    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    def reset_handler() -> ArenaObservation:
        return env.reset()

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
        action, source = _generate_suggestion(obs)
        return {"action": action, "source": source}

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
          <div class="actions"><button id="resetBtn" class="secondary">Start Episode</button><button id="suggestBtn" class="ghost">Suggest Action</button><button id="stepBtn" class="primary">Submit Action</button><button id="clearBtn" class="danger">Clear Form</button></div>
        </div>
        <div class="panel"><div class="label">Result</div><div id="resultBox" class="box">No actions submitted yet.</div></div>
      </div>
    </div>
  </div>
  <div id="toast" class="toast"></div>
  <script>
    const rewardHistory = []; const actionLog = []; let observation = null; let episodeCount = 0; let completedEpisodes = 0;
    const $ = (id) => document.getElementById(id); const rewardValueEl = $("rewardValue"); const resultBoxEl = $("resultBox");
    const modeSelectEl = $("modeSelect"); const verdictEl = $("verdict"); const explanationEl = $("explanation"); const cuesFoundEl = $("cuesFound"); const responseEl = $("response"); const safetyBoundaryEl = $("safetyBoundary"); const boundaryPresetEl = $("boundaryPreset");
    function toast(msg) { const t = $("toast"); t.textContent = msg; t.classList.add("show"); setTimeout(() => t.classList.remove("show"), 1500); }
    function setBusy(v) { for (const id of ["resetBtn","stepBtn","suggestBtn","clearBtn"]) $(id).disabled = v; }
    function syncActionControls() {
      const done = !!(observation && observation.done);
      $("stepBtn").disabled = done;
      $("suggestBtn").disabled = done;
    }
    function updateKpis() { $("kpiEpisodes").textContent = String(episodeCount); $("kpiActions").textContent = String(actionLog.length); $("kpiDone").textContent = String(completedEpisodes); const avg = rewardHistory.length ? rewardHistory.reduce((a,b)=>a+b,0)/rewardHistory.length : 0; $("kpiAvg").textContent = avg.toFixed(3); }
    function setReward(v) { rewardValueEl.textContent = Number(v).toFixed(3); rewardValueEl.className = "reward " + (v >= 0.7 ? "good" : v >= 0.35 ? "mid" : "bad"); rewardHistory.push(v); if (rewardHistory.length > 60) rewardHistory.shift(); drawSparkline(); updateKpis(); }
    function drawSparkline() { const canvas = $("spark"); const ctx = canvas.getContext("2d"); const ratio = window.devicePixelRatio || 1; canvas.width = canvas.clientWidth * ratio; canvas.height = canvas.clientHeight * ratio; ctx.setTransform(ratio,0,0,ratio,0,0); const w = canvas.clientWidth; const h = canvas.clientHeight; ctx.clearRect(0,0,w,h); if (!rewardHistory.length) return; ctx.strokeStyle = "rgba(255,255,255,.18)"; ctx.lineWidth = 1; for (const y of [0.25,0.5,0.75]) { ctx.beginPath(); ctx.moveTo(0,h*y); ctx.lineTo(w,h*y); ctx.stroke(); } const step = w / Math.max(1, rewardHistory.length - 1); const grad = ctx.createLinearGradient(0,0,w,0); grad.addColorStop(0,"#00d8c2"); grad.addColorStop(1,"#7c6dff"); ctx.strokeStyle = grad; ctx.lineWidth = 2.4; ctx.beginPath(); rewardHistory.forEach((r,i) => { const x = i*step; const y = h - Math.max(0,Math.min(1,r))*h; if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y); }); ctx.stroke(); }
    function clearErrors() { $("errorSummary").classList.remove("show"); $("errorSummary").textContent = ""; for (const id of ["verdictErr","boundaryErr","explanationErr","cuesErr"]) $(id).textContent = ""; for (const id of ["verdict","safetyBoundary","explanation","cuesFound"]) $(id).classList.remove("error"); }
    function validateForm() { clearErrors(); const errors = []; const explanation = explanationEl.value.trim(); const boundary = safetyBoundaryEl.value.trim(); const cues = cuesFoundEl.value.split(",").map((x)=>x.trim()).filter(Boolean); if (!verdictEl.value) { errors.push("Verdict is required."); $("verdictErr").textContent = "Choose a verdict."; verdictEl.classList.add("error"); } if (explanation.length < 16) { errors.push("Explanation should be at least 16 characters."); $("explanationErr").textContent = "Add evidence-based reasoning."; explanationEl.classList.add("error"); } if (!boundary) { errors.push("Safety boundary cannot be empty."); $("boundaryErr").textContent = "Set a clear safety boundary."; safetyBoundaryEl.classList.add("error"); } if (cues.length === 0) { errors.push("Add at least one cue."); $("cuesErr").textContent = "Add at least one cue."; cuesFoundEl.classList.add("error"); } if (errors.length) { const summary = $("errorSummary"); summary.textContent = "Please fix: " + errors.join(" "); summary.classList.add("show"); return false; } return true; }
    async function suggestAction() { if (!observation) { toast("Start episode first"); return; } if (observation.done) { toast("Episode complete. Start a new episode."); syncActionControls(); return; } setBusy(true); try { const res = await fetch("suggest", { method:"POST" }); const out = await res.json(); if (!res.ok) throw new Error(out.detail || "Suggest failed"); const action = out.action || {}; verdictEl.value = action.verdict || "unknown"; explanationEl.value = action.explanation || ""; cuesFoundEl.value = Array.isArray(action.cues_found) ? action.cues_found.join(", ") : ""; responseEl.value = action.response || ""; safetyBoundaryEl.value = action.safety_boundary || boundaryPresetEl.value.trim() || "Defensive analysis only."; toast("Suggestion loaded from model"); } catch (e) { resultBoxEl.textContent = "Suggest error: " + e.message; toast("Suggestion unavailable"); } finally { setBusy(false); syncActionControls(); } }
    function pushEvent(title, details, reward, done) { const item = { at:new Date().toLocaleTimeString(), title, details, reward, done }; actionLog.unshift(item); const host = $("timeline"); host.innerHTML = actionLog.slice(0,50).map((e)=>( '<div class="event"><div class="meta">' + e.at + " • " + e.title + (e.done ? " • episode complete" : "") + '</div><div>' + e.details.replace(/</g,"&lt;").replace(/>/g,"&gt;") + '</div><div class="meta">reward: ' + Number(e.reward || 0).toFixed(3) + '</div></div>' )).join(""); updateKpis(); }
    async function checkHealth() { try { const res = await fetch("health"); if (!res.ok) throw new Error("bad"); $("apiStatus").textContent = "online"; $("healthDot").style.background = "var(--ok)"; $("healthDot").style.boxShadow = "0 0 14px rgba(87,226,157,.7)"; } catch (_e) { $("apiStatus").textContent = "offline"; $("healthDot").style.background = "var(--bad)"; $("healthDot").style.boxShadow = "0 0 14px rgba(255,107,136,.6)"; } }
    async function resetScenario() { setBusy(true); clearErrors(); try { const res = await fetch("reset", { method:"POST" }); if (!res.ok) throw new Error("Reset failed"); observation = await res.json(); episodeCount += 1; $("rolePill").textContent = observation.role; $("scenarioPill").textContent = observation.scenario_id; $("episodePill").textContent = String(episodeCount); $("stepPill").textContent = String(observation.turn_index); $("incomingMessage").textContent = observation.incoming_message; $("contextBox").textContent = ["Channel: " + observation.channel, "Persona: " + observation.persona, "Organization: " + observation.organization, "Task: " + observation.task, "", "Policy:", observation.policy_excerpt || "(none)", "", "Thread Context:", observation.thread_context || "(none)"].join("\\n"); resultBoxEl.textContent = "Episode initialized. Prepare action and submit."; safetyBoundaryEl.value = boundaryPresetEl.value.trim() || "Defensive analysis only."; pushEvent("Episode started", "Scenario " + observation.scenario_id + " loaded.", 0, false); toast("Episode loaded"); } catch (e) { resultBoxEl.textContent = "Error: " + e.message; } finally { setBusy(false); syncActionControls(); } }
    async function submitAction() { if (!observation) { toast("Start episode first"); return; } if (observation.done) { toast("Episode complete. Start a new episode."); syncActionControls(); return; } if (!validateForm()) return; setBusy(true); try { const payload = { verdict: verdictEl.value, explanation: explanationEl.value.trim(), cues_found: cuesFoundEl.value.split(",").map((x)=>x.trim()).filter(Boolean), response: responseEl.value.trim(), safety_boundary: safetyBoundaryEl.value.trim() }; const res = await fetch("step", { method:"POST", headers:{ "Content-Type":"application/json" }, body: JSON.stringify(payload) }); const out = await res.json(); if (!res.ok) throw new Error(out.detail || "Step failed"); observation = out.observation; $("stepPill").textContent = String(observation.turn_index); $("incomingMessage").textContent = observation.incoming_message; const reward = Number(out.reward || 0); setReward(reward); if (out.done) completedEpisodes += 1; const breakdown = out?.observation?.reward_breakdown || {}; resultBoxEl.textContent = ["done: " + out.done, "reward: " + reward.toFixed(3), "", "breakdown:", JSON.stringify(breakdown, null, 2)].join("\\n"); pushEvent("Action submitted", "Verdict: " + payload.verdict + " | cues: " + payload.cues_found.slice(0,3).join(", "), reward, !!out.done); if (out.done) toast("Episode complete"); else toast("Turn scored"); } catch (e) { resultBoxEl.textContent = "Error: " + e.message; } finally { setBusy(false); syncActionControls(); } }
    function clearForm() { explanationEl.value = ""; cuesFoundEl.value = ""; responseEl.value = ""; verdictEl.value = "unknown"; safetyBoundaryEl.value = boundaryPresetEl.value.trim() || "Defensive analysis only."; clearErrors(); toast("Form cleared"); }
    function persistPrefs() { localStorage.setItem("sea_mode", modeSelectEl.value); localStorage.setItem("sea_boundary", boundaryPresetEl.value); }
    function loadPrefs() { modeSelectEl.value = localStorage.getItem("sea_mode") || "manual"; boundaryPresetEl.value = localStorage.getItem("sea_boundary") || "Defensive analysis only."; safetyBoundaryEl.value = boundaryPresetEl.value; }
    function bootParticles() { const canvas = $("particles"); const ctx = canvas.getContext("2d"); let w = 0; let h = 0; const dots = Array.from({ length:42 }).map(() => ({ x:Math.random(), y:Math.random(), r:Math.random()*2.2+.8, vx:(Math.random()-0.5)*0.0005, vy:(Math.random()-0.5)*0.0005 })); function resize() { w = canvas.clientWidth = window.innerWidth; h = canvas.clientHeight = window.innerHeight; const dpr = window.devicePixelRatio || 1; canvas.width = w*dpr; canvas.height = h*dpr; ctx.setTransform(dpr,0,0,dpr,0,0); } function tick() { ctx.clearRect(0,0,w,h); for (const d of dots) { d.x += d.vx; d.y += d.vy; if (d.x < 0 || d.x > 1) d.vx *= -1; if (d.y < 0 || d.y > 1) d.vy *= -1; const x = d.x*w; const y = d.y*h; const g = ctx.createRadialGradient(x,y,0,x,y,d.r*8); g.addColorStop(0,"rgba(122,109,255,.35)"); g.addColorStop(1,"rgba(0,216,194,0)"); ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x,y,d.r*8,0,Math.PI*2); ctx.fill(); } requestAnimationFrame(tick); } resize(); window.addEventListener("resize", resize); requestAnimationFrame(tick); }
    $("resetBtn").addEventListener("click", resetScenario); $("stepBtn").addEventListener("click", submitAction); $("suggestBtn").addEventListener("click", suggestAction); $("clearBtn").addEventListener("click", clearForm); modeSelectEl.addEventListener("change", persistPrefs); boundaryPresetEl.addEventListener("change", () => { safetyBoundaryEl.value = boundaryPresetEl.value; persistPrefs(); }); window.addEventListener("resize", drawSparkline); window.addEventListener("keydown", (e) => { if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "enter") submitAction(); });
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
