from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.responses import RedirectResponse
try:
    from openenv.core.env_server import create_web_interface_app
except ImportError:
    create_web_interface_app = None

from social_engineer_arena.models import ArenaAction, ArenaObservation, ArenaState
from social_engineer_arena.server.environment import SocialEngineerArenaEnvironment


env = SocialEngineerArenaEnvironment()


def attach_showcase_routes(app: FastAPI) -> None:
    @app.get("/")
    def root_redirect() -> RedirectResponse:
        # Make the showcase UI the default Space landing page.
        return RedirectResponse(url="/arena")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/reset")
    def api_reset() -> ArenaObservation:
        return env.reset()

    @app.post("/api/step")
    def api_step(action: ArenaAction) -> dict:
        observation, reward, done = env.step(action)
        return {"observation": observation, "reward": reward, "done": done}

    @app.get("/api/state")
    def api_state() -> ArenaState:
        return env.state

    @app.get("/arena", response_class=HTMLResponse)
    def arena_showcase() -> str:
        return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SocialEngineerArena - Live Demo</title>
  <style>
    :root {
      --bg0: #0a0a16;
      --bg1: #15152f;
      --accent: #7c5cff;
      --accent2: #16e0bd;
      --text: #f7f7ff;
      --muted: #a8adc9;
      --danger: #ff5f79;
      --safe: #56e39f;
      --warn: #ffcf5a;
      --glass: rgba(255, 255, 255, 0.06);
      --border: rgba(255, 255, 255, 0.15);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, Segoe UI, system-ui, sans-serif;
      color: var(--text);
      background: radial-gradient(circle at 20% 20%, #1f1f45 0%, transparent 40%),
                  radial-gradient(circle at 80% 0%, #113e48 0%, transparent 35%),
                  linear-gradient(120deg, var(--bg0), var(--bg1));
      min-height: 100vh;
      overflow-x: hidden;
    }
    .bg-anim::before, .bg-anim::after {
      content: "";
      position: fixed;
      width: 320px;
      height: 320px;
      border-radius: 50%;
      filter: blur(64px);
      z-index: -1;
      animation: drift 12s ease-in-out infinite alternate;
    }
    .bg-anim::before { background: #6c4cff88; top: 10%; left: -60px; }
    .bg-anim::after { background: #00e0c088; bottom: 2%; right: -80px; animation-delay: 1.5s; }
    @keyframes drift { from { transform: translateY(-20px); } to { transform: translateY(40px); } }
    .wrap { max-width: 1200px; margin: 24px auto; padding: 0 16px 32px; }
    .hero {
      display: grid;
      grid-template-columns: 1.3fr .7fr;
      gap: 16px;
      margin-bottom: 16px;
    }
    .card {
      background: var(--glass);
      border: 1px solid var(--border);
      backdrop-filter: blur(14px);
      border-radius: 18px;
      box-shadow: 0 10px 40px rgba(0,0,0,.28);
      padding: 16px;
    }
    h1 { margin: 0 0 10px; font-size: 28px; letter-spacing: .3px; }
    .subtitle { color: var(--muted); margin-bottom: 12px; }
    .stats { display: flex; gap: 10px; flex-wrap: wrap; }
    .pill {
      background: rgba(255,255,255,.08);
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 13px;
    }
    .layout {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }
    .label { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; margin-bottom: 6px; }
    .box {
      background: rgba(0,0,0,.2);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 10px;
      min-height: 110px;
      white-space: pre-wrap;
      line-height: 1.45;
    }
    textarea, select, input {
      width: 100%;
      border: 1px solid var(--border);
      background: rgba(0,0,0,.25);
      color: var(--text);
      border-radius: 12px;
      padding: 10px;
      font: inherit;
      outline: none;
      transition: border-color .2s, transform .2s;
    }
    textarea:focus, select:focus, input:focus { border-color: #7f9bff; transform: translateY(-1px); }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 10px; }
    .btns { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px; }
    button {
      border: 0;
      border-radius: 12px;
      color: white;
      cursor: pointer;
      padding: 10px 14px;
      font-weight: 600;
      transition: transform .18s ease, box-shadow .2s ease, opacity .2s;
    }
    button:hover { transform: translateY(-2px) scale(1.01); }
    button:disabled { opacity: .55; cursor: default; transform: none; }
    .primary { background: linear-gradient(90deg, var(--accent), #9b4dff); box-shadow: 0 8px 24px rgba(124,92,255,.35); }
    .secondary { background: linear-gradient(90deg, #0ea9a2, #0ac59f); box-shadow: 0 8px 24px rgba(22,224,189,.25); }
    .ghost { background: rgba(255,255,255,.1); border: 1px solid var(--border); }
    .reward {
      font-size: 34px;
      font-weight: 800;
      letter-spacing: .4px;
      margin: 6px 0;
      transition: color .2s ease, text-shadow .2s ease;
    }
    .reward.good { color: var(--safe); text-shadow: 0 0 16px #56e39f66; }
    .reward.mid { color: var(--warn); text-shadow: 0 0 16px #ffcf5a55; }
    .reward.bad { color: var(--danger); text-shadow: 0 0 16px #ff5f7955; }
    .small { font-size: 12px; color: var(--muted); }
    .spark { height: 120px; width: 100%; }
    .toast {
      position: fixed;
      right: 14px;
      top: 14px;
      background: rgba(30,30,50,.9);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 10px 12px;
      opacity: 0;
      transform: translateY(-10px);
      transition: all .25s;
    }
    .toast.show { opacity: 1; transform: translateY(0); }
    @media (max-width: 1000px) {
      .hero, .layout { grid-template-columns: 1fr; }
      .row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body class="bg-anim">
  <div class="wrap">
    <div class="hero">
      <div class="card">
        <h1>SocialEngineerArena Live Showcase</h1>
        <div class="subtitle">Red-team / blue-team simulation with animated scoring and real API calls.</div>
        <div class="stats">
          <div class="pill">Mode: <strong id="modePill">Unknown</strong></div>
          <div class="pill">Scenario: <strong id="scenarioPill">-</strong></div>
          <div class="pill">Episode: <strong id="episodePill">0</strong></div>
        </div>
      </div>
      <div class="card">
        <div class="label">Reward Meter</div>
        <div id="rewardValue" class="reward mid">0.0000</div>
        <canvas id="spark" class="spark"></canvas>
        <div class="small">Animated reward trend from recent actions.</div>
      </div>
    </div>

    <div class="layout">
      <div class="card">
        <div class="label">Incoming Message</div>
        <div id="incomingMessage" class="box">Click "Start New Scenario".</div>
        <div class="label" style="margin-top:10px;">Context</div>
        <div id="contextBox" class="box"></div>
      </div>

      <div class="card">
        <div class="row">
          <div>
            <div class="label">Verdict</div>
            <select id="verdict">
              <option value="unknown">unknown</option>
              <option value="safe">safe</option>
              <option value="phishing">phishing</option>
              <option value="pretexting">pretexting</option>
            </select>
          </div>
          <div>
            <div class="label">Safety Boundary</div>
            <input id="safetyBoundary" value="Defensive classification only." />
          </div>
        </div>
        <div class="label">Explanation</div>
        <textarea id="explanation" rows="4" placeholder="Explain concrete evidence and subtle cues..."></textarea>
        <div class="label">Cues Found (comma separated)</div>
        <input id="cuesFound" placeholder="lookalike sender, urgency, bypasses process" />
        <div class="label">Response</div>
        <textarea id="response" rows="3" placeholder="Optional defender response or attacker simulation text"></textarea>
        <div class="btns">
          <button id="resetBtn" class="secondary">Start New Scenario</button>
          <button id="stepBtn" class="primary">Submit Action</button>
          <button id="autoBtn" class="ghost">Autofill Defender Demo</button>
        </div>
        <div id="resultBox" class="box" style="margin-top:10px;"></div>
      </div>
    </div>
  </div>

  <div id="toast" class="toast"></div>

  <script>
    const rewardHistory = [];
    let observation = null;
    let episodeCount = 0;

    const $ = (id) => document.getElementById(id);
    const verdictEl = $("verdict");
    const explanationEl = $("explanation");
    const cuesFoundEl = $("cuesFound");
    const responseEl = $("response");
    const safetyBoundaryEl = $("safetyBoundary");
    const rewardValueEl = $("rewardValue");
    const resultBoxEl = $("resultBox");

    function toast(msg) {
      const t = $("toast");
      t.textContent = msg;
      t.classList.add("show");
      setTimeout(() => t.classList.remove("show"), 1700);
    }

    function setBusy(v) {
      $("resetBtn").disabled = v;
      $("stepBtn").disabled = v;
      $("autoBtn").disabled = v;
    }

    function setReward(v) {
      rewardValueEl.textContent = Number(v).toFixed(4);
      rewardValueEl.className = "reward " + (v >= 0.7 ? "good" : v >= 0.35 ? "mid" : "bad");
      rewardHistory.push(v);
      if (rewardHistory.length > 35) rewardHistory.shift();
      drawSparkline();
    }

    function drawSparkline() {
      const canvas = $("spark");
      const ctx = canvas.getContext("2d");
      canvas.width = canvas.clientWidth * (window.devicePixelRatio || 1);
      canvas.height = canvas.clientHeight * (window.devicePixelRatio || 1);
      ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);
      const w = canvas.clientWidth, h = canvas.clientHeight;
      ctx.clearRect(0, 0, w, h);
      if (!rewardHistory.length) return;
      const min = 0, max = 1;
      const step = w / Math.max(1, rewardHistory.length - 1);
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#7c5cff";
      ctx.beginPath();
      rewardHistory.forEach((r, i) => {
        const x = i * step;
        const y = h - ((r - min) / (max - min + 1e-9)) * h;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }

    async function resetScenario() {
      setBusy(true);
      try {
        const res = await fetch("/api/reset", { method: "POST" });
        if (!res.ok) throw new Error("Reset failed");
        observation = await res.json();
        episodeCount += 1;
        $("modePill").textContent = observation.role;
        $("scenarioPill").textContent = observation.scenario_id;
        $("episodePill").textContent = String(episodeCount);
        $("incomingMessage").textContent = observation.incoming_message;
        $("contextBox").textContent = [
          "Channel: " + observation.channel,
          "Persona: " + observation.persona,
          "Organization: " + observation.organization,
          "Task: " + observation.task,
          "",
          "Thread Context:",
          observation.thread_context,
        ].join("\\n");
        resultBoxEl.textContent = "Scenario loaded. Fill your action and submit.";
        toast("Scenario loaded");
      } catch (e) {
        resultBoxEl.textContent = "Error: " + e.message;
      } finally {
        setBusy(false);
      }
    }

    function autofillDemo() {
      verdictEl.value = "pretexting";
      explanationEl.value = "The sender address is lookalike, urgency is used, sensitive account change is requested, and normal approval process is bypassed.";
      cuesFoundEl.value = "lookalike sender, urgency, payroll change, bypasses process, secrecy";
      responseEl.value = "Please confirm this request out-of-band through the official HR verification flow.";
      safetyBoundaryEl.value = "Defensive analysis only.";
      toast("Autofill ready");
    }

    async function submitAction() {
      if (!observation) {
        toast("Start a scenario first");
        return;
      }
      setBusy(true);
      try {
        const payload = {
          verdict: verdictEl.value,
          explanation: explanationEl.value,
          cues_found: cuesFoundEl.value.split(",").map(x => x.trim()).filter(Boolean),
          response: responseEl.value,
          safety_boundary: safetyBoundaryEl.value,
        };
        const res = await fetch("/api/step", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (!res.ok) throw new Error("Step failed");
        const out = await res.json();
        setReward(out.reward ?? 0);
        const breakdown = out?.observation?.reward_breakdown || {};
        resultBoxEl.textContent = "Done: " + out.done + "\\nReward: " + (out.reward ?? 0).toFixed(4) + "\\n\\nBreakdown:\\n" + JSON.stringify(breakdown, null, 2);
        toast("Action scored");
      } catch (e) {
        resultBoxEl.textContent = "Error: " + e.message;
      } finally {
        setBusy(false);
      }
    }

    $("resetBtn").addEventListener("click", resetScenario);
    $("stepBtn").addEventListener("click", submitAction);
    $("autoBtn").addEventListener("click", autofillDemo);
    window.addEventListener("resize", drawSparkline);

    resetScenario();
  </script>
</body>
</html>"""


if os.getenv("ENABLE_WEB_INTERFACE", "true").lower() == "true" and create_web_interface_app:
    # openenv-core expects an environment class/factory, not an instantiated object.
    app = create_web_interface_app(SocialEngineerArenaEnvironment, ArenaAction, ArenaObservation)
    attach_showcase_routes(app)
else:
    app = FastAPI(title="SocialEngineerArena")
    attach_showcase_routes(app)


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
