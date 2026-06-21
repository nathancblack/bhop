"""Export an agent run as a self-contained, embeddable HTML player.

Runs a trained model (or scripted bhop) through the environment, captures the
per-tick trajectory, and writes a single .html file with the data inlined and a
vanilla-JS canvas animation. No dependencies, no server -- open it in any
browser or drop it into a personal site with an <iframe>.

Usage:
    # Best trained model:
    python scripts/export_run.py --model-path models/bhop_10m_continuous \
        --output site/bhop_run.html

    # Scripted bhop (no model needed -- shows the exploit itself):
    python scripts/export_run.py --scripted --output site/bhop_scripted.html

    # A different env (e.g. the collision corridor):
    python scripts/export_run.py --model-path models/bhop_corridor_2m \
        --env-id bhop/BhopCorridor-v0 --output site/bhop_corridor.html
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import gymnasium as gym

import bhop  # noqa: F401 -- triggers env registration
from bhop.physics import Q3Physics

# 320 ups is Q3's nominal ground speed cap (sv_maxspeed). Bunnyhopping is
# interesting precisely because it exceeds this -- the player renders it as a
# reference line in the speed gauge.
SPEED_CAP = 320.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an agent run as an HTML player")
    parser.add_argument("--output", required=True, help="Output .html path")
    parser.add_argument("--model-path", default=None, help="Trained SB3 model (omit with --scripted)")
    parser.add_argument("--scripted", action="store_true", help="Use scripted bhop instead of a model")
    parser.add_argument("--env-id", default="bhop/BhopFlat-v0", help="Gymnasium env id")
    parser.add_argument("--ticks", type=int, default=1000, help="Ticks to simulate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--title", default=None, help="Title shown in the player")
    parser.add_argument(
        "--deterministic", action="store_true",
        help="Use the deterministic policy (default: stochastic, which bhops harder)",
    )
    return parser.parse_args()


def _record(phys: Q3Physics, rec: dict, info_speed: float) -> None:
    rec["x"].append(round(float(phys.position[0]), 2))
    rec["y"].append(round(float(phys.position[1]), 2))
    rec["speed"].append(round(float(info_speed), 1))
    rec["vz"].append(round(float(phys.velocity[2]), 1))
    rec["on_ground"].append(bool(phys.on_ground))
    rec["yaw"].append(round(float(np.degrees(phys.yaw)), 1))


def run_scripted(n_ticks: int, geometry=None) -> dict:
    """Hand-written bunnyhop, matching the canonical physics test.

    A bhop is a release-jump-airstrafe cycle: jump must be released for one
    tick before it can fire again, then the player strafes right while slowly
    rotating yaw (0.4 deg/tick) for the whole airborne arc. Holding jump every
    tick would only ever produce a single hop.
    """
    phys = Q3Physics(geometry=geometry)
    rec = _empty_rec()
    yaw_rate = np.radians(0.4)

    # Build up ground speed to the 320 cap first (forward, no strafe).
    for _ in range(min(150, n_ticks)):
        phys.tick(forward_move=127, right_move=0, jump=False, yaw_delta=0)
        _record(phys, rec, phys.horizontal_speed)

    # Then bhop cycles until we run out of ticks.
    while len(rec["x"]) < n_ticks:
        # Release jump for one tick (on ground), realigning aim to velocity.
        phys.yaw = float(np.arctan2(phys.velocity[1], phys.velocity[0]))
        phys.tick(forward_move=0, right_move=0, jump=False, yaw_delta=0)
        _record(phys, rec, phys.horizontal_speed)
        # Jump, then air-strafe until we land again.
        phys.tick(forward_move=0, right_move=127, jump=True, yaw_delta=yaw_rate)
        _record(phys, rec, phys.horizontal_speed)
        for _ in range(200):
            if phys.on_ground or len(rec["x"]) >= n_ticks:
                break
            phys.tick(forward_move=0, right_move=127, jump=False, yaw_delta=yaw_rate)
            _record(phys, rec, phys.horizontal_speed)

    # Trim any overshoot from the inner loop.
    for k in rec:
        del rec[k][n_ticks:]
    return rec


def run_model(model_path: str, env_id: str, n_ticks: int, seed: int,
              deterministic: bool) -> dict:
    from stable_baselines3 import PPO

    env = gym.make(env_id, max_episode_steps=n_ticks)
    env.reset(seed=seed)
    model = PPO.load(model_path, device="cpu")
    phys = env.unwrapped._physics

    rec = _empty_rec()
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        _record(phys, rec, info["speed"])
    env.close()
    return rec


def _empty_rec() -> dict:
    return {"x": [], "y": [], "speed": [], "vz": [], "on_ground": [], "yaw": []}


def _walls_from_env(env_id: str) -> list[list[float]]:
    """Return wall AABBs [x0,y0,x1,y1] (top-down) for the env's geometry, if any."""
    if env_id == "bhop/BhopFlat-v0":
        return []
    env = gym.make(env_id)
    geo = getattr(env.unwrapped._physics, "geometry", None)
    env.close()
    if geo is None:
        return []
    walls = []
    for b in geo.brushes:
        # Skip floor brushes (those whose top is at/below z=0); keep walls.
        if float(b.maxs[2]) <= 0:
            continue
        walls.append([float(b.mins[0]), float(b.mins[1]),
                      float(b.maxs[0]), float(b.maxs[1])])
    return walls


def build_html(rec: dict, title: str, walls: list[list[float]]) -> str:
    speeds = rec["speed"]
    payload = {
        "title": title,
        "speedCap": SPEED_CAP,
        "frametimeMs": int(Q3Physics.FRAMETIME * 1000),
        "stats": {
            "ticks": len(speeds),
            "mean": round(float(np.mean(speeds)), 1),
            "max": round(float(np.max(speeds)), 1),
            "final": round(float(speeds[-1]), 1),
            "jumps": _count_jumps(rec["on_ground"]),
        },
        "walls": walls,
        "rec": rec,
    }
    data_json = json.dumps(payload, separators=(",", ":"))
    return _HTML_TEMPLATE.replace("/*__DATA__*/", data_json)


def _count_jumps(on_ground: list[bool]) -> int:
    return sum(1 for a, b in zip(on_ground[:-1], on_ground[1:]) if a and not b)


# ---------------------------------------------------------------------------
# Self-contained player. Data is injected at /*__DATA__*/.
# Vanilla JS + canvas -- no external requests, no build step.
# ---------------------------------------------------------------------------
_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>bhop run</title>
<style>
  :root { --bg:#0d1117; --panel:#161b22; --line:#30363d; --fg:#e6edf3;
          --accent:#58a6ff; --hot:#ff7b50; --cap:#f0c674; --ground:#3fb950; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }
  .wrap { max-width:880px; margin:0 auto; padding:18px; }
  h1 { font-size:16px; font-weight:600; margin:0 0 2px; }
  .sub { color:#8b949e; font-size:12px; margin-bottom:14px; }
  .stage { position:relative; width:100%; aspect-ratio:16/10; background:var(--panel);
           border:1px solid var(--line); border-radius:10px; overflow:hidden; }
  canvas { position:absolute; inset:0; width:100%; height:100%; }
  .hud { position:absolute; top:12px; left:12px; right:12px; display:flex;
         gap:10px; flex-wrap:wrap; pointer-events:none; }
  .chip { background:rgba(13,17,23,.72); border:1px solid var(--line);
          border-radius:8px; padding:6px 10px; font-variant-numeric:tabular-nums; }
  .chip b { font-size:18px; }
  .chip.spd b { color:var(--accent); }
  .chip.spd.over b { color:var(--hot); }
  .gaugewrap { position:absolute; left:12px; right:12px; bottom:12px;
               pointer-events:none; }
  .gauge { height:10px; background:#0d1117; border:1px solid var(--line);
           border-radius:6px; overflow:hidden; position:relative; }
  .gaugefill { height:100%; width:0%; background:linear-gradient(90deg,var(--accent),var(--hot)); }
  .capmark { position:absolute; top:-3px; bottom:-3px; width:2px; background:var(--cap); }
  .caplabel { position:absolute; transform:translateX(-50%); top:-18px;
              font-size:10px; color:var(--cap); white-space:nowrap; }
  .controls { display:flex; align-items:center; gap:12px; margin-top:12px; }
  button { background:var(--accent); color:#04101f; border:0; border-radius:8px;
           padding:8px 16px; font-weight:600; cursor:pointer; font-size:14px; }
  button:active { transform:translateY(1px); }
  input[type=range] { flex:1; accent-color:var(--accent); }
  .legend { color:#8b949e; font-size:11px; margin-top:10px; display:flex;
            gap:16px; flex-wrap:wrap; }
  .legend span::before { content:"\2014"; margin-right:5px; font-weight:700; }
  .legend .air::before { color:var(--hot); }
  .legend .gnd::before { color:var(--ground); }
  .legend .cap::before { color:var(--cap); }
</style>
</head>
<body>
<div class="wrap">
  <h1 id="title">bhop run</h1>
  <div class="sub" id="subtitle"></div>
  <div class="stage">
    <canvas id="cv"></canvas>
    <div class="hud">
      <div class="chip spd" id="spdchip">speed <b id="spd">0</b> ups</div>
      <div class="chip">jumps <b id="jmp">0</b></div>
      <div class="chip">tick <b id="tk">0</b></div>
    </div>
    <div class="gaugewrap">
      <div class="gauge">
        <div class="gaugefill" id="gfill"></div>
        <div class="capmark" id="capmark"></div>
        <div class="caplabel" id="caplabel">320 cap</div>
      </div>
    </div>
  </div>
  <div class="controls">
    <button id="play">&#9654; Play</button>
    <input type="range" id="scrub" min="0" max="100" value="0">
  </div>
  <div class="legend">
    <span class="air">airborne (strafing for speed)</span>
    <span class="gnd">on ground</span>
    <span class="cap">320 ups cap</span>
  </div>
</div>
<script>
const DATA = /*__DATA__*/;
(function(){
  const rec = DATA.rec, n = rec.x.length, cap = DATA.speedCap;
  const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
  const elSpd=document.getElementById('spd'), elJmp=document.getElementById('jmp'),
        elTk=document.getElementById('tk'), elFill=document.getElementById('gfill'),
        elSpdChip=document.getElementById('spdchip'), scrub=document.getElementById('scrub'),
        playBtn=document.getElementById('play');

  document.getElementById('title').textContent = DATA.title;
  const s = DATA.stats;
  document.getElementById('subtitle').textContent =
    `${s.ticks} ticks (${(s.ticks*DATA.frametimeMs/1000).toFixed(1)}s) · `+
    `mean ${s.mean} · max ${s.max} · final ${s.final} ups · ${s.jumps} jumps`;

  // World bounds (with padding) -> fit into canvas.
  let minX=Math.min(...rec.x), maxX=Math.max(...rec.x),
      minY=Math.min(...rec.y), maxY=Math.max(...rec.y);
  for(const w of DATA.walls){ minX=Math.min(minX,w[0]); minY=Math.min(minY,w[1]);
    maxX=Math.max(maxX,w[2]); maxY=Math.max(maxY,w[3]); }
  const padW=(maxX-minX||1)*0.06, padH=(maxY-minY||1)*0.06;
  minX-=padW; maxX+=padW; minY-=padH; maxY+=padH;

  let W=0,H=0,scale=1,offX=0,offY=0;
  function resize(){
    const r=cv.getBoundingClientRect(), dpr=window.devicePixelRatio||1;
    cv.width=r.width*dpr; cv.height=r.height*dpr; ctx.setTransform(dpr,0,0,dpr,0,0);
    W=r.width; H=r.height;
    scale=Math.min(W/(maxX-minX), H/(maxY-minY));
    // center the world in the canvas
    offX=(W-(maxX-minX)*scale)/2; offY=(H-(maxY-minY)*scale)/2;
    draw(frame);
  }
  // World X is "forward", Y is "left". Map X->screen X, Y->screen Y (flip Y up).
  function sx(x){ return offX+(x-minX)*scale; }
  function sy(y){ return H-(offY+(y-minY)*scale); }

  let frame=0, playing=false, last=0;

  function draw(f){
    ctx.clearRect(0,0,W,H);
    // walls
    ctx.fillStyle='rgba(139,148,158,.18)'; ctx.strokeStyle='rgba(139,148,158,.5)';
    for(const w of DATA.walls){
      const x=sx(w[0]), y2=sy(w[3]), ww=(w[2]-w[0])*scale, hh=(w[3]-w[1])*scale;
      ctx.fillRect(x,y2,ww,hh); ctx.strokeRect(x,y2,ww,hh);
    }
    // path so far, colored by ground/air
    ctx.lineWidth=2; ctx.lineJoin='round';
    for(let i=1;i<=f;i++){
      ctx.beginPath();
      ctx.moveTo(sx(rec.x[i-1]),sy(rec.y[i-1]));
      ctx.lineTo(sx(rec.x[i]),sy(rec.y[i]));
      ctx.strokeStyle = rec.on_ground[i] ? 'rgba(63,185,80,.85)' : 'rgba(255,123,80,.9)';
      ctx.stroke();
    }
    // faint preview of the whole remaining path
    ctx.strokeStyle='rgba(139,148,158,.18)'; ctx.lineWidth=1; ctx.beginPath();
    ctx.moveTo(sx(rec.x[f]),sy(rec.y[f]));
    for(let i=f+1;i<n;i++) ctx.lineTo(sx(rec.x[i]),sy(rec.y[i]));
    ctx.stroke();
    // agent
    const ax=sx(rec.x[f]), ay=sy(rec.y[f]);
    ctx.beginPath(); ctx.arc(ax,ay,6,0,Math.PI*2);
    ctx.fillStyle = rec.on_ground[f] ? '#3fb950' : '#ff7b50'; ctx.fill();
    ctx.strokeStyle='#fff'; ctx.lineWidth=1.5; ctx.stroke();
    // facing tick
    const yr=rec.yaw[f]*Math.PI/180;
    ctx.beginPath(); ctx.moveTo(ax,ay);
    ctx.lineTo(ax+Math.cos(yr)*14, ay-Math.sin(yr)*14);
    ctx.strokeStyle='#fff'; ctx.lineWidth=2; ctx.stroke();

    // HUD
    const sp=rec.speed[f];
    elSpd.textContent=Math.round(sp); elTk.textContent=f;
    elJmp.textContent=jumpsUpTo(f);
    const over = sp>cap; elSpdChip.classList.toggle('over',over);
    elFill.style.width=Math.min(100, sp/(cap*3)*100)+'%';
    scrub.value=(f/(n-1)*100);
  }

  // precompute cumulative jump counts
  const cumJumps=new Int32Array(n);
  for(let i=1;i<n;i++) cumJumps[i]=cumJumps[i-1]+((rec.on_ground[i-1]&&!rec.on_ground[i])?1:0);
  function jumpsUpTo(f){ return cumJumps[f]; }

  // place the cap marker on the gauge (gauge spans 0..3*cap)
  document.getElementById('capmark').style.left=(cap/(cap*3)*100)+'%';
  document.getElementById('caplabel').style.left=(cap/(cap*3)*100)+'%';

  function tick(t){
    if(!playing) return;
    if(!last) last=t;
    const dt=t-last;
    if(dt>=DATA.frametimeMs){ last=t; frame=(frame+1)%n; draw(frame); }
    requestAnimationFrame(tick);
  }
  playBtn.onclick=()=>{
    playing=!playing;
    playBtn.innerHTML = playing ? '&#10073;&#10073; Pause' : '&#9654; Play';
    if(playing){ last=0; requestAnimationFrame(tick); }
  };
  scrub.oninput=()=>{ playing=false; playBtn.innerHTML='&#9654; Play';
    frame=Math.round(scrub.value/100*(n-1)); draw(frame); };

  window.addEventListener('resize',resize);
  resize();
})();
</script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()

    geometry = None
    if args.env_id != "bhop/BhopFlat-v0":
        env = gym.make(args.env_id)
        geometry = getattr(env.unwrapped._physics, "geometry", None)
        env.close()

    if args.scripted:
        default_title = "Scripted bunnyhop (hand-written inputs)"
        rec = run_scripted(args.ticks, geometry=geometry)
    else:
        if not args.model_path:
            raise SystemExit("Provide --model-path or use --scripted")
        default_title = f"Trained PPO agent — {os.path.basename(args.model_path)}"
        rec = run_model(args.model_path, args.env_id, args.ticks, args.seed,
                        args.deterministic)

    title = args.title or default_title
    walls = _walls_from_env(args.env_id)
    html = build_html(rec, title, walls)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(html)

    sp = np.array(rec["speed"])
    print(f"Wrote {args.output}")
    print(f"  {len(sp)} ticks · mean {sp.mean():.1f} · max {sp.max():.1f} "
          f"· final {sp[-1]:.1f} ups · {_count_jumps(rec['on_ground'])} jumps")


if __name__ == "__main__":
    main()
