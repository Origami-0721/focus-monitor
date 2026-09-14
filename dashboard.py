"""实时专注面板 —— 本地网页，每 3 秒自动刷新。

    uv run dashboard.py            # 起服务并打开浏览器
    uv run focus.py --dashboard    # 同上（或从托盘菜单"实时面板"进）

只监听 127.0.0.1，不对外网开放。页面每 3 秒抓一次 /live，只换动态区域，
不整页重载，所以不会闪、不会丢滚动位置。

页面上的「数据 N 秒前」是关键：记录程序要是崩了或没启动，数字会一直涨，
一眼就能看出问题 —— 之前托盘的绿点悄悄消失，就是因为没有任何地方能告诉你
进程已经死了。
"""

from __future__ import annotations

import html
import http.server
import re
import threading
import time
import urllib.parse
import webbrowser
from collections import defaultdict

import focus
from focus import STATES
from report import (FLOW_MIN, _dur, _hm, _mdhm, _sessions, _streaks, _timed,
                    load)

DEFAULT_PORT = 8787
LIVE_WINDOW = 30 * 3600   # 实时面板只看最近 30 小时，够覆盖"今天"且不必全表扫
STALE_AFTER = 90.0        # 超过这么久没新样本，就认为记录程序挂了

_server: http.server.ThreadingHTTPServer | None = None


# ─────────────────── 动态片段 ───────────────────

def _today_bounds(now: float) -> tuple[float, float]:
    lt = time.localtime(now)
    midnight = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))
    return midnight, midnight + 86400


_NAV = ('<div class="nav"><a href="/">实时面板</a>'
        '<a href="/rate">自述评分</a>'
        '<a href="/settings">设置</a><a href="/report">完整报告</a></div>')


def _rate_page(msg: str = "") -> str:
    """自述评分页。

    刻意不显示任何实测数据 —— 看到「这段测出 85%」就会被锚定，
    那收集到的就不是独立评价，没法用来验证测量准不准。
    """
    import ratings

    now = time.time()
    items = _timed(load(since=now - 2 * 86400))
    todo = ratings.pending(items, now)
    done = sorted(ratings.ratings_map().items(), reverse=True)[:12]

    banner = f'<div class="alert good">{html.escape(msg)}</div>' if msg else ""
    hist = "".join(
        f'<tr><td>{_mdhm(bs)} – {_hm(bs + ratings.BLOCK)}</td>'
        f'<td class="num">{r["score"]} / 5</td>'
        f'<td>{html.escape(r["note"]) or "<span class=\"muted\">—</span>"}</td></tr>'
        for bs, r in done) or \
        '<tr><td colspan="3" class="muted">还没有评分记录</td></tr>'

    if not todo:
        todo_html = (
            '<div class="alert good">当前没有待评分的时段。'
            '完整记录满 30 分钟、且其中有效数据 ≥10 分钟，这里就会出现新的评分项。</div>')
    else:
        rows = "".join(
            f'<form method="post" action="/rate" class="rrow">'
            f'<input type="hidden" name="block_start" value="{p["start"]:.0f}">'
            f'<span class="rt">{_hm(p["start"])} – {_hm(p["end"])}</span>'
            f'<span class="rb">'
            + "".join(f'<button name="score" value="{s}">{s}</button>'
                      for s in (1, 2, 3, 4, 5))
            + '</span></form>'
            for p in todo[:14])
        todo_html = (
            f'<p class="sub">共 {len(todo)} 个时段待评分，按最新的在前。'
            f'凭记忆选一个数就行，别去看报告里的数字。</p>'
            f'<div class="rlist">{rows}</div>')

    return f"""<div class="wrap">{_NAV}{banner}
<h1>自述评分</h1>
<p class="sub"><b>凭感觉打分，不要参考任何实测数据。</b>
这里刻意不显示测量结果 —— 看到了就会被锚定，那收集到的评价就没法用来验证工具准不准了。</p>

<h2>待评分</h2>
{todo_html}

<h2>评分记录</h2>
<table class="hist"><thead><tr><th>时段</th><th>自评</th><th>备注</th></tr></thead>
<tbody>{hist}</tbody></table>
<p class="note">1 = 完全没在状态，3 = 一般，5 = 非常投入、进入心流。
评分可在报告页和实测投入率做相关性对照 —— 那才是「数据和你感受对不对得上」的答案。</p>
</div>"""


def rate_color(pct: float) -> str:
    """投入率配色：高绿、中蓝、低橙。只写一处，免得每张卡各配一套。"""
    return "#22c55e" if pct >= 70 else "#38bdf8" if pct >= 45 else "#f59e0b"


def _live_html() -> str:
    focus.maybe_reload_config()      # 设置页改完，面板下一个 3 秒周期就反映出来
    ENGAGED = focus.ENGAGED
    rows = load(since=time.time() - LIVE_WINDOW)
    if not rows:
        return (f'<div class="wrap">{_NAV}<h1>实时专注面板</h1>'
                '<p class="muted">还没有任何数据。先跑 '
                '<code>uv run focus.py</code>。</p></div>')

    items = _timed(rows)
    last = items[-1]
    last_ts = last[0] + last[1]
    lag = max(0.0, time.time() - last_ts)
    label, color = STATES.get(last[2], ("未知", "#64748b"))

    # 记录程序是否还活着 —— 这是这个面板最重要的一个数字
    # 校准状态：复用 focus.ear_threshold 的同一套逻辑，只是数据源换成库里的 ear 列
    ears = [x[7] for x in items if x[9]][-600:]
    if len(ears) >= focus.EAR_MIN_SAMPLES:
        cal = (f'睁眼基线 {focus.ear_threshold(ears) / focus.EAR_RATIO:.3f}'
               f' · 已自适应')
    else:
        cal = f'校准中 {len(ears)}/{focus.EAR_MIN_SAMPLES}'

    if lag > STALE_AFTER:
        health = (f'<span class="warn">数据已停滞 {_dur(lag)} —— '
                  f'记录程序似乎没在运行</span>')
    else:
        health = (f'<span class="ok">记录中 · {lag:.0f} 秒前更新</span>'
                  f' · <span class="muted">{cal}</span>')

    # 当前这次连续投入（往回数）
    run = 0.0
    i = len(items) - 1
    while i >= 0 and items[i][2] in ENGAGED:
        run += items[i][1]
        i -= 1
    if run <= 0:
        flow = ('<div class="big muted">—</div>'
                '<div class="sub">当前没有在投入</div>')
    elif run >= FLOW_MIN:
        flow = (f'<div class="big" style="color:#22c55e">{_dur(run)}</div>'
                f'<div class="pbar"><i style="width:100%;background:#22c55e"></i></div>'
                f'<div class="sub"><b style="color:#22c55e">已进入心流</b>'
                f'（阈值 {_dur(FLOW_MIN)}）</div>')
    else:
        pct = run / FLOW_MIN * 100
        # 越接近阈值越亮，让人一眼看出"还差多少"
        c = "#22c55e" if pct >= 70 else "#38bdf8" if pct >= 35 else "#64748b"
        flow = (f'<div class="big" style="color:{c}">{_dur(run)}</div>'
                f'<div class="pbar"><i style="width:{pct:.0f}%;background:{c}"></i></div>'
                f'<div class="sub">还差 <b>{_dur(FLOW_MIN - run)}</b> 进入心流</div>')

    # 当前会话
    sess = _sessions(items)
    if sess:
        s = sess[-1]
        s_start = s[0][0]
        s_act = sum(x[1] for x in s if x[2] != "away")
        s_eng = sum(x[1] for x in s if x[2] in ENGAGED)
        s_rate = s_eng / s_act * 100 if s_act else 0.0
        sess_html = (f'<div class="big">{_dur(s_act)}</div>'
                     f'<div class="sub">{_hm(s_start)} 坐下 · 投入 {_dur(s_eng)}'
                     f' · {s_rate:.0f}%</div>')
    else:
        sess_html = '<div class="big muted">—</div><div class="sub">还没形成会话</div>'

    # 今天
    day0, day1 = _today_bounds(time.time())
    today = [x for x in items if day0 <= x[0] < day1]
    t_act = sum(x[1] for x in today if x[2] != "away")
    t_eng = sum(x[1] for x in today if x[2] in ENGAGED)
    t_streak = len(_streaks(today))
    t_rate = t_eng / t_act * 100 if t_act else 0.0

    # 最近 60 分钟色带：按分钟聚合。
    # 直接逐条渲染的话一小时是 3600 个色块 —— 糊成一团，DOM 还重。
    buckets60: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    cutoff = time.time() - 3600
    for x in items:
        if x[0] >= cutoff:
            buckets60[int(x[0] // 60)][x[2]] += x[1]
    cells60 = []
    for minute, group in sorted(buckets60.items()):
        dom = max(group, key=group.get)
        cells60.append(
            f'<span class="seg" '
            f'style="background:{STATES.get(dom, ("", "#334155"))[1]}" '
            f'title="{_hm(minute * 60)} {STATES.get(dom, ("?",))[0]}"></span>')
    tape = "".join(cells60) or '<span class="muted">最近一小时没有记录</span>'

    # 今天按小时
    hour_act: dict[int, float] = defaultdict(float)
    hour_eng: dict[int, float] = defaultdict(float)
    for x in today:
        h = time.localtime(x[0]).tm_hour
        if x[2] != "away":
            hour_act[h] += x[1]
        if x[2] in ENGAGED:
            hour_eng[h] += x[1]
    peak = max(hour_act.values()) if hour_act else 1.0
    bars = "".join(
        f'<div class="hbar"><div class="hbg">'
        f'<i style="height:{max(3, round(hour_act[h] / peak * 70))}px"></i>'
        f'<b style="height:{max(0, round(hour_eng[h] / peak * 70))}px"></b></div>'
        f'<span>{h:02d}</span></div>'
        for h in sorted(hour_act)) or '<p class="muted">今天还没有记录</p>'

    return f"""<div class="wrap">
{_NAV}
<div class="top">
  <div class="now">
    <span class="dot" style="background:{color}"></span>
    <span class="st">{label}</span>
  </div>
  <div class="health">{health}</div>
</div>

<div class="cards">
  <div class="card"><div class="k">当前连续投入</div>{flow}</div>
  <div class="card"><div class="k">本次会话</div>{sess_html}</div>
  <div class="card"><div class="k">今天累计投入</div>
    <div class="big" style="color:{rate_color(t_rate)}">{_dur(t_eng)}</div>
    <div class="sub">活跃 {_dur(t_act)} ·
      <b style="color:{rate_color(t_rate)}">{t_rate:.0f}%</b> ·
      心流 {t_streak} 段</div>
  </div>
</div>

<h2>最近 60 分钟</h2>
<div class="tape">{tape}</div>

<h2>今天各小时</h2>
<div class="hbars">{bars}</div>
<p class="note">深色柱 = 活跃时长，浅色内柱 = 其中的有效投入。
数据只在本机 127.0.0.1 上，不对外网开放。</p>
</div>"""


# ─────────────────── 设置页 ───────────────────

_FIELDS = [
    ("判定阈值", [
        ("YAW_TOL", "头部左右偏容差（度）", "超过这个角度就算没在看屏幕"),
        ("PITCH_TOL", "抬头/低头容差（度）",
         "笔记本摄像头在屏幕上方，正常注视本就有 15~20°，别设太小"),
        ("DESK_PITCH_MAX", "伏案俯仰上限（度）",
         "低头超过上面那个容差、又不超过这个值 → 判「伏案」；再低判走神"),
        ("EAR_CLOSED", "闭眼 EAR 阈值", "低于此值算闭眼。戴眼镜被误判就调到 0.15"),
        ("EAR_SUSTAIN", "持续闭眼判疲劳（秒）", ""),
        ("AWAY_FACE", "人脸消失判离开（秒）", ""),
        ("AWAY_IDLE", "键鼠空闲判离开（秒）", ""),
        ("TILT_WARN", "坐姿不良肩线倾角（度）", ""),
    ]),
    ("性能", [
        ("FACE_FPS", "人脸检测频率（次/秒）", "越高越吃 CPU"),
        ("POSE_FPS", "姿态检测频率（次/秒）", "动作慢，2 就够"),
        ("PROC_WIDTH", "送进模型的画面宽度", "越小越快"),
        ("AWAY_WRITE_EVERY", "离开时落库间隔（秒）",
         "离开期间降频写库，避免整夜待机撑爆数据库"),
    ]),
]

_TEXTAREAS = [
    ("WORK_APPS", "工作应用（每行一个进程名）",
     "必须是小写带 .exe，例如 code.exe。不在这里的应用一律算「中性」，"
     "既不算专注也不算走神"),
    ("DISTRACT_KEYWORDS", "分心关键词（每行一个）",
     "匹配窗口标题，中文英文都行。游戏 exe 名和中文标题一定要加，"
     "否则会被静默归到中性（绝区零踩过这个坑）"),
    ("STUDY_KEYWORDS", "学习豁免关键词（每行一个）",
     "本来要判分心的标题里含这些词 → 改判工作。用来救「B 站看 C++ 课」这类"),
]


def _parse_form(form: dict) -> tuple[dict, list[str]]:
    """表单 → 配置。返回 (配置, 错误列表)。"""
    cur = focus.current_config()
    cfg: dict = {}
    errors: list[str] = []
    for key, caster in focus._SCALARS.items():
        raw = (form.get(key) or [""])[0].strip()
        try:
            cfg[key] = caster(raw)
        except (TypeError, ValueError):
            cfg[key] = cur[key]
            errors.append(f"{key} 不是有效数字：{raw!r}")
    for key, _, _ in _TEXTAREAS:
        raw = (form.get(key) or [""])[0]
        cfg[key] = [s.strip() for s in re.split(r"[\n,，;；]", raw) if s.strip()]
    cfg["DESKWORK_IS_ENGAGED"] = "DESKWORK_IS_ENGAGED" in form
    if not errors:
        errors.extend(focus.validate_config(cfg))
    return cfg, errors


def _settings_page(cfg: dict | None = None, errors: list[str] | None = None,
                   saved: bool = False, reset: bool = False) -> str:
    cfg = focus.current_config() if cfg is None else cfg
    errors = errors or []
    blocks = []

    for group, fields in _FIELDS:
        rows = ""
        for key, label, hint in fields:
            h = f'<span class="hint">{html.escape(hint)}</span>' if hint else ""
            rows += (f'<label class="fld"><span class="fl">{html.escape(label)}</span>'
                     f'<input type="number" step="any" name="{key}" '
                     f'value="{cfg.get(key, "")}">{h}</label>')
        blocks.append(f'<h2>{html.escape(group)}</h2><div class="fields">{rows}</div>')

    checked = " checked" if cfg.get("DESKWORK_IS_ENGAGED") else ""
    blocks.append(
        '<h2>行为</h2><div class="fields">'
        f'<label class="fld chk"><input type="checkbox" '
        f'name="DESKWORK_IS_ENGAGED"{checked}>'
        '<span class="fl">伏案计入专注时长</span>'
        '<span class="hint">低头看书/写作业算不算投入。前置摄像头分不清'
        '「低头看书」和「低头玩手机」—— 发现伏案时长虚高就取消勾选</span></label></div>')

    for key, label, hint in _TEXTAREAS:
        val = html.escape("\n".join(cfg.get(key, [])))
        blocks.append(f'<h2>{html.escape(label)}</h2>'
                      f'<textarea name="{key}" rows="10" spellcheck="false">{val}'
                      f'</textarea><p class="note">{html.escape(hint)}</p>')

    if errors:
        items = "".join(f"<li>{html.escape(e)}</li>" for e in errors)
        banner = f'<div class="alert bad"><b>没有保存，请修正：</b><ul>{items}</ul></div>'
    elif saved:
        banner = ('<div class="alert good">已保存。监视进程、报告、面板都会在几秒内'
                  '自动生效，不用重启。</div>')
    elif reset:
        banner = '<div class="alert good">已恢复默认设置。</div>'
    else:
        banner = ""

    return f"""<div class="wrap">{_NAV}{banner}
<h1>设置</h1>
<p class="sub">保存后写入 <code>config.json</code>，各进程自动重新加载。</p>
<form method="post" action="/settings">
{"".join(blocks)}
<div class="acts">
  <button type="submit">保存</button>
  <button type="submit" formaction="/settings/reset" class="ghost">恢复默认</button>
</div>
</form>
<p class="note">「恢复默认」会删掉 config.json，回到代码里的初始值。</p>
</div>"""


_PAGE = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>实时专注面板</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; padding:26px 20px 50px; background:#0f172a; color:#e2e8f0;
        font:15px/1.6 "Segoe UI",system-ui,-apple-system,"Microsoft YaHei",sans-serif; }
  .wrap { max-width:900px; margin:0 auto; }
  h1 { font-size:22px; margin:0 0 4px; }
  h2 { font-size:14px; margin:28px 0 12px; color:#94a3b8; font-weight:600;
       letter-spacing:.04em; }
  .top { display:flex; justify-content:space-between; align-items:center;
        flex-wrap:wrap; gap:10px; margin-bottom:22px; }
  .now { display:flex; align-items:center; gap:12px; }
  .dot { width:18px; height:18px; border-radius:50%; }
  .st { font-size:30px; font-weight:700; }
  .health { font-size:13px; color:#64748b; }
  .health a { color:#38bdf8; text-decoration:none; }
  .ok { color:#22c55e; }
  .warn { color:#f59e0b; font-weight:600; }
  .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:14px; }
  .card { background:#1e293b; border:1px solid #334155; border-radius:12px; padding:16px 18px; }
  .card .k { color:#94a3b8; font-size:13px; margin-bottom:6px; }
  .big { font-size:26px; font-weight:700; }
  .sub { font-size:12px; color:#64748b; margin-top:4px; }
  .pbar { background:#0f172a; height:6px; border-radius:3px; overflow:hidden;
          margin:8px 0 4px; }
  .pbar i { display:block; height:100%; border-radius:3px; }
  .tape { display:flex; flex-wrap:wrap; gap:1px; background:#1e293b;
          border-radius:10px; padding:10px; min-height:34px; align-items:center; }
  .seg { width:9px; height:22px; border-radius:2px; }
  .hbars { display:flex; gap:6px; align-items:flex-end; background:#1e293b;
           border-radius:10px; padding:12px; min-height:100px; }
  .hbar { flex:1; display:flex; flex-direction:column; align-items:center; gap:4px; }
  .hbar span { font-size:10px; color:#64748b; }
  .hbg { width:100%; height:70px; display:flex; flex-direction:column;
         justify-content:flex-end; align-items:center; }
  .hbg i { display:block; width:100%; background:#1d4ed8; border-radius:2px 2px 0 0; }
  .hbg b { display:block; width:100%; background:#22c55e; border-radius:0; margin-top:-1px; }
  .muted { color:#64748b; }
  .note { color:#64748b; font-size:12px; margin-top:12px; }
  code { background:#1e293b; padding:1px 5px; border-radius:4px; font-size:13px; }
  .nav { display:flex; gap:18px; margin-bottom:18px; padding-bottom:12px;
         border-bottom:1px solid #1e293b; }
  .nav a { color:#94a3b8; text-decoration:none; font-size:14px; }
  .nav a:hover { color:#e2e8f0; }
  .sub { color:#64748b; font-size:13px; margin:0 0 22px; }
  .fields { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr));
            gap:12px; }
  .fld { display:flex; flex-direction:column; gap:5px; background:#1e293b;
         border:1px solid #334155; border-radius:10px; padding:12px 14px; }
  .fld .fl { font-size:13px; color:#cbd5e1; }
  .fld .hint { font-size:11px; color:#64748b; line-height:1.5; }
  .fld.chk { flex-direction:row; align-items:flex-start; gap:10px; flex-wrap:wrap; }
  .fld.chk .fl { font-weight:600; }
  .fld.chk .hint { flex-basis:100%; }
  input[type=number] { background:#0f172a; border:1px solid #475569; color:#e2e8f0;
        border-radius:6px; padding:7px 10px; font:inherit; font-size:14px;
        width:100%; }
  input[type=number]:focus, textarea:focus { outline:none; border-color:#38bdf8; }
  input[type=checkbox] { width:17px; height:17px; accent-color:#22c55e; margin-top:2px; }
  textarea { width:100%; background:#0f172a; border:1px solid #475569; color:#e2e8f0;
        border-radius:10px; padding:12px 14px; font:13px/1.6 ui-monospace,
        Consolas,"Cascadia Mono",monospace; resize:vertical; }
  .acts { display:flex; gap:12px; margin-top:26px; }
  button { background:#22c55e; color:#052e16; border:none; border-radius:8px;
        padding:10px 24px; font:inherit; font-size:15px; font-weight:600;
        cursor:pointer; }
  button:hover { filter:brightness(1.1); }
  button.ghost { background:transparent; color:#94a3b8; border:1px solid #475569;
        font-weight:400; }
  button.ghost:hover { color:#e2e8f0; border-color:#64748b; }
  .alert { border-radius:10px; padding:14px 18px; margin-bottom:20px; font-size:14px; }
  .alert ul { margin:8px 0 0; padding-left:20px; }
  .alert.good { background:#052e16; border:1px solid #22c55e; color:#86efac; }
  .alert.bad { background:#431407; border:1px solid #f59e0b; color:#fdba74; }
  .rlist { display:flex; flex-direction:column; gap:8px; }
  .rrow { display:flex; align-items:center; gap:16px; background:#1e293b;
          border:1px solid #334155; border-radius:10px; padding:10px 16px; }
  .rt { font-size:15px; color:#cbd5e1; font-variant-numeric:tabular-nums;
        min-width:132px; }
  .rb { display:flex; gap:6px; }
  .rb button { background:#0f172a; color:#94a3b8; border:1px solid #475569;
        border-radius:8px; width:42px; padding:7px 0; font-size:14px;
        font-weight:600; cursor:pointer; }
  .rb button:hover { background:#22c55e; color:#052e16; border-color:#22c55e; }
  /* 不固定列宽的话三列会塌在一起：<th> 浏览器默认居中，加上列宽自动分配，
     表头会挤成一坨（"自评备注" 连成一个词）。 */
  .hist th, .hist td { text-align:left; }
  .hist th:nth-child(1), .hist td:nth-child(1) { width:230px; }
  .hist th:nth-child(2), .hist td:nth-child(2) { width:80px; text-align:right; }
</style></head><body>
<div id="live">__LIVE__</div>
__JS__
</body></html>"""

# 只有实时面板要轮询。设置页套同一个外壳但必须不带它 ——
# 否则 3 秒后表单会被实时面板覆盖掉。
_POLL_JS = """<script>
  let fail = 0;
  async function tick() {
    try {
      const r = await fetch('/live', {cache: 'no-store'});
      if (!r.ok) throw new Error(r.status);
      document.getElementById('live').innerHTML = await r.text();
      fail = 0;
    } catch (e) {
      if (++fail === 2) {
        document.getElementById('live').innerHTML =
          '<div class="wrap"><p class="warn">面板与服务失去连接，' +
          '请确认 dashboard 进程还在运行。</p></div>';
      }
    }
  }
  setInterval(tick, 3000);
</script>"""


def _shell(body: str, poll: bool = False) -> str:
    """把内容装进带样式的完整页面。"""
    return _PAGE.replace("__LIVE__", body).replace("__JS__", _POLL_JS if poll else "")


# ─────────────────── HTTP ───────────────────

class _Handler(http.server.BaseHTTPRequestHandler):
    def _send(self, body: str, code: int = 200) -> None:
        raw = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(raw)

    def _redirect(self, to: str) -> None:
        """303 让浏览器把 POST 换成 GET，刷新页面不会重复提交。"""
        self.send_response(303)
        self.send_header("Location", to)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802  (stdlib 命名)
        raw_path, _, qs = self.path.partition("?")
        query = urllib.parse.parse_qs(qs)
        try:
            if raw_path == "/live":
                self._send(_live_html())
            elif raw_path == "/settings":
                self._send(_shell(_settings_page(saved="saved" in query,
                                                 reset="reset" in query)))
            elif raw_path == "/rate":
                self._send(_shell(_rate_page(
                    "已记录。" if "done" in query else "")))
            elif raw_path == "/report":
                from report import build_html
                self._send(build_html(load()))
            elif raw_path == "/":
                self._send(_shell(_live_html(), poll=True))
            else:
                self._send("<h1>404</h1>", 404)
        except Exception as exc:                       # 单次请求出错不该带崩服务
            self._send(f"<h1>500</h1><pre>{html.escape(str(exc))}</pre>", 500)

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        n = int(self.headers.get("Content-Length") or 0)
        form = urllib.parse.parse_qs(
            self.rfile.read(n).decode("utf-8", "replace"), keep_blank_values=True)
        try:
            if path == "/settings/reset":
                focus.reset_config()
                self._redirect("/settings?reset=1")
                return
            if path == "/rate":
                import ratings
                try:
                    bs = float((form.get("block_start") or [""])[0])
                    sc = int((form.get("score") or [""])[0])
                except ValueError:
                    self._send(_shell(_rate_page("提交的数据不合法，请重新点一次。")), 400)
                    return
                if bs <= 0 or not 1 <= sc <= 5:     # 表单是用户输入，挡一道
                    self._send(_shell(_rate_page("评分必须是 1–5。")), 400)
                    return
                ratings.save(bs, sc)
                self._redirect("/rate?done=1")
                return
            if path != "/settings":
                self._send("<h1>404</h1>", 404)
                return
            cfg, errors = _parse_form(form)
            if errors:                                 # 有问题就把填的内容原样退回
                self._send(_shell(_settings_page(cfg, errors)))
                return
            focus.save_config(cfg)                     # 立刻 apply，监视进程随后也会重载
            self._redirect("/settings?saved=1")
        except Exception as exc:
            self._send(f"<h1>500</h1><pre>{html.escape(str(exc))}</pre>", 500)

    def log_message(self, *args) -> None:
        pass                                           # 别把控制台刷爆


def _make_server(port: int) -> http.server.ThreadingHTTPServer:
    """只绑 127.0.0.1 —— 这些是摄像头推出来的数据，不能对外网开放。"""
    last: Exception | None = None
    for p in range(port, port + 20):
        try:
            return http.server.ThreadingHTTPServer(("127.0.0.1", p), _Handler)
        except OSError as exc:
            last = exc
    raise SystemExit(f"端口 {port}–{port + 19} 都占用了：{last}")


def serve_background(port: int = DEFAULT_PORT, open_browser: bool = True) -> str:
    """后台线程起服务，供托盘调用。重复调用只会再打开一次浏览器。"""
    global _server
    if _server is None:
        _server = _make_server(port)
        threading.Thread(target=_server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{_server.server_port}/"
    if open_browser:
        webbrowser.open(url)
    return url


def base_url() -> str:
    """当前面板的真实地址。

    端口不一定等于 DEFAULT_PORT —— 被占用时会往后顺延，所以任何地方
    都别硬编码 8787。
    """
    port = _server.server_port if _server is not None else DEFAULT_PORT
    return f"http://127.0.0.1:{port}/"


def main() -> None:
    url = serve_background()
    print(f"实时面板: {url}\nCtrl+C 停止")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    main()
