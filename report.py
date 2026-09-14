"""把 focus.db 渲染成一份自包含的 HTML 报告。

不引 matplotlib，图表全用内联 SVG + CSS —— 少一个依赖，报告还是单文件。

分析模型（这是回答"我几点最容易进入心流"的地方）：
  会话   一次"坐下来用电脑"的连续时段。样本空档 >2 分钟（待机/关机）或
         连续离开 >15 分钟（人走了），都判定为会话结束。
  心流   会话内连续投入 ≥15 分钟，允许中间有 ≤60 秒的短暂中断（挠头、喝水
         不该掐断一段专注）。"投入"的构成见 focus.ENGAGED —— 默认是
         focused + neutral + deskwork，即"看着屏幕"或"低头伏案"。
  进入耗时 = 心流片段起点 − 会话起点。这个数是按时段聚合的主角。
"""

from __future__ import annotations

import html
import math
import sqlite3
import time
import webbrowser
from collections import defaultdict
from pathlib import Path

import focus
import ratings
from focus import DB_PATH, STATES

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "report.html"

MAX_GAP = 5.0           # 样本间隔超过这个秒数就不计入任何状态（程序没在跑）
AWAY_SAMPLE_CAP = 40.0  # 离开期间降频到 30 秒一条，容差给到 40 秒
SESSION_GAP = 120.0     # 空档超过 2 分钟 → 新会话（待机 / 关机 / 崩溃）
SESSION_AWAY = 900.0    # 连续离开超过 15 分钟 → 本次会话结束
FLOW_MIN = 900.0        # 连续专注 ≥15 分钟算进入心流
FLOW_GAP = 60.0         # 心流片段内允许 ≤60 秒的短暂中断
MIN_HOUR_DATA = 600.0   # 某个钟点至少累积 10 分钟才参与"黄金时段"排名，否则样本太少


def load(db_path: Path = DB_PATH, since: float | None = None) -> list[tuple]:
    """读样本。since 是时间戳下界 —— 实时面板每次轮询都读，不能全表扫。"""
    sql = ("SELECT ts,state,app,title,yaw,pitch,ear,tilt,scale,present,idle "
           "FROM samples")
    args: tuple = ()
    if since is not None:
        sql += " WHERE ts >= ?"
        args = (since,)
    conn = sqlite3.connect(db_path)
    try:
        # 首次运行（还没建过表）时返回空，让面板显示"还没有任何数据"，
        # 而不是 500 —— 空环境是最常见的新手场景
        if not conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='samples'").fetchone():
            return []
        return conn.execute(sql + " ORDER BY ts", args).fetchall()
    finally:
        conn.close()


def _dur(sec: float) -> str:
    sec = int(sec)
    if sec >= 3600:
        return f"{sec // 3600}小时{sec % 3600 // 60}分"
    if sec >= 60:
        return f"{sec // 60}分{sec % 60}秒"
    return f"{sec}秒"


def _hm(ts: float) -> str:
    return time.strftime("%H:%M", time.localtime(ts))


def _mdhm(ts: float) -> str:
    return time.strftime("%m-%d %H:%M", time.localtime(ts))


_BROWSER_SUFFIX = (" - google chrome", " - microsoft edge",
                   " - mozilla firefox", " - brave", " - opera")


def _label(app: str, title: str) -> str:
    """排行要的是"在干什么"，不是"用哪个浏览器"。

    标题里通常带浏览器后缀，剥掉它，剩下的才是有效信息。
    """
    t = (title or "").strip()
    low = t.lower()
    for suf in _BROWSER_SUFFIX:
        if low.endswith(suf):
            t = t[: -len(suf)].strip()
            break
    return (t or app or "(未知)")[:48]


def _is_looking(present: int | bool, yaw: float, pitch: float) -> bool:
    """这一秒人是不是正看着屏幕。

    用来拆分走神的两种成因：只有"确实在看着屏幕"的走神才归因给前台应用。
    转头或人离开画面时前台开着什么应用，跟走神没有因果关系 —— 不拆的话会得出
    "快速设置是你最大分心源"这种荒谬结论。

    这个还原是精确的：decide() 里"看着屏幕 + 分心应用"必定输出 distracted，
    所以用存下来的 yaw/pitch/present 反查，一个不多一个不少。
    """
    return (bool(present) and abs(yaw) <= focus.YAW_TOL
            and abs(pitch) <= focus.PITCH_TOL)


# ─────────────────── 会话 / 心流片段 ───────────────────

def _timed(rows: list[tuple]) -> list[tuple]:
    """给每条样本补上"它代表多少秒"，供后续按时间加权。

    返回 (ts, dur, state, app, title, yaw, pitch, ear, tilt, present)。
    """
    out = []
    for i, r in enumerate(rows):
        ts = r[0]
        nxt = rows[i + 1][0] if i + 1 < len(rows) else ts + 1.0
        # 离开期间是每 30 秒才落一条，间隔上限必须放宽，否则离开时长会被严重算少
        cap = AWAY_SAMPLE_CAP if r[1] == "away" else MAX_GAP
        dur = max(0.0, min(nxt - ts, cap))
        out.append((ts, dur, r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[9]))
    return out


def _sessions(items: list[tuple]) -> list[list[tuple]]:
    """按"空档"和"长时间离开"把样本切成会话。

    dropping 必须只在"人回来"时复位：之前用另一个标志在 else 分支复位，
    结果离开期间的样本又被塞进了新会话。
    """
    sessions: list[list[tuple]] = []
    cur: list[tuple] = []
    away_run = 0.0
    dropping = False

    for it in items:
        if cur and it[0] - cur[-1][0] > SESSION_GAP:
            sessions.append(cur)          # 待机 / 关机造成的空档
            cur, away_run, dropping = [], 0.0, False

        if it[2] == "away":
            if dropping:
                continue                  # 人还没回来，这段不归任何会话
            away_run += it[1]
            if away_run > SESSION_AWAY:
                while cur and cur[-1][2] == "away":
                    cur.pop()             # 会话在离开的那一刻截断
                if cur:
                    sessions.append(cur)
                cur, dropping = [], True
                continue
            cur.append(it)
        else:
            away_run = 0.0
            dropping = False
            cur.append(it)

    if cur:
        sessions.append(cur)
    return [s for s in sessions if s]


def _streaks(items: list[tuple]) -> list[tuple]:
    """找出心流片段，返回 (起, 止, 有效专注秒数, 片段总秒数)。"""
    ENGAGED = focus.ENGAGED     # 局部绑定：这个集合会被设置页改，不能缓存
    out: list[tuple] = []
    n = len(items)
    i = 0
    while i < n:
        if items[i][2] not in ENGAGED:
            i += 1
            continue
        start = last = i
        gap = 0.0
        j = i + 1
        while j < n:
            # 时间上断开（待机 / 关机 / 程序没跑）必须算断点。
            # 只看状态不行：_timed 把长空档的时长截断到几秒，于是 6 小时的停机
            # 在 gap 累加器里只值 5 秒，两段 3 分钟的专注会被拼成一段"心流"。
            if items[j][0] - items[j - 1][0] > SESSION_GAP:
                break
            if items[j][2] in ENGAGED:
                last, gap = j, 0.0
            else:
                gap += items[j][1]
                if gap > FLOW_GAP:
                    break
            j += 1
        if items[last][0] + items[last][1] - items[start][0] >= FLOW_MIN:
            engaged = sum(items[k][1] for k in range(start, last + 1)
                          if items[k][2] in ENGAGED)
            out.append((items[start][0], items[last][0] + items[last][1],
                        engaged, items[last][0] + items[last][1] - items[start][0]))
        i = last + 1
    return out


# ─────────────────── 渲染小件 ───────────────────

def _donut(parts: list[tuple[str, float, str]]) -> str:
    total = sum(p[1] for p in parts)
    if total <= 0:
        return '<p class="muted">没有数据</p>'
    r, sw = 70.0, 26.0
    circ = 2 * math.pi * r
    arcs, offset = [], 0.0
    for label, sec, color in parts:
        if sec <= 0:
            continue
        length = sec / total * circ
        arcs.append(
            f'<circle cx="100" cy="100" r="{r}" fill="none" stroke="{color}" '
            f'stroke-width="{sw}" stroke-dasharray="{length:.2f} {circ - length:.2f}" '
            f'stroke-dashoffset="{-offset:.2f}" transform="rotate(-90 100 100)">'
            f'<title>{html.escape(label)} {sec / total * 100:.1f}%</title></circle>')
        offset += length
    return (f'<svg viewBox="0 0 200 200" class="donut" role="img" '
            f'aria-label="状态分布">{"".join(arcs)}'
            f'<text x="100" y="94" text-anchor="middle" class="donut-big">'
            f'{_dur(total)}</text>'
            f'<text x="100" y="116" text-anchor="middle" class="donut-sub">'
            f'总计</text></svg>')


def _app_table(mapping: dict[str, float], total: float, empty: str) -> str:
    top = sorted(mapping.items(), key=lambda kv: -kv[1])[:8]
    rows = "".join(
        f'<tr><td>{html.escape(a)}</td><td class="num">{_dur(d)}</td>'
        f'<td class="num">{d / max(total, 1) * 100:.0f}%</td></tr>'
        for a, d in top) or f'<tr><td colspan="3" class="muted">{empty}</td></tr>'
    return ('<table><thead><tr><th>窗口 / 应用</th><th class="num">时长</th>'
            f'<th class="num">占比</th></tr></thead><tbody>{rows}</tbody></table>')


def _rate_cell(pct: float) -> str:
    color = "#22c55e" if pct >= 70 else "#38bdf8" if pct >= 45 else "#f59e0b"
    return (f'<div class="rc"><div class="bar"><i style="width:{pct:.0f}%;'
            f'background:{color}"></i></div><span>{pct:.0f}%</span></div>')


# ─────────────────── 主渲染 ───────────────────

def build_html(rows: list[tuple]) -> str:
    focus.maybe_reload_config()          # 报告用当前设置，不是进程启动时的
    ENGAGED = focus.ENGAGED              # 局部绑定，下面所有引用都取最新值
    TILT_WARN = focus.TILT_WARN
    if not rows:
        return ("<!DOCTYPE html><html lang='zh-CN'><meta charset='utf-8'>"
                "<title>专注度报告</title>"
                "<body style='font:16px system-ui;background:#0f172a;color:#e2e8f0;"
                "padding:60px;text-align:center'><h1>还没有数据</h1>"
                "<p>先跑 <code>uv run focus.py</code> 记录一段时间。</p></body></html>")

    items = _timed(rows)
    sessions = _sessions(items)

    # ── 全局时长 ──
    dur: dict[str, float] = defaultdict(float)
    app_dur: dict[str, float] = defaultdict(float)
    work_dur: dict[str, float] = defaultdict(float)
    gaze_away = 0.0        # 走神里"人没看屏幕"的那部分，不归因给任何应用
    hour_active: dict[int, float] = defaultdict(float)
    hour_engaged: dict[int, float] = defaultdict(float)
    hour_active_days: dict[int, set] = defaultdict(set)   # 每钟点活跃过的日期
    day_active: dict[str, float] = defaultdict(float)     # 每天活跃时长
    day_engaged: dict[str, float] = defaultdict(float)    # 每天投入时长
    hour_sessions: dict[int, int] = defaultdict(int)
    hour_entry: dict[int, list[float]] = defaultdict(list)
    tilt_bad = 0.0
    yaws: list[float] = []
    pitches: list[float] = []
    ears: list[float] = []
    switches = 0
    prev_state = None

    for (ts, dt, state, app, title, yaw, pitch, ear, tilt, present) in items:
        dur[state] += dt
        lt = time.localtime(ts)
        day = time.strftime("%Y-%m-%d", lt)
        if state != "away":
            hour_active[lt.tm_hour] += dt
            hour_active_days[lt.tm_hour].add(day)
            day_active[day] += dt
        if state in ENGAGED:
            hour_engaged[lt.tm_hour] += dt
            day_engaged[day] += dt
        if present:
            yaws.append(yaw)
            pitches.append(pitch)
            ears.append(ear)
        if tilt and tilt > TILT_WARN:
            tilt_bad += dt
        if state == "distracted":
            # 走神拆成两块：看着屏幕的才归因给应用，其余的算"人没看屏幕"
            if _is_looking(present, yaw, pitch):
                app_dur[_label(app, title)] += dt
            else:
                gaze_away += dt
        elif state == "focused":
            work_dur[_label(app, title)] += dt
        if prev_state is not None and state != prev_state:
            switches += 1
        prev_state = state

    total = sum(dur.values())
    active = total - dur["away"]
    focus_rate = dur["focused"] / active * 100 if active > 0 else 0.0
    engaged_rate = sum(dur[s] for s in ENGAGED) / active * 100 if active > 0 else 0.0
    looking_rate = sum(dur[s] for s in ("focused", "neutral")) / active * 100 \
        if active > 0 else 0.0
    order = ["focused", "neutral", "deskwork", "distracted", "drowsy", "away"]
    app_distract = sum(app_dur.values())   # 走神里归因给应用的那部分，做排行分母

    # ── 会话与心流 ──
    sess_rows, all_streaks = [], []
    for s in sessions:
        s_start, s_end = s[0][0], s[-1][0] + s[-1][1]
        s_act = sum(x[1] for x in s if x[2] != "away")
        s_foc = sum(x[1] for x in s if x[2] == "focused")
        if s_act < 60:
            continue                       # 不到 1 分钟的碎片不列入
        st = _streaks(s)
        hour_sessions[time.localtime(s_start).tm_hour] += 1
        entry = (st[0][0] - s_start) if st else None
        if entry is not None:
            hour_entry[time.localtime(s_start).tm_hour].append(entry)
        for k in st:
            all_streaks.append((k[0], k[1], k[2], k[3], s_start))
        sess_rows.append((s_start, s_end, s_act, s_foc, len(st), entry))

    sess_rows.sort(key=lambda r: -r[0])
    all_streaks.sort(key=lambda r: -r[0])

    sess_html = "".join(
        f'<tr><td>{_mdhm(a)}</td><td class="num">{_dur(b - a)}</td>'
        f'<td class="num">{_dur(c)}</td><td class="num">{d / c * 100:.0f}%</td>'
        f'<td class="num">{e}</td>'
        f'<td class="num">{_dur(f) if f is not None else "—"}</td></tr>'
        for a, b, c, d, e, f in sess_rows[:40]) or \
        '<tr><td colspan="6" class="muted">还没有足够长的会话（≥1 分钟）</td></tr>'

    streak_html = "".join(
        f'<tr><td>{_mdhm(a)}</td><td class="num">{_dur(d)}</td>'
        f'<td class="num">{c / d * 100:.0f}%</td>'
        f'<td class="num">{_dur(a - s)}</td></tr>'
        for a, b, c, d, s in all_streaks[:25]) or \
        '<tr><td colspan="4" class="muted">还没有 ≥15 分钟的心流片段</td></tr>'

    # ── 黄金时段 ──
    ranked = [(h, hour_active[h]) for h in hour_active if hour_active[h] >= MIN_HOUR_DATA]
    ranked.sort(key=lambda kv: -hour_engaged[kv[0]] / kv[1])
    skipped = [h for h in hour_active if hour_active[h] < MIN_HOUR_DATA]
    hour_rows = []
    for i, (h, _) in enumerate(ranked[:10]):
        entry = hour_entry.get(h)
        cls = ' class="best"' if i == 0 else ""     # 反斜杠不能出现在 f-string 表达式里
        hour_rows.append(
            f'<tr{cls}>'
            f'<td>{h:02d}:00</td><td class="num">{_dur(hour_active[h])}</td>'
            f'<td>{_rate_cell(hour_engaged[h] / hour_active[h] * 100)}</td>'
            f'<td class="num">{hour_sessions.get(h, 0)}</td>'
            f'<td class="num">{_dur(sum(entry) / len(entry)) if entry else "—"}</td>'
            f'</tr>')
    hour_html = "".join(hour_rows) or \
        (f'<tr><td colspan="5" class="muted">还没有任何钟点累积到 '
         f'{_dur(MIN_HOUR_DATA)} 的数据</td></tr>')

    legend = "".join(
        f'<div class="row"><span class="dot" style="background:{STATES[s][1]}"></span>'
        f'<span class="lbl">{STATES[s][0]}</span>'
        f'<div class="bar"><i style="width:{dur[s] / total * 100:.1f}%;'
        f'background:{STATES[s][1]}"></i></div>'
        f'<span class="val">{_dur(dur[s])} · {dur[s] / total * 100:.0f}%</span></div>'
        for s in order if dur[s] > 0)

    # ── 时间轴 ──
    buckets: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for ts, dt, state, *_ in items:
        buckets[int(ts // 60)][state] += dt
    cells = []
    for minute in sorted(buckets):
        dom = max(buckets[minute], key=buckets[minute].get)  # type: ignore[arg-type]
        cells.append(f'<span class="cell" style="background:{STATES[dom][1]}" '
                     f'title="{_mdhm(minute * 60)} {STATES[dom][0]}"></span>')

    # 每小时归一成"平均每天"：只早上用电脑的人，上午柱不再被几天撑虚高。
    # 原始累计 hour_active 仍留给黄金时段排序，柱状图只看平均强度。
    hour_avg = {h: hour_active[h] / len(hour_active_days[h])
                for h in hour_active}
    peak = max(hour_avg.values()) if hour_avg else 1.0
    hours_bar = "".join(
        f'<div class="hbar"><span class="hv">{_dur(hour_avg[h])}</span>'
        f'<div class="hbg"><i style="height:{max(3, round(hour_avg[h] / peak * 110))}px"'
        f' title="{h:02d} 点 · 平均每天 {_dur(hour_avg[h])}"></i></div>'
        f'<span class="hl">{h:02d}</span></div>'
        for h in range(24) if h in hour_avg)

    # 近 14 个有记录的自然日：每天投入时长柱，悬停看当天专注率
    day_keys = sorted(day_active)[-14:]
    if day_keys:
        day_peak = max(day_engaged.get(d, 0) for d in day_keys) or 1.0
        day_bars = "".join(
            f'<div class="hbar"><span class="hv">{_dur(day_engaged.get(d, 0))}</span>'
            f'<div class="hbg"><i style="height:'
            f'{max(3, round(day_engaged.get(d, 0) / day_peak * 110))}px"'
            f' title="{d} 投入 {_dur(day_engaged.get(d, 0))} · '
            f'专注率 {day_engaged.get(d, 0) / day_active[d] * 100:.0f}%"></i></div>'
            f'<span class="hl">{d[5:]}</span></div>'
            for d in day_keys)
    else:
        day_bars = ""

    # ── 自述对照：自评分 vs 实测投入率 ──
    pairs = ratings.paired(items)
    corr = ratings.correlation(pairs)
    by_score: dict[int, list[float]] = defaultdict(list)
    for _bs, sc, measured in pairs:
        by_score[sc].append(measured)
    self_rows = "".join(
        f'<tr><td>{s} / 5</td><td class="num">{len(v)}</td>'
        f'<td>{_rate_cell(sum(v) / len(v) * 100)}</td></tr>'
        for s, v in sorted(by_score.items()))
    corr_txt = f"r = {corr:.2f}" if corr is not None else "样本不足"
    self_html = (
        f'<p class="sub" style="margin:0 0 14px">配对样本 {len(pairs)} 条 · '
        f'相关系数 <b>{corr_txt}</b> —— {html.escape(ratings.verdict(corr, len(pairs)))}</p>'
        + (f'<table><thead><tr><th>你的自评</th><th class="num">样本数</th>'
           f'<th>平均实测投入率</th></tr></thead><tbody>{self_rows}</tbody></table>'
           if by_score else
           '<p class="muted">还没有可配对的评分。面板 →「自述评分」给过去的时段打分后，'
           '这里会出现对照表。</p>')
        + '<p class="note">实测投入率 = (专注 + 中性 + 伏案) ÷ 有效时长。'
          '如果相关性强，说明这套判定和你的真实感受是一回事，黄金时段那张表才可信；'
          '如果对不上，就先调阈值，别急着信报告里的其他结论。</p>')

    avg = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731
    days = len({time.strftime("%Y-%m-%d", time.localtime(r[0])) for r in rows})
    best = f"{ranked[0][0]:02d}:00 前后" if ranked else "数据不足"

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>专注度报告</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; padding:32px 20px 64px; background:#0f172a; color:#e2e8f0;
        font:15px/1.6 "Segoe UI",system-ui,-apple-system,"Microsoft YaHei",sans-serif; }}
  .wrap {{ max-width:980px; margin:0 auto; }}
  h1 {{ font-size:24px; margin:0 0 4px; }}
  h2 {{ font-size:16px; margin:36px 0 14px; color:#94a3b8; font-weight:600;
       letter-spacing:.04em; }}
  .sub {{ color:#64748b; margin:0 0 28px; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:14px; }}
  .card {{ background:#1e293b; border:1px solid #334155; border-radius:12px; padding:16px 18px; }}
  .card .k {{ color:#94a3b8; font-size:13px; }}
  .card .v {{ font-size:26px; font-weight:700; margin-top:6px; }}
  .hl-card {{ background:linear-gradient(135deg,#14532d,#1e293b); border-color:#22c55e; }}
  .grid {{ display:grid; grid-template-columns:220px 1fr; gap:36px; align-items:center; }}
  @media (max-width:640px) {{ .grid {{ grid-template-columns:1fr; }} }}
  .donut {{ width:100%; max-width:220px; }}
  .donut-big {{ fill:#e2e8f0; font-size:20px; font-weight:700; }}
  .donut-sub {{ fill:#64748b; font-size:12px; }}
  .row {{ display:grid; grid-template-columns:14px 60px 1fr 130px; gap:10px;
         align-items:center; margin-bottom:12px; }}
  .dot {{ width:12px; height:12px; border-radius:3px; }}
  .lbl {{ font-size:14px; }}
  .bar {{ background:#0f172a; height:10px; border-radius:5px; overflow:hidden; }}
  .bar i {{ display:block; height:100%; border-radius:5px; }}
  .val {{ font-size:13px; color:#94a3b8; text-align:right; font-variant-numeric:tabular-nums; }}
  .rc {{ display:flex; align-items:center; gap:9px; }}
  .rc .bar {{ flex:1; }}
  .rc span {{ font-size:13px; color:#cbd5e1; width:38px; text-align:right;
             font-variant-numeric:tabular-nums; }}
  .tl {{ display:flex; flex-wrap:wrap; gap:2px; }}
  .cell {{ width:9px; height:22px; border-radius:2px; }}
  table {{ width:100%; border-collapse:collapse; background:#1e293b;
          border-radius:12px; overflow:hidden; }}
  th,td {{ padding:10px 16px; text-align:left; border-bottom:1px solid #334155; }}
  th {{ color:#94a3b8; font-size:13px; font-weight:600; }}
  tr:last-child td {{ border-bottom:none; }}
  .num {{ text-align:right; font-variant-numeric:tabular-nums; color:#cbd5e1; }}
  .hbars {{ display:flex; gap:10px; align-items:flex-end; background:#1e293b;
           border-radius:12px; padding:16px; }}
  .hbar {{ flex:1; display:flex; flex-direction:column; align-items:center; gap:5px; }}
  .hl {{ font-size:11px; color:#64748b; }}
  .hbg {{ width:100%; height:110px; display:flex; align-items:flex-end; }}
  .hbg i {{ display:block; width:100%; background:#3b82f6; border-radius:3px 3px 0 0; }}
  .hv {{ font-size:11px; color:#94a3b8; font-variant-numeric:tabular-nums; }}
  .muted {{ color:#64748b; }}
  .note {{ color:#64748b; font-size:13px; margin-top:12px; }}
  /* 次要板块折叠。用原生 details，不需要 JS，也不影响首屏加载。 */
  details {{ border:1px solid #334155; border-radius:12px; margin:14px 0; }}
  details > summary {{ cursor:pointer; padding:13px 20px; font-size:15px;
        font-weight:600; color:#94a3b8; list-style:none; user-select:none; }}
  details > summary::-webkit-details-marker {{ display:none; }}
  details > summary::before {{ content:"▸ "; color:#64748b; }}
  details[open] > summary {{ color:#e2e8f0; }}
  details[open] > summary::before {{ content:"▾ "; color:#38bdf8; }}
  details .dbody {{ padding:2px 20px 16px; }}
  details .dbody h3 {{ font-size:14px; margin:20px 0 10px; color:#94a3b8;
        font-weight:600; }}
  /* 黄金时段第一名：整行打绿底，往上一眼就能看到 */
  tr.best td {{ background:#052e16; }}
  tr.best td:first-child {{ color:#22c55e; font-weight:700; }}
  .pbar {{ background:#0f172a; height:6px; border-radius:3px; overflow:hidden;
           margin:7px 0 4px; }}
  .pbar i {{ display:block; height:100%; border-radius:3px; }}
</style></head><body><div class="wrap">
<h1>专注度报告</h1>
<p class="sub">{_mdhm(rows[0][0])} – {_mdhm(rows[-1][0])} · {days} 天 · {len(rows)} 条样本</p>

<div class="cards">
  <div class="card hl-card"><div class="k">最佳时段</div>
    <div class="v" style="color:#22c55e;font-size:23px">{best}</div></div>
  <div class="card"><div class="k">有效投入率</div>
    <div class="v" style="color:#38bdf8">{engaged_rate:.0f}%</div></div>
  <div class="card"><div class="k">心流片段（≥15 分钟）</div>
    <div class="v">{len(all_streaks)}</div></div>
  <div class="card"><div class="k">会话数</div><div class="v">{len(sess_rows)}</div></div>
</div>

<h2>黄金时段</h2>
<table><thead><tr><th>时段</th><th class="num">活跃时长</th><th>有效投入占比</th>
<th class="num">会话数</th><th class="num">平均进入心流耗时</th></tr></thead>
<tbody>{hour_html}</tbody></table>
<p class="note">按"有效投入占比"排序（{'+'.join(sorted(ENGAGED))} ÷ 活跃时长），只统计累积
≥{_dur(MIN_HOUR_DATA)} 的钟点。{f"数据不足被排除的钟点：{', '.join(f'{h:02d}' for h in sorted(skipped))}。" if skipped else ""}
"平均进入心流耗时"是从坐下到第一段 ≥15 分钟投入开始的间隔，只统计该钟点开始的会话。</p>

<h2>自述对照（数据准不准）</h2>
{self_html}

<details><summary>会话明细与心流片段</summary><div class="dbody">
<h3>会话明细</h3>
<table><thead><tr><th>开始时间</th><th class="num">跨度</th><th class="num">活跃</th>
<th class="num">专注率</th><th class="num">心流片段</th><th class="num">进入心流耗时</th></tr></thead>
<tbody>{sess_html}</tbody></table>
<p class="note">会话 = 一次"坐下来用电脑"的连续时段。样本空档 &gt;{_dur(SESSION_GAP)}（待机/关机）
或连续离开 &gt;{_dur(SESSION_AWAY)}，都判定为会话结束。{"仅显示最近 40 次。" if len(sess_rows) > 40 else ""}</p>

<h3>心流片段</h3>
<table><thead><tr><th>开始时间</th><th class="num">持续</th>
<th class="num">有效专注占比</th><th class="num">距会话开始</th></tr></thead>
<tbody>{streak_html}</tbody></table>
<p class="note">心流 = 连续专注 ≥{_dur(FLOW_MIN)}，允许中间有 ≤{_dur(FLOW_GAP)} 的短暂中断。
{"仅显示最近 25 段。" if len(all_streaks) > 25 else ""}</p>
</div></details>

<h2>状态分布</h2>
<div class="grid">{_donut([(STATES[s][0], dur[s], STATES[s][1]) for s in order])}
<div>{legend}</div></div>
<p class="note">看着屏幕（专注 + 中性）占活跃时长的 <b>{looking_rate:.0f}%</b>；
加上伏案（低头看书/写作业），有效投入占
<b style="color:#22c55e;font-size:16px">{engaged_rate:.0f}%</b>。
伏案由头部俯仰角估算，前置摄像头分不清"低头看书"和"低头玩手机" —— 觉得虚高就把
focus.py 里的 DESKWORK_IS_ENGAGED 改成 False。</p>

<details><summary>全天活跃分布与每日投入</summary><div class="dbody">
<h3>全天活跃分布</h3>
<div class="hbars">{hours_bar}</div>
<p class="note">每根柱 = 该钟点<em>平均每天</em>的活跃时长。分母是"该钟点有记录的天数"
（只部分天用电脑的钟点，不会被少有的几天撑虚高），所以这是你的作息节律，不是总量。</p>
<h3>近 14 天每日投入</h3>
<div class="hbars">{day_bars}</div>
<p class="note">每日投入 = 计入投入的状态（{'+'.join(sorted(ENGAGED))}）时长。
悬停看当天专注率。{"仅显示最近 14 个有记录的自然日。" if day_keys else ""}</p>
<h3>时间轴</h3>
<div class="tl">{"".join(cells)}</div>
<p class="note">每格 1 分钟，颜色对应主导状态。共 {len(cells)} 分钟。</p>
</div></details>

<details><summary>应用排行</summary><div class="dbody">
<h3>专注应用排行</h3>
{_app_table(work_dur, dur["focused"], "没有专注记录")}
<h3>分心应用排行</h3>
{_app_table(app_dur, app_distract, "没有应用导致的分心")}
<p class="note">走神共 {_dur(dur["distracted"])}，拆成两块：
<b>应用导致的 {_dur(app_distract)}</b>（上表，你确实看着屏幕时被它拉走），
以及 <b>人没看屏幕的 {_dur(gaze_away)}</b>（转头或人离开画面 —— 此时前台开着什么
跟走神没有因果关系，所以不归因给任何应用）。
两者解法相反：前者靠屏蔽应用，后者靠休息或换任务。</p>
</div></details>

<details><summary>身体信号与坐姿</summary><div class="dbody">
<div class="cards">
  <div class="card"><div class="k">平均头部偏航</div><div class="v">{avg(yaws):.1f}°</div></div>
  <div class="card"><div class="k">平均头部俯仰</div><div class="v">{avg(pitches):.1f}°</div></div>
  <div class="card"><div class="k">平均眼开度 EAR</div><div class="v">{avg(ears):.3f}</div></div>
  <div class="card"><div class="k">坐姿不良时长</div><div class="v">{_dur(tilt_bad)}</div></div>
</div>
<p class="note">坐姿由前置摄像头肩线倾角估算（&gt;{TILT_WARN:.0f}° 记为不良）。
正面视角看不到驼背，这不是脊柱检测。状态切换 {switches} 次。</p>
</div></details>

</div></body></html>"""


def export_csv(rows: list[tuple]) -> Path:
    """把样本导出成 Excel 能直接打开的 CSV（UTF-8 BOM + 表头）。

    ts 记成本地时间可读串；标题字段可能带逗号/引号/换行，按 CSV 规则转义。
    """
    out = ROOT / f"focus-{time.strftime('%Y%m%d')}.csv"
    lines = ["时间,状态,应用,窗口标题,偏航,俯仰,EAR,肩倾,置信度,在画面,空闲秒"]
    for ts, state, app, title, yaw, pitch, ear, tilt, scale, present, idle in rows:
        lines.append(",".join([
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)),
            _csv_cell(state), _csv_cell(app), _csv_cell(title),
            f"{yaw:.1f}", f"{pitch:.1f}", f"{ear:.3f}", f"{tilt:.1f}",
            f"{scale:.3f}", "1" if present else "0", f"{idle:.0f}",
        ]))
    out.write_text("\n".join(lines), encoding="utf-8-sig")
    return out


def _csv_cell(value) -> str:
    """窗口标题可能带着逗号、引号、换行，包起来才不会把一列拆成好几列。"""
    s = str(value)
    if any(c in s for c in ',"\n'):
        s = '"' + s.replace('"', '""') + '"'
    return s


def main() -> None:
    rows = load()
    OUT.write_text(build_html(rows), encoding="utf-8")
    print(f"报告已生成: {OUT}")
    if rows:
        csv_out = export_csv(rows)
        print(f"数据已导出: {csv_out}")
        webbrowser.open(OUT.as_uri())


if __name__ == "__main__":
    main()
