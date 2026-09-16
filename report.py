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
import sys
import time
from collections import defaultdict
from pathlib import Path

import focus
import ratings
from focus import DB_PATH, STATES

# 打包成 exe 后 __file__ 指向临时解压目录，报告写进去等于没生成。
# frozen 时以 exe 所在目录为根，报告和 CSV 落在用户看得见的地方。
if getattr(sys, "frozen", False):
    ROOT = Path(sys.executable).resolve().parent
else:
    ROOT = Path(__file__).resolve().parent
OUT = ROOT / "report.html"

MAX_GAP = 5.0           # 样本间隔超过这个秒数就不计入任何状态（程序没在跑）
AWAY_SAMPLE_CAP = 40.0  # 离开期间降频到 30 秒一条，容差给到 40 秒
SESSION_GAP = 120.0     # 空档超过 2 分钟 → 新会话（待机 / 关机 / 崩溃）
SESSION_AWAY = 900.0    # 连续离开超过 15 分钟 → 本次会话结束
FLOW_MIN = 900.0        # 连续投入 ≥15 分钟才算心流片段
FLOW_GAP = 60.0         # 心流片段内允许 ≤60 秒的短暂中断
FLOW_FOCUS_RATIO = 0.5  # 片段里"专注"（工作应用）至少要占一半
MIN_HOUR_DATA = 600.0   # 某个时段至少累积 10 分钟才参与排名，否则样本太少
# 一个时段至少要这么多个"有进入耗时记录"的会话才给平均值。
# 只有 1~2 个样本时求平均没有意义，直接显示"数据不足"比印个假数字诚实。
MIN_BAND_SESSIONS = 3

# 时段分档：4 段 → 每 3 小时 → 每小时。选哪一档由数据量决定（见 pick_granularity）。
# 以前这里只写死了一档粗分，注释和 README 都承诺"样本够了会逐步压缩"，
# 但没有任何代码做这件事 —— 于是"我几点最容易进入心流"这个核心问题，
# 永远只能用 6 小时宽的格子回答。现在真的会随样本量细化。
GRAN_COARSE, GRAN_3H, GRAN_HOUR = "coarse", "3h", "hour"

_BAND_PREFIX = {GRAN_COARSE: "", GRAN_3H: "3h", GRAN_HOUR: "1h"}


def _bands(gran: str) -> tuple[tuple[str, int], ...]:
    """某一档的 (时段名, 起始小时) 表。时段名带前缀，避免不同档之间串味。"""
    pre = _BAND_PREFIX[gran]
    if gran == GRAN_HOUR:
        return tuple((f"{pre}{h:02d}", h) for h in range(24))
    if gran == GRAN_3H:
        return tuple((f"{pre}{h:02d}", h) for h in range(0, 24, 3))
    return tuple((f"{pre}{n}", s) for n, s in
                 (("凌晨", 0), ("上午", 6), ("下午", 12), ("晚上", 18)))


def _band(ts: float, gran: str = GRAN_COARSE) -> str:
    """时间戳落在哪个时段。"""
    table = _bands(gran)
    h = time.localtime(ts).tm_hour
    for name, start in reversed(table):
        if h >= start:
            return name
    return table[0][0]


def pick_granularity(item_hours: dict[int, set[str]]) -> str:
    """按样本量挑最细的可用档位。

    规则：一档里**每一个有数据的桶**都必须至少有 MIN_BAND_SESSIONS 个
    会话，才允许用这一档 —— 因为只有那样，每个桶的"平均进入耗时"才不是
    印刷出来的假数字。任何一档里混着"样本不足"的桶，就退回上一档。

    传入的是 {钟点: 该钟点出现过的日期集合}，日期集合的基数当会话数的
    代理（同一天同一钟点两次坐下仍算一次，但作为分档门槛足够了）。
    """
    if not item_hours:
        return GRAN_COARSE
    for gran in (GRAN_HOUR, GRAN_3H):
        buckets: dict[str, set[str]] = defaultdict(set)
        for h, days in item_hours.items():
            buckets[_band_from_hour(h, gran)] |= days
        if buckets and all(len(d) >= MIN_BAND_SESSIONS for d in buckets.values()):
            return gran
    return GRAN_COARSE


def _band_from_hour(h: int, gran: str) -> str:
    """已知钟点，直接取时段名（省掉一次 localtime）。"""
    table = _bands(gran)
    for name, start in reversed(table):
        if h >= start:
            return name
    return table[0][0]


def load(db_path: Path | None = None, since: float | None = None) -> list[tuple]:
    """读样本。since 是时间戳下界 —— 实时面板每次轮询都读，不能全表扫。

    默认路径在**调用时**取，不写成 `db_path=DB_PATH` 参数默认值。
    参数默认值在 import 时就绑定死了，而 focus.DB_PATH 是允许被改的
    （自检、测试、多库对照都靠这个）—— 写死之后那些场景会静默读错文件，
    返回空列表，看起来像"没数据"，很难查。
    """
    db_path = DB_PATH if db_path is None else db_path
    sql = ("SELECT ts,state,app,title,yaw,pitch,ear,tilt,scale,present,idle "
           "FROM samples")
    args: tuple = ()
    if since is not None:
        sql += " WHERE ts >= ?"
        args = (since,)
    # 走 focus.open_db：统一拿到 WAL + busy_timeout。
    # 报告要对全表做聚合，这是最长的读事务 —— 不设 busy_timeout 的话
    # 采集线程会在它持锁期间撞锁失败（而采集线程崩掉是静默的，见 Monitor.run）。
    conn = focus.open_db(db_path)
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


def _day_key(ts: float) -> str:
    """自然日字符串。时间轴、每日柱状图共用，保证分组口径一致。

    **按本地时区算，不能用 UTC。** 用 UTC 的话，东八区凌晨 0–8 点的记录
    会被归到"前一天"，跨天分隔线直接错位。
    """
    return time.strftime("%Y-%m-%d", time.localtime(ts))


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
        total = items[last][0] + items[last][1] - items[start][0]
        if total >= FLOW_MIN:
            span = range(start, last + 1)
            engaged = sum(items[k][1] for k in span if items[k][2] in ENGAGED)
            focused = sum(items[k][1] for k in span if items[k][2] == "focused")
            # 光"看着屏幕"不算心流。要求这段里至少一半时间在真正的工作应用上 ——
            # 否则"坐下来盯着桌面发呆 15 分钟"也算，这正是旧定义下会印出
            # "进入心流耗时 0 秒"的根源（会话一开头就是 neutral，立刻满足条件）。
            if focused >= total * FLOW_FOCUS_RATIO:
                out.append((items[start][0], items[last][0] + items[last][1],
                            engaged, total, focused))
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


def _app_table(mapping: dict[str, float], total: float, empty: str,
               total_label: str = "占比", limit: int = 15) -> str:
    """应用时长排行。

    表头的分母必须写出来。原来只有"占比"两个字，读者自然会以为分母是
    "总时长"，而实际是"该状态的总时长" —— 专注应用占 60% 的意思
    是"专注时间里 60% 在它上面"，不是"全天 60%"。
    差一个字，结论能反过来说。

    limit 从 8 提到 15：截断本身是合理的（排行不需要长尾），但原来
    砍到 8 之后没有任何提示，用户看到的就是"记录不全"。现在配合
    「应用使用记录」那张全量表，这里只作为快速榜。
    """
    ranked = sorted(mapping.items(), key=lambda kv: -kv[1])
    top = ranked[:limit]
    rows = "".join(
        f'<tr><td>{html.escape(a)}</td><td class="num">{_dur(d)}</td>'
        f'<td class="num">{d / max(total, 1) * 100:.0f}%</td></tr>'
        for a, d in top) or f'<tr><td colspan="3" class="muted">{empty}</td></tr>'
    cut = ranked[limit:]
    tail = ""
    if cut:
        tail = (f'<p class="note">另有 {len(cut)} 个未列出，'
                f'合计 {_dur(sum(v for _, v in cut))}。'
                f'要完整清单看上面的「应用使用记录」。</p>')
    return ('<table><thead><tr><th>窗口 / 应用</th><th class="num">时长</th>'
            f'<th class="num" title="分母 = {html.escape(total_label)}">'
            f'占比<sup>*</sup></th></tr></thead><tbody>{rows}</tbody></table>{tail}')


def _all_app_table(all_app: dict[str, float],
                   all_app_state: dict[str, dict[str, float]],
                   app_first: dict[str, float],
                   app_last: dict[str, float],
                   all_app_days: dict[str, set],
                   all_app_gaze: dict[str, float],
                   active_total: float,
                   limit: int = 40) -> str:
    """全部应用使用记录：每个应用 + 时长 + 占比 + 出现时段 + 主要状态。

    为什么要单独一张表（而不是把 _app_table 的 [:8] 去掉）：
    原来只有"专注应用"和"分心应用"两张榜，**中性状态的应用完全不出现** ——
    查资料、翻文件、看文档这些"用着电脑但不算专注也不算分心"的时间，
    加起来往往有数百分钟，在报告里却一片空白。用户看到的就是"记录不全"。

    另外补上「时段」列：只有总时长，读者没法判断这些时间落在一天里的哪一段，
    也就没法回答"我下午到底在干嘛"。
    """
    ranked = sorted(all_app.items(), key=lambda kv: -kv[1])
    if not ranked:
        return '<p class="muted">这段时间没有应用使用记录。</p>'

    shown = ranked[:limit]
    rows = []
    for lbl, secs in shown:
        st = all_app_state.get(lbl, {})
        # 主要状态取时长最大的那个；顺带标出是否"全部集中在某一个状态"
        dom = max(st.items(), key=lambda kv: kv[1])[0] if st else ""
        dom_name, dom_color = STATES.get(dom, ("未知", "#64748b"))
        mix = len([1 for v in st.values() if v > 0])
        dom_note = (f'<span class="sfx">{mix} 种</span>' if mix > 1 else "")

        first, last = app_first.get(lbl), app_last.get(lbl)
        days = len(all_app_days.get(lbl, ()))
        if days > 1:
            # 跨天时 first/last 分属不同日期，直接拼"首时刻 – 末时刻"会得到
            # "14:00 – 11:15" 这种看起来像倒着走的区间（昨天下午到今早）。
            # 跨天就该说"出现在哪几天"，而不是假装它是一个连续区间。
            span = f'{days} 天'
            span_note = ""
        else:
            span = (f'{_hm(first)} – {_hm(last)}' if first and last else "—")
            span_note = ""

        # 人没看屏幕的占比够高时提醒一句。不写"分心"——这只是"没看屏幕"，
        # 可能是在看书、也可能是离开了，报告没有资格替用户下这个判断。
        away_secs = all_app_gaze.get(lbl, 0.0)
        gaze_note = ""
        if secs > 0 and away_secs / secs >= 0.3:
            gaze_note = (f'<span class="sfx">'
                         f'其中 {away_secs / secs * 100:.0f}% 未看屏幕</span>')

        rows.append(
            f'<tr><td>{html.escape(lbl)}</td>'
            f'<td class="num">{_dur(secs)}</td>'
            f'<td class="num">{secs / max(active_total, 1) * 100:.1f}%</td>'
            f'<td class="num">{span}{span_note}</td>'
            f'<td><span class="dot" style="background:{dom_color}"></span>'
            f'{dom_name}{dom_note}{gaze_note}</td></tr>')

    hidden = ranked[limit:]
    more = ""
    if hidden:
        more = (f'<p class="note">另有 {len(hidden)} 个应用未列出，'
                f'合计 {_dur(sum(v for _, v in hidden))}'
                f'（都在 1 分钟以下，通常是系统弹窗或短暂切换）。</p>')

    return ('<table><thead><tr><th>应用 / 窗口</th><th class="num">时长</th>'
            '<th class="num">占活跃</th><th class="num">时段</th>'
            f'<th>主要状态</th></tr></thead><tbody>{"".join(rows)}</tbody>'
            f'</table>{more}')


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
    # 「所有应用使用记录」用的全量累计。
    # 和上面两个分开：work_dur / app_dur 只服务"排行"（专注 / 分心两种归因），
    # 而用户想看的是"我今天到底开过哪些应用、各用了多久" —— 那必须包含
    # neutral（中性）和 deskwork（伏案），否则 GitHub 看了 45 分钟、
    # 记事本写了 40 分钟，在报告里一个字都找不到。
    all_app: dict[str, float] = defaultdict(float)
    # 每个应用按状态拆开：同一个应用可能既在专注时用、也在走神时用，
    # 只给一个总数会误导（微信 40 分钟全是走神，和 40 分钟全是工作，意义相反）。
    all_app_state: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float))
    # 首次 / 最后出现时刻，用来显示"这条记录对应一天里的哪一段"。
    app_first: dict[str, float] = {}
    app_last: dict[str, float] = {}
    all_app_days: dict[str, set] = defaultdict(set)
    # 应用时长里"人没看屏幕"的那部分。用于表里标注，不是分心归因。
    all_app_gaze: dict[str, float] = defaultdict(float)
    gaze_away = 0.0        # 走神里"人没看屏幕"的那部分，不归因给任何应用
    hour_active: dict[int, float] = defaultdict(float)
    hour_engaged: dict[int, float] = defaultdict(float)
    hour_active_days: dict[int, set] = defaultdict(set)   # 每钟点活跃过的日期
    day_active: dict[str, float] = defaultdict(float)     # 每天活跃时长
    day_engaged: dict[str, float] = defaultdict(float)    # 每天投入时长
    hour_sessions: dict[int, int] = defaultdict(int)
    hour_entry: dict[int, list[float]] = defaultdict(list)
    # ── 时段统计（分档由数据量决定，见 pick_granularity）──
    # 活跃/投入按"样本实际发生的时刻"算；会话数和进入耗时按"坐下的时刻"算。
    # 两者回答的问题不同：前者是"这段时间我人在不在线"，后者是"我几点坐下来
    # 更容易快速进入状态"。混用会让同一行自相矛盾（旧表里 14:00 那行就是：
    # 99% 投入、21 分钟活跃，却 0 个会话）。
    #
    # 分档要先用「钟点 → 出现过的日期」定下来，所以这里先扫一遍收集
    # item_hours，再决定 gran，然后才真正分桶。两趟比一趟慢不了多少
    # （items 本来就只有报告窗口内那么多行），但换来的是"承诺过的细化"。
    item_hours: dict[int, set[str]] = defaultdict(set)
    for ts, dt, state, *_ in items:
        if state != "away" and dt > 0:
            item_hours[time.localtime(ts).tm_hour].add(
                time.strftime("%Y-%m-%d", time.localtime(ts)))
    gran = pick_granularity(item_hours)

    band_active: dict[str, float] = defaultdict(float)
    band_engaged: dict[str, float] = defaultdict(float)
    band_sessions: dict[str, int] = defaultdict(int)
    band_entry: dict[str, list[float]] = defaultdict(list)   # 进入心流耗时
    band_flow: dict[str, list[float]] = defaultdict(list)    # 心流片段持续时长
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
        b = _band(ts, gran)
        if state != "away":
            hour_active[lt.tm_hour] += dt
            hour_active_days[lt.tm_hour].add(day)
            day_active[day] += dt
            band_active[b] += dt
        if state in ENGAGED:
            hour_engaged[lt.tm_hour] += dt
            day_engaged[day] += dt
            band_engaged[b] += dt
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

        # 全量应用记录：只要人不在离开状态、且前台确实有东西，就计入。
        # 不按状态过滤 —— 用户问的是"我用了什么"，不是"我专注时用了什么"。
        #
        # 注意这里**故意**包含"转头时前台开着的应用"（gaze-away 那种情况）。
        # 排行表必须排除它（否则会得出"快速设置是你最大分心源"的荒谬结论），
        # 但使用记录不含它就会漏掉真实时间 —— 人转头去做别的事时，
        # 电脑上确实还开着那个窗口。两者目的不同，口径也就该不同：
        # 排行回答"什么在拉走我"，记录回答"我用了什么"。
        if state != "away" and dt > 0 and (app or title):
            lbl = _label(app, title)
            all_app[lbl] += dt
            all_app_state[lbl][state] += dt
            all_app_days[lbl].add(day)
            if lbl not in app_first or ts < app_first[lbl]:
                app_first[lbl] = ts
            end = ts + dt
            if lbl not in app_last or end > app_last[lbl]:
                app_last[lbl] = end
            # 人没看屏幕的时间单独累计，表里标出来。这不是分心归因，
            # 只是告诉读者"这段里有一部分人不在看屏幕"，避免过度解读。
            if not _is_looking(present, yaw, pitch):
                all_app_gaze[lbl] += dt
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
        b = _band(s_start, gran)           # 按"坐下"的时刻归属，不按心流发生的时刻
        band_sessions[b] += 1
        entry = (st[0][0] - s_start) if st else None
        if entry is not None:
            band_entry[b].append(entry)
        for k in st:
            band_flow[b].append(k[3])      # 片段总时长
            all_streaks.append((k[0], k[1], k[2], k[3], s_start, k[4]))
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
        f'<td class="num">{e / d * 100:.0f}%</td>'
        f'<td class="num">{_dur(a - s)}</td></tr>'
        for a, b, c, d, s, e in all_streaks[:25]) or \
        '<tr><td colspan="4" class="muted">还没有 ≥15 分钟的心流片段</td></tr>'

    # ── 时段（分档 gran 已由数据量决定：4 段 / 3 小时 / 1 小时）──
    # 只列"有数据"的时段：按小时分档时 24 行里有大半是空的，全列出来
    # 会把真正有信号的那几行冲淡。空时段不参与排名也不显示。
    ranked = [(b, band_active[b]) for b in band_active
              if band_active[b] >= MIN_HOUR_DATA]
    ranked.sort(key=lambda kv: -band_engaged[kv[0]] / kv[1])
    skipped = [b for b in band_active if band_active[b] < MIN_HOUR_DATA]
    band_rows = []
    for i, (b, _) in enumerate(ranked):
        entries = band_entry.get(b, [])
        flows = band_flow.get(b, [])
        cls = ' class="best"' if i == 0 else ""
        # 样本不够就不给平均值 —— 1 个样本不叫"平均"，印出来是误导
        entry_txt = (_dur(sum(entries) / len(entries))
                     if len(entries) >= MIN_BAND_SESSIONS
                     else f'<span class="muted">样本不足（{len(entries)}）</span>')
        flow_txt = (_dur(sum(flows) / len(flows))
                    if len(flows) >= MIN_BAND_SESSIONS
                    else f'<span class="muted">样本不足（{len(flows)}）</span>')
        band_rows.append(
            f'<tr{cls}><td>{b}</td>'
            f'<td class="num">{_dur(band_active[b])}</td>'
            f'<td>{_rate_cell(band_engaged[b] / band_active[b] * 100)}</td>'
            f'<td class="num">{band_sessions.get(b, 0)}</td>'
            f'<td class="num">{entry_txt}</td>'
            f'<td class="num">{flow_txt}</td></tr>')
    hour_html = "".join(band_rows) or \
        (f'<tr><td colspan="6" class="muted">还没有任何时段累积到 '
         f'{_dur(MIN_HOUR_DATA)} 的数据</td></tr>')

    GRAN_LABEL = {GRAN_COARSE: "4 段", GRAN_3H: "3 小时", GRAN_HOUR: "1 小时"}
    gran_note = (
        f'当前按 <b>{GRAN_LABEL[gran]}</b> 分档。'
        + (f'每个格子都已攒够 {MIN_BAND_SESSIONS} 次会话，所以进一步细化到'
           + ("半小时" if gran == GRAN_HOUR else "1 小时") + "也不会全是「样本不足」。"
           if gran == GRAN_HOUR else
           f'等每个格子都攒够 {MIN_BAND_SESSIONS} 次会话，会自动细到'
           + ("1 小时。" if gran == GRAN_3H else "3 小时，再到 1 小时。"))
        if gran != GRAN_COARSE else
        f'当前按 <b>4 段</b>分档 —— 样本还不够细。等每个时段都攒够 '
        f'{MIN_BAND_SESSIONS} 次会话，这张表会自动细到 3 小时、再到 1 小时，'
        f'那时「几点」才真正回答得出来。')

    legend = "".join(
        f'<div class="row"><span class="dot" style="background:{STATES[s][1]}"></span>'
        f'<span class="lbl">{STATES[s][0]}</span>'
        f'<div class="bar"><i style="width:{dur[s] / total * 100:.1f}%;'
        f'background:{STATES[s][1]}"></i></div>'
        f'<span class="val">{_dur(dur[s])} · {dur[s] / total * 100:.0f}%</span></div>'
        for s in order if dur[s] > 0)

    # ── 时间轴 ──
    # 按自然日分组。原来是一条不换行的长条，跨天完全看不出来 —— 昨天 23:58
    # 和今天 00:01 挨在一起，中间那道"隔夜"的信息丢了，读起来像一直在用电脑。
    buckets: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for ts, dt, state, *_ in items:
        buckets[int(ts // 60)][state] += dt
    day_cells: dict[str, list[str]] = defaultdict(list)
    for minute in sorted(buckets):
        dom = max(buckets[minute], key=buckets[minute].get)  # type: ignore[arg-type]
        day = _day_key(minute * 60)
        day_cells[day].append(
            f'<span class="cell" style="background:{STATES[dom][1]}" '
            f'title="{_mdhm(minute * 60)} {STATES[dom][0]}"></span>')
    # 跨天时插一条分隔行，日子多的先显示（最新在上）。
    tl_days = sorted(day_cells, reverse=True)[:7]
    tl_parts = []
    for day in tl_days:
        tl_parts.append(
            f'<div class="tlrow"><span class="tlday">{day}</span>'
            f'<div class="tl">{"".join(day_cells[day])}</div></div>')
    timeline_html = "".join(tl_parts)
    tl_minutes = sum(len(v) for v in day_cells.values())

    # 每小时归一成"平均每天"：只早上用电脑的人，上午柱不再被几天撑虚高。
    # 原始累计 hour_active 仍留给黄金时段排序，柱状图只看平均强度。
    #
    # 分母理论上是 0 会 ZeroDivisionError。hour_active 和 hour_active_days
    # 是同一处代码同时写的，所以构造不出真实触发路径（第三轮审查把它列进
    # "未能确认的项"）。但报告是"点一下就该出来"的东西，为一个假设中的路径
    # 让整页崩掉不划算 —— 这里直接跳过没有活跃日记录的钟点。
    hour_avg = {h: hour_active[h] / len(days)
                for h, days in hour_active_days.items()
                if days and h in hour_active}
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

    # ── 主卡片：三态，别印一个撑不住的结论 ──
    # 旧版一旦某个时段过 600 秒就直接印"最佳时段：上午"，可那一行可能
    # 还是「样本不足（0）」—— 卡片说"这是你的黄金时段"，下面的表却拿不出
    # 任何会话和进入耗时。那是报告在替数据吹牛。改成分三种情况。
    #
    # 口径必须和表格排名一致：表格是按**有效投入占比**排的、第一行打绿底，
    # 卡片却去显示「平均进入心流耗时」，于是出现过"卡片上的数字比第二名
    # 还差"的读法 —— 同一张报告里两个第一名。现在卡片的第一行永远先说
    # 排名依据（投入占比），进入耗时降为补充说明。
    if not ranked:
        hero_k = "最佳时段"
        hero_v = "数据不足"
        hero_v_css = "color:#64748b;font-size:20px"
        hero_note = ("还没有任何时段累积到 " + _dur(MIN_HOUR_DATA) + " 的记录。"
                     "再记录几天就会出来。")
    else:
        b0 = ranked[0][0]
        e0 = band_entry.get(b0, [])
        rate0 = band_engaged[b0] / band_active[b0] * 100
        # 样本太薄时投入占比本身也不稳（几个样本就能凑出 100%），
        # 所以只给它加分档提示，不吹成"最佳"。
        thin = band_active[b0] < 2 * MIN_HOUR_DATA
        hero_k = ("最佳时段 · " if not thin else "投入占比最高 · ") + b0
        hero_v = f"投入占比 {rate0:.0f}%"
        hero_v_css = ("color:#22c55e;font-size:23px" if not thin
                      else "color:#38bdf8;font-size:22px")
        if len(e0) >= MIN_BAND_SESSIONS:
            hero_note = (f"活跃 {_dur(band_active[b0])}、"
                         f"{band_sessions.get(b0, 0)} 次会话；"
                         f"坐下到第一段心流平均 {_dur(sum(e0) / len(e0))}"
                         f"（{len(e0)} 次会话）。")
        else:
            hero_note = (f"活跃 {_dur(band_active[b0])}、"
                         f"{band_sessions.get(b0, 0)} 次会话。"
                         f"该时段进入心流的样本还差 "
                         f"{max(0, MIN_BAND_SESSIONS - len(e0))} 次，"
                         f"所以暂时答不出「多快进入心流」。")
        if thin:
            hero_note += " 这个时段样本还不多，占比可能还会变。"

    # ── 自述对照：这是全报告的信任基础，提到顶部而不是埋在中部 ──
    # README 自己说 r < 0.4 时"别急着信报告里的其他结论"，那就不该让用户
    # 先读完几十行分析才看到它。原来它用 .sub（全文最暗的灰）渲染。
    if corr is None:
        trust_tone = ("<div class='trust pending'><b>数据可信度尚未验证</b> · "
                      f"已配对 {len(pairs)}/{ratings.MIN_PAIRS} 个自述评分 —— "
                      "先把这个凑够，再信下面的结论更有意义。</div>")
    elif corr >= 0.7:
        trust_tone = (f"<div class='trust good'><b>数据可信</b> · 自评和实测"
                      f"相关系数 r = {corr:.2f}，两者是一致的。</div>")
    elif corr >= 0.4:
        trust_tone = (f"<div class='trust mid'><b>大致对得上</b> · r = {corr:.2f}，"
                      "大方向一致，但阈值还有调整空间。</div>")
    else:
        trust_tone = (f"<div class='trust bad'><b>先别信其他结论</b> · "
                      f"r = {corr:.2f} 说明测量和你的感受对不上，"
                      "请先按「校准」一节调阈值。</div>")

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
  /* 每天一行：左边日期标签固定宽度，右边是当天的格子。 */
  .tlrow {{ display:flex; align-items:flex-start; gap:10px; margin-bottom:6px; }}
  .tlday {{ width:78px; flex:none; font-size:11px; color:#64748b;
            font-variant-numeric:tabular-nums; padding-top:3px; }}
  .tlrow .tl {{ flex:1; min-width:0; }}
  .cell {{ width:9px; height:22px; border-radius:2px; }}
  table {{ width:100%; border-collapse:collapse; background:#1e293b;
          border-radius:12px; overflow:hidden; }}
  th,td {{ padding:10px 16px; text-align:left; border-bottom:1px solid #334155; }}
  th {{ color:#94a3b8; font-size:13px; font-weight:600; }}
  tr:last-child td {{ border-bottom:none; }}
  .num {{ text-align:right; font-variant-numeric:tabular-nums; color:#cbd5e1; }}
  /* 表格里的次要说明（"3 天"、"2 种"）：跟在主值后面，靠右一小截。 */
  .sfx {{ color:#64748b; font-size:11px; margin-left:6px; }}
  /* 状态圆点：在表格里也要显示，之前只在状态分布那张图里定义了 .dot */
  td .dot {{ display:inline-block; width:8px; height:8px; border-radius:2px;
             margin-right:6px; vertical-align:middle; }}
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
  /* 可信度横幅：全报告的信任基础，放在最上面而不是埋在中部 */
  .trust {{ border-radius:10px; padding:12px 18px; margin:0 0 20px; font-size:14px;
            border:1px solid; }}
  .trust.good {{ background:#052e16; border-color:#22c55e; color:#86efac; }}
  .trust.mid {{ background:#422006; border-color:#f59e0b; color:#fcd34d; }}
  .trust.bad {{ background:#450a0a; border-color:#ef4444; color:#fca5a5; }}
  .trust.pending {{ background:#1e293b; border-color:#475569; color:#94a3b8; }}
  .card .n {{ font-size:12px; color:#94a3b8; margin-top:6px; line-height:1.5; }}
</style></head><body><div class="wrap">
<h1>专注度报告</h1>
<p class="sub">{_mdhm(rows[0][0])} – {_mdhm(rows[-1][0])} · {days} 天 · {len(rows)} 条样本</p>

{trust_tone}

<div class="cards">
  <div class="card hl-card"><div class="k">{hero_k}</div>
    <div class="v" style="{hero_v_css}">{hero_v}</div>
    <div class="n">{hero_note}</div></div>
  <div class="card"><div class="k">有效投入率</div>
    <div class="v" style="color:#38bdf8">{engaged_rate:.0f}%</div>
    <div class="n">基于 {_dur(active)} 活跃时长</div></div>
  <div class="card"><div class="k">心流片段（≥15 分钟）</div>
    <div class="v">{len(all_streaks)}</div>
    <div class="n">共 {len(sess_rows)} 次会话</div></div>
  <div class="card"><div class="k">测量可信度</div>
    <div class="v" style="font-size:22px">{corr_txt}</div>
    <div class="n">{len(pairs)}/{ratings.MIN_PAIRS} 个自述评分</div></div>
</div>

<h2>黄金时段</h2>
<table><thead><tr><th>时段</th><th class="num">活跃时长</th><th>有效投入占比</th>
<th class="num">会话数</th><th class="num">平均进入心流耗时</th>
<th class="num">平均心流时长</th></tr></thead>
<tbody>{hour_html}</tbody></table>
<p class="note">
<b>排名依据是「有效投入占比」</b>（绿色那行 = 占比最高），不是「进入心流耗时」——
后者受单次异常影响太大。{gran_note} 活跃不足 {_dur(MIN_HOUR_DATA)} 的时段不参与排名
{f"（已排除：{', '.join(skipped)}）" if skipped else ""} —— 拿几秒样本编出"100% 专注"的假排名没有意义。<br>
「平均进入心流耗时」= 从坐下到第一段心流之间的间隔，<b>按"坐下"的时刻归属</b>；
「平均心流时长」= 心流片段本身持续多久。<br>
心流 = 连续投入 ≥{_dur(FLOW_MIN)}，<b>且其中「专注」（工作应用）占比 ≥{FLOW_FOCUS_RATIO:.0%}</b> ——
否则"坐下来盯着屏幕发呆"也会算进去，那正是以前会印出"进入心流耗时 0 秒"的原因。<br>
两个平均值都要求该时段至少有 {MIN_BAND_SESSIONS} 个样本，不够就写"样本不足"，不印假数字。</p>

<h2>自述对照（数据准不准）</h2>
<p class="sub" style="margin:0 0 14px">配对样本 {len(pairs)} 条 · 相关系数
<b>{corr_txt}</b> —— {html.escape(ratings.verdict(corr, len(pairs)))}</p>
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
<th class="num">其中专注占比</th><th class="num">距会话开始</th></tr></thead>
<tbody>{streak_html}</tbody></table>
<p class="note">心流 = 连续投入 ≥{_dur(FLOW_MIN)}，允许中间有 ≤{_dur(FLOW_GAP)} 的短暂中断，
且其中在工作应用上的时间 ≥{FLOW_FOCUS_RATIO:.0%}。
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
{timeline_html}
<p class="note">每格 1 分钟，颜色对应主导状态，按自然日分行（最新在上，最多 7 天）。
共 {tl_minutes} 分钟。断行处就是跨天，中间那段没有记录。
{"显示最近 7 天；更早的天数未展开。" if len(day_cells) > len(tl_days) else ""}</p>
</div></details>

<details open><summary>应用使用记录</summary><div class="dbody">
<h3>全部应用（{len(all_app)} 个）</h3>
{_all_app_table(all_app, all_app_state, app_first, app_last, all_app_days,
                 all_app_gaze, active)}
<p class="note">包含<strong>所有</strong>状态下的应用使用，不只是专注和分心 ——
查资料、翻文件、写文档这些中性时间也在里面，否则你会觉得"记录少了一大块"。
「占活跃」的分母是全天活跃时长 {_dur(active)}（已扣掉离开）。
「时段」是该应用当天首次和最后一次出现的时刻，多天时另标天数。
「未看屏幕」只表示摄像头当时没拍到你在看屏幕（转头、低头或离开），
<strong>不等于是分心</strong> —— 也可能是在看纸质材料。</p>
</div></details>

<details><summary>应用排行</summary><div class="dbody">
<h3>专注应用排行</h3>
{_app_table(work_dur, dur["focused"], "没有专注记录",
            f"专注总时长 {_dur(dur['focused'])}")}
<h3>分心应用排行</h3>
{_app_table(app_dur, app_distract, "没有应用导致的分心",
            f"应用导致的分心总时长 {_dur(app_distract)}")}
<p class="note"><b><sup>*</sup> 占比的分母不是全天，而是各自那一栏的总时长</b> ——
"专注应用 60%" 是说专注时间里六成花在它上面，不是全天六成。
两栏分母不同，跨栏比较没有意义。</p>
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

    数值列一律走 _num()：库里这几列**允许为 NULL**（老版本写进去的行、
    或某帧算不出指标时），直接 f"{yaw:.1f}" 会抛
    `TypeError: unsupported format string passed to NoneType.__format__`，
    而且是导出到一半才抛 —— 用户拿到的是个截断的、能打开但少一大半的 CSV，
    比明确报错更危险。空值统一写成空字符串，Excel 也不会把它显示成 0。
    """
    out = ROOT / f"focus-{time.strftime('%Y%m%d')}.csv"
    lines = ["时间,状态,应用,窗口标题,偏航,俯仰,EAR,肩倾,置信度,在画面,空闲秒"]
    for ts, state, app, title, yaw, pitch, ear, tilt, scale, present, idle in rows:
        lines.append(",".join([
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)),
            _csv_cell(state), _csv_cell(app), _csv_cell(title),
            _num(yaw, 1), _num(pitch, 1), _num(ear, 3), _num(tilt, 1),
            _num(scale, 3), "1" if present else "0", _num(idle, 0),
        ]))
    out.write_text("\n".join(lines), encoding="utf-8-sig")
    return out


def _num(value, digits: int) -> str:
    """数值列的空值安全格式化：None → 空串，而不是崩掉整次导出。"""
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""


def _csv_cell(value) -> str:
    """窗口标题可能带着逗号、引号、换行，包起来才不会把一列拆成好几列。"""
    s = str(value)
    if any(c in s for c in ',"\n'):
        s = '"' + s.replace('"', '""') + '"'
    return s


def main() -> Path | None:
    """生成报告文件（exe 旁的 report.html + CSV）。不自动开浏览器。

    展示由应用窗口承担（窗口 /report 页直接渲染报告 HTML），
    这里的产出是「可以带走/分享/打印的文档」。返回报告路径。
    """
    focus.use_safe_console()      # 报告里的中文提示在西文代码页上会抛异常
    rows = load()
    OUT.write_text(build_html(rows), encoding="utf-8")
    print(f"报告已生成: {OUT}")
    if rows:
        # CSV 是附加产物，不该拖垮报告本身：磁盘满 / 文件被 Excel 占着
        # （Windows 上很常见）都会让 write_text 抛异常。以前这一抛会带着
        # 整个 main() 一起挂掉，调用方连 OUT 都拿不到，面板就白屏了。
        try:
            csv_out = export_csv(rows)
            print(f"数据已导出: {csv_out}")
        except OSError as exc:
            print(f"CSV 导出失败（报告本身已生成）：{exc}")
    return OUT


if __name__ == "__main__":
    main()
