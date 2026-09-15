"""自述对照：收集你对每个时段的专注感受，用来验证测量准不准。

为什么值得单独记一张表：工具现在只会输出数字，没有任何东西能证明那些数字
和你的真实感受对得上。有了自评分，才谈得上"准不准"。

一条硬约束：**评分界面不能显示任何实测数据**。要是写着"这段测出 85% 专注"，
你会被那个数字锚定，那就不是在验证，只是在复读。所以评分页只给时间范围。

模块不 import report —— 样本由调用方传进来（report._timed(...)），否则
report 和 ratings 会互相 import 成环。
"""

from __future__ import annotations

import contextlib
import sqlite3
import time

import focus

BLOCK = 1800.0          # 30 分钟一块，按整点/半点切
MIN_ACTIVE = 600.0      # 块内至少 10 分钟有效数据才值得评
# 超过这么久就不再提醒 —— 想不起当时什么感觉了。
# 原来给 8 小时，实际是让人回忆"上午十点那半小时"，回忆不出来只能瞎填，
# 而瞎填的数据会直接污染相关系数。3 小时是回忆得住的上限。
RECENT = 3 * 3600
# 少于这么多配对样本就不算相关系数。
#
# 为什么从 8 抬到 30：蒙特卡洛跑过一遍（3000 次，纯随机数据）——
# n=8 时 |r|≥0.4 的假阳性率 **30.8%**，|r|≥0.7 也有 5.2%。
# 也就是说八个点算出来的"中等相关"，三次里有一次是纯噪声。
# n=30 时同样的门槛降到约 3%，才配得上 verdict() 里那句"数据可信"。
# 这是产品的信誉问题：宁可告诉用户"样本还不够"，也不能拿噪声当结论。
MIN_PAIRS = 30

SCHEMA = """
CREATE TABLE IF NOT EXISTS ratings(
    block_start REAL PRIMARY KEY,
    score INTEGER NOT NULL,
    note TEXT DEFAULT '',
    rated_at REAL NOT NULL
);
"""

# 显式"跳过"的块。和评分分开放：跳过不等于 0 分，也不等于没评过 ——
# 它表示"这段时间我根本没状态可评"。没有这个出口的话，用户只能
# 要么瞎填一个分，要么看着待评列表永远消不下去。
SKIP_SCHEMA = """
CREATE TABLE IF NOT EXISTS rating_skips(
    block_start REAL PRIMARY KEY,
    skipped_at REAL NOT NULL
);
"""


@contextlib.contextmanager
def _conn():
    """用完必须显式 close。

    `with sqlite3.connect(...) as conn` 只管事务提交，**不关连接** —— 那样每次
    调用都会漏一个句柄，Windows 上还会把文件锁住（自检里正是这么撞出来的）。
    """
    conn = focus.open_db(focus.DB_PATH)
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def skips() -> set[float]:
    """被显式跳过的块。"""
    with _conn() as conn:
        conn.executescript(SKIP_SCHEMA)
        return {r[0] for r in conn.execute(
            "SELECT block_start FROM rating_skips")}


def skip(block_start: float) -> None:
    """标记某个块"不评"。待评列表靠它才能被清干净。"""
    with _conn() as conn:
        conn.executescript(SKIP_SCHEMA)
        conn.execute(
            "INSERT INTO rating_skips(block_start,skipped_at) VALUES(?,?) "
            "ON CONFLICT(block_start) DO UPDATE SET skipped_at=excluded.skipped_at",
            (block_start, time.time()))


def blocks(items: list[tuple]) -> dict[float, tuple[float, float]]:
    """把样本按 30 分钟聚合成 {块起点: (有效时长, 投入时长)}。"""
    out: dict[float, tuple[float, float]] = {}
    for ts, dt, state, *_ in items:
        bs = (ts // BLOCK) * BLOCK
        active, engaged = out.get(bs, (0.0, 0.0))
        if state != "away":
            active += dt
        if state in focus.ENGAGED:
            engaged += dt
        out[bs] = (active, engaged)
    return out


def ratings_map() -> dict[float, dict]:
    with _conn() as conn:
        return {r[0]: {"score": r[1], "note": r[2] or "", "rated_at": r[3]}
                for r in conn.execute(
                    "SELECT block_start,score,note,rated_at FROM ratings")}


def save(block_start: float, score: int, note: str = "") -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO ratings(block_start,score,note,rated_at) VALUES(?,?,?,?) "
            "ON CONFLICT(block_start) DO UPDATE SET "
            "score=excluded.score,note=excluded.note,rated_at=excluded.rated_at",
            (block_start, int(score), note[:500], time.time()))


def pending(items: list[tuple], now: float | None = None) -> list[dict]:
    """待评分的块，最新的在前。只保留数据够、又还新鲜的。"""
    now = time.time() if now is None else now
    done = ratings_map()
    skipped = skips()
    out = []
    for bs, (active, _eng) in sorted(blocks(items).items(), reverse=True):
        if bs in done or bs in skipped or active < MIN_ACTIVE:
            continue
        if now - (bs + BLOCK) > RECENT:
            continue
        out.append({"start": bs, "end": bs + BLOCK, "active": active})
    return out


def is_ratable(items: list[tuple], block_start: float,
               now: float | None = None) -> bool:
    now = time.time() if now is None else now
    if now - (block_start + BLOCK) > RECENT:
        return False
    if block_start in ratings_map() or block_start in skips():
        return False
    return blocks(items).get(block_start, (0.0, 0.0))[0] >= MIN_ACTIVE


def paired(items: list[tuple]) -> list[tuple]:
    """把「自评分」和「实测投入率」配成对：(块起点, 自评分, 实测投入率)。

    这是整个验证的核心数据 —— 有它才能谈相关性。
    """
    agg = blocks(items)
    out = []
    for bs, r in sorted(ratings_map().items()):
        active, engaged = agg.get(bs, (0.0, 0.0))
        if active < MIN_ACTIVE:
            continue
        out.append((bs, r["score"], engaged / active))
    return out


def correlation(pairs: list[tuple]) -> float | None:
    """皮尔逊相关系数。样本不够就返回 None —— 别拿三个点算相关。"""
    if len(pairs) < MIN_PAIRS:
        return None
    xs = [p[1] for p in pairs]
    ys = [p[2] for p in pairs]
    n = len(pairs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx < 1e-9 or dy < 1e-9:
        return None          # 评分全一样，算不出相关
    return num / (dx * dy)


def verdict(r: float | None, n: int) -> str:
    """把相关系数翻译成人话。

    措辞刻意保守：n 刚过 MIN_PAIRS 时，即使 r 看起来很高，
    95% 置信区间也可能宽到跨过零点（n=30、r=0.6 时区间约 ±0.34）。
    所以结论里必须带一句"样本量"提醒 —— 否则用户会把一个
    统计上还站不住的数字当成定论，然后据此去"重新校准"阈值，
    反而把本来正常的判定改坏。
    """
    if r is None:
        return (f"还差一些样本（已配对 {n} 条，至少需要 {MIN_PAIRS} 条，"
                "而且自评分不能全一样）")

    # 样本量不足时的置信度提示。用 Fisher z 变换的粗略区间：
    # se ≈ 1/sqrt(n-3)，95% 半宽 ≈ 1.96/sqrt(n-3)。
    half = 1.96 / (n - 3) ** 0.5 if n > 4 else 1.0
    half = min(half, 1.0)
    loose = "；样本量偏少，这个结论还可能变" if half > 0.25 else ""

    if r >= 0.7:
        return f"强相关 —— 测出来的和你感受到的是一回事，数据可信{loose}"
    if r >= 0.4:
        return f"中等相关 —— 大方向对得上，但阈值还有调的空间{loose}"
    if r > -0.4:
        return "基本不相关 —— 测量结果和你的感受对不上，阈值需要重新校准"
    return "负相关 —— 这很反常，可能是某个判定反了，建议先查阈值"
