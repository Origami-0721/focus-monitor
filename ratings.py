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
# 少于这么多**有效样本**就不算相关系数。
#
# 为什么从 8 抬到 30：蒙特卡洛跑过一遍（3000 次，纯随机数据）——
# n=8 时 |r|≥0.4 的假阳性率 **30.8%**，|r|≥0.7 也有 5.2%。
# 也就是说八个点算出来的"中等相关"，三次里有一次是纯噪声。
# n=30 时同样的门槛降到约 3%，才配得上 verdict() 里那句"数据可信"。
# 这是产品的信誉问题：宁可告诉用户"样本还不够"，也不能拿噪声当结论。
#
# **比的是有效样本量（`effective_n`）而不是配对条数**：那个 3% 是"n 个独立
# 样本"下的数，降权之后 90 条低权重的配对并不等于 90 个独立样本。
MIN_PAIRS = 30

# 判断"档间方向对不对"时，一个自评档至少要几条配对。
#
# 只有一条配对的档，它的"档均值"就是那一条自己 —— 一条离群的评分（比如某次
# 自评 3 分、实测却只有 5%）就能把整段序列弄成"非单调"，于是报告会去说
# "你的判据反了，该调阈值"。**指错方向比不说话更糟**：用户会照着一句错话
# 把一套本来正确的判据改坏。所以档位不够时宁可返回"说不清"。
MIN_BUCKET = 2

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
    """待评分的块，最新的在前。只保留数据够、又还新鲜的。

    **块必须已经过完**（`now >= bs + BLOCK`）。原来没有这一条，而列表又是
    "最新的在前" —— 于是**正在进行的那半小时永远排在第一位**，用户打开评分页
    第一眼看到的就是它，顺手就打了分。

    实测就是这么发生的：90 条评分里 **22 条**是在块结束前打的（提前最多的
    18.9 分钟）。那种评分的对象是"半场"，而实测投入率是按**整块**算的，
    两者根本不是同一段时间 —— 混进相关系数里等于自己掺噪声，
    而且掺的方向还偏（块刚过半时人往往还在状态，评 5 分、实测也高，
    于是把 r 抬高）。

    页面顶部那句"完整记录满 30 分钟"说的本来就是这个意思，只是代码没做到。
    """
    now = time.time() if now is None else now
    done = ratings_map()
    skipped = skips()
    out = []
    for bs, (active, _eng) in sorted(blocks(items).items(), reverse=True):
        if bs in done or bs in skipped or active < MIN_ACTIVE:
            continue
        if now < bs + BLOCK:
            continue                 # 这半小时还没过完，先别问
        if now - (bs + BLOCK) > RECENT:
            continue
        out.append({"start": bs, "end": bs + BLOCK, "active": active})
    return out


def is_ratable(items: list[tuple], block_start: float,
               now: float | None = None) -> bool:
    now = time.time() if now is None else now
    if now < block_start + BLOCK:
        return False             # 块还没过完，没有可评的对象
    if now - (block_start + BLOCK) > RECENT:
        return False
    if block_start in ratings_map() or block_start in skips():
        return False
    return blocks(items).get(block_start, (0.0, 0.0))[0] >= MIN_ACTIVE


def _pair_weight(active: float, elapsed: float) -> float:
    """这个配对有多可信（0~1）。两项相乘，各自有依据 —— 不是拍脑袋的旋钮。

    1) **数据量。** 实测那一侧是 `engaged / active` 这个**比值**，它的方差近似
       `p(1-p)/active` —— 也就是"块里有效数据越多，这个比值越准"。按 `active`
       加权就是（近似）逆方差加权，和统计教科书里的做法一致。只有
       `MIN_ACTIVE`（10 分钟）的块拿三分之一权重，满 30 分钟的块才满权。

    2) **回忆延迟。** 自评分是事后补的，隔得越久越接近"编一个"。项目已经定了
       `RECENT`（3 小时）这条线，这里只是把它从"能不能打分"变成"这个分算多少
       数"：刚打完满权，拖到 3 小时边界降到 0。

    **为什么不改成"砍掉低权重的配对"。** 实测过：按"块内数据不足
    1.5×MIN_ACTIVE 就剔掉"砍一刀，r 从 0.271 **掉到 0.206**；改砍 2×MIN_ACTIVE
    又升回 0.324。同一类操作给出相反结果，说明"砍在哪一刀"没有依据，涨跌都是
    运气 —— 那种提升不能信。降权不引入这种断崖：它连续地把不可信的部分压小，
    而不是赌一个阈值。
    """
    w_data = min(1.0, active / BLOCK)
    w_recall = max(0.0, 1.0 - elapsed / RECENT)
    return w_data * w_recall


def effective_n(weights: list[float]) -> float:
    """有效样本量 `(Σw)² / Σw²`。

    降权之后"实际相当于多少条独立样本"—— 判断结论稳不稳该看这个数，而不是
    配对的条数。

    **它衡量的是权重有多不均匀，不是权重有多小。** 全部乘同一个常数不影响它：
    90 条各自 0.3，和 90 条各自 1.0，等效样本量都是 90（只是整体缩了 3 倍，
    而相关系数本来就与整体缩放无关）。真正拉低它的是**悬殊**：27 条满权 +
    63 条只有 0.01，等效出来是 28 条 —— 那 63 条基本只是陪衬。
    """
    sw = sum(weights)
    sw2 = sum(w * w for w in weights)
    return (sw * sw / sw2) if sw2 > 1e-12 else 0.0


def _pair_rows(items: list[tuple]) -> tuple[list[tuple], dict[str, list[float]]]:
    """一次遍历，分出「可用的配对」和「被舍弃的块（按原因归类）」。

    两个出口必须出自同一次判断。分成两趟各判一次的话，迟早会漂移成
    "报告说丢了 3 条、实际丢了 5 条"那种对不上的数 —— 而那种错没有任何报错，
    只会让人对报告里的每个数字都不敢信。

    原因分三类，因为**对用户的含义完全不同**（见 `dropped()`）。
    """
    agg = blocks(items)
    kept: list[tuple] = []
    drop: dict[str, list[float]] = {"early": [], "late": [], "thin": []}
    for bs, r in sorted(ratings_map().items()):
        active, engaged = agg.get(bs, (0.0, 0.0))
        if active < MIN_ACTIVE:
            drop["thin"].append(bs)
            continue
        elapsed = r["rated_at"] - (bs + BLOCK)
        if elapsed < 0:
            drop["early"].append(bs)
            continue
        if elapsed > RECENT:
            drop["late"].append(bs)
            continue
        kept.append((bs, r["score"], engaged / active,
                     _pair_weight(active, elapsed)))
    return kept, drop


def paired(items: list[tuple]) -> list[tuple]:
    """把「自评分」和「实测投入率」配成对：(块起点, 自评分, 实测投入率, 权重)。

    这是整个验证的核心数据 —— 有它才能谈相关性。

    **三条硬性舍弃**（前两条是项目本来就认的规矩，只是以前只在"能不能打分"
    那一侧拦，算的时候没拦；第三条是上面那个 `pending()` 缺陷留下的历史数据）：

    - 块内有效数据不足 `MIN_ACTIVE`（10 分钟）→ 实测值本身没有依据。
    - 打分时刻早于块结束 → **块还没过完就先评了**，人还没过完那半小时。
    - 打分时刻晚于「块结束 + `RECENT`」→ 拖过 3 小时才补的，回忆不出来只能
      瞎填。`RECENT` 从 8 小时收紧到 3 小时之后，老数据里就留着窗口外的
      评分，所以这一侧必须自己再拦一次。
    """
    return _pair_rows(items)[0]


def dropped(items: list[tuple]) -> dict[str, list[float]]:
    """没进对照的评分块，按原因分组：

    - `early`：评分时间早于时段结束 —— 评的是半场，实测按整块算。
    - `late`：补分太晚（超过 `RECENT`）。
    - `thin`：那个时段有效数据不足 `MIN_ACTIVE`。

    报告要把这些数印出来。用户打了 90 个分、报告只认 67 个，中间那 23 个
    必须有个交代 —— 否则就是**静默丢数据**，他会以为程序漏了、或者以为
    自己的评分没保存成功。
    """
    return _pair_rows(items)[1]


def wpearson(xs: list[float], ys: list[float],
             ws: list[float]) -> float | None:
    """加权皮尔逊相关系数，**不带任何门槛**。退化输入返回 None。

    单独抽出来是为了**只有一份公式**。`scripts/corr_robust.py` 要拿它做
    留一法和 bootstrap —— 那些场景**必须**能在样本不足时也算出数
    （"样本不够时这个数有多不稳"正是要测的东西），所以不能走
    `correlation()` 的 `MIN_PAIRS` 门槛。

    但"不能走门槛"不等于"该抄一份公式"：抄一份的话，以后改了加权口径，
    稳健性检验就会在**检验另一个数**，而且不会有任何报错 —— 它会一本正经地
    告诉你"报告上那个数很稳"，稳的是它自己那份公式。
    """
    sw = sum(ws)
    if sw < 1e-9 or len(xs) < 3:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    num = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    dx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs)) ** 0.5
    dy = sum(w * (y - my) ** 2 for w, y in zip(ws, ys)) ** 0.5
    if dx < 1e-9 or dy < 1e-9:
        return None          # 评分全一样，算不出相关
    return num / (dx * dy)


def correlation(pairs: list[tuple]) -> float | None:
    """**加权**皮尔逊相关系数。样本不够就返回 None —— 别拿三个点算相关。

    权重是 `paired()` 的第 4 项。不用等权的理由见 `_pair_weight`：块内只有
    10 分钟数据的配对、和拖了两小时才补的评分，都不该跟一个满 30 分钟、
    刚打完的分平起平坐。

    门槛看**有效样本量**（`n_eff < MIN_PAIRS` 就拒绝），不看条数 ——
    因为 `MIN_PAIRS = 30` 是拿蒙特卡洛标出来的**假阳性率**门槛，
    而假阳性率取决于有效样本量。
    """
    ws = [p[3] for p in pairs]
    if effective_n(ws) < MIN_PAIRS:
        return None
    return wpearson([p[1] for p in pairs], [p[2] for p in pairs], ws)


def measured_diag(pairs: list[tuple]) -> dict:
    """实测侧还剩多少区分度 —— 用来区分「判据定错了」和「单位太短、分不出来」。

    **为什么非要有这个函数。** r 偏低时报告原来只有一句建议："阈值需要重新调"。
    可 r 低有两个完全不同的成因，修法相反：

    - **判据错了**：自评高的人实测反而低（档间不单调）→ 该去查阈值。
    - **单位太短**：方向明明是对的，但**单个 30 分钟块**本身抖得厉害 →
      这时候去动阈值是**把一套正确的判据改坏**，越调越糟。

    实测量过（67 条配对）：档内池化 σ = 26.9%，档间差平均只有 15.2%，
    最高的两档（4 分 vs 5 分）只差 **0.9 个百分点** —— 档内抖动是档间差异的
    1.77 倍。而实测投入率本身的 σ 有 28.9%，**根本不缺区分度**（不是"饱和"）：
    缺的是"单块这个单位"的分辨率。同一个库按更长的窗口重聚，r 单调上升：
    30 分钟 0.285 → 3 小时 0.360 → 6 小时 0.547 → 整天 0.631。
    r 随聚合粒度单调上升，正是"单点噪声主导"的指纹。

    返回的键（空 dict = 档位不够，说不清）：

    - `n_buckets`：参与判断的档数（每档至少 `MIN_BUCKET` 条）。
    - `within`：**档内**加权池化 σ（同一自评分内部的实测波动）。
    - `gap`：相邻档之间差的绝对值的平均（想分辨的信号）。
    - `top_gap`：最高两档之差 —— 最常被拿来区分"还行 / 很好"。
    - `mono`：档均值是否随自评分**非递减**（方向对不对）。

    `within`/`gap` 都归一在 0~1 尺度上（调用方乘 100 当百分点印）。
    """
    acc: dict[int, list[tuple[float, float]]] = {}
    for _bs, sc, measured, w in pairs:
        acc.setdefault(sc, []).append((measured, w))
    st: dict[int, tuple[float, float, float]] = {}
    for sc, v in acc.items():
        # 只有一条配对的档，它的"均值"就是那一条自己 —— 一条离群的评分能把
        # 整段序列弄成"非单调"，然后报告就会去说"你的判据反了，去调阈值"。
        # 宁可说"看不出来"，也不能指错方向（那会让人把对的判据改坏）。
        if len(v) < MIN_BUCKET:
            continue
        sw = sum(w for _m, w in v)
        if sw <= 1e-12:
            continue
        mean = sum(m * w for m, w in v) / sw
        var = sum(w * (m - mean) ** 2 for m, w in v) / sw
        st[sc] = (mean, sw, var ** 0.5)
    if len(st) < 2:
        return {}
    ks = sorted(st)
    means = [st[k][0] for k in ks]
    gaps = [means[i + 1] - means[i] for i in range(len(means) - 1)]
    num = sum(st[k][1] * st[k][2] ** 2 for k in ks)
    den = sum(st[k][1] for k in ks)
    return {
        "n_buckets": len(ks),
        "within": (num / den) ** 0.5 if den > 1e-12 else 0.0,
        "gap": sum(abs(g) for g in gaps) / len(gaps),
        "top_gap": abs(gaps[-1]),
        "mono": all(g >= 0 for g in gaps),
    }


def diag_kind(diag: dict | None) -> str:
    """把诊断归成三类成因，`verdict()` / 横幅按它分叉措辞。

    - `"threshold"`：档间**不**单调 —— 自评高的人实测反而低，判据方向可疑，
      这种情况才该把人指去调阈值。
    - `"unit"`：档间单调（方向对），但档内抖动 ≥ 档间差 —— 是"单个半小时
      太短"，调阈值没用。
    - `""`：说不清（档位不够，或者方向对且档间差本来就够大）。
      **默认落到原来的措辞**，不新增结论 —— 诊断没把握时不要乱说话。
    """
    if not diag or diag.get("n_buckets", 0) < 2:
        return ""
    if not diag.get("mono"):
        return "threshold"
    if diag["within"] >= diag["gap"]:
        return "unit"
    return ""


def verdict(r: float | None, n: float, diag: dict | None = None) -> str:
    """把相关系数翻译成人话。

    `n` 是**有效样本量**（`effective_n()` 的结果，可能带小数），不是配对条数 ——
    降权之后 90 条只有 0.3 权重的配对，和 27 条满权样本一样虚，
    置信区间该按后者算。

    `diag` 是 `measured_diag()` 的结果（可选）。传了它，措辞就按**成因**分叉：
    r 低而档间单调时不再说"阈值需要重新调" —— 那种情况动阈值只会把一套正确的
    判据改坏，真正的原因是对照的单位太短（见 `measured_diag` 里那份实测）。
    不传（`None`）时行为和以前**逐字一致**，老调用点不受影响。

    措辞刻意保守：n 刚过 MIN_PAIRS 时，即使 r 看起来很高，
    95% 置信区间也可能宽到跨过零点（n=30、r=0.6 时区间约 ±0.34）。
    所以结论里必须带一句"样本量"提醒 —— 否则用户会把一个
    统计上还站不住的数字当成定论，然后跑去动阈值，
    反而把本来正常的判定改坏。

    措辞里**不许出现「校准」**：程序没有校准这个功能（EAR 基线自动学），
    用户读到"重新校准"会去找一个不存在的东西 —— 报告横幅那边踩过这个坑，
    实测反馈原话是「没找到校准交互」。要指路就指到"报告顶部的横幅"，
    具体改哪个设置项由横幅负责（它在调用方那边，就在本节上方）。
    """
    if r is None:
        return (f"还差一些样本（等效 {n:.0f} 条，至少需要 {MIN_PAIRS} 条，"
                "而且自评分不能全一样）")

    # 样本量不足时的置信度提示。用 Fisher z 变换的粗略区间：
    # se ≈ 1/sqrt(n-3)，95% 半宽 ≈ 1.96/sqrt(n-3)。
    half = 1.96 / (n - 3) ** 0.5 if n > 4 else 1.0
    half = min(half, 1.0)
    loose = "；样本量偏少，这个结论还可能变" if half > 0.25 else ""
    kind = diag_kind(diag)

    if r >= 0.7:
        return f"强相关 —— 测出来的和你感受到的是一回事，数据可信{loose}"
    if r >= 0.4:
        if kind == "unit":
            return ("中等相关 —— 大方向对得上；但单个半小时太短、抖动大，"
                    f"这个数被压低了，阈值不用动{loose}")
        return f"中等相关 —— 大方向对得上，但阈值还有调的空间{loose}"
    # 下面两条**才是真正会推着用户去动阈值**的结论，所以样本量提醒更不能漏。
    # 原来恰好只给上面两条"结论不错"的分支加了 loose，把这两条漏了 ——
    # 而 docstring 里写的就是"防止用户据此去重新校准阈值"，
    # 规则写了却没落到该落的地方。
    #
    # 也不写"重新校准"：那是把人指到一个不存在的功能（见 docstring）。
    # 这里只指"报告顶部的横幅"，横幅就在本节上方，位置是准的；
    # 设置项的名字由横幅那边管，不在第二个文件里再抄一遍。
    _WHERE = "（在哪调见报告顶部的横幅）"
    if r > -0.4:
        if kind == "unit":
            # 这是本轮新增的分支，也是原来最误导的那一种：r 低就一口咬定
            # "阈值需要重新调"。而逐档表明明显示方向是对的（自评越高、实测
            # 越高），成因是单块太短 —— 照原来那句话去调阈值，等于把一套
            # 正确的判据改坏。
            #
            # 这一支**排在样本量分支前面**：样本少时那句"再攒一些评分看看"
            # 在成因是"单位太短"时是误导 —— 再攒评分不会让 r 变大，只会让
            # 这个数更确定。所以先说清成因，样本量提醒降级成附带一句。
            return ("基本不相关 —— 但逐档看，你的自评越高、实测投入率也越高，"
                    "方向是对的；这个数偏低是「单个半小时太短」造成的："
                    "同一档内部的波动比档与档之间的差还大。阈值不用动，"
                    "下面那行诊断里有具体数字"
                    + ("；另外样本量也还偏少，这个数本身还会变。" if half > 0.25
                       else "。"))
        if half > 0.25:
            return ("基本不相关 —— 测量结果和你的感受对不上"
                    "；但样本量偏少，先别急着动阈值，再攒一些评分看看")
        return ("基本不相关 —— 测量结果和你的感受对不上"
                f"，阈值需要重新调{_WHERE}")
    if kind == "unit":
        return ("负相关 —— 这很反常，可能是某个判定反了"
                "；不过逐档看方向是对的，先按下面那行诊断核对一遍")
    return ("负相关 —— 这很反常，可能是某个判定反了"
            + ("；样本量偏少，这个结论还可能变" if half > 0.25
               else f"，建议先查阈值{_WHERE}"))
