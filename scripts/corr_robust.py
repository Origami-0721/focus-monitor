# -*- coding: utf-8 -*-
"""自述对照的相关系数，到底站不站得住 —— 只读分析工具。

**为什么需要它。** 报告上印着一个 r，但它只给点估计，不给"这个数有多稳"。
实测踩到过一次：把配对按越来越长的窗口重聚时，r 看起来"单调上升"
（30 分钟 +0.29 → 3 小时 +0.36 → 6 小时 +0.55 → 整天 +0.63），
差点据此得出"粒度越粗越准"的结论。补做三项检验之后发现
**整天那个 +0.63 是一天撑起来的**：

| 窗口 | n | Pearson | Spearman | 留一法 | bootstrap 95% |
|---|---|---|---|---|---|
| 30 分钟 | 67 | +0.294 | +0.336 | [+0.228, +0.358] | [+0.038, +0.541] |
| 6 小时 | 24 | +0.614 | +0.540 | [+0.478, +0.701] | [+0.114, +0.841] |
| 整天 | 13 | +0.740 | **+0.291** | [+0.512, +0.821] | **[−0.244, +0.935]** |

整天那行的 Pearson 和 Spearman 差了 **0.449** —— 秩相关几乎归零，说明
"相关"完全来自一两个极值点，而不是整体趋势。同一批数据里，
去掉 09-15 那一天，Pearson 从 +0.74 掉到 +0.20 附近。

**所以这个工具要回答的是：报告上那个数，去掉任意一个点会不会变天。**

三项检验各管一件事：

1. **Spearman vs Pearson**：差得多 ⇒ 少数离群点在主导（Pearson 对极值敏感）。
2. **留一法**：每次去掉一个点重算，看摆动幅度。摆动大 ⇒ 结论不稳。
3. **Bootstrap**：有放回重抽，看区间跨不跨 0。跨 0 ⇒ 连"正相关"都还说不准。

用法：

    uv run python scripts/corr_robust.py                 # 用默认的 focus.db
    uv run python scripts/corr_robust.py --db <路径>

**它只读。** 库先经 `sqlite3.backup()` 拷到临时目录再读 —— WAL 模式下直接复制
主文件会丢掉最近的写入，拿一份"看起来正常"的旧数据算出错误结论。
"""
from __future__ import annotations

import argparse
import random
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import focus  # noqa: E402
import ratings  # noqa: E402
import report  # noqa: E402

# 随机种子固定：这份输出会被贴进讨论里，别人重跑要能拿到同一串数。
SEED = 20261002
BOOT = 2000


def wpearson(xs, ys, ws):
    """加权皮尔逊 —— **直接借 `ratings.wpearson`**，不在这里抄一份公式。

    它不带样本量门槛，正是这里需要的：留一法和 bootstrap 都**必须**能在
    样本不足时也算出数（"样本不够时这个数有多不稳"就是要测的东西）。
    抄一份公式的话，以后改了加权口径，这个工具就会去**检验另一个数**，
    而且不会有任何报错 —— 它会一本正经地报告"报告上那个数很稳"。
    """
    return ratings.wpearson(xs, ys, ws)


def rank(v):
    """平均秩（处理并列）。"""
    order = sorted(range(len(v)), key=lambda i: v[i])
    out = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def spearman(xs, ys):
    return wpearson(rank(xs), rank(ys), [1.0] * len(xs))


def build(pairs, agg, W):
    """把配对按窗口 W 重聚成 (标签, x, y, 权重)。

    W == BLOCK 时**直接用 pairs** —— 报告上印的就是它，这里必须逐字复现，
    否则"检验报告上那个数"就变成了"检验另一个数"。
    """
    if W == ratings.BLOCK:
        return ([time.strftime("%m-%d %H:%M", time.localtime(p[0])) for p in pairs],
                [p[1] for p in pairs], [p[2] for p in pairs], [p[3] for p in pairs])
    win: dict[float, list[tuple[float, float]]] = {}
    for bs, sc, _y, _w in pairs:
        win.setdefault((bs // W) * W, []).append((sc, agg.get(bs, (0.0, 0.0))[0]))
    labels, xs, ys, ws = [], [], [], []
    for key in sorted(win):
        lst = win[key]
        sw = sum(a for _s, a in lst)
        if sw < ratings.MIN_ACTIVE:
            continue
        eng = act = 0.0
        for bs, _s, _y, _w in pairs:
            if (bs // W) * W == key:
                a, e = agg.get(bs, (0.0, 0.0))
                act += a
                eng += e
        if act <= 0:
            continue
        labels.append(time.strftime("%m-%d %H:%M", time.localtime(key)))
        xs.append(sum(s * a for s, a in lst) / sw)
        ys.append(eng / act)
        # 粗窗口里单块的降权已经没有意义（块被合并了），按窗口总有效时长加权。
        ws.append(act)
    return labels, xs, ys, ws


def analyse(xs, ys, ws, rnd):
    """三项检验，返回 (r, rho, 留一法区间, bootstrap 区间, ≤0 比例, 最敏感点)。"""
    n = len(xs)
    r = wpearson(xs, ys, ws)
    rho = spearman(xs, ys)
    loo = []
    for k in range(n):
        v = wpearson([x for i, x in enumerate(xs) if i != k],
                     [y for i, y in enumerate(ys) if i != k],
                     [w for i, w in enumerate(ws) if i != k])
        loo.append((v, k))
    lv = [v for v, _ in loo if v is not None]
    boot = []
    for _ in range(BOOT):
        idx = [rnd.randrange(n) for _ in range(n)]
        if len({*idx}) < 3:
            continue
        v = wpearson([xs[i] for i in idx], [ys[i] for i in idx],
                     [ws[i] for i in idx])
        if v is not None:
            boot.append(v)
    boot.sort()
    blo, bhi = boot[int(len(boot) * .025)], boot[int(len(boot) * .975)]
    neg = sum(1 for v in boot if v <= 0) / len(boot)
    swing = max((r - v for v, _ in loo if v is not None), default=0.0)
    worst = max((t for t in loo if t[0] is not None), key=lambda t: r - t[0],
                default=(None, None))
    return r, rho, (min(lv), max(lv)), (blo, bhi), neg, swing, worst[1]


def main() -> int:
    ap = argparse.ArgumentParser(description="自述对照相关系数的稳健性检验")
    ap.add_argument("--db", default=None, help="focus.db 路径（默认用项目里的）")
    args = ap.parse_args()

    focus.use_safe_console()
    src_path = Path(args.db) if args.db else Path(focus.DB_PATH)
    if not src_path.exists():
        print(f"找不到库：{src_path}")
        return 1

    tmp = Path(tempfile.mkdtemp(prefix="fmrob_"))
    dst = tmp / "focus.db"
    try:
        s = sqlite3.connect(f"file:{src_path}?mode=ro", uri=True)
        d = sqlite3.connect(dst)
        try:
            s.backup(d)          # 不能 copy2：WAL 里还有最近的写入
        finally:
            d.close()
            s.close()
        keep = focus.DB_PATH
        focus.DB_PATH = dst
        try:
            rows = report.load()
            items = report._timed(rows)
            pairs = ratings.paired(items)
            agg = ratings.blocks(items)

            if len(pairs) < 5:
                print(f"配对只有 {len(pairs)} 条，样本太少，检验没有意义。")
                return 1

            n_eff = ratings.effective_n([p[3] for p in pairs])
            print(f"库：{src_path}")
            print(f"配对 {len(pairs)} 条，有效样本量 {n_eff:.1f}"
                  f"（门槛 {ratings.MIN_PAIRS}）")
            print(f"报告上印的 r = {ratings.correlation(pairs)}")
            print()

            rnd = random.Random(SEED)
            print(f"{'窗口':>8} {'n':>4} {'r':>8} {'ρ':>8} {'|r-ρ|':>6} "
                  f"{'留一法区间':>19} {'bootstrap 95%':>19} {'≤0':>6} {'摆动':>7}")
            print("-" * 96)
            rowsout = []
            for label, W in (("30 分钟", ratings.BLOCK), ("1 小时", 3600),
                             ("3 小时", 10800), ("6 小时", 21600),
                             ("整天", 86400)):
                labels, xs, ys, ws = build(pairs, agg, W)
                if len(xs) < 5:
                    print(f"{label:>8} {len(xs):>4}   （点太少，跳过）")
                    continue
                r, rho, (lo, hi), (blo, bhi), neg, swing, _k = analyse(
                    xs, ys, ws, rnd)
                print(f"{label:>8} {len(xs):>4} {r:>+8.3f} {rho:>+8.3f} "
                      f"{abs(r - rho):>6.3f} {f'[{lo:+.3f},{hi:+.3f}]':>19} "
                      f"{f'[{blo:+.3f},{bhi:+.3f}]':>19} "
                      f"{neg * 100:>5.1f}% {swing * 100:>6.1f}")
                rowsout.append((label, len(xs), r, rho, lo, hi, blo, bhi, neg))

            print("\n怎么读这张表：")
            print("  |r-ρ| 大（≳0.15）  → Pearson 是被少数离群点撑起来的，"
                  "看 ρ 更靠谱")
            print("  留一法摆动大       → 去掉一个点就变天，这个数不该拿来做决定")
            print("  bootstrap 区间跨 0 → 连'正相关'都还说不准")
            print("  ⚠ n 少的那些行天然更容易跨 0 —— 那是样本量的问题，"
                  "不是相关不存在。别把'没测出来'读成'没有'。")
            bad = [t for t in rowsout if t[4] < 0 or t[8] > .05]
            if bad:
                print("\n⚠ 下面这些行没站住（区间跨 0 或 ≥5% 的重抽给出 r≤0）：")
                for t in bad:
                    print(f"    {t[0]}：n={t[1]}  r={t[2]:+.3f}  "
                          f"bootstrap [{t[6]:+.3f},{t[7]:+.3f}]  "
                          f"r≤0 占 {t[8] * 100:.1f}%")
                print("  这些数**不要**写进报告或拿去改阈值。")
            return 0
        finally:
            focus.DB_PATH = keep
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
