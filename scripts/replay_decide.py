"""拿真实历史数据重放**当前**的 `decide()`，看判据改动会把状态分布改成什么样。

**为什么需要它。** 自检只能证明"函数本身符合预期"，证明不了"换到你的数据上
会发生什么"。而走神/疲劳这类判据的阈值（`DISTRACT_SWITCH_RATE`、`YAW_TOL`、
`PITCH_TOL` …）一旦调错，症状是**静默的**：状态分布整体偏移，没有任何报错，
自检也全绿。实测就是这么被抓过一次 —— 旧的"按应用类型直接判走神"把 68.5%
的走神样本判错了（在 B 站看高数课被判成走神），而自检完全覆盖不到。

所以改 `decide()` 或那几个阈值之后，**跑一遍这个脚本**，对着数字确认：
- "由走神变回来"的那批，是不是他真正在学的东西；
- "变成走神"的那批（误伤检查），是不是本来就不该算走神。

**它只读。** 库是先用 `sqlite3` 的 `backup()` 拷到临时目录再读的 ——
WAL 模式下直接复制主文件会丢掉最近的写入，拿一份"看起来正常"的旧数据
算出错误结论。

用法：

    uv run python scripts/replay_decide.py --db <你的 focus.db>
    uv run python scripts/replay_decide.py            # 默认用本仓库的 focus.db
    uv run python scripts/replay_decide.py --db ... --top 60

改动前后对比：`git stash` 一下再跑一次，两份输出对着看。

**近似之处**（库里没存这些中间量，只能在重放时估）：
- `closed_for` 没有逐条落库 → 按 0 处理；原本是 `drowsy` 的行沿用原状态
  （那是 EAR 链路算出来的，这里重算不了）。
- `away_for` 没有逐条落库 → 用"连续 `present=0` 的时长"估。
- `idle` / `present` / `yaw` / `pitch` / `app` / `title` 都是库里真有的。
"""

from __future__ import annotations

import argparse
import collections
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import focus  # noqa: E402

STATES = ("focused", "neutral", "deskwork", "distracted", "drowsy", "away")


def _load_config(db: Path, explicit: Path | None) -> None:
    """载入配置：显式指定的优先，其次库旁边那份 `config.json`，再其次默认值。

    必须走这一步：阈值就在配置里，用默认值去重放一份调过阈值的库，
    得到的分布和用户实际看到的对不上。
    """
    path = explicit or (db.parent / "config.json")
    if not path.exists():
        print(f"（没有 {path.name}，用默认阈值）")
        return
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    focus.apply_config(cfg)
    print(f"已载入 {path}：{sorted(cfg)}")


def _copy_db(src: Path, dst: Path) -> None:
    """用 sqlite3 的 backup() 复制 —— 别用文件复制，WAL 下会静默丢最近的写入。"""
    for suf in ("", "-wal", "-shm"):
        Path(str(dst) + suf).unlink(missing_ok=True)
    s = sqlite3.connect(src)
    try:
        o = sqlite3.connect(dst)
        try:
            s.backup(o)
        finally:
            o.close()
    finally:
        s.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", type=Path, default=ROOT / "focus.db",
                    help="要重放的 focus.db（只读，会先拷一份到临时目录）")
    ap.add_argument("--config", type=Path, default=None,
                    help="config.json（默认找库旁边那份）")
    ap.add_argument("--top", type=int, default=30,
                    help="每类最多列多少条（默认 30）")
    args = ap.parse_args()

    focus.use_safe_console()

    if not args.db.exists():
        print(f"找不到库：{args.db}", file=sys.stderr)
        return 2

    # 先把数据读出来（顺便把"这个库还不能重放"这类问题挡在前面），
    # 再打配置信息 —— 否则在管道里 stdout 是缓冲的、stderr 不是，
    # 报错会跑到配置信息前面去，看着莫名其妙。
    with tempfile.TemporaryDirectory() as td:
        dst = Path(td) / "replay.db"
        _copy_db(args.db, dst)
        conn = sqlite3.connect(dst)
        try:
            try:
                rows = conn.execute(
                    "SELECT ts, state, app, title, yaw, pitch, present, idle "
                    "FROM samples ORDER BY ts").fetchall()
            except sqlite3.OperationalError as exc:
                print(f"这个库里没有 samples 表（{exc}）—— 还没真正采集过？",
                      file=sys.stderr)
                return 2
        finally:
            conn.close()

    if not rows:
        print("库里一条样本都没有。", file=sys.stderr)
        return 2

    _load_config(args.db, args.config)
    print(f"DISTRACT_SWITCH_RATE = {focus.DISTRACT_SWITCH_RATE}  "
          f"YAW_TOL = {focus.YAW_TOL}  PITCH_TOL = {focus.PITCH_TOL}  "
          f"DESK_PITCH_MAX = {focus.DESK_PITCH_MAX}")

    switch_at: collections.deque[float] = collections.deque()
    last_win: tuple[str, str] | None = None
    away_since: float | None = None

    matrix: collections.Counter = collections.Counter()
    new_dist: collections.Counter = collections.Counter()
    old_dist: collections.Counter = collections.Counter()
    changed: collections.Counter = collections.Counter()

    for ts, old, app, title, yaw, pitch, present, idle in rows:
        old_dist[old] += 1

        # 切换频率：最近 DISTRACT_SWITCH_WINDOW 秒里换了几次前台窗口。
        # 第一行不算"切换" —— 没有前一扇窗口可比。
        win = (app, title)
        if last_win is not None and win != last_win:
            switch_at.append(ts)
        last_win = win
        while switch_at and ts - switch_at[0] > focus.DISTRACT_SWITCH_WINDOW:
            switch_at.popleft()
        rate = len(switch_at) * (60.0 / focus.DISTRACT_SWITCH_WINDOW)

        if present:
            away_since = None
        elif away_since is None:
            away_since = ts
        away_for = (ts - away_since) if away_since else 0.0

        if old == "drowsy":
            new = "drowsy"          # 见文件头：这条重算不了，沿用
        else:
            new = focus.decide(
                face_present=bool(present), yaw=yaw or 0.0, pitch=pitch or 0.0,
                closed_for=0.0, idle_sec=idle or 0.0,
                app_kind=focus.classify_app(app or "", title or ""),
                away_for=away_for, switch_rate=rate)

        new_dist[new] += 1
        matrix[(old, new)] += 1
        if old != new:
            changed[(old, new, focus.classify_app(app or "", title or ""),
                     (title or "")[:44])] += 1

    total = len(rows)
    print(f"\n== 重放 {total} 条样本 ==")
    print(f"{'状态':<10}{'改前':>9}{'改后':>9}{'变化':>9}")
    for st in STATES:
        a, b = old_dist.get(st, 0), new_dist.get(st, 0)
        print(f"{st:<10}{a:>9}{b:>9}{b - a:>+9}")

    print("\n== 状态迁移（只列有变化的）==")
    for (o, n), cnt in matrix.most_common():
        if o != n:
            print(f"  {o:11s} → {n:11s} {cnt:7d}  ({100.0 * cnt / total:5.2f}%)")

    # 误伤检查：新判据**新判出来**的走神。这份清单应该越短越好，
    # 而且逐条看标题应该都"确实像在走神"。
    print(f"\n== 变成走神的（误伤检查，最多 {args.top} 条）==")
    shown = 0
    for (o, n, kind, t), cnt in changed.most_common():
        if n == "distracted" and o != "distracted":
            print(f"  {cnt:6d}  {o:11s} → {n:11s}  [{kind}] {t}")
            shown += 1
            if shown >= args.top:
                break
    if not shown:
        print("  （没有 —— 新判据一条新的走神都没判出来）")

    # 原来的受害者：以前被判走神、现在回来了。这批应该都是他真正在学的东西。
    print(f"\n== 由走神变回来的（原 bug 的受害者，最多 {args.top} 条）==")
    shown = 0
    for (o, n, kind, t), cnt in changed.most_common():
        if o == "distracted" and n != "distracted":
            print(f"  {cnt:6d}  {o:11s} → {n:11s}  [{kind}] {t}")
            shown += 1
            if shown >= args.top:
                break
    if not shown:
        print("  （没有）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
