"""复现「自评分打不开」的真正机制：delete 模式下**写者被读者饿死**。

用户日志里的那一行是死因，不是症状：

    File "focus.py", line 764, in run
      conn.commit()
    sqlite3.OperationalError: database is locked

注意死的是**采集线程的 commit**，不是面板的读。delete 模式下：
  · 读事务持 SHARED，写事务持 RESERVED，两者**可以共存**；
  · 但 commit 要把 RESERVED 升级成 EXCLUSIVE，而 EXCLUSIVE 要求
    **没有任何 SHARED 读者**。
所以只要面板侧有人在长读（`/report` 对 10MB 库做全表聚合、`_pending_ratings`
每几秒 load 一次两天窗口），采集线程的 commit 就会被挡在 busy_timeout 之外
→ 抛异常 → 线程死 → 连接没关、写事务还开着 → **之后每一条读都撞同一个锁**，
持续 1 小时 49 分，而进程还活着、桌面图标也还是绿的。

这个脚本把那条时间线原样跑两遍（delete / WAL），证明：
  ① delete：写者提交失败，就是用户遇到的那个异常；
  ② WAL：同一时间线写者提交成功 —— 读不阻塞写，这才是根治。
"""
from __future__ import annotations

import multiprocessing as mp
import sqlite3
import tempfile
import time
from pathlib import Path

HOLD = 8.0          # 读事务持有多久（模拟 /report 对 10MB 库做全表聚合）
BUSY = 5.0          # 采集线程的 busy_timeout（老代码就是 5 秒）


def reader(path: str, holding, done) -> None:
    """面板侧的长读（report.load 全表聚合）。"""
    conn = sqlite3.connect(path, timeout=30.0)
    try:
        conn.execute("BEGIN")
        conn.execute("SELECT count(*), sum(ts) FROM samples").fetchone()
        holding.set()                       # 此刻起，读事务持有 SHARED
        time.sleep(HOLD)
        conn.commit()
    finally:
        conn.close()
    done.set()


def writer(path: str, holding, out) -> None:
    """采集线程的一次攒批提交。"""
    conn = sqlite3.connect(path, timeout=BUSY)
    conn.execute(f"PRAGMA busy_timeout={int(BUSY * 1000)}")
    holding.wait(10)
    t0 = time.time()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("INSERT INTO samples VALUES(?,?)", (time.time(), "focused"))
        conn.commit()                       # ← 用户日志里就是死在这一行
        out.put(f"成功（{time.time() - t0:.1f}s）")
    except Exception as exc:                # noqa: BLE001  —— 就是要看它怎么失败
        out.put(f"失败 {type(exc).__name__}: {exc}（{time.time() - t0:.1f}s）")
    finally:
        conn.close()


def scenario(mode: str) -> None:
    with tempfile.TemporaryDirectory() as td:
        db = str(Path(td) / "focus.db")
        c = sqlite3.connect(db)
        c.execute(f"PRAGMA journal_mode={mode}")
        c.execute("CREATE TABLE samples(ts REAL, state TEXT)")
        c.executemany("INSERT INTO samples VALUES(?,?)",
                      [(time.time(), "focused")] * 20000)
        c.commit()
        got = c.execute("PRAGMA journal_mode").fetchone()[0]
        c.close()

        holding, done = mp.Event(), mp.Event()
        out: mp.Queue = mp.Queue()
        rp = mp.Process(target=reader, args=(db, holding, done))
        wp = mp.Process(target=writer, args=(db, holding, out))
        rp.start()
        wp.start()
        wp.join(90)
        rp.join(90)
        print(f"  journal_mode={got!r:8}  采集线程提交: {out.get()}")


def main() -> int:
    print(f"同一条时间线：面板正在长读（{HOLD:.0f}s），采集线程此时提交一批")
    for mode in ("delete", "WAL"):
        scenario(mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
