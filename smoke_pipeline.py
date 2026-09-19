# -*- coding: utf-8 -*-
"""管线冒烟：采集循环的接线有没有真的接上。

`--selftest` 测的是纯逻辑（`decide()`、`ear_threshold()` 这些函数本身），
`smoke.py` 测的是渲染。两者都**测不出"没人喂数据给逻辑"**这一类故障。

为什么非要有这个文件：采集循环在一次重构里被删掉了 `vision.read_face` /
`vision.read_posture` 和四个缓冲区的填充，**消费端却留着**。后果是
`face_buf` 永远为空 → `present` 恒为 False → `decide()` 只会返回
distracted / away —— 专注、中性、伏案、疲劳四种状态一个都出不来，
而自检全绿。这个洞在 main 上活了好几轮才被发现。

所以这里用**假摄像头 + 假 Vision** 真跑一遍 `Monitor.run()`，然后查库里
到底出现了哪些状态。不需要摄像头、不需要下载模型、不联网。

两个场景：

  ① 恒定输入 —— 永远"看到一张正对屏幕的脸"，状态必须判成「专注」。
     盯的是"数据到底有没有流进 decide()"。

  ② 脚本化时间线 —— 睁眼 → 闭眼 → 转头出画 → 回来还闭着眼。
     盯的是"闭眼计时会不会跨过丢脸那段盲区接着累加"。会的话，人只是
     转了下头，回来 1 秒就被记一条「疲劳」（改之前实测能复现）。
     这一段同时验反面：一直闭着眼不丢脸时「疲劳」**必须**照样判得出来 ——
     修 bug 不能顺手把功能一起修没了。

  ③ 暂停 → 恢复 —— 暂停时循环在开头就 continue，根本走不到视觉那一段，
     所以"只在丢脸时复位"救不了它。恢复时那个陈旧的起点会直接命中。
     ②③ 一起才能钉住"复位要放在重新开始观察的那一侧"。

用法:
    python smoke_pipeline.py
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

import focus

# 场景①跑够 2 次结算（每秒一次）就行；留点余量给慢机器
DURATION = 4.0


class _FakeCap:
    """假摄像头：永远返回一帧纯色图。"""

    def read(self):
        return True, np.zeros((480, 640, 3), np.uint8)

    def release(self):
        pass

    def isOpened(self):
        return True


class _FakeVision:
    """假视觉：永远"看到一张正对屏幕的脸"，所以状态应该判成 专注。

    yaw=0 在容差内、pitch=5 小于 PITCH_TOL → 看着屏幕；
    ear=0.30 明显高于闭眼阈值 → 不算疲劳。
    """

    face_calls = 0
    pose_calls = 0

    def __init__(self):
        pass

    def read_face(self, frame):
        _FakeVision.face_calls += 1
        return {"yaw": 0.0, "pitch": 5.0, "ear": 0.30, "scale": 1.0}

    def read_posture(self, frame):
        _FakeVision.pose_calls += 1
        return {"tilt": 3.0, "hand_eye": False}


# ────────────────────── 场景②：脚本化时间线 ──────────────────────
#
# 时间轴（相对本段开始，单位秒）：
#
#   0.0 – 1.2   睁眼 ear=0.32   攒够基线样本，自适应阈值定在 0.32×0.67 ≈ 0.214
#   1.2 – 2.0   闭眼 ear=0.10   closed_since ≈ 1.2；最长只累计 0.8 秒，
#                               够不到 EAR_SUSTAIN，所以这段本身不该出「疲劳」
#   2.0 – 3.8   丢脸（返回 None）1.8 秒的盲区。长度 ≥ 一个结算窗口，
#                               所以**必定**盖住至少一个 present=False 的结算
#   3.8 – 8.0   回来，仍然闭眼   计时应该从 3.8 秒**重新开始**
#
# 判据：第一条「疲劳」必须落在"脸回来之后 ≥ 1.5 秒"。
#   有 bug（计时跨盲区累加）：closed_for 从 1.2 起算，回来的第一个结算窗口
#       就已经 ≥ 2.0 → 疲劳出现在回来之后 0 ~ 1.0 秒 → 断言失败。
#   修好后：closed_for 从 3.8 起算，最早 3.8+2.0 = 5.8 → 落在 2.0 秒之后。
#
# 为了让这一段别跑太久，把"攒够基线"和"闭眼多久算疲劳"都调小：
# 两个常量都是在函数里现读的模块级全局，所以改 focus 上的值就生效。
_OPEN_UNTIL = 1.2
_CLOSED_UNTIL = 2.0
_RETURN_AT = 3.8
_TIMELINE_DURATION = 8.0
_EAR_OPEN = 0.32
_EAR_CLOSED = 0.10

_T0 = 0.0          # 本段起点，main() 里在起线程之前赋值


class _ScriptedVision:
    """按 elapsed 演出"睁眼 → 闭眼 → 转头出画 → 回来还闭着眼"。"""

    return_ts: float | None = None    # 丢脸之后第一次重新看到脸的时刻

    def __init__(self):
        pass

    def read_face(self, frame):
        t = time.time() - _T0
        if t < _OPEN_UNTIL:
            ear = _EAR_OPEN
        elif t < _CLOSED_UNTIL:
            ear = _EAR_CLOSED
        elif t < _RETURN_AT:
            return None               # 转头出画：脸不在了
        else:
            if _ScriptedVision.return_ts is None:
                _ScriptedVision.return_ts = time.time()
            ear = _EAR_CLOSED         # 回来了，眼睛还是闭着
        return {"yaw": 0.0, "pitch": 5.0, "ear": ear, "scale": 1.0}

    def read_posture(self, frame):
        return {"tilt": 3.0, "hand_eye": False}


# ────────────────────── 场景③：暂停 → 恢复 ──────────────────────
#
# 暂停那条分支在循环**开头**就 `continue` 了，根本走不到视觉那一段 ——
# 所以"只在丢脸时复位"这种写法救不了它：暂停期间 closed_since 没人动，
# 恢复时 `closed_since or now` 直接沿用暂停前的起点，人一恢复就被记「疲劳」。
#
# 时间轴（相对本段开始，单位秒）：
#
#   0.0 – 1.2   睁眼          基线建立
#   1.2 – 2.0   闭眼          closed_since ≈ 1.2
#   2.0 – 4.0   **暂停**      摄像头已释放，循环在开头 continue
#   4.0 – 8.0   恢复，仍闭眼   计时应该从 4.0 之后**重新开始**
#
# 判据同场景②：第一条「疲劳」必须落在"恢复之后 ≥ 1.5 秒"。
#
# 为什么要单独测这一条：复位写在"丢脸"那一支也能让场景②变绿，
# 但暂停这条路径根本不经过那一支。两个场景一起才能钉住
# "复位要放在重新开始观察的那一侧"这个设计决定。
_PAUSE_AT = 2.0
_RESUME_AT = 4.0
_PAUSE_TIMELINE_DURATION = 8.0

_resume_ts = 0.0        # 实际解除暂停的时刻，main() 之外由 _resume() 写


class _ClosedAfterPauseVision:
    """睁眼一会儿 → 之后一直闭着眼（暂停期间它根本不会被调用）。"""

    def __init__(self):
        pass

    def read_face(self, frame):
        t = time.time() - _T0
        ear = _EAR_OPEN if t < _OPEN_UNTIL else _EAR_CLOSED
        return {"yaw": 0.0, "pitch": 5.0, "ear": ear, "scale": 1.0}

    def read_posture(self, frame):
        return {"tilt": 3.0, "hand_eye": False}


def _resume(mon) -> None:
    global _resume_ts
    mon.paused = False
    _resume_ts = time.time()


def _run(vision_cls, db: Path, duration: float, arm=None) -> None:
    """把外部依赖全部换掉（摄像头、模型、前台窗口、键鼠空闲）后真跑一遍采集循环。

    `arm(mon)` 用来在 run() 之前挂上定时器（暂停/恢复这类按时间轴驱动的动作）。
    """
    global _T0
    focus.DB_PATH = db
    focus.ensure_models = lambda on_progress=None: None
    focus.Vision = vision_cls
    focus.open_camera = lambda preferred=None: _FakeCap()
    focus.active_window = lambda: ("code.exe", "focus.py - Visual Studio Code")
    focus.idle_seconds = lambda: 0.0

    mon = focus.Monitor()
    # 时间轴的原点必须在这里取，**不能留在调用方**：调用方是在 `Monitor()`
    # 构造之前取的，而"跑多久"的计时器是从下面这一行才开始的 —— 两个原点
    # 之间隔着 Monitor 构造那一段。本机是几毫秒，CI 上偶尔要一两秒，正好吃掉
    # 场景②的余量（drowsy 最早 5.8 秒才出现，采集窗口只有 8.0 秒）→ 偶发红，
    # 而且红在"判不出疲劳"这种看起来像逻辑坏了的断言上，极难查。
    _T0 = time.time()
    if arm is not None:
        arm(mon)
    threading.Timer(duration, mon.stop).start()
    mon.run()


def _rows(db: Path) -> list[tuple]:
    conn = sqlite3.connect(db)
    try:
        return conn.execute(
            "SELECT ts, state, ear FROM samples ORDER BY ts").fetchall()
    finally:
        conn.close()


def _check_constant_input() -> str:
    """场景①：恒定喂"正对屏幕的脸"，必须产出「专注」。"""
    db = Path(tempfile.mkdtemp()) / "pipeline-smoke.db"
    _run(_FakeVision, db, DURATION)
    rows = _rows(db)

    dist: dict[str, int] = {}
    for _, st, _e in rows:
        dist[st] = dist.get(st, 0) + 1
    ears = sum(e or 0.0 for _, _, e in rows)

    # ① 视觉函数必须真的被调用过
    assert _FakeVision.face_calls > 0, (
        "采集循环从没调用过 read_face —— 视觉链路断了（生产端被删了、消费端还在）")
    assert _FakeVision.pose_calls > 0, "采集循环从没调用过 read_posture"
    # ② 必须真的落了样本
    assert len(rows) >= 2, f"只落了 {len(rows)} 条样本，采集循环没正常跑起来"
    # ③ 关键：正对屏幕 + 工作应用，必须产出「专注」，不能只有走神/离开
    assert "focused" in dist, (
        f"喂了正对屏幕的人脸却判不出「专注」，实际状态分布：{dist}"
        "（视觉数据没有流进 decide()）")
    # ④ EAR 必须真的写进库了，否则是"读到了但没存"
    assert ears > 0, f"落库的 ear 合计为 {ears}，说明视觉结果没传到写库那一步"

    return (f"read_face {_FakeVision.face_calls} 次 / "
            f"read_posture {_FakeVision.pose_calls} 次，"
            f"{len(rows)} 条样本，状态 {dist}")


def _check_face_loss_timeline() -> str:
    """场景②：丢脸那段盲区不该让闭眼计时接着累加，但真疲劳仍要判得出来。"""
    global _T0
    _ScriptedVision.return_ts = None

    saved = focus.EAR_MIN_SAMPLES, focus.EAR_SUSTAIN
    focus.EAR_MIN_SAMPLES = 8     # 0.8 秒就能把基线攒出来
    focus.EAR_SUSTAIN = 2.0
    try:
        db = Path(tempfile.mkdtemp()) / "pipeline-timeline.db"
        _T0 = 0.0                 # 真值由 _run() 在起线程前取（见那里的注释）
        _run(_ScriptedVision, db, _TIMELINE_DURATION)
        rows = _rows(db)
    finally:
        focus.EAR_MIN_SAMPLES, focus.EAR_SUSTAIN = saved

    drowsy = [ts for ts, st, _e in rows if st == "drowsy"]
    assert _ScriptedVision.return_ts is not None, (
        "脚本化的时间线没跑到「回来」那一段 —— 采集循环可能提前停了"
        f"（只落了 {len(rows)} 条样本）")
    assert drowsy, (
        f"一直闭着眼（3.8 秒起）却始终判不出「疲劳」，实际状态分布 "
        f"{sorted({st for _t, st, _e in rows})} —— 丢脸作废的逻辑把真疲劳也清掉了")

    offset = drowsy[0] - _ScriptedVision.return_ts
    assert offset >= 1.5, (
        f"脸回来之后 {offset:.1f} 秒就记了「疲劳」：闭眼计时跨过了丢脸那段盲区"
        f"接着累加（人只是转了下头，眼睛只被观察到闭了 0.8 秒）")

    return (f"{len(rows)} 条样本，{len(drowsy)} 条疲劳，"
            f"第一条在脸回来之后 {offset:.1f} 秒")


def _check_resume_after_pause() -> str:
    """场景③：暂停再恢复之后，闭眼计时不该拿着暂停前的起点接着算。"""
    global _T0, _resume_ts
    saved = focus.EAR_MIN_SAMPLES, focus.EAR_SUSTAIN
    focus.EAR_MIN_SAMPLES = 8
    focus.EAR_SUSTAIN = 2.0
    try:
        db = Path(tempfile.mkdtemp()) / "pipeline-pause.db"
        _resume_ts = 0.0
        _T0 = 0.0                 # 真值由 _run() 在起线程前取（见那里的注释）

        def arm(mon):
            threading.Timer(_PAUSE_AT,
                            lambda: setattr(mon, "paused", True)).start()
            threading.Timer(_RESUME_AT, lambda: _resume(mon)).start()

        _run(_ClosedAfterPauseVision, db, _PAUSE_TIMELINE_DURATION, arm=arm)
        rows = _rows(db)
    finally:
        focus.EAR_MIN_SAMPLES, focus.EAR_SUSTAIN = saved

    drowsy = [ts for ts, st, _e in rows if st == "drowsy"]
    assert _resume_ts > 0, (
        f"时间线没跑到「恢复」那一段（只落了 {len(rows)} 条样本）")
    assert drowsy, (
        f"恢复之后一直闭着眼却判不出「疲劳」，实际状态分布 "
        f"{sorted({st for _t, st, _e in rows})}，只落了 {len(rows)} 条样本")

    # _resume_ts 记的是"解除暂停"那一刻，而循环最多晚 0.5 秒（暂停时的 sleep）
    # 才真正回到采集 —— 这个偏差只会让 offset 更大，所以是安全方向。
    offset = drowsy[0] - _resume_ts
    assert offset >= 1.5, (
        f"恢复之后 {offset:.1f} 秒就记了「疲劳」：闭眼计时沿用了暂停前的起点"
        f"（人一恢复就被判疲劳）")

    return (f"{len(rows)} 条样本，{len(drowsy)} 条疲劳，"
            f"第一条在恢复之后 {offset:.1f} 秒")


def main() -> int:
    focus.use_safe_console()

    summary1 = _check_constant_input()
    summary2 = _check_face_loss_timeline()
    summary3 = _check_resume_after_pause()

    print(f"管线冒烟通过\n"
          f"  ① 恒定输入：{summary1}\n"
          f"  ② 丢脸时间线：{summary2}\n"
          f"  ③ 暂停恢复：{summary3}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
