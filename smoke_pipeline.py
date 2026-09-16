# -*- coding: utf-8 -*-
"""管线冒烟：采集循环的视觉链路有没有真的接上。

`--selftest` 测的是纯逻辑（`decide()`、`ear_threshold()` 这些函数本身），
`smoke.py` 测的是渲染。两者都**测不出"没人喂数据给逻辑"**这一类故障。

为什么非要有这个文件：采集循环在一次重构里被删掉了 `vision.read_face` /
`vision.read_posture` 和四个缓冲区的填充，**消费端却留着**。后果是
`face_buf` 永远为空 → `present` 恒为 False → `decide()` 只会返回
distracted / away —— 专注、中性、伏案、疲劳四种状态一个都出不来，
而自检全绿。这个洞在 main 上活了好几轮才被发现。

所以这里用**假摄像头 + 假 Vision** 真跑一遍 `Monitor.run()`，然后查库里
到底出现了哪些状态。不需要摄像头、不需要下载模型、不联网。

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

# 跑够 2 次结算（每秒一次）就行；留点余量给慢机器
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
        return 3.0


def main() -> int:
    focus.use_safe_console()

    db = Path(tempfile.mkdtemp()) / "pipeline-smoke.db"
    # 把外部依赖全部换掉：摄像头、模型、前台窗口、键鼠空闲
    focus.DB_PATH = db
    focus.ensure_models = lambda on_progress=None: None
    focus.Vision = _FakeVision
    focus.open_camera = lambda preferred=None: _FakeCap()
    focus.active_window = lambda: ("code.exe", "focus.py - Visual Studio Code")
    focus.idle_seconds = lambda: 0.0

    mon = focus.Monitor()
    threading.Timer(DURATION, mon.stop).start()
    mon.run()

    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            "SELECT state, COUNT(*), SUM(ear) FROM samples GROUP BY state").fetchall()
        total = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    finally:
        conn.close()

    dist = {st: n for st, n, _ in rows}
    ears = sum(e or 0.0 for _, _, e in rows)

    # ① 视觉函数必须真的被调用过
    assert _FakeVision.face_calls > 0, (
        "采集循环从没调用过 read_face —— 视觉链路断了（生产端被删了、消费端还在）")
    assert _FakeVision.pose_calls > 0, "采集循环从没调用过 read_posture"
    # ② 必须真的落了样本
    assert total >= 2, f"只落了 {total} 条样本，采集循环没正常跑起来"
    # ③ 关键：正对屏幕 + 工作应用，必须产出「专注」，不能只有走神/离开
    assert "focused" in dist, (
        f"喂了正对屏幕的人脸却判不出「专注」，实际状态分布：{dist}"
        "（视觉数据没有流进 decide()）")
    # ④ EAR 必须真的写进库了，否则是"读到了但没存"
    assert ears > 0, f"落库的 ear 合计为 {ears}，说明视觉结果没传到写库那一步"

    print(f"管线冒烟通过 - read_face {_FakeVision.face_calls} 次 / "
          f"read_posture {_FakeVision.pose_calls} 次，"
          f"{total} 条样本，状态 {dist}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
