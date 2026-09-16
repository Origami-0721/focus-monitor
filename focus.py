"""专注度监视器 —— 前置摄像头 + 屏幕使用，双路判定，后台托盘常驻。

用法:
    uv run focus.py                    # 开启监视（托盘图标，退出用托盘菜单）
    uv run focus.py --install-startup  # 装开机自启，之后不用手动开
    uv run focus.py --uninstall-startup
    uv run focus.py --toggle           # 切换开/关（桌面开关快捷方式调的就是它）
    uv run focus.py --install-shortcut # 在桌面建一个开/关切换快捷方式
    uv run focus.py --stop             # 停掉正在运行的实例
    uv run focus.py --dashboard        # 实时面板（本地网页，每 3 秒自动刷新）
    uv run focus.py --report           # 生成 HTML 报告并打开
    uv run focus.py --selftest         # 跑自检，不开摄像头
    uv run focus.py --camera 1         # 指定摄像头序号

长期记录: 装上开机自启后就不用管了。脚本会一直跑，人离开或电脑待机都不影响 ——
"会话"是在报告里按数据空档和长时间离开自动切出来的，不需要程序自己开关。
同一时间只允许一个实例（重复启动会被单实例锁挡掉）。

隐私: 全程本地处理，不保存任何图像、不联网上传。磁盘上只有 focus.db 里的
数值指标和窗口标题。

判定的三个信号:
  1. 头部朝向  —— FaceLandmarker + solvePnP 解出偏航/俯仰角，判断有没有看屏幕
  2. 眼睛闭合  —— 眼睛纵横比 EAR，持续闭眼判为疲劳
  3. 坐姿      —— PoseLandmarker 肩线倾角 + 人脸尺度（离屏幕距离）

已知局限: 前置摄像头从正面拍，看不到驼背的侧面轮廓。所以坐姿用的是
"肩线倾斜 + 是否越凑越近"，不是真正的脊柱弯曲检测。别把它当医学结论。
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import json
import logging
import math
import os
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque
from logging.handlers import RotatingFileHandler
from pathlib import Path

import cv2
import numpy as np

# 直接 `python focus.py` 跑时本模块叫 __main__，而 report / dashboard 里的
# `import focus` 会把同一个文件再加载一遍，产生第二个模块对象 —— 于是
# apply_config 改的是 __main__，报告读的是另一个副本，设置根本不共享。
# 这个别名让两边指向同一个模块对象（正常 import 时 sys.modules 里已有，无事发生）。
sys.modules.setdefault("focus", sys.modules[__name__])

# 打包成 exe 后 __file__ 指向临时解压目录，文件会写进 _MEI 缓存、一退出就没。
# frozen 时以 exe 所在目录为根：focus.db / 日志 / 报告 / 模型都落在用户看得见的地方。
if getattr(sys, "frozen", False):
    ROOT = Path(sys.executable).resolve().parent
else:
    ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
DB_PATH = ROOT / "focus.db"
LOG_PATH = ROOT / "focus.log"
PID_PATH = ROOT / "focus.pid"      # 运行中实例的 PID，供 --stop 用

log = logging.getLogger("focus")


def setup_log(verbose: bool = False) -> None:
    """pythonw 启动时没有控制台，出什么事都看不见 —— 必须落盘。

    日志滚动保留 3 份 × 1MB，长期挂着也不会把磁盘写满。
    """
    handlers: list[logging.Handler] = [
        RotatingFileHandler(LOG_PATH, maxBytes=1_000_000, backupCount=3,
                            encoding="utf-8")]
    if verbose:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=logging.INFO, handlers=handlers,
                        format="%(asctime)s %(levelname)s %(message)s",
                        force=True)
    # 采集线程崩了默认是静默的：栈打到 stderr，而 pythonw 根本没有 stderr。
    # 挂个全局钩子把它写进日志 —— 这是排查"绿点为什么没了"的唯一线索。
    def _hook(args) -> None:
        log.error("线程 %s 异常退出", getattr(args.thread, "name", "?"),
                  exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

    threading.excepthook = _hook


def use_safe_console() -> None:
    """让控制台输出在任何代码页下都能显示，且永不抛 UnicodeEncodeError。

    中文提示在中文 Windows（GBK 控制台）上一直没问题，所以这个坑藏了很久。
    它是在 GitHub Actions 的 Windows runner 上才暴露的：那里是西文代码页，
    `print("自检通过")` 直接抛 UnicodeEncodeError —— 一次断言全过的自检被判
    成红色失败。更麻烦的是它崩在**最后一行**，前面输出看着一切正常，光看日志
    根本猜不到是编码问题。

    做法分两步，都是为了"本地照旧、CI 可读"：
      1. 当前代码页表示得了中文 → 原样不动，本地 GBK 控制台的显示不受影响；
         表示不了（cp1252/cp437 这类西文页）→ 切到 UTF-8。CI 的日志查看器
         按 UTF-8 渲染，于是断言失败时那句中文提示才真的能看懂。
      2. 一律加上 errors="replace" 兜底：编不出来的字符退化成 '?'，
         但绝不再把一次成功的运行变成异常退出。
    """
    probe = "自检"
    for stream in (sys.stdout, sys.stderr):
        if stream is None:              # pythonw 启动时没有控制台
            continue
        enc = getattr(stream, "encoding", None)
        if enc:
            try:
                probe.encode(enc)
            except (UnicodeEncodeError, LookupError):
                enc = "utf-8"           # 西文代码页 → 换成 UTF-8
        else:
            enc = "utf-8"
        try:
            stream.reconfigure(encoding=enc, errors="replace")
        except (AttributeError, ValueError, OSError):
            # 流被替换成不支持重配置的对象（测试替身、某些重定向），忽略即可。
            pass

MODEL_URLS = {
    "face_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task",
    "pose_landmarker_lite.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}

# ───────────────────────── 调参区 ─────────────────────────
# 硬件和环境差异全部在这几个数上调，别去改下面的判定逻辑。
FACE_FPS = 10          # 人脸检测频率，越高越吃 CPU
POSE_FPS = 2           # 姿态检测频率，动作慢，2Hz 足够
PROC_WIDTH = 480       # 送进模型的画面宽度，越小越快

YAW_TOL = 25.0         # 头部左右偏超过这个角度 → 没在看屏幕
# 俯仰阈值比偏航宽：笔记本摄像头在屏幕上方，正常看屏幕时俯仰角本就不是 0，
# 实测中位数 15.8°。卡 20° 会把正常注视判成走神。
PITCH_TOL = 30.0
# 比 PITCH_TOL 低、又不低于这个角度 → 判为"伏案"（低头对桌子，看书 / 写作业）。
# 再低就是脸快贴桌上了，多半是低头玩手机，判走神。这个界限是估的，
# 实测你伏案时的 pitch 值，按实际分布调。
DESK_PITCH_MAX = 65.0
# 闭眼阈值不再写死，改成"本人睁眼基线的比例" —— 见 ear_threshold()。
# EAR_CLOSED 退化成校准样本不足时的出厂兜底值。
EAR_CLOSED = 0.19      # 出厂兜底：基线还没建立起来时用这个
EAR_BASELINE_PCT = 75  # 用近期 EAR 的这个百分位当"睁眼基线"（抗犯困拖塌）
EAR_RATIO = 0.67       # 闭眼阈值 = 睁眼基线 × 这个比例
EAR_MIN_SAMPLES = 60   # 基线样本少于此数就退回出厂值
EAR_SUSTAIN = 3.0      # 持续闭眼多少秒算疲劳
AWAY_FACE = 20.0       # 人脸消失多少秒算离开
AWAY_IDLE = 180.0      # 键鼠无操作多少秒算离开
TILT_WARN = 12.0       # 肩线倾斜超过多少度算坐姿不良
AWAY_WRITE_EVERY = 30.0  # 离开期间每 30 秒才落一条，否则整夜待机能把数据库撑爆
# 攒多少条样本才 commit 一次。每条单独 commit 在 journal_mode=delete 下
# 等于每条一次 fsync 级写盘 + 一次持写锁，读数进程（面板每 3 秒轮询）
# 撞锁的概率被放大了十几倍。攒批后写入次数降两个量级，
# 代价是崩溃时最多丢这么多条样本（每条约 2 秒，即最多丢几十秒）。
SAMPLE_COMMIT_EVERY = 15
# 连续投入超过这么久就别弹评分提醒 —— 在人正专注的时候打断是本末倒置，
# 等自然断点（走神 / 离开 / 疲劳）再问。
FLOW_QUIET = 300.0
# 连续投入时最长闭嘴多久。到点即使仍在心流里也放行一次提醒，
# 否则"一整天不间断专注"的人永远收不到评分样本（见 should_prompt_rating）。
PROMPT_CEILING = 2 * 3600.0

WORK_APPS = {
    "code.exe", "code - insiders.exe", "cursor.exe", "devenv.exe", "idea64.exe",
    "pycharm64.exe", "sublime_text.exe", "notepad++.exe", "notepad.exe",
    "windowsterminal.exe", "powershell.exe", "cmd.exe", "wt.exe", "git-bash.exe",
    "python.exe", "pythonw.exe", "wps.exe", "winword.exe", "excel.exe",
    "powerpnt.exe", "acrobat.exe", "sumatrapdf.exe", "obsidian.exe", "notion.exe",
    "typora.exe", "onedrive.exe", "explorer.exe", "mstsc.exe", "xmind.exe",
}
# 这两个列表是日常最需要维护的东西：没被列进去的应用一律算"其他"（中性），
# 既不会算专注也不会算走神。实测踩过的坑：游戏 exe 名（如 zenlesszonezero.exe）
# 和中文标题都不匹配任何默认关键词，会被静默归到"中性"。
# 分类在采集时就写入数据库了，改完这里要重新记录才生效。
DISTRACT_KEYWORDS = (
    "抖音", "哔哩哔哩", "bilibili", "youtube", "微博", "weibo", "小红书",
    "知乎", "淘宝", "京东", "爱奇艺", "腾讯视频", "优酷", "直播", "游戏",
    "steam", "原神", "genshin", "绝区零", "zenless", "崩坏", "星穹铁道",
    "王者荣耀", "英雄联盟", "漫画", "小说", "贴吧", "虎扑",
    "netflix", "twitch", "reddit", "instagram", "tiktok",
)
# 学习豁免表：只在"本来要判分心"时才启用（见 classify_app）。
# 目的是救回"在 B 站看 C++ 课"这种情况 —— 平台是娱乐的，内容不是。
# 别加"第""讲"这种单字，太泛，"【第5期】游戏实况"会被误救。
STUDY_KEYWORDS = (
    "c++", "cpp", "python", "java", "javascript", "typescript", "rust",
    "golang", "kotlin", "swift", "sql", "linux", "docker", "git", "leetcode",
    "教程", "课程", "公开课", "网课", "mooc", "lecture", "tutorial",
    "算法", "数据结构", "编译原理", "操作系统", "计算机网络", "计网",
    "考研", "习题", "作业", "复习", "论文", "答辩", "文献",
    "documentation", "文档", "手册", "api 参考", "网课笔记",
)

STATES = {
    "focused":    ("专注", "#22c55e"),
    "neutral":    ("中性", "#38bdf8"),
    "deskwork":   ("伏案", "#14b8a6"),
    "distracted": ("走神", "#f59e0b"),
    "drowsy":     ("疲劳", "#a855f7"),
    "away":       ("离开", "#64748b"),
}

# 伏案（低头看书 / 写作业）算不算"投入时长"。算，因为它确实是你想要的产出时间；
# 但前置摄像头分不清"低头看书"和"低头玩手机"，所以这个开关留给你 ——
# 哪天发现伏案时长虚高，把这里改成 False，伏案就只统计不计入专注率。
DESKWORK_IS_ENGAGED = True
ENGAGED = frozenset({"focused", "neutral", "deskwork"} if DESKWORK_IS_ENGAGED
                    else {"focused", "neutral"})

# ───────────────── 配置覆盖（config.json）─────────────────
# 上面这些常量就是默认值。config.json 里出现的键会覆盖它们。
# 改完不需要重启进程：监视循环、报告、实时面板都会定期调用
# maybe_reload_config()，靠文件 mtime 判断有没有变。
CONFIG_PATH = ROOT / "config.json"

# 版本号。这里和 pyproject.toml 的 version 必须一致（发布清单里有一步专门核对）。
# 冻结成 exe 后，bug 报告唯一能问到的版本信息就是它。
__version__ = "0.2.2"

_SCALARS: dict[str, type] = {
    "FACE_FPS": int, "POSE_FPS": int, "PROC_WIDTH": int,
    "YAW_TOL": float, "PITCH_TOL": float, "DESK_PITCH_MAX": float,
    "EAR_CLOSED": float, "EAR_SUSTAIN": float,
    "AWAY_FACE": float, "AWAY_IDLE": float,
    "TILT_WARN": float, "AWAY_WRITE_EVERY": float,
}
_LIST_KEYS = ("WORK_APPS", "DISTRACT_KEYWORDS", "STUDY_KEYWORDS")

# 导入时的快照，供"恢复默认"用 —— apply_config 之后 globals() 就不是默认值了
_DEFAULTS: dict = {k: globals()[k] for k in _SCALARS}
_DEFAULTS.update({k: globals()[k] for k in _LIST_KEYS})
_DEFAULTS["DESKWORK_IS_ENGAGED"] = DESKWORK_IS_ENGAGED


def current_config() -> dict:
    """当前生效的配置，用于渲染设置页。"""
    cfg = {k: globals()[k] for k in _SCALARS}
    cfg.update({k: sorted(globals()[k]) for k in _LIST_KEYS})
    cfg["DESKWORK_IS_ENGAGED"] = DESKWORK_IS_ENGAGED
    return cfg


def validate_config(cfg: dict) -> list[str]:
    """返回错误列表。

    这些值配错了不会报错，只会让行为变得莫名其妙（比如俯仰阈值超过伏案上限，
    就永远不会出现"伏案"状态）。它们是用户输入，属于信任边界，必须挡一道。
    """
    errs = []
    if cfg["PITCH_TOL"] >= cfg["DESK_PITCH_MAX"]:
        errs.append("俯仰阈值必须小于伏案上限，否则永远不会判出「伏案」")
    if not 0 < cfg["EAR_CLOSED"] < 1:
        errs.append("闭眼 EAR 阈值必须在 0 和 1 之间")
    if cfg["AWAY_FACE"] >= AWAY_IDLE:
        errs.append("「人脸消失判离开」的秒数应该小于「键鼠空闲判离开」的秒数")
    for k in ("FACE_FPS", "POSE_FPS", "PROC_WIDTH"):
        if cfg[k] <= 0:
            errs.append(f"{k} 必须为正数")
    if cfg["EAR_SUSTAIN"] <= 0 or cfg["AWAY_FACE"] <= 0:
        errs.append("疲劳/离开的持续秒数必须为正数")
    return errs


def apply_config(cfg: dict) -> None:
    global DESKWORK_IS_ENGAGED, ENGAGED
    for name, caster in _SCALARS.items():
        if name in cfg:
            globals()[name] = caster(cfg[name])
    for name in _LIST_KEYS:
        if name in cfg:
            vals = [str(v).strip() for v in cfg[name] if str(v).strip()]
            cur = globals()[name]
            globals()[name] = set(vals) if isinstance(cur, set) else tuple(vals)
    if "DESKWORK_IS_ENGAGED" in cfg:
        DESKWORK_IS_ENGAGED = bool(cfg["DESKWORK_IS_ENGAGED"])
    # ENGAGED 是由 DESKWORK_IS_ENGAGED 推出来的，必须跟着重算
    ENGAGED = frozenset({"focused", "neutral", "deskwork"} if DESKWORK_IS_ENGAGED
                        else {"focused", "neutral"})


def load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            log.exception("config.json 解析失败，本次用默认值")
    return {}


def save_config(cfg: dict) -> None:
    """写 config.json 并立即生效。

    先写临时文件再 os.replace()：直接 write_text 是 truncate + write，
    不是原子的 —— 并发的 load_config 可能读到半个 JSON，那时 load_config
    返回 {}，于是 maybe_reload_config 会把「默认值」当成新配置应用一遍，
    还打印一句"已重新加载（0 项）"。配置就这么无声地丢了。
    """
    tmp = CONFIG_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    os.replace(tmp, CONFIG_PATH)         # 同盘替换是原子的
    apply_config(cfg)


_config_mtime = 0.0
# apply_config / maybe_reload_config 会被采集线程和 HTTP 线程同时调用，
# 而 apply_config 要逐个写十几个全局量 —— 中途被读到就是一条用新阈值
# 配旧容差的样本。用锁把「改」和「读-改-mtime」都串起来。
_config_lock = threading.RLock()


def _config_stamp() -> tuple[float, int]:
    """配置文件的版本戳：mtime + 大小。

    只看 st_mtime 不够 —— 同一秒内的两次保存可能拿到相同的 st_mtime
    （秒级分辨率），第二次编辑就被静默丢弃。
    """
    try:
        st = CONFIG_PATH.stat()
        return (st.st_mtime, st.st_size)
    except OSError:
        return (0.0, 0)


def maybe_reload_config() -> bool:
    """config.json 变动了就重新应用；没变就立刻返回（只做一次 stat）。"""
    global _config_mtime
    with _config_lock:
        stamp = _config_stamp()
        if stamp == _config_mtime:
            return False
        _config_mtime = stamp
        if not CONFIG_PATH.exists():
            apply_config(_DEFAULTS)      # 文件被删掉 = 恢复默认，不能什么都不做
            log.info("config.json 不存在，已恢复默认配置")
            return True
        cfg = load_config()
        errs = validate_config(cfg) if cfg else []
        if errs:
            log.warning("config.json 有 %d 处问题，仍按原样应用：%s",
                        len(errs), "；".join(errs))
        apply_config(cfg)
        log.info("配置已重新加载（%d 项）", len(cfg))
        return True


def reset_config() -> None:
    with _config_lock:
        if CONFIG_PATH.exists():
            CONFIG_PATH.unlink()
        apply_config(_DEFAULTS)


maybe_reload_config()      # 导入时立即生效一次
# ──────────────────────────────────────────────────────────


# ══════════════════════ Win32：当前窗口 + 空闲时长 ══════════════════════
# 用 ctypes 直接调，省掉 pywin32 依赖。

class _LASTINPUTINFO(ctypes.Structure):
    _fields_ = [("cbSize", wt.UINT), ("dwTime", wt.DWORD)]


_user32 = ctypes.WinDLL("user32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

_user32.GetForegroundWindow.restype = wt.HWND
_user32.GetWindowTextLengthW.argtypes = [wt.HWND]
_user32.GetWindowTextW.argtypes = [wt.HWND, wt.LPWSTR, ctypes.c_int]
_user32.GetWindowThreadProcessId.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]
_user32.GetWindowThreadProcessId.restype = wt.DWORD
_user32.GetLastInputInfo.argtypes = [ctypes.POINTER(_LASTINPUTINFO)]
_kernel32.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
_kernel32.OpenProcess.restype = wt.HANDLE
_kernel32.QueryFullProcessImageNameW.argtypes = [
    wt.HANDLE, wt.DWORD, wt.LPWSTR, ctypes.POINTER(wt.DWORD)]
_kernel32.CloseHandle.argtypes = [wt.HANDLE]
_kernel32.GetTickCount64.restype = ctypes.c_ulonglong

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def active_window() -> tuple[str, str]:
    """返回 (进程名小写, 窗口标题)。拿不到就返回空串。"""
    hwnd = _user32.GetForegroundWindow()
    if not hwnd:
        return "", ""

    pid = wt.DWORD()
    _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))

    n = _user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(n + 1)
    _user32.GetWindowTextW(hwnd, buf, n + 1)

    exe = ""
    h = _kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
    if h:
        path = ctypes.create_unicode_buffer(260)
        size = wt.DWORD(260)
        if _kernel32.QueryFullProcessImageNameW(h, 0, path, ctypes.byref(size)):
            exe = path.value.rsplit("\\", 1)[-1].lower()
        _kernel32.CloseHandle(h)
    return exe, buf.value


def idle_seconds() -> float:
    """键鼠无操作时长（秒）。"""
    info = _LASTINPUTINFO()
    info.cbSize = ctypes.sizeof(info)
    if not _user32.GetLastInputInfo(ctypes.byref(info)):
        return 0.0
    # GetTickCount64 低 32 位减 dwTime，天然处理 49.7 天回绕
    return ((_kernel32.GetTickCount64() & 0xFFFFFFFF) - info.dwTime) / 1000.0


_mutex_handle = None


def acquire_singleton() -> bool:
    """同一时间只允许一个实例。

    装上开机自启之后，如果又手动开一次，第二个进程抢不到摄像头、还会往同一个
    数据库里写重复数据。用命名互斥体挡住。句柄要一直握着，不能释放。
    """
    global _mutex_handle
    _kernel32.CreateMutexW.restype = wt.HANDLE
    _kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, wt.BOOL, wt.LPCWSTR]
    ctypes.set_last_error(0)
    _mutex_handle = _kernel32.CreateMutexW(
        None, False, "Local\\focus-monitor-singleton")
    return ctypes.get_last_error() != 183        # ERROR_ALREADY_EXISTS


# ══════════════════════ 判定逻辑（纯函数，可自测） ══════════════════════

def ear_from(pts: list[tuple[float, float]]) -> float:
    """眼睛纵横比。pts 是 6 个眼周点，顺序 [外角, 上1, 上2, 内角, 下2, 下1]。

    EAR = (|p1-p5| + |p2-p4|) / (2 * |p0-p3|)
    睁眼约 0.25~0.35，闭眼掉到 0.1~0.2。
    """
    p = [np.asarray(q, dtype=np.float64) for q in pts]
    vert = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    horiz = np.linalg.norm(p[0] - p[3])
    return float(vert / (2.0 * horiz)) if horiz > 1e-6 else 0.0


def ear_threshold(history: list[float]) -> float:
    """闭眼阈值 = 本人睁眼基线的比例。

    固定阈值在不同人 / 不同光照下会系统性偏。实测：暗光下睁眼 EAR 从 0.285
    抬到 0.329，而阈值钉死在 0.19 —— 等于要求"闭得更狠才算闭眼"，疲劳直接漏报。
    换成相对基线后，暗光、眼型、戴不戴眼镜、换摄像头都能自动跟上，不需要手工校准。

    取较高的百分位而不是中位数：万一用户真的困了十分钟，中位数会跟着塌下去，
    阈值跟着塌就永远判不出疲劳。高百分位锚在"偶尔还是会把眼睛睁大"的那一侧。
    """
    if len(history) < EAR_MIN_SAMPLES:
        return EAR_CLOSED                      # 样本不够，先拿出厂值
    ordered = sorted(history)
    idx = min(len(ordered) - 1, int(len(ordered) * EAR_BASELINE_PCT / 100))
    return max(0.05, ordered[idx] * EAR_RATIO)


def should_prompt_rating(engaged_run: float, ratable: bool,
                         since_last: float = 0.0) -> bool:
    """要不要弹评分提醒。

    抽成独立函数是为了能断言 —— 这条规则是明确要求的行为：
    **正在连续投入时不准打扰**，等断点再问。

    但"不打扰"必须有个上限（since_last ≥ PROMPT_CEILING 时强行放行）。
    纯靠 engaged_run < FLOW_QUIET 有个反直觉的后果：一个上午都在连续投入、
    只在 alt-tab 那几十秒里断一下的人，_engaged_run 几乎从不归零，
    于是**最专注的那些时段一个评分都收不到**。样本被系统性偏向前半天
    被打断的时段 —— 而相关性验证的正是"专注"，这个偏差方向刚好最坏。

    到点还是没断点就放行一次：托盘气泡不抢焦点、不阻塞输入，
    比"永远收不到训练数据"划算得多。
    """
    if not ratable:
        return False
    return engaged_run < FLOW_QUIET or since_last >= PROMPT_CEILING


def shoulder_tilt(lx: float, ly: float, rx: float, ry: float) -> float:
    """肩线相对水平的夹角（度）。参数是左右肩的像素坐标。

    dx 必须取绝对值：MediaPipe 的 11/12 是"人的左右肩"，投影到画面里水平
    顺序是反的，直接把 dx 喂进 atan2 会得到 180° 附近的值（实测中位 173.7°），
    于是"坐姿不良"会永远成立。
    """
    return abs(math.degrees(math.atan2(ry - ly, abs(rx - lx))))


def classify_app(exe: str, title: str) -> str:
    """把当前窗口分成 work / distract / other 三类。"""
    hay = f"{exe} {title}".lower()
    if any(k in hay for k in DISTRACT_KEYWORDS):
        # 学习豁免只在这条分支生效：本来要判分心的，如果内容看着是学习就算工作。
        # 放在这里而不是最前面，是为了不去干扰正常工作应用的判断。
        if any(k in hay for k in STUDY_KEYWORDS):
            return "work"
        return "distract"
    if exe in WORK_APPS:
        return "work"
    return "other"


def decide(*, face_present: bool, yaw: float, pitch: float, closed_for: float,
           idle_sec: float, app_kind: str, away_for: float) -> str:
    """输入这一秒的聚合指标，输出状态。状态机全部规则都在这。"""
    if idle_sec >= AWAY_IDLE or away_for >= AWAY_FACE:
        return "away"
    if not face_present:
        return "distracted"          # 短暂丢脸，还不算离开
    if closed_for >= EAR_SUSTAIN:
        return "drowsy"
    if abs(yaw) <= YAW_TOL:
        if abs(pitch) <= PITCH_TOL:
            # 看着屏幕
            return {"work": "focused", "distract": "distracted"}.get(app_kind, "neutral")
        if PITCH_TOL < pitch <= DESK_PITCH_MAX:
            # 正对桌子低头 —— 看书 / 写作业，也可能是玩手机（前置摄像头分不出）。
            # 下界必须是 PITCH_TOL 而不是负无穷：仰头看天花板不算伏案。
            return "deskwork"
    return "distracted"              # 脸在，但转头看别处 / 低得太狠了


# ══════════════════════ 视觉检测 ══════════════════════

# MediaPipe FaceMesh 眼周点序号
L_EYE = [362, 385, 387, 263, 373, 380]
R_EYE = [33, 160, 158, 133, 153, 144]

# solvePnP 用的通用人脸 3D 模型点（毫米），配对上面对应的 2D 点
_PNP_3D = np.array([
    (0.0, 0.0, 0.0),        # 1   鼻尖
    (0.0, -63.6, -12.5),    # 152 下巴
    (-43.3, 32.7, -26.0),   # 33  右眼外角
    (43.3, 32.7, -26.0),    # 263 左眼外角
    (-28.9, -28.9, -24.1),  # 61  嘴角
    (28.9, -28.9, -24.1),   # 291 嘴角
], dtype=np.float64)
_PNP_IDX = [1, 152, 33, 263, 61, 291]

# 头姿求解方法。同一坐姿下的对照实测（各 20 秒）：
#   ITERATIVE  退化 2.9%   yaw 抖动 2.12°   pitch 抖动 2.20°   pitch 范围 10.7~23.3
#   EPNP       退化 0.0%   yaw 抖动 0.89°   pitch 抖动 0.25°   pitch 范围 14.3~15.9
# ITERATIVE 只有 6 个对应点时容易收敛到"鼻子在相机背后"的退化解 —— 而退化时
# 旧代码把角度留在 0，也就是"正在正视前方"这个自信的错误值。EPNP 全胜，用它。
PNP_FLAGS = cv2.SOLVEPNP_EPNP


def ensure_models(on_progress=None) -> None:
    """首次运行下载 .task 模型（约 4MB + 6MB）。

    on_progress(text): 可选回调。打包成 --windowed 后 print 全进黑洞，
    首次运行会看起来像"双击了但什么都没发生" —— 启动时给个托盘气泡，
    用户才知道程序在工作而不是坏了。
    """
    MODELS.mkdir(exist_ok=True)
    for name, url in MODEL_URLS.items():
        dst = MODELS / name
        if dst.exists():
            continue
        print(f"下载模型 {name} ...")
        if on_progress:
            on_progress(f"正在下载模型 {name}（首次运行需要，约 11MB，请稍候）")
        tmp = dst.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dst)


class Vision:
    """封装 FaceLandmarker + PoseLandmarker。mediapipe 在这才 import，让自检跑得快。"""

    def __init__(self) -> None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = __import__("mediapipe")
        self.face = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(MODELS / "face_landmarker.task")),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
            ))
        self.pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(MODELS / "pose_landmarker_lite.task")),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
            ))
        self._seq = 0
        self._last_yaw = 0.0        # 上一次求解成功的角度，退化时兜底用
        self._last_pitch = 0.0
        self.pnp_fail = 0           # solvePnP 失败计数，供诊断
        self.pnp_total = 0

    def _ts(self) -> int:
        self._seq += 1
        return self._seq * 33  # 毫秒，只要单调递增

    def _img(self, frame_bgr: np.ndarray):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                              data=np.ascontiguousarray(rgb))

    def read_face(self, frame_bgr: np.ndarray) -> dict | None:
        """返回 {yaw, pitch, ear, scale}，没检测到脸返回 None。"""
        h, w = frame_bgr.shape[:2]
        res = self.face.detect_for_video(self._img(frame_bgr), self._ts())
        if not res.face_landmarks:
            return None
        lm = res.face_landmarks[0]

        def px(i: int) -> tuple[float, float]:
            return lm[i].x * w, lm[i].y * h

        # 头部姿态：用鼻尖在相机坐标系里的方向角，不去解欧拉角。
        # 欧拉角分解在俯仰上会整体翻转 180°（实测中位数 171.7°），而且依赖
        # 人脸 3D 模型的尺寸标定；atan2(tx,tz) 对整体尺度不敏感，稳得多。
        img_pts = np.array([px(i) for i in _PNP_IDX], dtype=np.float64)
        focal = float(w)
        cam = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], np.float64)
        ok, _, tvec = cv2.solvePnP(_PNP_3D, img_pts, cam, np.zeros((4, 1)),
                                   flags=PNP_FLAGS)
        self.pnp_total += 1
        tx, ty, tz = (float(v) for v in tvec.flatten()) if ok else (0.0, 0.0, 0.0)
        if ok and tz > 1e-6:
            self._last_yaw = yaw = math.degrees(math.atan2(tx, tz))    # 左右转头
            self._last_pitch = pitch = math.degrees(math.atan2(ty, tz))  # 抬头/低头
        else:
            # 关键：不能留 0。0° 的含义是"正在正视前方"——那是一个自信的错误值，
            # 会把失败帧全变成"专注看屏幕"。宁可沿用上一帧，至少是真实观测过的。
            self.pnp_fail += 1
            yaw, pitch = self._last_yaw, self._last_pitch

        ear = (ear_from([px(i) for i in L_EYE]) + ear_from([px(i) for i in R_EYE])) / 2.0
        scale = math.dist(px(33), px(263)) / w   # 两眼外角距 / 画面宽 → 离屏幕远近
        return {"yaw": yaw, "pitch": pitch, "ear": ear, "scale": scale}

    def read_posture(self, frame_bgr: np.ndarray) -> float | None:
        """返回肩线倾角（度），看不到肩膀返回 None。"""
        h, w = frame_bgr.shape[:2]
        res = self.pose.detect_for_video(self._img(frame_bgr), self._ts())
        if not res.pose_landmarks:
            return None
        lm = res.pose_landmarks[0]
        ls, rs = lm[11], lm[12]
        if ls.visibility < 0.5 or rs.visibility < 0.5:
            return None
        return shoulder_tilt(ls.x * w, ls.y * h, rs.x * w, rs.y * h)


# ══════════════════════ 采集线程 ══════════════════════

# 运行时共享状态放模块级而不是 Monitor 实例上：dashboard 可能独立运行
# （uv run dashboard.py）或作为托盘子进程被 import，不能依赖拿到 Monitor 对象。
_runtime = {"paused": False, "heartbeat": 0.0}


def set_paused(paused: bool) -> None:
    _runtime["paused"] = paused


def is_paused() -> bool:
    return _runtime["paused"]


def beat() -> None:
    """采集线程每转一圈调一次。

    托盘图标靠它区分"正常记录"和"线程已经死了"：以前 Monitor.run() 抛异常
    只是 log 一下就 return，托盘照常刷着最后一次的状态色 —— 看起来完全健康。
    """
    _runtime["heartbeat"] = time.time()


def heartbeat_age() -> float:
    """距离上次采集还有多久（秒）。从未采集过返回一个很大的数。"""
    hb = _runtime.get("heartbeat", 0.0)
    return float("inf") if not hb else max(0.0, time.time() - hb)


def is_stale(after: float = 90.0) -> bool:
    """采集是否已经停摆（且不是主动暂停 —— 暂停是预期行为，不算故障）。"""
    return not is_paused() and heartbeat_age() > after


SCHEMA = """
CREATE TABLE IF NOT EXISTS samples(
    ts REAL, state TEXT, app TEXT, title TEXT,
    yaw REAL, pitch REAL, ear REAL, tilt REAL, scale REAL,
    present INTEGER, idle REAL
);
CREATE INDEX IF NOT EXISTS idx_ts ON samples(ts);
"""


def open_camera(preferred: int | None = None) -> cv2.VideoCapture:
    """打开一个能出彩色画面的摄像头。Windows Hello 红外头是灰度的，跳过。"""
    order = [preferred] if preferred is not None else [0, 1, 2, 3]
    for idx in order:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue
        ok, frame = cap.read()
        if ok and frame is not None and frame.ndim == 3 and frame.shape[1] >= 320:
            print(f"摄像头 #{idx} 已打开  {frame.shape[1]}x{frame.shape[0]}")
            return cap
        cap.release()
    raise RuntimeError("没找到可用摄像头，用 --camera N 手动指定序号")


# ─────────────────── 数据库连接 ───────────────────
# 这个库同时被四个地方碰：采集线程（写）、托盘今日统计（读）、
# dashboard /live 每 3 秒（读）、/report 全表扫（读）。
# 全部走这一个入口，PRAGMA 才有一处可改，不会漏掉某个连接点。

def open_db(path: Path | None = None, readonly: bool = False,
            timeout: float = 15.0) -> sqlite3.Connection:
    """打开 focus.db，统一设好并发相关的 PRAGMA。

    为什么必须集中：不设 PRAGMA 时 SQLite 默认 journal_mode=delete，
    **读写互斥** —— 读事务持锁期间写操作会阻塞满 timeout 然后抛
    `database is locked`。采集循环里的这一抛会让采集永久停止，
    所以这不是性能调优，是正确性问题。

    - `journal_mode=WAL`：读不再阻塞写、写不再阻塞读。这是本场景最关键的一条，
      因为读数进程很多而写只有一路。
    - `busy_timeout`：撞锁时先等而不是立刻失败。15 秒足够覆盖
      普通写入；真正的长事务问题要靠 WAL + 攒批提交解决。
    - `synchronous=NORMAL`：WAL 下这个档位是安全的（断电最多丢最近若干事务，
      而我们的样本本来就是"丢了也没关系"的传感器数据），换来显著更少的 fsync。
    """
    path = DB_PATH if path is None else path
    if readonly:
        # 只读连接不设 journal_mode —— 那是写操作，只读库上会被拒。
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout)
        conn.execute(f"PRAGMA busy_timeout={int(timeout * 1000)}")
        return conn

    conn = sqlite3.connect(path, timeout=timeout)
    conn.execute(f"PRAGMA busy_timeout={int(timeout * 1000)}")
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.DatabaseError as exc:
        # 网络盘 / 只读挂载上 WAL 可能建不起来。降级继续跑，
        # 但记一笔 —— 否则"为什么还是经常撞锁"会查不出来。
        log.warning("无法启用 WAL（降级为默认日志模式）：%s", exc)
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _size(n: float) -> str:
    """字节数转人能读的形式。"""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _stamp(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def backup_db(out: Path | None = None) -> Path:
    """把 focus.db 备份成一个独立文件，返回备份路径。

    为什么不用 shutil.copy：WAL 模式下 focus.db 和 focus.db-wal 是**两个文件**，
    还没 checkpoint 的最新样本全在 -wal 里。单拷 focus.db 会安静地丢掉最近一段
    数据 —— 备份出来的库能打开、能查询，只是少了一截，最难发现的那种坏法。
    sqlite3 的 backup API 在事务层面取一致快照，能同时看见 WAL 的内容，
    而且**可以在采集线程正在写的时候安全执行**，不用先停监视。

    备份不做保留策略：它可能正是你唯一的救命文件，轮转掉就白备份了。
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(f"还没有数据库可备份：{DB_PATH}")
    if out is None:
        out = DB_PATH.with_name(f"{DB_PATH.stem}-backup-{time.strftime('%Y%m%d-%H%M%S')}.db")

    src = open_db(DB_PATH)
    try:
        dst = sqlite3.connect(out)
        try:
            src.backup(dst)
            dst.commit()
        finally:
            dst.close()
    finally:
        src.close()
    print(f"已备份: {out}  ({_size(out.stat().st_size)})")
    return out


def compact_db(days: float, assume_yes: bool = False) -> None:
    """删除 `days` 天以前的原始样本并回收磁盘空间。

    这是全项目唯一会**删数据**的操作，所以设了三道闸门：
      1. 默认只预演：把要删的行数、时间范围打出来就结束，真要删必须显式 --yes。
      2. 真删之前**强制先备份**，备份失败就中止 —— 不允许出现"删了但没备份"。
      3. 备份路径和删除范围都打印出来，用户随时能自己找回。

    为什么不做"聚合成小时级摘要"：报告的每一个数字都来自同一条样本序列，
    `_timed()` 按相邻样本的时间差算时长、上限 5 秒。把一小时压成一行之后，
    这行只能贡献 5 秒，所有时长统计都会塌掉。要让聚合数据进报告，就得给
    报告开第二条数据通路，让每个板块都能同时读两种口径 —— 那是重构级别的改动，
    而本项目只有 --selftest 一套测试，不值得为"少一个功能"冒这个险。
    所以这里选择**明确地删**，而不是做一个"看起来还在、其实已经失真"的聚合。
    """
    if days <= 0:
        print("保留天数必须为正数。想保留全部，就别执行 --compact。")
        return
    if not DB_PATH.exists():
        print("还没有数据库，无需整理。")
        return

    cutoff = time.time() - days * 86400
    conn = open_db(DB_PATH)
    try:
        # 库文件存在 ≠ 有样本表：老版本留下的、或者只被 ratings 建过表的
        # focus.db 都可能没有 samples。这时直接 SELECT 会抛一个很难看的
        # OperationalError，而用户只是想整理一下磁盘。
        has_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='samples'"
        ).fetchone()
        if not has_table:
            print("库里还没有样本表（可能还没真正开始采集过），无需整理。")
            return
        total = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        n, oldest, newest = conn.execute(
            "SELECT COUNT(*), MIN(ts), MAX(ts) FROM samples WHERE ts < ?",
            (cutoff,)).fetchone()
        if not n:
            print(f"没有超过 {days:g} 天的样本（共 {total} 行），无需整理。")
            return
        print(f"共 {total} 行，其中 {n} 行早于 {days:g} 天前"
              f"（{_stamp(oldest)} ~ {_stamp(newest)}）。")
        if not assume_yes:
            print("以上只是预演，没有删除任何数据。")
            print("确认后重新执行并加上 --yes 才会真正删除（删除前会自动备份）。")
            return

        try:
            backup_db()
        except (OSError, sqlite3.Error) as exc:
            print(f"备份失败，已中止，未删除任何数据：{exc}")
            return

        before = DB_PATH.stat().st_size
        conn.execute("DELETE FROM samples WHERE ts < ?", (cutoff,))
        conn.commit()
        # VACUUM 必须在事务外，且会重建整个库 —— 这才是真正把文件缩小的一步，
        # 只 DELETE 的话空间只是进了 freelist，文件大小纹丝不动。
        # 单独兜一层：删除已经提交了，VACUUM 失败（比如监视正持有读事务）
        # 不能被报成"整理失败" —— 否则用户以为数据还在，再跑一次发现行数
        # 对不上，反而更慌。
        try:
            conn.execute("VACUUM")
        except sqlite3.Error as exc:
            print(f"数据已删除，但回收空间失败：{exc}")
            print("（监视还在运行的话，先 --stop 再单独跑一次即可。）")
    finally:
        conn.close()

    after = DB_PATH.stat().st_size
    print(f"完成：删除 {n} 行，{_size(before)} → {_size(after)}"
          f"（回收 {_size(max(0.0, before - after))}）。")
    print("注意：这段历史在报告里就没有了。想保留请先留着上面那份备份。")


class Monitor(threading.Thread):
    """后台采集：读摄像头 → 算指标 → 每秒落一条样本。"""

    def __init__(self, camera: int | None = None) -> None:
        super().__init__(daemon=True)
        self.camera = camera
        self.running = False
        self.paused = False
        self.state = "away"
        self.on_state = lambda s: None      # 托盘在这里挂回调更新图标
        self.on_block_end = lambda bs: None  # 一个 30 分钟块结束且值得评时触发
        self.on_progress = lambda msg: None  # 首次下载模型等长任务，给用户一个提示
        self.on_fatal = lambda msg: None     # 启动致命失败：托盘弹气泡 + 打开面板
        self._last_prompted = 0.0
        self._engaged_since: float | None = None   # 当前这段连续投入从何时开始
        self._engaged_run = 0.0
        # 睁眼基线的滚动样本（10Hz × 180 秒 = 3 分钟）。
        # 窗口不能太长：实测暗光基线 0.33、亮光 0.23，差 40%。窗口设 10 分钟的话，
        # 开关灯之后基线要 10 分钟才跟上，这期间的疲劳判定整个是错的。
        # 配合 75 百分位（而不是中位数），即使这 3 分钟里用户一直犯困也拖不塌基线。
        self._ear_hist: deque[float] = deque(maxlen=1800)
        self._ear_thr = EAR_CLOSED                 # 当前生效的闭眼阈值

    def run(self) -> None:
        log.info("采集线程启动 camera=%s", self.camera)
        try:
            ensure_models(on_progress=self.on_progress)
            vision = Vision()
            cap = open_camera(self.camera)
        except Exception as exc:
            log.exception("初始化失败，采集没有开始")
            # 只写日志等于没报错：--windowed 下没人看得到，而托盘图标照旧躺着，
            # 用户只会觉得"开着但没数据"。必须显式告诉用户。
            if self.on_fatal:
                try:
                    self.on_fatal(f"启动失败：{exc}")
                except Exception:
                    log.exception("上报启动失败时又出错")
            return

        conn = open_db()
        conn.executescript(SCHEMA)
        conn.commit()

        self.running = True
        self._seq = 0
        last_face = last_pose = last_flush = time.time()
        last_away_write = 0.0
        last_beat = time.time()
        last_rating_check = time.time()
        n_rows = 0
        fails = 0
        db_fails = 0          # 连续撞锁次数，用于退避；成功一次即清零
        face_buf: deque[dict] = deque(maxlen=64)
        tilt_buf: deque[float] = deque(maxlen=8)
        scale_hist: deque[float] = deque(maxlen=600)
        closed_since: float | None = None
        away_since: float | None = None
        lean = 0.0

        while self.running:
            # 暂停必须真的把摄像头释放掉。以前只是跳过后处理，read 照跑，
            # 指示灯还亮着 —— 用户以为关了其实没关。
            if self.paused:
                if cap is not None:
                    cap.release()
                    cap = None
                    log.info("已暂停，摄像头已释放")
                beat()                # 暂停时也不刷"停摆"告警：这是预期状态
                time.sleep(0.5)
                continue
            if cap is None:
                try:
                    cap = open_camera(self.camera)
                    log.info("已恢复，摄像头已重新打开")
                except RuntimeError:
                    time.sleep(3)
                    continue

            ok, frame = cap.read()
            if not ok:
                # 休眠唤醒后、或摄像头被别的程序抢走时，read 会一直失败。
                # 置空交给循环开头重开，重开逻辑只有一处。
                fails += 1
                if fails > 100:
                    cap.release()
                    cap = None
                    fails = 0
                    continue
                time.sleep(0.05)
                continue
            fails = 0

            now = time.time()
            beat()                    # 心跳：告诉托盘"我还活着"，见 is_stale()
            h, w = frame.shape[:2]
            proc = cv2.resize(frame, (PROC_WIDTH, int(h * PROC_WIDTH / w)))

            # 一次帧处理出错绝不能带走整个采集线程。
            # 以前循环体整段裸露在外，一次 sqlite3 "database is locked" 就会
            # 沿 run() 抛出、被 excepthook 记一行日志、线程结束 —— 用户只看到
            # 图标慢慢变黄，唯一线索是 focus.log 里一行栈。
            # 现在把每帧的处理包起来：撞锁退避重试，其他异常跳过这一帧。
            #
            # 只在"快要落库"那一段（每秒结算）里包，不包整个循环体：
            # 摄像头 read / 重开 那些分支有自己的 continue 语义，
            # 混进 try 里会改变它们的退出路径，得不偿失。
            try:
                # 每秒结算一次
                if now - last_flush >= 1.0:
                    last_flush = now
                    maybe_reload_config()      # 设置页改完立即生效，不用重启
                    self._ear_thr = ear_threshold(self._ear_hist)
                    if now - last_beat >= 600:
                        # 心跳：进程要是被静默干掉，日志里至少能看出它活到几点
                        last_beat = now
                        log.info("心跳：本次运行累计 %d 条样本，当前状态 %s",
                                 n_rows, self.state)
                    if now - last_rating_check >= 300:
                        last_rating_check = now
                        self._prompt_rating(now)
                    present = len(face_buf) > 0
                    yaw = float(np.median([f["yaw"] for f in face_buf])) if face_buf else 0.0
                    pitch = float(np.median([f["pitch"] for f in face_buf])) if face_buf else 0.0
                    ear = float(np.median([f["ear"] for f in face_buf])) if face_buf else 0.0
                    tilt = float(np.median(tilt_buf)) if tilt_buf else 0.0
                    closed_for = (now - closed_since) if closed_since else 0.0
                    away_for = (now - away_since) if away_since else 0.0
                    exe, title = active_window()
                    kind = classify_app(exe, title)

                    st = decide(face_present=present, yaw=yaw, pitch=pitch,
                                closed_for=closed_for, idle_sec=idle_seconds(),
                                app_kind=kind, away_for=away_for)

                    # 离开期间降频落库。**判断必须在 INSERT 之前** ——
                    # 原来这段写在 commit() 之后，样本早就落库了，continue
                    # 只能跳过状态更新，等于降频从未生效（整夜待机照样每 2 秒
                    # 一条，一晚一万四千行）。降频要真的省下写入，就必须
                    # 在写库之前决定写不写。
                    #
                    # 例外：状态**刚**变成 away 的那一条永远要写，否则"离开"
                    # 这个事件本身就不落库了，报告里会看到"专注 → 直接没有"。
                    away_throttled = (st == "away" and self.state == "away"
                                      and now - last_away_write < AWAY_WRITE_EVERY)
                    if away_throttled:
                        # 样本不写库，但缓冲清理照旧 —— 内存操作，跟落库无关。
                        face_buf.clear()
                    else:
                        conn.execute(
                            "INSERT INTO samples VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                            (now, st, exe, title[:200], yaw, pitch, ear, tilt,
                             lean, 1 if present else 0, idle_seconds()))
                        n_rows += 1
                        last_away_write = now
                        # 攒批提交：每 SAMPLE_COMMIT_EVERY 条才 commit 一次。
                        # 每条单独 commit 在 journal_mode=delete 下等于每条一次
                        # fsync + 一次持写锁，把撞锁概率放大十几倍。
                        if n_rows % SAMPLE_COMMIT_EVERY == 0:
                            conn.commit()

                        # 跟踪"这段连续投入持续了多久"，评分提醒靠它决定要不要闭嘴
                        if st in ENGAGED:
                            if self._engaged_since is None:
                                self._engaged_since = now
                            self._engaged_run = now - self._engaged_since
                        else:
                            self._engaged_since = None
                            self._engaged_run = 0.0

                        if st != self.state:
                            self.state = st
                            self.on_state(st)

                        face_buf.clear()
                        # tilt_buf 不清空：留成 4 秒滚动窗口，中位数才压得住单帧跳变
                db_fails = 0
            except sqlite3.OperationalError as exc:
                # 撞锁是预期内的（读数进程很多），退避后继续，不致命。
                # 退避上限 5 秒：长时间锁定通常意味着别的进程开了长事务，
                # 无限指数退避会让采集"看起来"死掉，反而更难查。
                db_fails += 1
                if db_fails in (1, 10, 100) or db_fails % 500 == 0:
                    log.warning("写库失败第 %d 次（已退避重试）：%s",
                                db_fails, exc)
                time.sleep(min(0.5 * db_fails, 5.0))
                continue
            except Exception:
                # 非数据库异常（模型、解码、算法）几乎总是瞬时或单帧的，
                # 记日志后继续 —— 让一个坏帧杀掉采集是明显的错误取舍。
                log.exception("本帧处理出错，已跳过")
                time.sleep(0.05)
                continue

        if cap is not None:        # 暂停状态下退出时它已经是 None 了
            cap.release()
        # finally 里关连接：上面 try 里 continue 不会走到这儿，
        # 但真要是有没接住的异常穿出去，至少别把写连接和 WAL 文件晾着。
        try:
            conn.commit()          # 把不足一批的尾巴补上，否则最后十几条白采
        except sqlite3.Error:
            log.exception("退出前补提交失败")
        conn.close()
        log.info("采集线程已停止")

    def stop(self) -> None:
        self.running = False

    @property
    def ear_baseline(self) -> float:
        """当前估计的睁眼基线。返回 0 表示样本还不够、还在校准中。"""
        if len(self._ear_hist) < EAR_MIN_SAMPLES or not EAR_RATIO:
            return 0.0
        return self._ear_thr / EAR_RATIO

    def _prompt_rating(self, now: float) -> None:
        """提醒打分 —— 但绝不在人正专注的时候打断。

        两个条件同时满足才弹：① 最近有个数据够、还没评的块；② 人此刻已经
        脱离连续投入。也就是说等到自然断点（走神 / 离开 / 疲劳）才问。
        一直专注就一直不问 —— 那本来就是你想要的状态，没什么好打分的。

        ratings / report 在这里才 import：focus 是底层模块，不该在顶层依赖
        它们（而且 report 反过来 import focus，顶层引会成环）。
        """
        try:
            import ratings
            import report
            prev = (int(now // ratings.BLOCK) - 1) * ratings.BLOCK
            if prev <= self._last_prompted:
                return
            items = report._timed(report.load(since=prev))
            ratable = ratings.is_ratable(items, prev, now)
            since_last = now - self._last_prompted if self._last_prompted else 0.0
            if not should_prompt_rating(self._engaged_run, ratable, since_last):
                return                      # 正在心流里，这一轮先不打扰
            self._last_prompted = prev
            self.on_block_end(prev)
        except Exception:
            log.exception("评分提醒检查失败")


# ══════════════════════ 托盘 ══════════════════════

# 图标形状表：只靠颜色不行 —— 16×16 的点上 #14b8a6 / #22c55e / #38bdf8 几乎分不出，
# 色盲用户更是丢掉绿/琥珀的全部区别。形状是颜色之外的第二条通道。
_ICON_SHAPE = {
    "focused":    "solid",    # 实心圆 = 专注
    "neutral":    "half",     # 半环   = 中性
    "deskwork":   "bar",      # 横杠   = 伏案
    "distracted": "tri",      # 三角   = 走神
    "drowsy":     "ring",     # 空心环 = 疲劳
    "away":       "dash",     # 短横   = 离开
    "stale":      "alert",    # 感叹号 = 记录停了（在记录却收不到样本）
}


def _icon_image(color: str, shape: str = "solid"):
    """画托盘图标。shape 见 _ICON_SHAPE —— 颜色 + 形状两条通道。"""
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    box = (8, 8, 56, 56)
    if shape == "solid":
        d.ellipse(box, fill=color)
    elif shape == "ring":
        d.ellipse(box, outline=color, width=9)
    elif shape == "half":
        # 左半实心、右半只留边框：和实心圆区分得开
        d.ellipse(box, outline=color, width=5)
        d.pieslice(box, 90, 270, fill=color)
    elif shape == "bar":
        d.rounded_rectangle((8, 24, 56, 40), radius=6, fill=color)
    elif shape == "dash":
        d.rounded_rectangle((14, 29, 50, 35), radius=3, fill=color)
    elif shape == "tri":
        d.polygon([(32, 8), (58, 54), (6, 54)], fill=color)
    else:  # alert
        d.rounded_rectangle((26, 10, 38, 40), radius=5, fill=color)
        d.ellipse((26, 46, 38, 58), fill=color)
    return img


def _panel_url(path: str = "") -> str:
    """面板地址。端口可能不是默认的，所以问 dashboard 要，别硬编码。

    托盘打开页面已走 window.open_page；这里保留是给浏览器回退等场合用。
    """
    import dashboard
    dashboard.serve_background(open_browser=False)
    return dashboard.base_url().rstrip("/") + path


def _pending_ratings() -> int:
    try:
        import ratings
        import report
        items = report._timed(report.load(since=time.time() - 2 * 86400))
        return len(ratings.pending(items))
    except Exception:
        log.exception("检查待评分失败")
        return 0


def _today_engaged_seconds() -> int:
    """今天累计投入秒数（ENGAGED 状态），只读连接，不碰主进程的写连接。"""
    _today = time.mktime(time.localtime()[:3] + (0, 0, 0, 0, 0, -1))
    try:
        conn = open_db(readonly=True)
        try:
            marks = ",".join("?" * len(ENGAGED))
            row = conn.execute(
                f"SELECT COUNT(*) FROM samples WHERE ts >= ? "
                f"AND state IN ({marks})",
                (_today, *ENGAGED)).fetchone()
            return int(row[0])
        finally:
            conn.close()
    except sqlite3.Error:
        return 0


def run_tray(mon: Monitor) -> None:
    import pystray

    def refresh(icon) -> None:
        # 采集停摆时覆盖成"警告"图标。
        # 只靠状态色有个致命盲区：线程死了以后状态就冻在最后一次的值上，
        # 若那时恰好是"中性"，用户看到的是一个和健康状态一模一样的蓝点。
        # 这正是当初"绿点为什么没了"排查不出来的原因。
        if is_stale():
            icon.icon = _icon_image("#f59e0b", "alert")
            icon.title = "专注监视 — 无数据（可能已停止记录）"
            return
        label, color = STATES.get(mon.state, STATES["away"])
        icon.icon = _icon_image(color, _ICON_SHAPE.get(mon.state, "solid"))
        icon.title = f"专注监视 — {label}"

    mon.on_state = lambda _s: refresh(icon)

    def on_block_end(block_start: float) -> None:
        """提醒打分。

        气泡是点不出反应的 —— pystray 的 _on_notify 只处理左键和右键，
        没有接 NIN_BALLOONUSERCLICK。所以文案里直接写清楚该点哪，
        别让用户去点那个气泡。
        """
        try:
            n = _pending_ratings()
            tip = (f"刚才那 30 分钟你觉得自己专注吗？"
                   f"点托盘图标可直接打（待评 {n} 个时段）" if n > 0
                   else "刚才那 30 分钟你觉得自己专注吗？点托盘图标打分")
            icon.notify(tip, "专注监视 · 该打个分了")
        except Exception:
            log.exception("托盘通知失败")

    mon.on_block_end = on_block_end

    def on_progress(msg: str) -> None:
        """首次运行下载模型之类的事，冒个泡，别让程序看起来像没反应。"""
        try:
            icon.notify(msg, "专注监视 · 正在准备")
        except Exception:
            log.exception("进度提示失败")

    def on_fatal(msg: str) -> None:
        """启动就失败：气泡说清楚，再把面板打开 —— 否则用户只看到图标躺着。"""
        try:
            icon.notify(f"{msg}\n点托盘图标看面板，或右键「打开日志」看详情",
                        "专注监视 · 启动失败")
        except Exception:
            log.exception("失败提示失败")
        try:
            import window
            window.open_page("panel")
        except Exception:
            log.exception("打开面板失败")

    mon.on_progress = on_progress
    mon.on_fatal = on_fatal

    def on_panel(icon, _item):
        import window                     # 延迟：开机自启时别拖 pythonnet 加载
        window.open_page("panel")

    def on_rate(icon, _item):
        import window                     # 延迟：开机自启时别拖 pythonnet 加载
        window.open_page("rate")

    def on_default(icon, _item):
        """左键单击图标：有待评分就去评分页，否则打开面板。

        这是 pystray 在 Windows 上唯一保证能触发的"特殊动作"
        （HAS_DEFAULT_ACTION），比气泡可靠得多。
        """
        (on_rate if _pending_ratings() > 0 else on_panel)(icon, _item)

    def on_report(icon, _item):
        """生成报告文件，再到应用窗口里展示。"""
        import report                   # 延迟 import，见 on_panel 注释
        report.main()
        window.open_page("report")

    def on_log(icon, _item):
        """用系统默认程序打开 focus.log。

        没这个入口的话，日志只能靠用户知道路径、手动翻文件夹 ——
        等于不存在。采集出错时该看的第一个文件必须一步可达。
        """
        try:
            if LOG_PATH.exists():
                os.startfile(LOG_PATH)          # noqa: S606 (Windows 专属)
            else:
                icon.notify(f"还没有日志文件：{LOG_PATH}", "专注监视")
        except Exception:
            log.exception("打开日志失败")

    def on_quit(icon, _item):
        mon.stop()
        # 给采集线程一个收尾的机会：不 join 的话，下面 quit_app 一销毁窗口、
        # 主线程立即退出进程，daemon 采集线程被硬拔，退出前的补提交
        # （conn.commit() 尾部样本）根本没机会执行 —— 最后 ≤15 条样本
        # （约半分钟）白采。join 只等它在循环里看到 running=False，
        # 正常时瞬时返回；摄像头阻塞等极端情况靠超时兜底。
        try:
            mon.join(timeout=3)
        except Exception:
            pass
        icon.stop()
        import window                     # 延迟：与其余菜单项保持一致
        window.quit_app()                 # 销毁窗口 → GUI 循环返回 → 进程真正退出

    def on_pause(icon, _item):
        mon.paused = not mon.paused
        set_paused(mon.paused)         # dashboard 独立运行时靠它区分"暂停"和"记录程序挂了"
        icon.title = "专注监视 — 已暂停" if mon.paused else "专注监视"
        return True

    def _state_text(_i) -> str:
        if is_stale():
            # 从未有过任何一次心跳时 heartbeat_age() 返回 inf ——
            # int(inf) 直接 OverflowError，会把托盘线程炸掉。实测路径：
            # 模型下载失败 / 摄像头打不开 → 采集线程没起来 → 90 秒后
            # 打开菜单，这一行必崩。这种情况要诚实地写"从未采集"。
            age = heartbeat_age()
            ago = ("从未采集" if age == float("inf")
                   else f"{int(age)} 秒前")
            return f"当前状态：无数据（{ago}）— 记录可能已停止"
        if is_paused():
            return "当前状态：已暂停"
        label = STATES.get(mon.state, ("未知", "#64748b"))[0]
        return f"当前状态：{label}"

    def _today_text(_i) -> str:
        import report                       # 只在函数里取 _dur，避免顶层 import 成环
        txt = f"今日投入：{report._dur(_today_engaged_seconds())}"
        n = _pending_ratings()
        return f"{txt}　·　待评分 {n}" if n > 0 else txt

    def _pause_text(_i) -> str:
        """说明这次点下去会发生什么，并且点明摄像头会被释放。

        暂停时真的释放摄像头是隐私上的加分项，但以前只在代码注释里写过，
        用户看不到 —— 写进菜单文案它才开始起作用。
        """
        return ("继续记录（重新打开摄像头）" if mon.paused
                else "暂停记录（关闭摄像头）")

    icon = pystray.Icon(
        "focus-monitor",
        _icon_image(STATES["away"][1], _ICON_SHAPE["away"]),
        "专注监视",
        menu=pystray.Menu(
            pystray.MenuItem(_state_text, None, enabled=False),
            pystray.MenuItem(_today_text, None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("打开主窗口（左键点我）", on_default,
                             default=True),
            pystray.MenuItem("自述评分", on_rate),
            pystray.MenuItem("完整报告", on_report),
            pystray.MenuItem(_pause_text, on_pause),
            pystray.MenuItem("打开日志", on_log),
            pystray.MenuItem("退出", on_quit),
        ))
    refresh(icon)
    threading.Thread(target=icon.run, daemon=True).start()

    # 看门狗：状态没变时 on_state 不会被调用，光靠它发现不了"线程悄悄死了"——
    # 必须有个独立的心跳，定期重新评估图标。30 秒一次，开销可以忽略。
    def _watchdog() -> None:
        while True:
            time.sleep(30)
            try:
                refresh(icon)
            except Exception:
                log.exception("托盘看门狗刷新失败")

    threading.Thread(target=_watchdog, daemon=True).start()

    # 主线程让给 pywebview：GUI 循环硬性要求主线程，托盘消息循环不受限。
    import window
    window.start_main()


# ══════════════════════ 开 / 关 / 桌面开关 ══════════════════════

ICON_DIR = ROOT / "icons"
LNK_NAME = "专注监视.lnk"
PYW = ROOT / ".venv" / "Scripts" / "pythonw.exe"

# 子进程一律不要弹控制台 —— 这些函数可能跑在 pythonw 下
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_DETACHED = (getattr(subprocess, "DETACHED_PROCESS", 0)
             | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))


def _desktop() -> Path:
    return Path(os.environ.get("USERPROFILE", "")) / "Desktop"


def ensure_icons() -> tuple[Path, Path]:
    """生成「开」「关」两个图标。

    快捷方式的图标是静态的，没法自己反映运行状态 —— 只能在每次切换时
    连图标一起重写 .lnk，Explorer 会立刻刷新。
    """
    from PIL import Image, ImageDraw
    ICON_DIR.mkdir(exist_ok=True)
    on, off = ICON_DIR / "on.ico", ICON_DIR / "off.ico"
    for p, color, bar in ((on, "#22c55e", False), (off, "#64748b", True)):
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.ellipse((5, 5, 59, 59), fill=color)
        if bar:                      # 关：中间一道横杠，和「开」一眼区分
            d.rectangle((15, 27, 49, 37), fill="#0f172a")
        img.save(p, format="ICO", sizes=[(16, 16), (32, 32), (48, 48), (64, 64)])
    return on, off


def _write_lnk(icon: Path) -> None:
    """重写桌面快捷方式。.lnk 只能走 COM，交给 PowerShell，并隐藏它的窗口。"""
    lnk = _desktop() / LNK_NAME
    ps = (
        "$s=(New-Object -ComObject WScript.Shell).CreateShortcut('%s');"
        "$s.TargetPath='%s';"
        "$s.Arguments='\"%s\" --toggle';"
        "$s.WorkingDirectory='%s';"
        "$s.IconLocation='%s,0';"
        "$s.Description='专注度监视（双击切换开/关）';"
        "$s.Save()"
    ) % (lnk, PYW, ROOT / "focus.py", ROOT, icon)
    subprocess.run(["powershell", "-NoProfile", "-NonInteractive",
                    "-Command", ps],
                   creationflags=_NO_WINDOW, capture_output=True)


def running_pid() -> int | None:
    """仍在运行的实例 PID；PID 文件在但进程已死则返回 None。

    只信 PID 文件不够 —— 被强杀时它不会被清理（这个坑踩过一次了）。
    """
    if not PID_PATH.exists():
        return None
    try:
        pid = int(PID_PATH.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                       capture_output=True, text=True,
                       creationflags=_NO_WINDOW)
    return pid if str(pid) in (r.stdout or "") else None


def stop_running() -> None:
    """停掉正在运行的实例。"""
    pid = running_pid()
    if pid is None:
        PID_PATH.unlink(missing_ok=True)
        print("没有找到运行中的实例。")
        return
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                   capture_output=True, creationflags=_NO_WINDOW)
    PID_PATH.unlink(missing_ok=True)     # 强杀不会走对方的 finally
    print(f"已停止专注监视（PID {pid}）。")


def wait_and_open(timeout: float = 120.0) -> None:
    """等面板就绪再打开应用窗口。

    模型加载要十几秒，启动后立刻打开只会看到「无法连接」。
    """
    try:
        from dashboard import DEFAULT_PORT as port
    except Exception:
        port = 8787
    url = f"http://127.0.0.1:{port}/"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2).close()
            import window               # 延迟 import，见 on_panel 注释
            window.open_page("panel")
            return
        except Exception:
            time.sleep(1.5)


def toggle() -> None:
    """桌面快捷方式的入口：切换开/关，并把图标改成当前状态。"""
    on_icon, off_icon = ensure_icons()
    if running_pid() is not None:
        stop_running()
        _write_lnk(off_icon)
        print("已关闭专注监视。")
        return
    # DETACHED_PROCESS 是必须的：不脱离的话，开关一退出监视进程会被一起带走
    subprocess.Popen([str(PYW), str(ROOT / "focus.py")], cwd=str(ROOT),
                     creationflags=_DETACHED, close_fds=True)
    _write_lnk(on_icon)
    subprocess.Popen([str(PYW), str(ROOT / "focus.py"), "--wait-open"],
                     cwd=str(ROOT), creationflags=_DETACHED, close_fds=True)
    print("已启动专注监视，面板就绪后会自动打开。")


def install_shortcut() -> None:
    """建桌面开关，并清掉早期的三个快捷方式。"""
    on_icon, off_icon = ensure_icons()
    d = _desktop()
    for old in ("专注监视面板.url", "启动专注监视.lnk", "停止专注监视.lnk"):
        p = d / old
        if p.exists():
            p.unlink()
            print(f"  已删除旧快捷方式: {old}")
    _write_lnk(on_icon if running_pid() is not None else off_icon)
    print(f"  已创建桌面开关: {d / LNK_NAME}")
    print("  双击 = 开/关切换；图标绿点=运行中，灰底横杠=已停止")


# ══════════════════════ 开机自启 ══════════════════════

STARTUP_DIR = (Path(os.environ.get("APPDATA", "")) /
               "Microsoft/Windows/Start Menu/Programs/Startup")
STARTUP_CMD = STARTUP_DIR / "focus-monitor.cmd"


def install_startup() -> None:
    """往"启动"文件夹放一个 .cmd。不需要管理员权限，删掉文件就算卸载。"""
    pyw = ROOT / ".venv" / "Scripts" / "pythonw.exe"
    if not pyw.exists():
        sys.exit(f"找不到 {pyw}\n先跑一次 `uv run --directory \"{ROOT}\" focus.py` "
                 "把虚拟环境建出来")
    STARTUP_DIR.mkdir(parents=True, exist_ok=True)
    # cmd.exe 按系统 ANSI 代码页读脚本，路径含中文时要用 gbk 写，否则乱码
    STARTUP_CMD.write_text(
        "@echo off\r\n"
        f'cd /d "{ROOT}"\r\n'
        f'start "" "{pyw}" "{ROOT / "focus.py"}"\r\n',
        encoding="gbk")
    print(f"已安装开机自启:\n  {STARTUP_CMD}")
    print("用 pythonw 启动，不会弹黑窗。取消：--uninstall-startup")


def uninstall_startup() -> None:
    if STARTUP_CMD.exists():
        STARTUP_CMD.unlink()
        print(f"已移除开机自启: {STARTUP_CMD}")
    else:
        print("当前没有安装开机自启")


# ══════════════════════ 自检 ══════════════════════

def selftest() -> None:
    """不碰摄像头，只验判定逻辑和报告渲染。"""
    # EAR：正三角形似的眼睛 vs 眯成一条缝
    open_eye = [(0, 0), (1, -3), (3, -3), (4, 0), (3, 3), (1, 3)]
    shut_eye = [(0, 0), (1, -0.3), (3, -0.3), (4, 0), (3, 0.3), (1, 0.3)]
    assert ear_from(open_eye) > 0.6, ear_from(open_eye)
    assert ear_from(shut_eye) < 0.2, ear_from(shut_eye)
    assert ear_from([(0, 0)] * 6) == 0.0        # 退化输入不炸

    # 肩线倾角：左右肩顺序反着喂也必须得到同样的结果，且水平时接近 0
    assert shoulder_tilt(0.0, 100.0, 200.0, 100.0) == 0.0        # 两肩等高
    assert shoulder_tilt(200.0, 100.0, 0.0, 100.0) == 0.0        # 顺序反过来
    a = shoulder_tilt(0.0, 100.0, 200.0, 120.0)
    b = shoulder_tilt(200.0, 120.0, 0.0, 100.0)
    assert abs(a - b) < 1e-9 and 5 < a < 7, (a, b)               # 倾斜约 5.7°
    assert shoulder_tilt(0.0, 0.0, 0.0, 0.0) == 0.0              # 退化输入不炸

    # 闭眼阈值随本人基线走 —— 暗光 / 眼型 / 眼镜 / 换摄像头都能自动跟上
    assert ear_threshold([]) == EAR_CLOSED, "样本不足应退回出厂值"
    assert ear_threshold([0.30] * 10) == EAR_CLOSED, "样本太少也退回出厂值"
    light_smp = [0.27, 0.28, 0.29, 0.30, 0.31] * 40      # 亮光，基线约 0.30
    dark_smp = [x + 0.044 for x in light_smp]            # 暗光实测整体上移 0.044
    t_light, t_dark = ear_threshold(light_smp), ear_threshold(dark_smp)
    assert t_dark > t_light, "暗光下阈值必须跟着抬高，否则疲劳漏报"
    assert abs((t_dark - t_light) - 0.044 * EAR_RATIO) < 0.01, (t_light, t_dark)
    assert t_light < min(light_smp), "阈值必须明显低于睁眼基线，否则睁着眼也算闭眼"
    # 困了十分钟不能把基线拖塌 —— 否则阈值跟着塌，永远判不出疲劳
    assert ear_threshold(light_smp + [0.10] * 300) > t_light * 0.9, \
        "长时间低 EAR 不该把基线整体拉下去"

    # 应用分类
    assert classify_app("code.exe", "focus.py - Visual Studio Code") == "work"
    assert classify_app("chrome.exe", "【4K】哔哩哔哩 直播") == "distract"
    assert classify_app("chrome.exe", "GitHub - Pull requests") == "other"
    assert classify_app("chrome.exe", "Bilibili") == "distract"   # 大小写无关
    # 学习豁免：平台是娱乐的，内容不是
    assert classify_app(
        "chrome.exe", "【C++】清华大学 程序设计基础 第3讲 - 哔哩哔哩") == "work"
    assert classify_app("chrome.exe", "算法导论 公开课 - 哔哩哔哩") == "work"
    assert classify_app("chrome.exe", "考研数学 复习 - 哔哩哔哩") == "work"
    # 豁免不能滥用：不带学习关键词的娱乐内容照常判分心
    assert classify_app("chrome.exe", "【4K】舞蹈区精选 - 哔哩哔哩") == "distract"
    assert classify_app("chrome.exe", "王者荣耀 直播 - 哔哩哔哩") == "distract"

    # 状态机：每条分支都要走到
    base = dict(face_present=True, yaw=0.0, pitch=0.0, closed_for=0.0,
                idle_sec=0.0, app_kind="work", away_for=0.0)
    assert decide(**base) == "focused"
    assert decide(**{**base, "app_kind": "other"}) == "neutral"
    assert decide(**{**base, "app_kind": "distract"}) == "distracted"
    assert decide(**{**base, "yaw": 60.0}) == "distracted"        # 转头
    assert decide(**{**base, "pitch": -40.0}) == "distracted"     # 仰头看天花板
    # 伏案：低头对着桌子，但没低到贴桌面
    assert decide(**{**base, "pitch": 45.0}) == "deskwork"
    assert decide(**{**base, "pitch": DESK_PITCH_MAX}) == "deskwork"      # 边界内
    assert decide(**{**base, "pitch": DESK_PITCH_MAX + 1}) == "distracted"  # 低太狠
    assert decide(**{**base, "pitch": PITCH_TOL + 0.1}) == "deskwork"
    # 伏案也要求没转头，转头了还是走神
    assert decide(**{**base, "pitch": 45.0, "yaw": 50.0}) == "distracted"
    # 疲劳优先于伏案：趴桌上睡着了不能算在学习
    assert decide(**{**base, "pitch": 45.0, "closed_for": EAR_SUSTAIN}) == "drowsy"
    assert decide(**{**base, "yaw": YAW_TOL}) == "focused"        # 边界值算专注
    # 实测正常注视的俯仰中位数就有 15.8°，阈值必须容得下
    assert decide(**{**base, "pitch": 15.8}) == "focused"
    assert decide(**{**base, "pitch": PITCH_TOL}) == "focused"
    # 注意 PITCH_TOL 以上不再是"走神"而是"伏案"，边界见下面的伏案断言
    assert decide(**{**base, "closed_for": EAR_SUSTAIN}) == "drowsy"
    assert decide(**{**base, "face_present": False}) == "distracted"
    assert decide(**{**base, "away_for": AWAY_FACE}) == "away"
    assert decide(**{**base, "idle_sec": AWAY_IDLE}) == "away"
    assert decide(**{**base, "idle_sec": AWAY_IDLE}) == "away"
    # 离开的优先级高于疲劳：人走了就别报疲劳
    assert decide(**{**base, "closed_for": 99.0, "away_for": 99.0}) == "away"

    # 报告渲染
    import report
    # 排行显示"在干什么"而不是"用哪个浏览器"，所以标题优先、浏览器后缀要剥掉
    assert report._label("chrome.exe", "哔哩哔哩 - Google Chrome") == "哔哩哔哩"
    assert report._label("code.exe", "focus.py - Visual Studio Code") \
        == "focus.py - Visual Studio Code"
    assert report._label("explorer.exe", "") == "explorer.exe"
    assert report._label("", "") == "(未知)"

    rows = [(t, "focused", "code.exe", "focus.py", 2.0, 1.0, 0.3, 4.0, 1.0, 1, 0.0)
            for t in range(1789000000, 1789000060)]
    # yaw 要在容差内 —— 只有"确实看着屏幕"的走神才归因给应用
    rows += [(t, "distracted", "chrome.exe", "哔哩哔哩", 5.0, 2.0, 0.3, 15.0, 1.0, 1, 0.0)
             for t in range(1789000060, 1789000090)]
    rows += [(t, "away", "explorer.exe", "", 0.0, 0.0, 0.0, 0.0, 1.0, 0, 300.0)
             for t in range(1789000090, 1789000120)]
    html = report.build_html(rows)
    for needle in ("专注", "走神", "离开", "哔哩哔哩", "focus.py",
                   "<svg", "时间轴", "专注应用排行", "分心应用排行",
                   "应用使用记录", "占活跃", "时段"):
        assert needle in html, f"报告缺少 {needle}"
    assert html.count("<html") == 1

    # ── 走神归因的拆分 ──
    assert report._is_looking(1, 0.0, 10.0)
    assert not report._is_looking(0, 0.0, 10.0)             # 人脸不在
    assert not report._is_looking(1, YAW_TOL + 5, 10.0)     # 转头
    assert not report._is_looking(1, 0.0, PITCH_TOL + 5)    # 低头

    def _row(ts, state, app, title, yaw, pitch):
        return (ts, state, app, title, yaw, pitch, 0.3, 1.0, 1.0, 1, 0.0)

    # 看着屏幕的走神 → 归因给 B 站；转头时的走神 → 绝不能归因给快捷设置
    mixed = ([_row(1000 + i, "distracted", "chrome.exe", "哔哩哔哩", 0.0, 5.0)
              for i in range(30)]
             + [_row(1030 + i, "distracted", "shellhost.exe", "快速设置", 60.0, 5.0)
                for i in range(30)])
    mh = report.build_html(mixed)
    assert "哔哩哔哩" in mh, "看着屏幕时的分心应用应出现在排行里"

    # "快速设置"现在会出现在「应用使用记录」里（那份记录的口径是"我用过什么"，
    # 转头时前台确实开着它），但**绝不能进「分心应用排行」** —— 排行回答的是
    # "什么在拉走我"，把转头时段算进去就会得出"快速设置是你最大分心源"。
    # 所以这里改成按板块断言，而不是断言整个 HTML 里没有这个名字。
    assert "应用使用记录" in mh, "报告缺少全量应用使用记录"
    rank_sec = mh.split("分心应用排行")[1] if "分心应用排行" in mh else ""
    assert "快速设置" not in rank_sec, \
        "转头时前台是什么应用，与走神无关，不该进分心排行"

    # ── 会话切分与心流片段 ──
    def _mk(start, n, state, step=1.0):
        """造 n 条样本，格式对齐 focus.db 的一行。"""
        return [(start + i * step, state, "code.exe", "t", 0.0, 0.0, 0.3, 1.0,
                 1.0, 0 if state == "away" else 1, 0.0) for i in range(n)]

    def _fl(*blocks):
        return report._streaks(report._timed([r for b in blocks for r in _mk(*b)]))

    assert len(_fl((0, 1800, "focused"))) == 1                 # 连做 30 分钟 → 1 段
    assert _fl((0, 600, "focused"), (600, 120, "distracted"),
               (720, 600, "focused")) == []                    # 两段 10 分钟，各自都不够
    assert len(_fl((0, 600, "focused"), (600, 30, "distracted"),
                   (630, 600, "focused"))) == 1                # 中断 30 秒 → 合并成 20 分钟
    assert len(_fl((0, 900, "focused"), (900, 120, "distracted"),
                   (1020, 900, "focused"))) == 2               # 中断 120 秒 → 断开成两段
    # 时间空档不能拼：3 分钟专注 + 6 小时停机 + 3 分钟专注，绝不是一段心流
    assert _fl((0, 180, "focused"), (21600, 180, "focused")) == []

    # 心流要求片段里"专注"（工作应用）占比 ≥50%。
    # 旧定义只看"投入"，而投入包含中性 —— 于是"坐下来盯着屏幕"就算心流，
    # 会话一开头就满足条件，印出"进入心流耗时 0 秒"。
    assert _fl((0, 1800, "neutral")) == [], "纯中性是坐着看屏幕，不算心流"
    assert _fl((0, 1800, "deskwork")) == [], "纯伏案也不算"
    assert len(_fl((0, 1800, "focused"))) == 1
    assert len(_fl((0, 900, "focused"), (900, 900, "neutral"))) == 1, "正好一半算过"
    assert _fl((0, 600, "focused"), (600, 1200, "neutral")) == [], "只占 1/3 不算"

    # 时段划分：边界值必须归下一段，错一格会让整天数据移位
    def _at(hour):
        return time.mktime((2026, 1, 1, hour, 0, 0, 0, 0, -1))
    for _h, _want in ((0, "凌晨"), (3, "凌晨"), (5, "凌晨"), (6, "上午"),
                      (11, "上午"), (12, "下午"), (17, "下午"),
                      (18, "晚上"), (23, "晚上")):
        assert report._band(_at(_h)) == _want, f"{_h} 点应属{_want}"

    def _sess(*blocks):
        return report._sessions(report._timed([r for b in blocks for r in _mk(*b)]))

    # 离开 20 分钟 → 切成两次会话
    assert len(_sess((0, 120, "focused"), (120, 40, "away", 30.0),
                     (1320, 120, "focused"))) == 2
    # 样本空档 10 分钟（待机）→ 切成两次会话
    assert len(_sess((0, 120, "focused"), (720, 120, "focused"))) == 2
    # 短暂离开 2 分钟 → 仍属同一次会话
    assert len(_sess((0, 120, "focused"), (120, 4, "away", 30.0),
                     (240, 120, "focused"))) == 1

    # ── 配置层（放最后：会改全局，跑完必须还原，否则污染上面的断言）──
    snapshot = current_config()
    assert not validate_config(snapshot), validate_config(snapshot)
    assert validate_config({**snapshot, "PITCH_TOL": 70.0, "DESK_PITCH_MAX": 65.0}), \
        "俯仰阈值 >= 伏案上限必须报错，否则「伏案」永远不会出现"
    assert validate_config({**snapshot, "EAR_CLOSED": 1.5})
    assert validate_config({**snapshot, "FACE_FPS": 0})
    assert validate_config({**snapshot, "AWAY_FACE": 999.0})   # 比键鼠空闲还大

    # DESKWORK_IS_ENGAGED 只决定"算不算投入"，不改变状态判定本身
    assert decide(**{**base, "pitch": 45.0}) == "deskwork"
    dk = [_row(3000 + i, "deskwork", "code.exe", "focus.py", 0.0, 45.0)
          for i in range(60)]
    # 那个数字在报告里是加粗高亮的，所以匹配带标记的片段而不是纯文本
    MARK = 'font-size:16px">'          # 有效投入率那一个 <b> 的专属样式
    apply_config({"DESKWORK_IS_ENGAGED": True})
    assert "deskwork" in ENGAGED
    assert MARK + "100%</b>" in report.build_html(dk), "开关打开时有效投入应为 100%"
    apply_config({"DESKWORK_IS_ENGAGED": False})
    assert "deskwork" not in ENGAGED, "关掉开关后伏案不该算投入"
    assert decide(**{**base, "pitch": 45.0}) == "deskwork", "状态判定不受开关影响"
    assert MARK + "0%</b>" in report.build_html(dk), "报告的有效投入要跟着开关走"

    # 关键词表是整体替换，不是往默认值里追加
    apply_config({"YAW_TOL": 40.0, "WORK_APPS": ["myapp.exe"]})
    assert YAW_TOL == 40.0
    assert WORK_APPS == {"myapp.exe"}
    assert classify_app("myapp.exe", "x") == "work"
    assert classify_app("code.exe", "x") == "other", "旧白名单必须被整体替换掉"

    apply_config(snapshot)                                     # 还原
    assert YAW_TOL == snapshot["YAW_TOL"]
    assert "code.exe" in WORK_APPS
    assert classify_app("code.exe", "focus.py") == "work"
    assert decide(**{**base, "pitch": 45.0}) == "deskwork"

    # ── 自述对照 ──
    import ratings

    def _blk(start, n, state, step=1.0):
        return [(start + i * step, state, "code.exe", "t", 0.0, 0.0, 0.3, 1.0,
                 1.0, 0 if state == "away" else 1, 0.0) for i in range(n)]

    # 块聚合：900 秒专注 + 900 秒走神 → 有效 1800、投入 900
    agg = ratings.blocks(report._timed(_blk(0, 900, "focused")
                                       + _blk(900, 900, "distracted")))
    assert agg[0.0] == (1800.0, 900.0), agg

    # 相关系数。样本数必须过 MIN_PAIRS（现为 30），否则 correlation()
    # 直接返回 None —— 那正是它该做的事，不是 bug。
    _n = ratings.MIN_PAIRS
    _perfect = [(i, (i % 5) + 1, ((i % 5) + 1) * 0.1) for i in range(_n)]
    assert ratings.correlation(_perfect) > 0.99
    _inverse = [(i, (i % 5) + 1, 1 - ((i % 5) + 1) * 0.1) for i in range(_n)]
    assert ratings.correlation(_inverse) < -0.99
    assert ratings.correlation([(0, 3, 0.5)] * _n) is None, "评分全一样时算不出相关"
    assert ratings.correlation([(0, 1, 0.5)] * 3) is None, "样本太少不给结论"
    assert ratings.correlation(_perfect[:_n - 1]) is None, \
        f"差一条就该拒绝（门槛 {_n}）"
    assert "强相关" in ratings.verdict(0.8, 20)
    assert "样本" in ratings.verdict(None, 3)
    # 样本量偏少时，结论后面必须挂置信度提示
    assert "还可能变" in ratings.verdict(0.8, ratings.MIN_PAIRS + 1), \
        "刚过门槛的高相关必须带样本量提醒"

    # 数据库往返。用临时库 —— 绝不能污染用户的真实 focus.db
    real_db = globals()["DB_PATH"]
    tmp_db = ROOT / "_selftest_ratings.db"
    globals()["DB_PATH"] = tmp_db
    try:
        ratings.save(0.0, 4)
        assert ratings.ratings_map()[0.0]["score"] == 4
        ratings.save(0.0, 2)                      # 同一块重评 → 覆盖而不是新增
        assert ratings.ratings_map()[0.0]["score"] == 2
        assert len(ratings.ratings_map()) == 1
        full = report._timed(_blk(0, 1800, "focused"))
        assert ratings.paired(full) == [(0.0, 2, 1.0)], ratings.paired(full)
        assert ratings.pending(full, now=1900.0) == [], "已评过的块不该再出现"
        # 数据太少的块不该被要求评分
        thin = report._timed(_blk(0, 300, "focused"))
        assert [p for p in ratings.pending(thin, 1900.0)] == []
    finally:
        globals()["DB_PATH"] = real_db
        tmp_db.unlink(missing_ok=True)

    # ── 备份与保留策略 ──
    # backup_db / compact_db 是全项目唯一会碰用户数据文件的路径，必须有断言兜着：
    # "备份少拷了一截"和"没先备份就删"都是不可逆的事故，靠人工检查是查不出来的。
    real_db = globals()["DB_PATH"]
    tmp_db = ROOT / "_selftest_compact.db"
    globals()["DB_PATH"] = tmp_db
    try:
        def _count(path: Path) -> int:
            c = sqlite3.connect(path)
            try:
                return c.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
            finally:
                c.close()

        now = time.time()
        conn = open_db(tmp_db)
        conn.executescript(SCHEMA)
        conn.execute("DELETE FROM samples")
        # 3 条 400 天前的旧样本 + 2 条刚采的
        for i in range(3):
            conn.execute("INSERT INTO samples VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                         (now - 400 * 86400 + i, "focused", "code.exe", "t",
                          0.0, 0.0, 0.3, 1.0, 1.0, 1, 0.0))
        for i in range(2):
            conn.execute("INSERT INTO samples VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                         (now - i, "focused", "code.exe", "t",
                          0.0, 0.0, 0.3, 1.0, 1.0, 1, 0.0))
        conn.commit()
        conn.close()
        assert _count(tmp_db) == 5

        # 备份出来的必须是**完整**的一份。WAL 下如果只拷 focus.db 主文件，
        # 还没 checkpoint 的样本会全部丢掉 —— 库能打开、能查，只是少一截。
        bk = backup_db(ROOT / "_selftest_bk.db")
        assert bk.exists(), "备份文件没生成"
        assert _count(bk) == 5, f"备份不完整：{_count(bk)} != 5"

        # 预演阶段一行都不许少
        compact_db(90, assume_yes=False)
        assert _count(tmp_db) == 5, "预演阶段就删了数据"

        # 真删：旧的清掉、新的留着
        compact_db(90, assume_yes=True)
        assert _count(tmp_db) == 2, f"应剩 2 条新样本，实际 {_count(tmp_db)}"
        # 真删之前必须留下备份 —— 目录里应该多出一个 backup 文件
        assert list(ROOT.glob("_selftest_compact-backup-*.db")), \
            "删数据前没有自动备份"

        # 保留天数非正 → 什么都不动（这是"想保留全部"的表达方式）
        compact_db(0, assume_yes=True)
        assert _count(tmp_db) == 2, "保留天数非正时不该删任何东西"

        # 备份文件必须被 .gitignore 排除：内容和 focus.db 一样敏感，
        # 而 focus.db / focus.db-* 这两条通配**盖不住** focus-backup-*.db。
        ignore = ROOT / ".gitignore"
        if ignore.exists():                       # 打包成 exe 后没有这个文件
            assert "focus-backup-*.db" in ignore.read_text(encoding="utf-8"), \
                "备份文件没被 .gitignore 排除，一次 git add -A 就泄露个人数据"

        # 库存在但没有 samples 表（老版本残留）时不能抛异常
        empty_db = ROOT / "_selftest_notable.db"
        sqlite3.connect(empty_db).close()
        globals()["DB_PATH"] = empty_db
        compact_db(90, assume_yes=True)           # 只该打印一句提示
    finally:
        globals()["DB_PATH"] = real_db
        for p in list(ROOT.glob("_selftest_compact*.db*")) + \
                list(ROOT.glob("_selftest_bk.db*")) + \
                list(ROOT.glob("_selftest_notable.db*")):
            p.unlink(missing_ok=True)

    # 评分提醒的时机：正在连续投入时不准打扰（这是明确要求的行为）
    assert not should_prompt_rating(FLOW_QUIET, True), "正在心流中不该弹提醒"
    assert not should_prompt_rating(FLOW_QUIET + 3600, True), \
        "心流中且未到上限 → 仍不打扰"
    assert should_prompt_rating(FLOW_QUIET + 3600, True, PROMPT_CEILING), \
        "心流中但已超过上限 → 放行一次，否则永远收不到这段的评分"
    assert should_prompt_rating(0.0, True), "刚断点、有可评的块 → 可以问"
    assert should_prompt_rating(FLOW_QUIET - 1, True), "阈值内仍可问"
    assert not should_prompt_rating(0.0, False), "没有可评的块就不弹"
    assert not should_prompt_rating(FLOW_QUIET + 3600, False, 999999), \
        "没有可评的块时，上限也不能把它放行"

    # 这行以前是"踩着自己写的规矩"崩的：上面注释声称刻意只用 ASCII，消息本身
    # 却是中文。中文 Windows 的 GBK 控制台打得出来，所以本机一直没暴露；到了
    # GitHub Actions 的西文代码页就抛 UnicodeEncodeError，把一次断言全过的自检
    # 判成红色失败，而且崩在最后一行，光看日志看不出是编码问题。
    # 现在不再靠"只写 ASCII"来规避，改由 use_safe_console() 兜住任何代码页。
    print("自检通过 - 全部断言成立")


# ══════════════════════ 入口 ══════════════════════

def main() -> None:
    # 必须最先做：argparse 的 --help、以及下面所有中文提示都走 stdout，
    # 在西文代码页上会直接抛 UnicodeEncodeError（详见 use_safe_console）。
    use_safe_console()
    ap = argparse.ArgumentParser(description="摄像头 + 屏幕使用双路专注度监视器")
    ap.add_argument("--camera", type=int, default=None, help="摄像头序号，默认自动挑")
    ap.add_argument("--report", action="store_true", help="生成 HTML 报告并打开")
    ap.add_argument("--dashboard", action="store_true", help="起实时面板（本地网页）")
    ap.add_argument("--stop", action="store_true", help="停掉正在运行的实例")
    ap.add_argument("--toggle", action="store_true", help="切换开/关（桌面快捷方式用）")
    ap.add_argument("--install-shortcut", action="store_true", help="装桌面开关快捷方式")
    ap.add_argument("--wait-open", action="store_true",
                    help=argparse.SUPPRESS)   # 内部用：等面板就绪再开浏览器
    ap.add_argument("--selftest", action="store_true", help="跑自检，不开摄像头")
    ap.add_argument("--backup", action="store_true",
                    help="把 focus.db 备份一份（监视运行中也能安全执行）")
    ap.add_argument("--out", default=None,
                    help="--backup 的备份路径，默认写在 focus.db 旁边")
    ap.add_argument("--compact", action="store_true",
                    help="删除旧样本并回收空间；默认只预演，加 --yes 才真删")
    ap.add_argument("--days", type=float, default=90.0,
                    help="--compact 的保留天数，默认 90")
    ap.add_argument("--yes", action="store_true",
                    help="--compact 真正执行删除（不加则只预演）")
    ap.add_argument("--install-startup", action="store_true", help="装开机自启")
    ap.add_argument("--uninstall-startup", action="store_true", help="卸载开机自启")
    ap.add_argument("--version", action="version",
                    version=f"%(prog)s {__version__}")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if args.backup:
        # 这两条命令用户会在"磁盘快满了"的时候才想起来跑，那时最不该看到的是
        # 一段 traceback —— 路径写错、库还没建、目标盘不可写都要给一句人话。
        try:
            backup_db(Path(args.out) if args.out else None)
        except (OSError, sqlite3.Error) as exc:
            sys.exit(f"备份失败：{exc}")
        return
    if args.compact:
        try:
            compact_db(args.days, assume_yes=args.yes)
        except sqlite3.Error as exc:
            sys.exit(f"整理失败：{exc}（监视还在运行的话，先 --stop 再试）")
        return
    if args.install_startup:
        install_startup()
        return
    if args.uninstall_startup:
        uninstall_startup()
        return
    if args.report:
        import report
        report.main()
        return
    if args.dashboard:
        import dashboard
        dashboard.main()
        return
    if args.stop:
        stop_running()
        return
    if args.wait_open:
        wait_and_open()
        return
    if args.toggle:
        toggle()
        return
    if args.install_shortcut:
        install_shortcut()
        return

    if sys.platform != "win32":
        sys.exit("托盘和窗口监视依赖 Win32 API，当前只支持 Windows")

    if not acquire_singleton():
        sys.exit("已经有一个实例在跑了（图标可能藏在任务栏的 ^ 折叠区里）")

    PID_PATH.write_text(str(os.getpid()), encoding="utf-8")
    setup_log()
    log.info("专注监视启动 camera=%s", args.camera)

    # 首次运行（还没有数据库）：把面板显示出来。
    # 窗口默认是隐藏的，托盘图标又常常藏在 ^ 折叠区里 —— 于是"双击了但
    # 什么都没发生"。开一次面板就把"在记录、状态是什么、摄像头通不通"
    # 一次全回答了，成本只是一个 if。
    first_run = not DB_PATH.exists()

    # 面板随监视一起起，这样桌面快捷方式随时点得开，不用先去托盘菜单。
    # 只监听 127.0.0.1；起不来也不影响采集，所以异常只记日志。
    try:
        import dashboard
        log.info("实时面板: %s", dashboard.serve_background(open_browser=False))
    except Exception:
        log.exception("实时面板启动失败，监视继续")

    mon = Monitor(camera=args.camera)
    mon.start()

    if first_run:
        # 模型要下十几秒且窗口层要等主线程 GUI 循环起来，所以丢到子线程，
        # 让它等面板就绪再导航，别卡住托盘启动。
        def _show_first_run() -> None:
            time.sleep(3.0)
            try:
                import window
                window.open_page("panel")
            except Exception:
                log.exception("首次运行打开面板失败")

        threading.Thread(target=_show_first_run, daemon=True).start()

    try:
        run_tray(mon)
    except KeyboardInterrupt:
        pass
    except BaseException:
        # 托盘跑在主线程，它一崩整个进程就没了。而 pythonw 没有 stderr，
        # 栈会直接消失 —— 这是"绿点为什么悄悄不见了"的头号嫌疑，必须落盘。
        log.exception("托盘异常退出")
        raise
    finally:
        mon.stop()
        PID_PATH.unlink(missing_ok=True)
        log.info("专注监视已退出")
    print("已退出，运行 `uv run focus.py --report` 看报告")


if __name__ == "__main__":
    main()
