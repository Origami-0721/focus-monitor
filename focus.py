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
import webbrowser
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
EAR_CLOSED = 0.19      # 眼睛纵横比低于此值算闭眼（戴眼镜可能要调到 0.16）
EAR_SUSTAIN = 3.0      # 持续闭眼多少秒算疲劳
AWAY_FACE = 20.0       # 人脸消失多少秒算离开
AWAY_IDLE = 180.0      # 键鼠无操作多少秒算离开
TILT_WARN = 12.0       # 肩线倾斜超过多少度算坐姿不良
AWAY_WRITE_EVERY = 30.0  # 离开期间每 30 秒才落一条，否则整夜待机能把数据库撑爆

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
    CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    apply_config(cfg)


_config_mtime = 0.0


def maybe_reload_config() -> bool:
    """config.json 变动了就重新应用；没变就立刻返回（只做一次 stat）。"""
    global _config_mtime
    try:
        mtime = CONFIG_PATH.stat().st_mtime
    except OSError:
        mtime = 0.0
    if mtime == _config_mtime:
        return False
    _config_mtime = mtime
    if not CONFIG_PATH.exists():
        apply_config(_DEFAULTS)          # 文件被删掉 = 恢复默认，不能什么都不做
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


def ensure_models() -> None:
    """首次运行下载 .task 模型（约 4MB + 6MB）。"""
    MODELS.mkdir(exist_ok=True)
    for name, url in MODEL_URLS.items():
        dst = MODELS / name
        if dst.exists():
            continue
        print(f"下载模型 {name} ...")
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
                                   flags=cv2.SOLVEPNP_ITERATIVE)
        yaw = pitch = 0.0
        if ok:
            tx, ty, tz = (float(v) for v in tvec.flatten())
            if tz > 1e-6:
                yaw = math.degrees(math.atan2(tx, tz))     # 左右转头
                pitch = math.degrees(math.atan2(ty, tz))   # 抬头 / 低头

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


class Monitor(threading.Thread):
    """后台采集：读摄像头 → 算指标 → 每秒落一条样本。"""

    def __init__(self, camera: int | None = None) -> None:
        super().__init__(daemon=True)
        self.camera = camera
        self.running = False
        self.paused = False
        self.state = "away"
        self.on_state = lambda s: None      # 托盘在这里挂回调更新图标

    def run(self) -> None:
        log.info("采集线程启动 camera=%s", self.camera)
        try:
            ensure_models()
            vision = Vision()
            cap = open_camera(self.camera)
        except Exception:
            log.exception("初始化失败，采集没有开始")
            return

        conn = sqlite3.connect(DB_PATH)
        conn.executescript(SCHEMA)
        conn.commit()

        self.running = True
        self._seq = 0
        last_face = last_pose = last_flush = time.time()
        last_away_write = 0.0
        last_beat = time.time()
        n_rows = 0
        fails = 0
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
            h, w = frame.shape[:2]
            proc = cv2.resize(frame, (PROC_WIDTH, int(h * PROC_WIDTH / w)))

            if now - last_face >= 1.0 / FACE_FPS:
                last_face = now
                got = vision.read_face(proc)
                if got:
                    face_buf.append(got)
                    scale_hist.append(got["scale"])
                    base = float(np.median(scale_hist)) if scale_hist else got["scale"]
                    lean = got["scale"] / base if base > 1e-6 else 1.0
                    closed_since = None if got["ear"] >= EAR_CLOSED else (closed_since or now)
                    away_since = None
                else:
                    away_since = away_since or now

            if now - last_pose >= 1.0 / POSE_FPS:
                last_pose = now
                tilt = vision.read_posture(proc)
                if tilt is not None:
                    tilt_buf.append(tilt)

            # 每秒结算一次
            if now - last_flush >= 1.0:
                last_flush = now
                maybe_reload_config()      # 设置页改完立即生效，不用重启
                if now - last_beat >= 600:
                    # 心跳：进程要是被静默干掉，日志里至少能看出它活到几点
                    last_beat = now
                    log.info("心跳：本次运行累计 %d 条样本，当前状态 %s",
                             n_rows, self.state)
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

                conn.execute(
                    "INSERT INTO samples VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (now, st, exe, title[:200], yaw, pitch, ear, tilt, lean,
                     1 if present else 0, idle_seconds()))
                conn.commit()
                n_rows += 1

                # 离开期间降频落库；状态刚变成 away 的那一条永远要写
                if st == "away" and self.state == "away" \
                        and now - last_away_write < AWAY_WRITE_EVERY:
                    face_buf.clear()
                    continue
                last_away_write = now

                if st != self.state:
                    self.state = st
                    self.on_state(st)

                face_buf.clear()
                # tilt_buf 不清空：留成 4 秒的滚动窗口，中位数才压得住单帧跳变

        if cap is not None:        # 暂停状态下退出时它已经是 None 了
            cap.release()
        conn.close()
        log.info("采集线程已停止")

    def stop(self) -> None:
        self.running = False


# ══════════════════════ 托盘 ══════════════════════

def _icon_image(color: str):
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((8, 8, 56, 56), fill=color)
    return img


def run_tray(mon: Monitor) -> None:
    import pystray

    def refresh(icon) -> None:
        label, color = STATES.get(mon.state, STATES["away"])
        icon.icon = _icon_image(color)
        icon.title = f"专注监视 — {label}"

    mon.on_state = lambda _s: refresh(icon)

    def on_dash(icon, _item):
        import dashboard
        dashboard.serve_background()

    def on_report(icon, _item):
        import report
        report.main()

    def on_quit(icon, _item):
        mon.stop()
        icon.stop()

    def on_pause(icon, _item):
        mon.paused = not mon.paused
        icon.title = "专注监视 — 已暂停" if mon.paused else "专注监视"
        return True

    icon = pystray.Icon(
        "focus-monitor",
        _icon_image(STATES["away"][1]),
        "专注监视",
        menu=pystray.Menu(
            pystray.MenuItem("实时面板", on_dash),
            pystray.MenuItem("生成报告", on_report),
            pystray.MenuItem("暂停/继续", on_pause),
            pystray.MenuItem("退出", on_quit),
        ))
    refresh(icon)
    icon.run()          # 阻塞在主线程，采集跑在 daemon 线程


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
    """等面板就绪再开浏览器。

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
            webbrowser.open(url)
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
                   "<svg", "时间轴", "专注应用排行", "分心应用排行"):
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
    assert "快速设置" not in mh, "转头时前台是什么应用，与走神无关，不该进排行"

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
    apply_config({"DESKWORK_IS_ENGAGED": True})
    assert "deskwork" in ENGAGED
    assert "有效投入占 100%" in report.build_html(dk)
    apply_config({"DESKWORK_IS_ENGAGED": False})
    assert "deskwork" not in ENGAGED, "关掉开关后伏案不该算投入"
    assert decide(**{**base, "pitch": 45.0}) == "deskwork", "状态判定不受开关影响"
    assert "有效投入占 0%" in report.build_html(dk), "报告的有效投入要跟着开关走"

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

    print("自检通过 ✓  所有断言成立")


# ══════════════════════ 入口 ══════════════════════

def main() -> None:
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
    ap.add_argument("--install-startup", action="store_true", help="装开机自启")
    ap.add_argument("--uninstall-startup", action="store_true", help="卸载开机自启")
    args = ap.parse_args()

    if args.selftest:
        selftest()
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

    # 面板随监视一起起，这样桌面快捷方式随时点得开，不用先去托盘菜单。
    # 只监听 127.0.0.1；起不来也不影响采集，所以异常只记日志。
    try:
        import dashboard
        log.info("实时面板: %s", dashboard.serve_background(open_browser=False))
    except Exception:
        log.exception("实时面板启动失败，监视继续")

    mon = Monitor(camera=args.camera)
    mon.start()
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
