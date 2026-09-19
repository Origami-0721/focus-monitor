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
import ast
import ctypes
import ctypes.wintypes as wt
import json
import logging
import math
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
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
# 面板的**真实**地址，由监视进程起好服务之后写进来。
#
# 为什么非要有这个文件：8787 被占用时服务会往后顺延（8788…8806，见
# dashboard._make_server），而桌面开关 fork 出来的 `--wait-open` 子进程是
# **另一个进程** —— 它拿不到 `dashboard.base_url()`，那个函数读的是本进程的
# `_server`，在子进程里是 None。硬编码 DEFAULT_PORT 的话，端口一顺延，
# 子进程就一直敲一个没人监听的地址，等满 120 秒然后放弃，用户看到的还是
# "双击了没反应"。（使用教程里早就写了"被占了会自动往后顺延"，
# 只有 wait_and_open 那一处没跟上。）
URL_PATH = ROOT / "focus.url"

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
# 人脸丢了多久就把"连续闭眼"的计时作废。见采集循环里的注释：
# 不作废的话，闭眼计时会跨过"转头出画"这段盲区继续累加。
# 留一点宽限（0.5 秒 = FACE_FPS 下的 5 帧）是为了容忍检测的单帧抖动 ——
# 丢一两帧就把计时清零，真犯困时反而永远攒不满 EAR_SUSTAIN。
FACE_LOSS_GRACE = 0.5
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

# ── 走神的判据 ──
# **应用类型永远不能单独判走神。** 实测被冤枉过：在 B 站看了半小时高数课，
# 因为标题里带"哔哩哔哩"就被判成走神。平台是娱乐的，内容不是 —— 而程序
# 只看得到进程名和标题、看不到内容，所以它没有资格下这个结论。
#
# 拿真实库（121286 条样本）核过：21765 条走神里，14909 条（68.5%）是
# "脸在画面 + 正对屏幕"，只可能来自应用类型那条规则；而"脸在但转头/低头"
# 只有 51 条。也就是说走神几乎全是这条规则造的，眼睛的动作反倒几乎没被用上。
#
# 所以应用类型降级成"只抬高、不贬低"：work 能升成「专注」，其余一律「中性」
# （中性算投入时长，见 ENGAGED）。走神只剩两条**行为**判据：
#   一、脸在画面但没朝向屏幕（转头 / 低得太狠）—— decide() 末尾那条，本来就在。
#   二、在几个页面之间来回切、哪个都没停住 —— 下面这个阈值。
DISTRACT_SWITCH_RATE = 20.0     # 次/分钟
DISTRACT_SWITCH_WINDOW = 60.0   # 回看窗口（秒）
# 20 是拿真实数据标定的，不是拍脑袋：他 66374 条非离开样本里，每 60 秒
# 切换次数的 p50=2 / p75=5 / p90=8 / p95=10 / p99=14 / max=25。
# 取 15 会命中 0.81% 的样本 —— 里面赫然有他写 C++ 的 Visual Studio 会话，
# 那是真在干活。刚因为误判被投诉过，这条新规则宁可欠触发，所以取 20
# （命中 0.11%：只有"平均每 3 秒换一个窗口、连着换满一分钟"才够得着）。
# 想更严就调小（12 会命中 2.9%），想让它完全不触发就调到 9999。
#
# **别指望它抓短视频。** 刷抖音那种场景窗口标题一直不变、切换次数是 0 ——
# 前置摄像头加窗口标题看不到内容，这类"什么都没干"它看不见。
# 同理，标题里带时钟/进度的应用会被它当成一直在切，真遇到就调大这个值。
# 另外，标题只多一个 `*`（编辑器的"已修改"标记）也算一次切换 ——
# 实测他的数据里这不构成问题（次数到 20 时，一分钟内确实见过 5 个以上
# 不同窗口），但这是个已知的粗糙处。

# ── 视疲劳：眼睛该歇会儿了 ──
# **这一块判的是"视疲劳 / 干眼"，不是"困"，两者别混。**
# 文献口径（PERCLOS 那套）：跟**困倦**对应的是单次闭眼变长、慢眨眼/半眨眼
# 变多、PERCLOS（闭眼时间占比）升高；而眨眼**次数**受干眼、风吹、认知负荷
# 干扰，屏幕阅读时普遍掉到 5~7 次/分，它反映的是"盯屏幕太久"。
# 所以这里的产出**不进状态机**：状态描述的是"我在干什么"（专注/走神/…），
# 而"眼睛该休息了"是关于生理负荷的**建议**。混进状态会把投入时长统计污染掉。
#
# 眨眼怎么数：EAR 跌破阈值 → 回升算一次闭合；闭合短于 BLINK_MAX_DUR 才算
# 眨眼，更长的算眯眼/微睡眠，不重复计（那是 EAR_SUSTAIN 那条管的事）。
# 只在**正对屏幕**时计入 —— 转头时眼区被压缩、EAR 假性变低，
# 全算成眨眼的话眨眼率会凭空翻几倍。这不是理论担心：他库里 71544 条
# 有脸样本里，ear 低于出厂闭眼线 0.19 的占 14.6%，远多于眨眼能解释的量。
BLINK_WINDOW = 60.0      # 眨眼率 / 揉眼率的统计窗口（秒）
BLINK_MAX_DUR = 0.6      # 单次闭合短于此算眨眼；更长算眯眼/微睡眠，不计
BLINK_MIN_GAP = 0.5      # 两次眨眼的最小间隔。真人间隔 ≥ 0.5 秒（一次眨眼
#                          0.1~0.4 秒 + 睁眼间隔），所以这个值不会吃掉真眨眼；
#                          它同时挡住"EAR 在阈值附近来回穿越被数成一串"。
#                          万一把眨眼率数高了也只是不提前提醒（见下面那条
#                          "只提前不延后"），不会反过来误报，是失败安全的方向。
BLINK_MIN_OBS = 30.0     # 窗口里至少观察到这么多秒才出数（不足则报 0 = 未知）
BLINK_LOW_RATE = 8.0     # 低于此次/分视为盯屏幕过久（屏幕阅读常态是 5~7）
BLINK_HIGH_RATE = 35.0   # 高于此次/分更像干眼/刺激；只用于文案，不单独触发
#
# 提醒的触发：**连续用眼时长**是主判据，眨眼率只能让提醒提前、不能让闭嘴 ——
# 沿用这个项目里"弱证据只抬高、不贬低"的纪律（见 DISTRACT_SWITCH_RATE 那段）。
# 理由很实在：眨眼检测依赖 EAR 阈值，比"坐了多久"脆弱得多；把它做成
# 必要条件的话，检测一旦失灵就变成永远不提醒 —— 静默失效最难发现。
EYE_BREAK_AFTER = 40 * 60.0   # 连续用眼这么久 → 提醒休息
EYE_BREAK_SOON = 25 * 60.0    # 若同时眨眼率偏低 → 提前到这么久
EYE_REMIND_EVERY = 20 * 60.0  # 两次提醒的最小间隔，别把人烦到关掉
EYE_REST_RESET = 5 * 60.0     # 离开屏幕这么久才算真的休息过，用眼时长清零
#
# 揉眼：**只记录，不参与任何判定。** 原因见 read_posture 上方那段 ——
# Pose 模型只给手腕、没有手指，"手在脸附近"和托腮/扶眼镜/挠头分不开。
# 先攒够真实数据再决定阈值，跟当初标定走神阈值是同一个套路。
HAND_EYE_RADIUS = 0.75   # 手腕到两眼中点的距离 < 此值 × 两眼外角距 → 手在眼周
HAND_EYE_HOLD = 3        # 连续命中这么多次姿态采样才算一次揉眼（2Hz → 1.5 秒）
RUB_MIN_GAP = 30.0       # 两次揉眼的最小间隔，防一次揉眼被拆成好几次

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
#
# **DISTRACT_KEYWORDS 不再判走神**（见常量区 DISTRACT_SWITCH_RATE 上方那段）。
# 它现在唯一的作用是"别硬说专注"：命中这里的应用，看着屏幕也只算「中性」。
# 于是这张表配漏了不再有害（漏了顶多算中性），配多了也不再冤枉人 ——
# 以前配错一个词就等于凭空给人扣一整天走神时长。想让应用完全不参与
# 判定，把表清空即可。
DISTRACT_KEYWORDS = (
    "抖音", "哔哩哔哩", "bilibili", "youtube", "微博", "weibo", "小红书",
    "知乎", "淘宝", "京东", "爱奇艺", "腾讯视频", "优酷", "直播", "游戏",
    "steam", "原神", "genshin", "绝区零", "zenless", "崩坏", "星穹铁道",
    "王者荣耀", "英雄联盟", "漫画", "小说", "贴吧", "虎扑",
    "netflix", "twitch", "reddit", "instagram", "tiktok",
)
# 学习豁免表：命中它就升成「专注」（见 classify_app）。
# 目的是认出"在 B 站看 C++ 课"这种情况 —— 平台是娱乐的，内容不是。
# 别加"第""讲"这种单字，太泛，"【第5期】游戏实况"会被误当成学习。
#
# 实测教训：**光靠关键词救不全，别指望它兜底。** 用户实际看的标题是
# 《高等数学》全程教学视频【宋浩老师】、"2 函数"、"18 运算符-算术运算符"，
# 一个词都没命中，全被判成走神。真正的修法是取消"应用类型判走神"，
# 这张表只是锦上添花 —— 所以它命不中也不再是致命问题。
STUDY_KEYWORDS = (
    "c++", "cpp", "python", "java", "javascript", "typescript", "rust",
    "golang", "kotlin", "swift", "sql", "linux", "docker", "git", "leetcode",
    "教程", "课程", "公开课", "网课", "mooc", "lecture", "tutorial",
    "算法", "数据结构", "编译原理", "操作系统", "计算机网络", "计网",
    "考研", "习题", "作业", "复习", "论文", "答辩", "文献",
    # 学科名：实测漏得最狠的一类。"数学"这种词只会在已经命中娱乐平台
    # （B 站/YouTube）时才起作用，所以误救风险很低。
    "数学", "高数", "线代", "概率论", "离散数学", "物理", "化学", "英语",
    "四级", "六级", "期末", "期中", "考试",
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

# 启动时要不要自动把面板窗口弹出来。默认**关**。
#
# 这是**应用自己**唯一一处"每次启动都弹一个窗口"的地方：桌面开关 fork 出来的
# --wait-open 子进程（见 _start_engine）。用户第一次报的就是它
# （"桌面一直弹「专注监视」的窗口，会打扰我工作"）。
#
# 但要注意：他第二次带截图报的「**每次让你改一点东西**的时候就是这样一直弹出
# 这个在最顶层」，**不是这一处** —— 那个是自检里的 wait_and_open 打到了正在
# 运行的这个实例上，见 selftest 里那一段的注释。两个都要修，别把后者当成
# "用户又抱怨了一遍"就跳过。
#
# "程序到底跑起来没有"不靠这个窗口回答 —— 桌面快捷方式的绿点回答（绿点只在
# 进程真的起来之后才写，见 toggle()）。想看的时候点托盘图标，那一下**不受这个
# 开关影响**：它管的是"自动"，不是"能不能看"。
# 首次运行也不弹：那会儿"看起来像没反应"这件事已经有托盘气泡兜着
# （见 run_tray 的 on_progress），不需要再拿一个窗口去顶。
AUTO_OPEN_PANEL = False

# 到点提醒打分（托盘气泡「专注监视 · 该打个分了」）。默认**开**。
#
# 为什么默认开：自述评分是唯一能验证数据准不准的东西，而气泡是唯一能让人想起
# 来去打分的方式 —— 关掉它等于把训练数据断掉（`should_prompt_rating` 已经做了
# "心流中不打扰、只在自然断点问"）。但它确实是一种打扰，所以留个开关。
#
# 关掉之后**只是不弹气泡**：待评时段照样在攒，托盘图标左键照样进评分页，
# 面板和报告也照常。想静音又不想丢数据，放心关。
# 它**不管**「眼睛该歇会儿了」那条 —— 那条有自己的开关（EYE_BREAK_REMIND）。
RATE_REMIND = True

# 「眼睛该歇会儿了」的托盘气泡（见 run_tray 的 on_eye_break）。默认**开**。
#
# 为什么不和 RATE_REMIND 合成一个"静音所有气泡"：两件事该不该响的判据不同。
# 打分提醒的价值是"趁你还记得住"，关掉就真没了；眼睛提醒的价值是"打断你抬头
# 看远处"，而这件事你自己也做得到。合成一个开关的话，想静音其中一个就得连
# 另一个一起关掉。
#
# 关掉之后**只是不弹气泡**：面板里那条「该让眼睛歇会儿了」的横幅照旧
# （它只在你自己打开面板时可见，不打扰人），日志里也照旧写 `视疲劳提醒：…`
# —— 留着这行是为了让"没提醒"还能分清是眼睛不累还是这条链路坏了。
EYE_BREAK_REMIND = True

# ───────────────── 配置覆盖（config.json）─────────────────
# 上面这些常量就是默认值。config.json 里出现的键会覆盖它们。
# 改完不需要重启进程：监视循环、报告、实时面板都会定期调用
# maybe_reload_config()，靠文件 mtime 判断有没有变。
CONFIG_PATH = ROOT / "config.json"

# 版本号。这里和 pyproject.toml 的 version 必须一致（发布清单里有一步专门核对）。
# 冻结成 exe 后，bug 报告唯一能问到的版本信息就是它。
__version__ = "0.2.3"

_SCALARS: dict[str, type] = {
    "FACE_FPS": int, "POSE_FPS": int, "PROC_WIDTH": int,
    "YAW_TOL": float, "PITCH_TOL": float, "DESK_PITCH_MAX": float,
    "EAR_CLOSED": float, "EAR_SUSTAIN": float,
    "AWAY_FACE": float, "AWAY_IDLE": float,
    "TILT_WARN": float, "AWAY_WRITE_EVERY": float,
    # 走神的行为判据，也是最该按自己习惯调的一个数（见常量区注释）
    "DISTRACT_SWITCH_RATE": float,
    # 视疲劳提醒。这三个直接决定"多久被念一次"，最该按自己习惯调；
    # 眨眼/揉眼的检测细节（窗口、去抖、手腕半径）不进设置页 ——
    # 它们是标定值，改错了只会让统计失真，用户没法判断好坏。
    "EYE_BREAK_AFTER": float, "EYE_BREAK_SOON": float,
    "EYE_REMIND_EVERY": float, "BLINK_LOW_RATE": float,
}
_LIST_KEYS = ("WORK_APPS", "DISTRACT_KEYWORDS", "STUDY_KEYWORDS")

# 布尔开关。**必须单独列出来**：设置页里标量是 number 输入框、开关是 checkbox，
# 渲染和取值各走一条路，所以自检里那张"配置键 ⇔ 字段"的表（它只覆盖
# _SCALARS / _LIST_KEYS）管不到它们。
#
# 少渲染一个 checkbox 的后果是这个开关**永远是关的**：_parse_form 靠
# `key in form` 取值，页面上没有那个框就恒为 False —— 而界面上你还看得见它、
# 还勾得上、勾完还提示"已保存"。所以自检里另有一条专门核对每个 _FLAGS
# 都真的被渲染成了 checkbox（见 selftest 的"设置页"一节）。
_FLAGS = ("DESKWORK_IS_ENGAGED", "AUTO_OPEN_PANEL", "RATE_REMIND",
          "EYE_BREAK_REMIND")

# 导入时的快照，供"恢复默认"用 —— apply_config 之后 globals() 就不是默认值了
_DEFAULTS: dict = {k: globals()[k] for k in _SCALARS}
_DEFAULTS.update({k: globals()[k] for k in _LIST_KEYS})
_DEFAULTS.update({k: globals()[k] for k in _FLAGS})


def current_config() -> dict:
    """当前生效的配置，用于渲染设置页。"""
    cfg = {k: globals()[k] for k in _SCALARS}
    cfg.update({k: sorted(globals()[k]) for k in _LIST_KEYS})
    cfg.update({k: globals()[k] for k in _FLAGS})
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
    # 配小了会把"正常来回切窗口"判成走神（0 的话**每条样本都判走神**，
    # 直接把数据毁掉）；配太大则这条判据等于不存在。见常量区的标定说明。
    if cfg["DISTRACT_SWITCH_RATE"] <= 0:
        errs.append("「页面切换频率」必须为正数，否则每条样本都会判成走神")
    # 视疲劳提醒：EYE_BREAK_SOON 是"眨眼率偏低时提前提醒"的那一档，
    # 它不小于 EYE_BREAK_AFTER 的话那个提前量就是空的 —— 不报错，
    # 只是眨眼这条判据**静默失效**，谁也不会发现。
    if cfg["EYE_BREAK_SOON"] > cfg["EYE_BREAK_AFTER"]:
        errs.append("「提前提醒的用眼时长」不能大于「常规提醒的用眼时长」，"
                    "否则眨眼率偏低这条判据永远不起作用")
    for k in ("EYE_BREAK_SOON", "EYE_BREAK_AFTER", "EYE_REMIND_EVERY",
              "BLINK_LOW_RATE"):
        if cfg[k] <= 0:
            errs.append(f"{k} 必须为正数")
    return errs


def apply_config(cfg: dict) -> None:
    global ENGAGED
    for name, caster in _SCALARS.items():
        if name in cfg:
            globals()[name] = caster(cfg[name])
    for name in _LIST_KEYS:
        if name in cfg:
            vals = [str(v).strip() for v in cfg[name] if str(v).strip()]
            cur = globals()[name]
            globals()[name] = set(vals) if isinstance(cur, set) else tuple(vals)
    for name in _FLAGS:
        if name in cfg:
            globals()[name] = bool(cfg[name])
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


def hand_near_eye(eye_xy: tuple[float, float] | None, eye_span: float,
                  wrist: tuple[float, float, float], w: int, h: int) -> bool:
    """手腕是不是在眼周。wrist = (x, y, visibility)，归一化坐标。

    抽成纯函数是为了能直接断言 —— 这段几何换算有三处最容易写错，而且写错
    不会报错，只会**一直判 False**（揉眼次数永远是 0，看着像"你从不揉眼"）：
      1. 归一化坐标是各向异性的（x 除以宽、y 除以高），不乘回像素就比距离，
         画面越扁判得越歪；
      2. 半径必须跟着**眼距**缩放，否则人往后一靠（脸变小）就再也判不出来；
      3. 低 visibility 的手腕要丢掉，那是模型在瞎猜。

    **它判的是"手在脸附近"，不是"在揉眼睛"。** Pose 只给手腕、没有手指，
    托腮、扶眼镜、挠头全都会命中 —— 所以这个值只用于计数，不参与判定。
    """
    if eye_xy is None or eye_span <= 0:
        return False
    wx, wy, vis = wrist
    if vis < 0.5:
        return False
    ex, ey = eye_xy
    return math.hypot((wx - ex) * w, (wy - ey) * h) < HAND_EYE_RADIUS * eye_span * w


class BlinkTracker:
    """眨眼计数与眨眼率。纯逻辑，不碰视觉 —— 自检直接喂 EAR 序列就能测。

    一次"眨眼" = EAR 跌破阈值后又回升，且闭合时长 ≤ BLINK_MAX_DUR。
    更长的闭合不算眨眼（那是眯眼/微睡眠，归 EAR_SUSTAIN 那条管），
    否则"困得睁不开眼"会被数成"眨眼很频繁"，方向正好反了。

    `frontal` 是"这一帧正对屏幕"。只有正对屏幕时完成的眨眼才计入 ——
    转头时眼区被压缩、EAR 假性变低，不排除掉的话眨眼率会凭空翻几倍。
    但**每帧都要喂**（哪怕不正对）：状态机必须跟着走，不然转头期间的
    闭合会在回来那一帧被当成一次超长闭合，把后面的计数全带歪。
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._open = True
        self._start = 0.0
        self._last = -1e9
        # (时刻, 是否正对屏幕, 这一帧是否完成了一次眨眼)
        self._win: deque[tuple[float, bool, bool]] = deque()

    def feed(self, now: float, ear: float, thr: float, frontal: bool) -> None:
        blink = False
        if ear < thr:
            if self._open:
                self._open = False
                self._start = now
        elif not self._open:
            self._open = True
            if (now - self._start <= BLINK_MAX_DUR and frontal
                    and now - self._last >= BLINK_MIN_GAP):
                blink = True
                self._last = now
        self._win.append((now, frontal, blink))
        while self._win and now - self._win[0][0] > BLINK_WINDOW:
            self._win.popleft()

    def rate(self) -> float:
        """次/分钟。观察时长不足 BLINK_MIN_OBS 时返回 0.0，含义是"还不知道"。

        分母是**窗口里真正观察到正脸的那些秒**，不是窗口长度。用窗口长度的话，
        人中途走开五分钟、回来只看了十秒，会被算成"十秒里眨了 0 次"，
        眨眼率虚低 → 立刻误报"眼睛该休息了"。
        """
        obs = sum(1 for _, frontal, _ in self._win if frontal) / float(FACE_FPS)
        if obs < BLINK_MIN_OBS:
            return 0.0
        return sum(1 for _, _, b in self._win if b) * 60.0 / obs


class RubTracker:
    """揉眼计数。**只记录，不参与判定**（原因见常量区那一段）。

    手腕在眼周连续命中 HAND_EYE_HOLD 次才算一次 —— 单帧命中太容易是
    抬手路过、或者模型抖动。两次计数之间还要隔 RUB_MIN_GAP，免得
    一次揉眼（手在眼周来回动）被拆成好几次。
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._run = 0
        self._last = -1e9
        self._win: deque[float] = deque()

    def feed(self, now: float, hand_eye: bool) -> None:
        if hand_eye:
            self._run += 1
            if self._run >= HAND_EYE_HOLD and now - self._last >= RUB_MIN_GAP:
                self._last = now
                self._win.append(now)
                # 记完必须归零。不归零的话 _run 只增不减，冷却时间一过
                # 单独一帧就凑够条件 —— "连续命中 N 次"形同虚设，
                # 而且完全静默：计数看着在涨，没人会觉得它错了。
                self._run = 0
        else:
            self._run = 0
        while self._win and now - self._win[0] > BLINK_WINDOW:
            self._win.popleft()

    def count(self) -> int:
        """最近一个窗口内揉了几次。"""
        return len(self._win)


def eye_break_threshold(blink_rate: float) -> float:
    """连续用眼多久该提醒休息。

    眨眼率只能让提醒**提前**，不能让它闭嘴 —— 它是弱证据（依赖 EAR 阈值，
    比"坐了多久"脆弱得多），做成必要条件的话，检测一旦失灵就变成永远不提醒，
    而静默失效最难发现。和"应用类型只抬高不贬低"是同一条纪律。

    blink_rate == 0 表示样本不足、还不知道，此时按常规阈值处理。
    """
    if 0 < blink_rate < BLINK_LOW_RATE:
        return EYE_BREAK_SOON
    return EYE_BREAK_AFTER


def eye_break_message(eye_run: float, blink_rate: float) -> str:
    """提醒文案。眨眼率测出来了就把证据摆出来，没测出来就只讲时间。"""
    mins = max(1, int(eye_run // 60))
    if blink_rate <= 0:
        why = f"已经连续盯着屏幕 {mins} 分钟"
    elif blink_rate < BLINK_LOW_RATE:
        why = (f"已经连续用眼 {mins} 分钟，"
               f"这阵子眨眼只有 {blink_rate:.0f} 次/分（正常 15~20）")
    else:
        why = f"已经连续用眼 {mins} 分钟，眨眼 {blink_rate:.0f} 次/分"
    return f"{why}。抬头看看 6 米外的东西 20 秒，让眼睛歇一下。"


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
    """把当前窗口分成 work / distract / other 三类。

    **这个分类只用来"抬高"，不用来"贬低"** —— 见 decide() 里的用法：
    work 能升成「专注」，distract 只降到「中性」，永远不会降成「走神」。
    理由见常量区 DISTRACT_SWITCH_RATE 上方那段：在 B 站看高数课被这个
    分类直接判成走神，是实测报上来的 bug。
    """
    hay = f"{exe} {title}".lower()
    if any(k in hay for k in DISTRACT_KEYWORDS):
        # 学习豁免：平台是娱乐的、内容不是（B 站看 C++ 课）。命中就直接升成
        # work，也就是「专注」。它以前只是"免死金牌"，现在是真的加分项 ——
        # 因为 distract 已经不再扣分了。
        if any(k in hay for k in STUDY_KEYWORDS):
            return "work"
        return "distract"
    if exe in WORK_APPS:
        return "work"
    return "other"


def decide(*, face_present: bool, yaw: float, pitch: float, closed_for: float,
           idle_sec: float, app_kind: str, away_for: float,
           switch_rate: float = 0.0) -> str:
    """输入这一秒的聚合指标，输出状态。状态机全部规则都在这。

    switch_rate 是"最近一分钟里活动窗口切换了几次"，由调用方在滚动窗口上
    算好（见采集循环）。它有默认值 0.0，所以老调用方（自检里的单点断言）
    不改也不会被这条误伤。

    **应用类型只抬高、不贬低**：work → 专注，其余一律 → 中性。娱乐应用
    不再直接判走神，理由见常量区 DISTRACT_SWITCH_RATE 上方那段。
    """
    if idle_sec >= AWAY_IDLE or away_for >= AWAY_FACE:
        return "away"
    if not face_present:
        return "distracted"          # 短暂丢脸，还不算离开
    if closed_for >= EAR_SUSTAIN:
        return "drowsy"

    # 在几个页面之间来回切、哪个都没停住 —— 典型的"翻来翻去但什么都没干"。
    # 这是纯行为判据，不看用的是哪个应用。阈值是拿真实数据标定的。
    if switch_rate >= DISTRACT_SWITCH_RATE:
        return "distracted"

    if abs(yaw) <= YAW_TOL:
        if abs(pitch) <= PITCH_TOL:
            # 看着屏幕。这里就是那个 bug 的位置：原来 distract 直接返回
            # "distracted"，于是「看 B 站 = 走神」。现在只有白名单里的工作
            # 应用能升成专注，其余（含娱乐应用、不认识的）都是中性 ——
            # 中性算投入时长（见 ENGAGED），所以看网课不会再被扣分。
            return "focused" if app_kind == "work" else "neutral"
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
        # 揉眼检测要用：两眼中点 + 眼距，都归一化到画面。
        # 必须存下来，因为姿态是 2Hz、人脸是 10Hz —— 揉眼那一次姿态采样
        # 只能配上"最近一次人脸"的眼睛位置，不能指望同一帧里有脸有手。
        self._eye_xy: tuple[float, float] | None = None
        self._eye_span = 0.0

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
        # 两眼中点（归一化）+ 眼距，给 read_posture 判"手是不是在眼周"用。
        # 存归一化值而不是像素：两个模型拿到的是同一张画面、同一套归一化，
        # 但 read_posture 只知道自己那帧的 w/h，拿归一化值才能换算回去。
        x1, y1 = px(33)
        x2, y2 = px(263)
        self._eye_xy = ((x1 + x2) / 2.0 / w, (y1 + y2) / 2.0 / h)
        self._eye_span = scale
        return {"yaw": yaw, "pitch": pitch, "ear": ear, "scale": scale}

    def read_posture(self, frame_bgr: np.ndarray) -> dict | None:
        """返回 {"tilt": 肩线倾角或 None, "hand_eye": 手是否在眼周}。

        **为什么把 tilt 拆出来允许为 None**：原来是"看不到肩膀就整条返回 None"。
        但揉眼时手常常正好挡住肩膀 —— 合并返回的话，"手在眼周"这个观测
        会在最需要它的时刻消失。两者互不依赖，就该分开报。

        **hand_eye 只是"手在脸附近"，不是"在揉眼睛"。** MediaPipe 的 Pose
        只给 33 个身体点，**没有手指**，只有手腕（15/16）。所以托腮、扶眼镜、
        挠头、喝水全都会命中。要真分得清得上 HandLandmarker（第三个模型，
        CPU 明显更高）。因此这个值**只用于计数**，先攒真实数据再定阈值 ——
        跟当初标定走神阈值是同一个套路。
        """
        h, w = frame_bgr.shape[:2]
        res = self.pose.detect_for_video(self._img(frame_bgr), self._ts())
        if not res.pose_landmarks:
            return None
        lm = res.pose_landmarks[0]
        ls, rs = lm[11], lm[12]
        tilt = (shoulder_tilt(ls.x * w, ls.y * h, rs.x * w, rs.y * h)
                if ls.visibility >= 0.5 and rs.visibility >= 0.5 else None)

        hand_eye = False
        if self._eye_xy is not None and self._eye_span > 0:
            for i in (15, 16):                      # 左手腕 / 右手腕
                wr = lm[i]
                if hand_near_eye(self._eye_xy, self._eye_span,
                                 (wr.x, wr.y, wr.visibility), w, h):
                    hand_eye = True
                    break
        return {"tilt": tilt, "hand_eye": hand_eye}


# ══════════════════════ 采集线程 ══════════════════════

# 运行时共享状态放模块级而不是 Monitor 实例上：dashboard 可能独立运行
# （uv run dashboard.py）或作为托盘子进程被 import，不能依赖拿到 Monitor 对象。
_runtime = {"paused": False, "heartbeat": 0.0, "eye": None}


def eye_status() -> dict | None:
    """视疲劳的实时状态，供面板渲染横幅。没采集过返回 None。

    和 is_paused() 一样的取舍：只有"dashboard 与采集在同一进程"时才读得到，
    独立 `uv run dashboard.py` 时读不到 —— 那就干脆不显示横幅，
    而不是显示一个永远不更新的旧值。
    """
    return _runtime.get("eye")


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
    present INTEGER, idle REAL,
    blink REAL, rub INTEGER
);
CREATE INDEX IF NOT EXISTS idx_ts ON samples(ts);
"""

# 落库的列顺序**只写在这一处**。原来 5 个 INSERT 都写的是位置参数
# `VALUES(?,?,?,…)`，加一列就得同时改 5 处，漏一处是"列数不匹配"报错，
# 更糟的是改错顺序 —— 那种错不会报，只会静默把值塞进错误的列。
SAMPLE_COLS = ("ts", "state", "app", "title", "yaw", "pitch", "ear",
               "tilt", "scale", "present", "idle", "blink", "rub")
_SAMPLE_INSERT = (f"INSERT INTO samples({','.join(SAMPLE_COLS)}) "
                  f"VALUES({','.join('?' * len(SAMPLE_COLS))})")

# 后加的列。CREATE TABLE IF NOT EXISTS 对**已存在**的表什么都不做，
# 所以用户手上那份十几万行的 focus.db 必须靠 ALTER 补列。
_MIGRATIONS = (
    ("blink", "REAL"),      # 每分钟眨眼次数；0 = 样本不足，还不知道
    ("rub", "INTEGER"),     # 最近一分钟揉眼次数（只记录，不参与判定）
)


def ensure_columns(conn: sqlite3.Connection) -> list[str]:
    """给老的 samples 表补上后加的列，返回这次真正补了哪些。

    没有它的话，升级后第一次落库就炸 —— 而且是运行一段时间才炸
    （库是旧格式、代码以为有新列），日志里只有一句
    `table samples has no column named blink`，很难联想到"该迁移了"。
    """
    have = {r[1] for r in conn.execute("PRAGMA table_info(samples)")}
    if not have:
        return []              # 表还不存在，executescript(SCHEMA) 会一次建全
    added = []
    for name, decl in _MIGRATIONS:
        if name not in have:
            conn.execute(f"ALTER TABLE samples ADD COLUMN {name} {decl}")
            added.append(name)
    return added


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

def journal_mode(conn: sqlite3.Connection) -> str:
    """读回连接当前实际的日志模式。"""
    row = conn.execute("PRAGMA journal_mode").fetchone()
    return str(row[0]).lower() if row else "?"


_wal_warned = False


def _switch_wal(conn: sqlite3.Connection) -> bool:
    """在 conn 上尝试切到 WAL，**读回实际模式**判断成败。

    为什么不能"发一条 PRAGMA 就当成功了"：`PRAGMA journal_mode=WAL` 需要
    **排他锁**，拿不到就抛 `database is locked` —— 而 `busy_timeout` 在这条
    PRAGMA 上**不起作用**。实测：旁边有一个持写锁的连接（BEGIN IMMEDIATE）
    时，它在 **0.0 秒**就抛了，一点不等。

    失败时必须如实返回 False。假装成功的话，库其实还留在
    `journal_mode=delete` —— 那个模式**读写互斥**，面板/报告每一条读都可能
    撞锁，用户看到的就是「自评分打不开：database is locked」，而且这个状态
    会一直持续下去（没人会再去切第二次）。
    """
    global _wal_warned
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.DatabaseError as exc:
        if not _wal_warned:
            _wal_warned = True
            log.warning(
                "切 WAL 失败（库仍是 %s）：%s\n"
                "  这个模式下读写互斥，面板和报告随时可能报 database is locked。"
                "采集线程攒批提交会连续持有写锁十几秒，运行期基本切不动 —— "
                "重启一次即可（启动时采集还没起来，没有竞争）。",
                journal_mode(conn), exc)
        return False
    return journal_mode(conn) == "wal"


def ensure_wal(timeout: float = 20.0) -> bool:
    """把 focus.db 切到 WAL 并**确认**成功；返回是否处于 WAL。

    **必须在采集线程起来之前调用**（main 里就是这么用的）。这是唯一能可靠
    切成功的时机：没有别的连接持锁。库一旦切过去就是持久的，之后每次启动
    只是读一下模式，几乎零成本。

    为什么值得单独立一个入口，而不是靠 open_db 每次顺手切：采集线程每
    SAMPLE_COMMIT_EVERY 条才 commit 一次，也就是**连续持有写锁十几秒**，
    那段时间里切模式必然失败。所以运行期的尝试只是兜底，
    真正的迁移必须发生在启动时。
    """
    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    try:
        before = journal_mode(conn)
        if before == "wal":
            return True
        deadline = time.time() + timeout
        while True:
            if _switch_wal(conn):
                log.info("focus.db 已切到 WAL（原模式 %s）", before)
                return True
            if time.time() >= deadline:
                return False
            time.sleep(0.5)
    finally:
        conn.close()


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
    # 先读回实际模式再决定要不要切。库已经是 WAL 时这一步只是读一下文件头，
    # 不去抢锁，所以挂在每个连接上也不心疼。
    if journal_mode(conn) != "wal":
        # 运行期切多半切不动（见 _switch_wal / ensure_wal 的说明），但值得试
        # 一次：这个进程可能没走 main（--report、--dashboard 单独跑），
        # 也可能上一次是旧版本进程留下的 delete 模式。
        if _switch_wal(conn):
            log.info("focus.db 已切到 WAL")
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
        self.on_eye_break = lambda run, blink: None   # 视疲劳提醒（眼睛该歇会儿了）
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
        # 老库要在这里补列，否则第一次落库才炸（见 ensure_columns 的说明）
        _added = ensure_columns(conn)
        if _added:
            log.info("数据库补列：%s", "、".join(_added))
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
        # 上一次真的看到脸的时刻。只用来判断"这次闭眼计时是不是跨过了盲区"。
        # 初值 0.0 而不是 now：脸一次都没见过时 now - 0.0 是个巨大的数，
        # 正好落进"丢脸超时"那一支，不会留下一个假的计时起点。
        last_face_seen = 0.0
        lean = 0.0
        # 窗口/页面切换的时刻，用来算"来回切换"的频率（走神判据之一）。
        # 只留最近 DISTRACT_SWITCH_WINDOW 秒，窗口外的即时丢掉。
        switch_at: deque[float] = deque()
        last_window: tuple[str, str] | None = None
        # ── 视疲劳（眼睛该歇会儿了）──
        # 这三个都是纯统计，不参与状态判定（见常量区那一段）。
        blinks = BlinkTracker()
        rubs = RubTracker()
        # 这一轮"连续用眼"从何时开始。只在真的离开过（away 持续够久）才清零，
        # 否则起身倒杯水回来就重新计时，永远攒不到 40 分钟。
        eyes_since: float | None = None
        last_eye_remind = 0.0

        try:
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
                    # ── 视觉：人脸 / 姿态 ──
                    #
                    # 这两段必须放在这个 try 里，不能放外面：except 那一支本来就写着
                    # "非数据库异常（模型、解码、算法）几乎总是瞬时或单帧的" ——
                    # 模型推理正是这里最可能抛异常的东西，一个坏帧不该杀掉采集。
                    #
                    # 为什么给这段写了这么多注释：它在一次采集循环重构里被整段删掉过
                    # （vision 调用 + 四个缓冲区的填充一起没了，消费端却留着）。
                    # 后果是 face_buf / tilt_buf / _ear_hist 永远为空 → present 恒为
                    # False → decide() 只会返回 distracted / away，**专注、中性、伏案、
                    # 疲劳四种状态一个都出不来**，而自检全绿 —— 因为它测的是 decide()
                    # 和 ear_threshold() 本身，测不出"输入根本没接上"。
                    # 现在有两道防线盯这个：自检末尾的结构性断言（搜"视觉链路断了"）
                    # 负责"生产端被删了"，CI 里的 smoke_pipeline.py 负责真跑一遍、
                    # 确认喂了正对屏幕的人脸之后确实产出「专注」。
                    if now - last_face >= 1.0 / FACE_FPS:
                        last_face = now
                        # 上一次真的看到脸已经过去太久了 —— 中间那段盲区里的闭眼计时
                        # 不该接着累加，作废它。
                        #
                        # closed_since 只由"看到脸且 EAR 低"推进，丢脸时原本不动它，
                        # 于是"闭眼 2 秒 → 转头出画 3 秒 → 回来还闭着眼"会被算成
                        # 连续闭眼 5 秒，人刚回来 1 秒就记一条「疲劳」（实测复现，
                        # 见 smoke_pipeline.py 里那段脚本化的时间线）。拦住它的那条
                        # away_for >= AWAY_FACE 是 20 秒，拦不住 3 秒的短转头。
                        #
                        # 判据是"盲区超过了宽限期"，不是"有没有掉帧"：单帧抖动
                        # （now - last_face_seen 只有 0.1 秒）不清零，真断了才清零。
                        # 留宽限是必须的 —— FACE_FPS=10 下单帧丢失很常见，掉一帧就
                        # 清零的话，真犯困时反而永远攒不满 EAR_SUSTAIN。
                        #
                        # 为什么不怕把真疲劳一起清掉：脸不在画面里的时候 decide()
                        # 本来就返回 distracted，closed_for 再大也轮不到它说话。
                        #
                        # 放在"重新开始观察"这一侧（而不是丢脸那一侧）是有意的：
                        # 丢脸、暂停后恢复、摄像头重开、休眠唤醒都归这一条管 ——
                        # 它们共同的特征就是"这一帧之前有一段时间没在观察"。
                        # 写在丢脸那一支的话，暂停那条路径根本走不到（它在循环开头
                        # 就 continue 了），恢复时那个陈旧的 closed_since 会直接命中。
                        if (closed_since is not None
                                and now - last_face_seen > FACE_LOSS_GRACE):
                            closed_since = None
                        got = vision.read_face(proc)
                        if got:
                            face_buf.append(got)
                            scale_hist.append(got["scale"])
                            self._ear_hist.append(got["ear"])
                            base = float(np.median(scale_hist)) if scale_hist else got["scale"]
                            lean = got["scale"] / base if base > 1e-6 else 1.0
                            closed_since = (None if got["ear"] >= self._ear_thr
                                            else (closed_since or now))
                            # 眨眼统计。**每一帧都要喂**（哪怕没正对屏幕）：
                            # 状态机得跟着走，否则转头期间的那次闭合会在回来
                            # 那一帧被当成一次超长闭合，把后面的计数带歪。
                            # 是否计入由 tracker 内部按 frontal 决定。
                            blinks.feed(now, got["ear"], self._ear_thr,
                                        abs(got["yaw"]) <= YAW_TOL)
                            away_since = None
                            last_face_seen = now
                        else:
                            away_since = away_since or now

                    if now - last_pose >= 1.0 / POSE_FPS:
                        last_pose = now
                        pose = vision.read_posture(proc)
                        if pose is not None:
                            # tilt 可能是 None（手挡住肩膀），那是正常的：
                            # 看不到肩膀就沿用上一帧的中位数，别把整个观测丢掉。
                            if pose["tilt"] is not None:
                                tilt_buf.append(pose["tilt"])
                            rubs.feed(now, pose["hand_eye"])

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

                        # 「在几个页面之间来回切」的度量：最近一分钟换了几次
                        # 前台窗口。这是纯行为判据，不看是哪个应用 ——
                        # 见 decide() 里对它的用法和常量区的标定说明。
                        #
                        # 只在每秒结算时看一次，所以同一秒内连切几次只会被记成
                        # 一次。真实的切换比这慢得多，够用，而且省掉一层采样。
                        if (exe, title) != last_window:
                            last_window = (exe, title)
                            switch_at.append(now)
                        while (switch_at
                               and now - switch_at[0] > DISTRACT_SWITCH_WINDOW):
                            switch_at.popleft()
                        switch_rate = len(switch_at) * (60.0 / DISTRACT_SWITCH_WINDOW)

                        st = decide(face_present=present, yaw=yaw, pitch=pitch,
                                    closed_for=closed_for, idle_sec=idle_seconds(),
                                    app_kind=kind, away_for=away_for,
                                    switch_rate=switch_rate)

                        # ── 视疲劳：眼睛该歇会儿了 ──
                        # **不参与 decide()**：状态描述的是"我在干什么"，
                        # 而"眼睛该休息"是关于生理负荷的建议，混进状态会把
                        # 投入时长统计污染掉（见常量区那一段）。
                        if st == "away" and away_for >= EYE_REST_RESET:
                            # 真的离开过这么久才算休息，用眼计时清零。
                            # 不这么写的话，起身倒杯水回来就重新计时，
                            # 40 分钟这条永远攒不满。
                            eyes_since = None
                        elif st != "away" and eyes_since is None:
                            eyes_since = now
                        eye_run = (now - eyes_since) if eyes_since else 0.0
                        blink_rate = blinks.rate()
                        rub_rate = rubs.count()
                        if (eye_run >= eye_break_threshold(blink_rate)
                                and now - last_eye_remind >= EYE_REMIND_EVERY):
                            last_eye_remind = now
                            self._announce_eye_break(eye_run, blink_rate)
                        # 面板要读（同一进程时；独立跑 dashboard.py 时读不到，
                        # 和 is_paused() 一样的取舍，见 _runtime 的注释）。
                        _runtime["eye"] = {"run": eye_run, "blink": blink_rate,
                                           "rub": rub_rate, "at": now}

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
                                _SAMPLE_INSERT,
                                (now, st, exe, title[:200], yaw, pitch, ear, tilt,
                                 lean, 1 if present else 0, idle_seconds(),
                                 blink_rate, rub_rate))
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

        finally:
            # 无论怎么退出都要**关掉写连接** —— 正常停止、还是没接住的异常穿出去。
            #
            # 这一条是"库被锁死两小时"那个事故的最后一道防线。实测的用户日志：
            # 采集线程死在 `conn.commit()` 上（database is locked），连接没关、
            # 写事务还开着，于是**接下来 1 小时 49 分里每一条读都撞同一个锁** ——
            # 面板、报告、自评分全部 500，而进程还活着、桌面图标也还在，
            # 看起来"在跑"。把清理放进 finally，线程怎么死都不会把库晾在锁定状态。
            #
            # 光靠循环体里的 try/except 是不够的：那个 try 只包住"每秒结算"那一段，
            # 摄像头 read、cv2.resize、重开摄像头都在它外面 —— 那些地方抛一次
            # 就会穿出去。这不是假想：这次就是这么死的。
            if cap is not None:        # 暂停状态下退出时它已经是 None 了
                cap.release()
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

    def _announce_eye_break(self, eye_run: float, blink_rate: float) -> None:
        """视疲劳提醒。

        和评分提醒不同，这条**不等自然断点** —— 它的全部意义就是打断你
        （让你抬头看远处）。所以冷却时间（EYE_REMIND_EVERY）是唯一的约束。

        回调抛异常不能连累采集：托盘没了、气泡弹不出来，都不该让采集线程死。
        这里和 _prompt_rating 一样包一层，而且**失败也要算作已提醒** ——
        否则通知坏掉之后会变成每转一圈重试一次，日志被刷爆。
        """
        msg = eye_break_message(eye_run, blink_rate)
        log.info("视疲劳提醒：%s", msg)
        try:
            self.on_eye_break(eye_run, blink_rate)
        except Exception:
            log.exception("视疲劳提醒回调失败")


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
    """面板地址。**端口可能不是默认的，所以别硬编码 8787。**

    优先读监视进程写下的 URL_PATH —— 8787 被占用时服务会顺延到 8788…8806，
    而**别的进程**（浏览器兜底、桌面开关 fork 的 --wait-open）拿不到
    `dashboard.base_url()` 的权威值：那个函数读的是本进程的 `_server`，
    在子进程里是 None，只会退回默认端口，于是等一个没人监听的地址。

    URL_PATH 还没有时才问 dashboard。注意那条路会**顺带把服务起起来**
    （`serve_background` 的语义），所以它只适合本进程/浏览器兜底用 ——
    子进程要走 `_persisted_panel_url()`，见那个函数的注释。

    path 拼在根地址后面（如 "/rate"）。
    """
    try:
        base = URL_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        base = ""
    if not base:
        import dashboard
        dashboard.serve_background(open_browser=False)
        base = dashboard.base_url()
    return base.rstrip("/") + path


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


# 页面路由 —— 和 window.py 里那份保持一致。这里**刻意不 import window**：
# 这个函数的全部意义就是"窗口层没了也要能把页面显示出来"。
_PAGE_PATHS = {"panel": "/", "rate": "/rate", "report": "/report"}


def open_page(page: str) -> None:
    """把面板 / 自述评分 / 完整报告显示出来 —— **窗口层不可用也必须能看**。

    优先走应用窗口（pywebview / WebView2）。窗口层加载不了时（没装
    pywebview、pythonnet 起不来、系统缺 WebView2 运行时）退到系统浏览器：
    这些页面本来就是本地 HTTP 服务，浏览器一样能看。

    为什么值得单独抽一个函数：托盘菜单里原本是各处自己
    `import window; window.open_page(...)`。窗口层一坏，**所有菜单项都变成
    "点了没反应"** —— 而它们正是用户唯一的入口（发布包是 --windowed 的，
    没有控制台）。更糟的是 `import window` 抛出的异常在菜单回调里没人接，
    等于把"打不开页面"升级成"托盘失灵"。

    `import window` 失败**必须**走浏览器兜底，不能只是记条日志就算了 ——
    "点了没反应"和"用浏览器打开了"对用户是天壤之别。

    另外这个 import 是**延迟**的（放在函数里而不是模块顶层）：
    pywebview 会拖起 pythonnet，开机自启时不该为它多花那几百毫秒 ——
    而绝大多数启动根本不会调用这个函数。
    """
    try:
        import window
        window.open_page(page)
        return
    except Exception:
        log.exception("应用窗口打不开（%s），改用系统浏览器", page)
    # 走 _panel_url 而不是硬编码 8787：端口被占时 dashboard 会往后顺延，
    # 写死端口的话兜底打开的会是一个连不上的地址 —— 比不打开更让人困惑。
    try:
        url = _panel_url(_PAGE_PATHS.get(page, "/"))
    except Exception:
        log.exception("连面板地址都拿不到，放弃打开 %s", page)
        return
    try:
        webbrowser.open(url)
        log.info("已改用系统浏览器打开：%s", url)
    except Exception:
        log.exception("系统浏览器也打不开：%s", url)


def run_tray(mon: Monitor) -> None:
    import pystray

    # 没有窗口层时，主线程靠它退出（有窗口时靠 quit_app 让 GUI 循环返回）。
    _quit = threading.Event()

    def refresh(icon) -> None:
        # 桌面开关的图标也在这里一起刷。托盘图标能自己反映"在跑但没数据"，
        # 而 .lnk 的图标是个静态文件 —— 只有我们主动重写它才会变。
        # 放在最前面：下面那条 is_stale 分支会 return，跟在后面就漏了
        # —— "采集线程死了"恰恰是桌面图标最该从绿变黄的时刻。
        # running=True 是确定的：托盘线程活着就说明本进程在跑。
        # 状态没变时它只 stat 一下，不会真的写文件。
        sync_shortcut_icon(running=True)

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

        受 `RATE_REMIND` 管（设置页「行为 → 到点提醒打分」）。
        门放在**这一层**（通知），不是放在 `Monitor._prompt_rating` 那一层
        （判定）：这个开关管的是"弹不弹气泡"，不是"要不要检测该打分了" ——
        判定那层还兼着推进 `_last_prompted`，掐掉它会让"关一阵再打开"变成
        一次性补一堆提醒。待评时段也照样在攒，只是不吵你。
        """
        # 放在最前面：`_pending_ratings()` 会开一次库、读两天的数据，
        # 静音的时候连这一步都不必跑。
        if not RATE_REMIND:
            return
        try:
            n = _pending_ratings()
            tip = (f"刚才那 30 分钟你觉得自己专注吗？"
                   f"点托盘图标可直接打（待评 {n} 个时段）" if n > 0
                   else "刚才那 30 分钟你觉得自己专注吗？点托盘图标打分")
            icon.notify(tip, "专注监视 · 该打个分了")
        except Exception:
            log.exception("托盘通知失败")

    mon.on_block_end = on_block_end

    def on_eye_break(eye_run: float, blink_rate: float) -> None:
        """视疲劳提醒：眼睛该歇会儿了。

        和评分提醒不同，这条**就是要打断你**（让你抬头看远处），
        所以不等自然断点，只受 EYE_REMIND_EVERY 冷却约束。

        受 `EYE_BREAK_REMIND` 管（设置页「行为 → 眼睛该歇会儿了」）。
        门和 RATE_REMIND 一样放在**这一层**（通知），不放在
        `Monitor._announce_eye_break` 那一层：那层还兼着写
        `视疲劳提醒：…` 那行日志，静音之后它**必须继续写** ——
        否则"没提醒"就分不清是眼睛不累、还是这条链路坏了。
        """
        if not EYE_BREAK_REMIND:
            return
        try:
            icon.notify(eye_break_message(eye_run, blink_rate),
                        "专注监视 · 眼睛该歇会儿了")
        except Exception:
            log.exception("视疲劳提醒失败")

    mon.on_eye_break = on_eye_break

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
        open_page("panel")

    mon.on_progress = on_progress
    mon.on_fatal = on_fatal

    def on_panel(icon, _item):
        open_page("panel")

    def on_rate(icon, _item):
        open_page("rate")

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
        open_page("report")

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

    def on_shortcut(icon, _item):
        """在桌面建「专注监视」开关。

        为什么要做成托盘菜单项：发布包是 `--windowed` 的，**没有控制台**，
        所以 exe 用户根本敲不了 `--install-shortcut`。而教程把桌面开关
        写成日常入口 —— 没有这个菜单项，那条路对 exe 用户就是不存在的。

        也**不要**让用户"右键 exe → 发送到桌面"：那样建出来的快捷方式
        没有 `--toggle` 参数，双击只会去起第二个实例、被单例拦住。
        开关必须由 _write_lnk() 生成（它走 relaunch_cmd()，冻结态才指得对）。
        """
        try:
            install_shortcut()
            icon.notify("已在桌面建好「专注监视」开关，双击即切换开/关",
                        "专注监视")
        except Exception as exc:
            log.exception("建桌面开关失败")
            icon.notify(f"建桌面开关失败：{exc}", "专注监视")

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
        _quit.set()                       # 没有窗口层时，主线程靠它退出
        try:
            import window                 # 延迟：与其余菜单项保持一致
            window.quit_app()             # 销毁窗口 → GUI 循环返回 → 进程真正退出
        except Exception:
            # 窗口层坏了不该把"退出"变成"退不掉"：_quit 已经置位，
            # 主线程会从 run_tray 返回，main() 的 finally 照常收尾。
            log.exception("销毁窗口失败，主线程会自行退出")

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
            pystray.MenuItem("在桌面建开关快捷方式", on_shortcut),
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
    #
    # **窗口层加载不了不能把整个程序带走。** 这里是 run_tray 的最后一步，
    # 异常穿出去会被 main() 的 `except BaseException` 接住然后 `raise` ——
    # 进程直接退出，而 pythonw 没有 stderr，用户看到的只是
    # "托盘图标闪一下就没了"。而采集线程和托盘都**不依赖**窗口层：
    # 没装 pywebview、pythonnet 加载失败、系统缺 WebView2 运行时，
    # 这些都只该让"窗口"这个功能降级，不该让记录停摆。
    # （实测事故：用户的 .venv 里没有 pywebview —— 他更新前那版是用浏览器
    #  开面板的，压根不需要这个依赖 —— 启动两秒后进程就没了。）
    try:
        import window
    except Exception:
        log.exception("窗口层加载不了，面板/报告将改用系统浏览器；"
                      "记录与托盘继续运行")
        window = None
    if window is None:
        # 没有窗口层就没有"GUI 循环"可等，主线程只能守着。
        # 直接 return 的话 main() 会走到结尾 —— 而托盘和采集都是 daemon 线程，
        # 主线程一退出它们就被硬拔掉，程序等于白启动。
        _quit.wait()
        return
    window.start_main()


# ══════════════════════ 开 / 关 / 桌面开关 ══════════════════════

ICON_DIR = ROOT / "icons"
LNK_NAME = "专注监视.lnk"
PYW = ROOT / ".venv" / "Scripts" / "pythonw.exe"


def relaunch_cmd(frozen: bool | None = None) -> list[str]:
    """「重新启动自己」的命令行 —— 桌面开关和 --wait-open 都靠它。

    两种形态，差别是**入口在哪**：

    - 源码运行：`<ROOT>/.venv/Scripts/pythonw.exe <ROOT>/focus.py`
      用 pythonw 而不是 python，是为了不弹控制台窗口。
    - 冻结成 exe：**exe 自己就是入口，不能再传 focus.py**。
      发布包里既没有 `.venv` 也没有 `focus.py` —— 照搬源码那套，
      `subprocess.Popen` 会抛 FileNotFoundError，而 exe 是 `--windowed`、
      没有控制台，用户看到的只是"双击了但什么都没发生"。
      更糟的是桌面快捷方式的 TargetPath 也是这个值，等于**开关直接是死的**。

    frozen 写成可注入的参数（而不是只读 `sys.frozen`）是为了让它能被测到：
    开发机上 `sys.frozen` 永远是 False，那条分支不抽成纯函数就永远没人验证过。
    """
    if frozen is None:
        frozen = bool(getattr(sys, "frozen", False))
    if frozen:
        return [str(sys.executable)]
    return [str(PYW), str(ROOT / "focus.py")]

# 子进程一律不要弹控制台 —— 这些函数可能跑在 pythonw 下
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_DETACHED = (getattr(subprocess, "DETACHED_PROCESS", 0)
             | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))


def _desktop() -> Path:
    return Path(os.environ.get("USERPROFILE", "")) / "Desktop"


# 桌面开关的三种图标。快捷方式是个独立的应用图标（圆底 + 一个记号），
# 不是托盘那种小圆点，所以形状得能一眼分辨：
#   绿圆       = 在记录
#   黄底感叹号 = 在跑，但没在记录（模型没下来、摄像头坏了、采集线程死了）
#   灰底横杠   = 没在跑
#
# 三个**不同的文件**是必须的：.lnk 的 IconLocation 存的是路径，Windows 按路径
# 缓存图标 —— 同一个路径只换内容，桌面往往不刷新。
_ICON_STATES: dict[str, tuple[str, str | None]] = {
    "on":    ("#22c55e", None),      # 绿点
    "alert": ("#f59e0b", "bang"),    # 黄底感叹号
    "off":   ("#64748b", "bar"),     # 灰底横杠
}


def ensure_icons() -> dict[str, Path]:
    """生成桌面开关的三个图标，返回 {状态: 路径}。

    快捷方式的图标是静态的，没法自己反映运行状态 —— 只能在状态变化时
    连图标一起重写 .lnk，Explorer 会立刻刷新。
    """
    from PIL import Image, ImageDraw
    ICON_DIR.mkdir(exist_ok=True)
    out: dict[str, Path] = {}
    for state, (color, mark) in _ICON_STATES.items():
        p = ICON_DIR / f"{state}.ico"
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.ellipse((5, 5, 59, 59), fill=color)
        if mark == "bar":            # 关：中间一道横杠，和「开」一眼区分
            d.rectangle((15, 27, 49, 37), fill="#0f172a")
        elif mark == "bang":         # 在跑但没记录：和托盘一样用感叹号
            d.rounded_rectangle((28, 14, 36, 40), radius=4, fill="#0f172a")
            d.ellipse((28, 45, 36, 53), fill="#0f172a")
        img.save(p, format="ICO", sizes=[(16, 16), (32, 32), (48, 48), (64, 64)])
        out[state] = p
    return out


def shortcut_state(*, running: bool, stale: bool) -> str:
    """桌面开关此刻该显示哪个图标。

    抽成纯函数是为了能断言 —— 这条规则正是用户报的那个 bug：
    **图标要反映"程序在不在跑"，不是"用户按过开关"。**

    注意 `running=False` 时 stale 不再有意义：进程都没了，一律 off。
    """
    if not running:
        return "off"
    return "alert" if stale else "on"


_lnk_state: str | None = None       # 上一次写进快捷方式的图标状态


def should_sync_shortcut(lnk_exists: bool, state: str,
                         last_state: str | None, force: bool) -> bool:
    """要不要现在去重写那个 .lnk。两条规则，都很容易写反：

    - **快捷方式不存在就别去建。** 不是每个用户都要桌面开关（教程里是可选
      步骤）。程序启动时也会刷一次图标 —— 少了这条判断，等于**替所有用户
      在桌面上凭空放一个快捷方式**，还是删了下次启动又长出来的那种。
    - **状态没变就别重写。** 看门狗每 30 秒调一次，_write_lnk 要起一次
      PowerShell（几百毫秒）并重写桌面文件，没必要。force 留给"快捷方式
      可能被用户删掉又重建了"的场合。

    抽成纯函数是为了能断言 —— 这两条都是"条件写反了也照样跑得通、
    只有用户看得见"的那类。
    """
    return lnk_exists and (force or state != last_state)


def sync_shortcut_icon(force: bool = False, running: bool | None = None) -> None:
    """把桌面开关的图标刷成程序**此刻真实的状态**。

    为什么必须由程序自己来刷：.lnk 的图标是个静态文件，只在写的那一刻成立。
    原来只有 toggle() 在按下开关时写一次绿点，而且**在确认程序起来之前**就写了；
    之后无论程序变成什么样都不再更新 —— 于是"网络出错导致模型下不下来、
    采集根本没起来"的时候，桌面图标依然是绿的，用户以为在记录，其实一条样本
    都没有。（用户报的"待机/断网后再开启，图标是绿的但没在运行"就是这个。）

    所以调用方要么传**观测到的** running（toggle 里是 _start_engine 的返回值），
    要么干脆别传、让它自己去查 —— 永远不要传"我以为它在跑"。
    程序内部（托盘看门狗、退出兜底）直接传 True/False，省掉一次 tasklist。
    """
    global _lnk_state
    try:
        if running is None:
            running = running_pid() is not None
        state = shortcut_state(running=running, stale=is_stale())
        lnk = _desktop() / LNK_NAME
        if not should_sync_shortcut(lnk.exists(), state, _lnk_state, force):
            return
        _write_lnk(ensure_icons()[state])
        _lnk_state = state
        log.info("桌面开关图标 -> %s", state)
    except Exception:
        # 图标刷不上不该影响监视本身
        log.exception("刷新桌面开关图标失败")


def _wait_until_running(timeout: float = 15.0, probe=None) -> bool:
    """等刚启动的实例真的把 PID 文件写出来；超时算失败。

    PID 文件是在 acquire_singleton() 之后**立刻**写的，早于加载模型，
    所以正常情况下一两秒就返回。会走到超时的只有"进程压根没起来"：
    入口路径不对、单例被一个残留进程占着、exe 启动即崩。

    `probe` 可注入是为了让自检能覆盖"起没起来"两支：开发机上真开着监视时
    running_pid() 恒为真，"失败"那一支永远没人跑过。
    """
    if probe is None:
        probe = running_pid
    deadline = time.time() + timeout
    while time.time() < deadline:
        if probe() is not None:
            return True
        time.sleep(0.5)
    return False


def _complain(title: str, text: str) -> None:
    """把"操作失败了"真的告诉用户。

    双击开关的是个脱离终端的进程，而发布包是 `--windowed` ——
    `sys.stdout` 是 None，`print` 等于什么都没做。所以这里必须弹窗，
    否则用户面对的就是"双击了，没反应，也没人告诉他为什么"。
    """
    log.error("%s：%s", title, text.replace("\n", " "))
    try:
        # MB_OK | MB_ICONWARNING | MB_SETFOREGROUND | MB_TOPMOST
        ctypes.windll.user32.MessageBoxW(None, text, title, 0x00050030)
    except Exception:
        log.exception("弹窗失败（非 Windows 或没有桌面会话）")


def _write_lnk(icon: Path, cmd: list[str] | None = None) -> None:
    """重写桌面快捷方式。.lnk 只能走 COM，交给 PowerShell，并隐藏它的窗口。

    目标必须走 relaunch_cmd()：冻结成 exe 之后入口是 exe 本身，
    写死 `.venv/Scripts/pythonw.exe` 的话这个快捷方式在发布包里是**死的**
    （TargetPath 指向一个不存在的文件，双击毫无反应、也没有任何报错）。

    `cmd` 可注入是为了让自检能证明"目标不是写死的"：只断言"命令里出现了
    pythonw.exe"是测不出来的 —— 写死路径的旧写法同样含 pythonw.exe。
    只有"换一个哨兵命令进去、它必须被原样采纳"才能把这两种写法区分开。

    先写临时文件再 os.replace：这个函数现在会被**程序运行时**反复调用
    （启动、看门狗、退出兜底），而桌面上那个 .lnk 很可能正被 Explorer
    或用户的操作占用 —— 直接 Save() 覆盖，中途失败就会留下一个 0 字节的
    快捷方式，双击没反应、右键删也删不干净。临时文件必须放在**同一个
    目录**（同一个卷），os.replace 才能是原子改名；放 %TEMP% 的话，
    桌面被重定向到别的盘时会直接 OSError 跨卷失败。
    临时名保留 .lnk 后缀：CreateShortcut 的路径参数按 .lnk 处理，
    换成 .tmp 后缀是没验证过的写法。
    """
    lnk = _desktop() / LNK_NAME
    if cmd is None:
        cmd = relaunch_cmd()
    args = " ".join([f'"{a}"' for a in cmd[1:]] + ["--toggle"])
    tmp = lnk.with_name(f"{LNK_NAME}.new.lnk")
    tmp.unlink(missing_ok=True)
    ps = (
        "$s=(New-Object -ComObject WScript.Shell).CreateShortcut('%s');"
        "$s.TargetPath='%s';"
        "$s.Arguments='%s';"
        "$s.WorkingDirectory='%s';"
        "$s.IconLocation='%s,0';"
        "$s.Description='专注度监视（双击切换开/关）';"
        "$s.Save()"
    ) % (tmp, cmd[0], args, ROOT, icon)
    try:
        subprocess.run(["powershell", "-NoProfile", "-NonInteractive",
                        "-Command", ps],
                       creationflags=_NO_WINDOW, capture_output=True,
                       # 这个函数现在也会被采集线程调到（状态一变就刷图标），
                       # PowerShell 万一卡住不能把采集一起拖死。
                       timeout=20)
    except subprocess.TimeoutExpired:
        tmp.unlink(missing_ok=True)
        raise OSError("PowerShell 超时，快捷方式没更新") from None
    if tmp.exists():
        os.replace(tmp, lnk)
    else:
        # PowerShell 那边没写出东西来（COM 被策略挡了、执行失败）。
        # 报错比留一个"看起来刷新了其实没刷"的假象好。
        tmp.unlink(missing_ok=True)
        raise OSError(f"PowerShell 没能写出快捷方式：{tmp}")


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
        URL_PATH.unlink(missing_ok=True)   # 没实例了，地址也不该留着
        print("没有找到运行中的实例。")
        return
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                   capture_output=True, creationflags=_NO_WINDOW)
    PID_PATH.unlink(missing_ok=True)     # 强杀不会走对方的 finally
    URL_PATH.unlink(missing_ok=True)     # 同上：留着会让下次的开关读到旧端口
    print(f"已停止专注监视（PID {pid}）。")


def show_panel(timeout: float = 20.0) -> None:
    """把面板窗口显示出来。**只能在监视进程里调用。**

    窗口是监视进程的（`window.start_main()` 在它的主线程上跑），别的进程
    碰不到 —— 所以"显示面板"这个动作必须由**持有窗口的那个进程**来做。
    见 wait_and_open 的注释：以前是在子进程里直接调 open_page()，而那边
    `_window` 永远是 None，于是每次都退到系统浏览器弹一个新网页。

    等 `_window` 建出来是必须的：面板服务比 GUI 循环起得早，"请显示面板"
    这个请求常常先到，那时窗口对象还不存在 —— 直接 show 又会退到浏览器，
    等于把这个 bug 换个地方复现一遍。
    """
    try:
        import window
    except Exception:
        log.exception("窗口层加载不了，改用系统浏览器")
        open_page("panel")
        return
    if not window.wait_until_ready(timeout):
        # 窗口层在、但 GUI 循环没起来（pythonnet / WebView2 卡住之类）。
        # 这时**必须**降级到浏览器，否则用户点了开关什么都看不到。
        log.warning("等了 %.0f 秒窗口还没建出来，改用系统浏览器", timeout)
        open_page("panel")
        return
    open_page("panel")


def _persisted_panel_url() -> str:
    """监视进程写下的面板根地址（末尾带 /）；没有就退回默认端口。

    **只读，绝不起服务。** 这是给 --wait-open 子进程用的，而子进程里
    `dashboard._server` 是 None —— 一旦走 `_panel_url()`，`serve_background`
    就会**在子进程里另起一个面板服务**。那之后子进程是在自己回自己的请求：
    `/show-panel` 会回 200 ok 却什么都不做（它自己的 `_panel_shower` 没注册），
    窗口永远不出现，**而日志里一切正常**。这种静默失败必须避开，所以这里
    只读文件、只做地址拼装，不碰任何服务。

    文件还没写（监视进程刚起来，PID 文件比地址先落盘）时退回默认端口 ——
    调用方是轮询，下一轮就读到真值了。
    """
    try:
        url = URL_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        url = ""
    if not url:
        try:
            from dashboard import DEFAULT_PORT as port
        except Exception:
            port = 8787
        url = f"http://127.0.0.1:{port}/"
    return url if url.endswith("/") else url + "/"


def wait_and_open(timeout: float = 120.0) -> None:
    """等面板就绪，再请**监视进程**把它的窗口显示出来。

    模型加载要十几秒，启动后立刻打开只会看到「无法连接」。

    这个函数跑在一个**独立子进程**里（桌面开关 fork 出来的），走的不是
    托盘那条路，所以它必须自己 setup_log()：不然它出的任何问题都进不了
    focus.log，而 exe 是 --windowed、连 stderr 都没有 ——
    用户看到的只是"双击了，什么都没打开"，没有任何线索可查。

    **它不能自己 open_page()。** 窗口属于监视进程，这个子进程里的
    `window._window` 永远是 None，`open_page` 于是每次都走"窗口层不可用"
    的兜底 —— 用系统浏览器弹一个新网页。用户报的「双击开关就不断开新网页
    弹出这个面板、真正的应用窗口却不见」就是这么来的：开关每点一次，
    就多一个浏览器标签页。
    """
    setup_log()

    # 第一段：只负责"等服务就绪"。**不要**把"显示面板"塞进同一个 try ——
    # 塞在一起的话，那一步失败会被当成"服务还没起来"，于是一路重试到
    # 超时、最后一声不吭地退出。这两个失败的原因完全不同，必须分开处理。
    #
    # 地址**每轮重读**（_persisted_panel_url 会读 URL_PATH）：端口是监视进程
    # 起好服务之后才知道的，而它可能比我们晚一步写（PID 文件比地址先落盘）。
    # 轮询里重读就自然收敛了，不用在启动顺序上加约定。
    #
    # 用 _persisted_panel_url 而**不是** _panel_url：后者会顺手在子进程里
    # 把面板服务起起来，于是我们请求的是自己 —— /show-panel 回 200 ok 却
    # 什么都不做，窗口永远不出现，而日志里一切正常。见那个函数的注释。
    deadline = time.time() + timeout
    url = ""
    while time.time() < deadline:
        url = _persisted_panel_url()
        try:
            # 必须把响应体读掉再关。只调 .close() 等于中途掐断连接，服务端
            # 写响应时会抛 ConnectionAbortedError，在用户日志里留下一串
            # "面板处理 GET / 出错"的假故障 —— 排查时会被这堆噪音带偏。
            with urllib.request.urlopen(url, timeout=2) as resp:
                resp.read()
            break
        except Exception:
            time.sleep(1.5)
    else:
        log.warning("等面板就绪超时（%.0f 秒），放弃自动打开窗口", timeout)
        return

    # 第二段：请监视进程把窗口显示出来。
    try:
        with urllib.request.urlopen(url + "show-panel", timeout=10) as resp:
            resp.read()
        log.info("已请求监视进程显示面板")
        return
    except Exception:
        log.exception("请求监视进程显示面板失败，退到本进程打开")

    # 兜底：老版本的监视进程没有这个路由（用户刚更新了开关、后台进程还是
    # 旧的）。这条路至少能让页面出现，哪怕它落在浏览器里 ——
    # 比"双击了没反应"好。
    open_page("panel")


def _start_engine(timeout: float = 15.0) -> bool:
    """拉起监视本体，并**确认它真的起来了**；起来了才返回 True。

    两个进程各司其职：本体负责采集和托盘（DETACHED，不随开关退出），
    `--wait-open` 那个只负责等面板就绪再开窗口（模型要下十几秒，
    它得阻塞着等）—— **但后者只在 AUTO_OPEN_PANEL 打开时才起**，
    默认不起（用户报过"每次启动都弹一个窗口，打扰工作"）。

    **先确认本体活了，再起第二个。** 顺序反过来的话，本体压根没起来时，
    第二个进程会照旧去等 120 秒面板、最后放弃 —— 用户多等两分钟，
    而且日志里会多一条误导性的"等面板超时"。

    OSError 要自己接住：入口文件不存在时 Popen 抛 FileNotFoundError，
    而开关跑在 `--windowed` 的进程里、连 stderr 都没有，异常直接消失，
    用户看到的还是"双击了但什么都没发生" —— 正是这个 bug 的原始症状。
    接住它，才能走到"置灰 + 弹窗告诉他为什么"。
    """
    try:
        # DETACHED_PROCESS 是必须的：不脱离的话，开关一退出监视进程会被一起带走
        subprocess.Popen(relaunch_cmd(), cwd=str(ROOT),
                         creationflags=_DETACHED, close_fds=True)
        if not _wait_until_running(timeout):
            return False
        # ── 自动弹面板（AUTO_OPEN_PANEL）──
        # 这是**应用自己**唯一一处"每次启动都弹一个窗口"的地方：只要双击一次
        # 桌面开关，就有一个 --wait-open 子进程在等模型下完、然后把窗口
        # show() 出来。起进程这件事本身不需要窗口作证，绿点已经作证了
        # （toggle 里紧随其后的 sync_shortcut_icon）。
        #
        # **用户第二次报的「每次让你改点东西就弹窗」不是这一处** ——
        # 那个是自检里裸奔的 wait_and_open 打到了正在运行的这个实例上
        # （见 selftest 里那段的注释）。这一处是"他自己启动应用"时会看到的
        # 那一个，也就是他最初描述的那个。两个都修了。
        #
        # 放在 _wait_until_running 之后：本体都没起来的时候，起这个子进程
        # 只会让它白等 120 秒（见上面的 docstring）。
        if not AUTO_OPEN_PANEL:
            log.info("AUTO_OPEN_PANEL 关闭，不自动弹面板"
                     "（要看的话点托盘图标，那条路不走这里）")
            return True
        subprocess.Popen(relaunch_cmd() + ["--wait-open"], cwd=str(ROOT),
                         creationflags=_DETACHED, close_fds=True)
    except OSError:
        log.exception("启动失败：入口不存在或不可执行（%s）", relaunch_cmd())
        return False
    return True


def toggle() -> None:
    """桌面快捷方式的入口：切换开/关，并把图标改成**实际**状态。

    用户报的 bug 就在这个函数里：原来是先写绿点、再起进程，之后不管
    起没起来都不再看一眼。断网/待机后模型下不下来、采集线程根本没起来，
    桌面图标却一直是绿的 —— 图标反映的是"用户按了开关"，
    不是"程序在跑"。现在绿点只在确认进程真的起来之后才写。
    """
    if running_pid() is not None:
        stop_running()
        sync_shortcut_icon(force=True, running=False)
        print("已关闭专注监视。")
        return

    if _start_engine():
        sync_shortcut_icon(force=True, running=True)
        # 文案必须跟着开关走。默认关着的时候还打"面板就绪后会自动打开"，
        # 就是在告诉用户一件不会发生的事 —— 而他刚抱怨过弹窗，
        # 会以为设置没生效、又去反复双击。
        print("已启动专注监视，面板就绪后会自动打开。" if AUTO_OPEN_PANEL else
              "已启动专注监视。面板没有自动弹出（设置里「启动时自动打开面板」"
              "是关的），点托盘图标就能看。")
        return

    # 没起来。把开关置灰，并且**必须**告诉用户为什么 —— 双击开关的是个
    # 没有终端的进程，print 到不了任何地方。
    sync_shortcut_icon(force=True, running=False)
    _complain(
        "专注监视没能启动",
        "程序没有真正跑起来，桌面开关已置灰。\n\n"
        "常见原因：\n"
        "· 网络不通，首次运行要下载的人脸模型没下完\n"
        "· 摄像头被别的程序占用\n"
        "· 上次的进程还卡着（先双击一次关掉，再双击开启）\n\n"
        "托盘图标右键 →「打开日志」，focus.log 里有具体原因。",
    )


def install_shortcut() -> None:
    """建桌面开关，并清掉早期的三个快捷方式。"""
    global _lnk_state
    icons = ensure_icons()
    d = _desktop()
    for old in ("专注监视面板.url", "启动专注监视.lnk", "停止专注监视.lnk"):
        p = d / old
        if p.exists():
            p.unlink()
            print(f"  已删除旧快捷方式: {old}")
    # 建的时候图标直接写成真实状态：程序正在跑（多半是托盘菜单里点的）
    # 就是绿点，否则灰杠。_lnk_state 也要跟着对，不然随后的看门狗会白写一次。
    running = running_pid() is not None
    state = shortcut_state(running=running, stale=is_stale())
    _write_lnk(icons[state])
    _lnk_state = state
    print(f"  已创建桌面开关: {d / LNK_NAME}")
    print("  双击 = 开/关切换；图标 绿点=在记录，黄叹号=在跑但没在记录，"
          "灰杠=已停止")


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

    # ── 眨眼 / 视疲劳 ──
    # 这些计数器是纯逻辑，直接喂 EAR 序列就能测，不需要摄像头。
    # 时间轴按 FACE_FPS=10 走，一帧 = 0.1 秒。
    _thr = 0.21
    _bt = BlinkTracker()
    _t = 0.0
    for _i in range(300):                       # 30 秒 = BLINK_MIN_OBS，刚够出数
        _bt.feed(_t, 0.10 if _i % 50 == 0 else 0.30, _thr, True)   # 每 5 秒眨一次
        _t += 0.1
    _r = _bt.rate()
    assert abs(_r - 12.0) < 0.5, (
        f"眨眼率算成 {_r:.1f}，应该是 12（30 秒里眨了 6 次）。分母必须是"
        "**窗口里真正观察到正脸的秒数**，不是窗口长度 60 秒 —— 用窗口长度的话，"
        "人走开一阵再回来只看了十秒，会被算成「十秒里眨了 0 次」，眨眼率虚低，"
        "立刻误报「眼睛该休息了」")

    # 长闭合不算眨眼：那是眯眼/微睡眠，归 EAR_SUSTAIN 那条管。
    # 不排除的话"困得睁不开眼"会被数成"眨眼很频繁"，方向正好反了。
    _bt2 = BlinkTracker()
    _t = 0.0
    for _i in range(300):
        _bt2.feed(_t, 0.10 if (_i % 50) < 10 else 0.30, _thr, True)  # 1 秒的长闭合
        _t += 0.1
    assert not any(b for _, _, b in _bt2._win), (
        "1 秒的长闭合被算成眨眼了 —— 那是眯眼/微睡眠。算成眨眼会让"
        "「越困眨眼越多」，方向和事实正好相反")

    # 非正对屏幕时完成的眨眼不计入：转头时眼区被压缩、EAR 假性变低。
    # 实测他库里 71544 条有脸样本中 ear 低于 0.19 的占 14.6%，
    # 远多于眨眼能解释的量 —— 不排除掉，眨眼率会凭空翻几倍。
    _bt3 = BlinkTracker()
    _t = 0.0
    for _i in range(300):
        _bt3.feed(_t, 0.10 if _i % 50 == 0 else 0.30, _thr, False)
        _t += 0.1
    assert not any(b for _, _, b in _bt3._win), \
        "转头期间的假性低 EAR 被算成眨眼了"

    # 观察时长不够时返回 0，含义是"还不知道"（不是"没眨眼"）
    _bt4 = BlinkTracker()
    for _i in range(10):                        # 只有 1 秒
        _bt4.feed(_i * 0.1, 0.30, _thr, True)
    assert _bt4.rate() == 0.0, "观察不足 BLINK_MIN_OBS 时不该给出一个具体的率"

    # 去抖：两次眨眼挨得比 BLINK_MIN_GAP 还近时只记一次
    _bt5 = BlinkTracker()
    for _ts, _ear in ((0.0, 0.30), (0.1, 0.10), (0.2, 0.30),
                      (0.3, 0.10), (0.4, 0.30),      # 距上次只 0.2 秒 → 不计
                      (0.5, 0.10), (0.9, 0.30)):     # 距上次 0.7 秒 → 计
        _bt5.feed(_ts, _ear, _thr, True)
    _n5 = sum(1 for _, _, b in _bt5._win if b)
    assert _n5 == 2, f"挨得比 BLINK_MIN_GAP 还近的两次眨眼被数成了 {_n5} 次"

    # 「手在眼周」的几何换算。三处最容易写错，而且写错了不会报错，
    # 只会**一直判 False**（揉眼次数永远是 0，看着像"你从不揉眼"）。
    _eye = (0.5, 0.4)
    assert hand_near_eye(_eye, 0.2, (0.5, 0.4, 1.0), 480, 270), "手就在眼睛上"
    # 这一条专门钉各向异性：归一化坐标 y 除以高、x 除以宽，不乘回像素的话
    # 竖直方向会被高估（h/w = 0.5625），这个位置会被误判成"手不在眼周"。
    assert hand_near_eye(_eye, 0.2, (0.5, 0.6, 1.0), 480, 270), (
        "同样的归一化距离，竖直方向没乘回像素 —— 画面越扁判得越歪")
    assert not hand_near_eye(_eye, 0.2, (0.5, 0.8, 1.0), 480, 270), "太远了"
    assert not hand_near_eye(_eye, 0.2, (0.5, 0.4, 0.3), 480, 270), \
        "visibility 太低的手腕是模型在瞎猜，必须丢掉"
    assert not hand_near_eye(None, 0.2, (0.5, 0.4, 1.0), 480, 270), \
        "还没有眼睛位置时不该判出手在眼周"
    # 半径必须跟着眼距缩放：同一个手腕位置，人往后一靠（脸变小）就不该再命中，
    # 否则"离屏幕远"会被判成"一直在揉眼"。
    assert hand_near_eye(_eye, 0.2, (0.5, 0.6, 1.0), 480, 270) is True
    assert not hand_near_eye(_eye, 0.1, (0.5, 0.6, 1.0), 480, 270), \
        "眼距缩小一半后半径没跟着缩 —— 离屏幕远会被判成一直在揉眼"

    # 揉眼：连续命中 HAND_EYE_HOLD 次才算一次，断了要重新数
    _rt = RubTracker()
    for _i in range(HAND_EYE_HOLD):
        _rt.feed(_i * 0.5, True)                    # 连续命中 → 记 1 次
    assert _rt.count() == 1, "连续命中 HAND_EYE_HOLD 次应该记一次揉眼"

    # 记完必须重新数：冷却过了之后，**单独一帧**不该立刻又记一次。
    # 不归零的话 _run 只增不减，冷却一过随便来一帧就凑够条件，
    # "连续命中 N 次"这个条件形同虚设 —— 而且完全静默，计数看着是涨的。
    _rt.feed(10.0, True)
    _rt.feed(10.0 + RUB_MIN_GAP, True)
    assert _rt.count() == 1, (
        "冷却一过，单独一帧就把揉眼数记上去了 —— 计数后 _run 没归零，"
        "「连续命中 N 次」形同虚设")

    # 真的又揉一次（连续命中 + 过了冷却）才该记第 2 次
    for _i in range(HAND_EYE_HOLD):
        _rt.feed(45.0 + _i * 0.5, True)
    assert _rt.count() == 2, "过了冷却又连续命中，应该记第 2 次"

    # 眨眼率只能让提醒**提前**，不能让它闭嘴
    assert eye_break_threshold(0.0) == EYE_BREAK_AFTER, "没测出来时按常规阈值"
    assert eye_break_threshold(18.0) == EYE_BREAK_AFTER, "眨眼正常时按常规阈值"
    assert eye_break_threshold(5.0) == EYE_BREAK_SOON, "眨眼率偏低应该提前提醒"
    for _rate in (0.0, 1.0, 5.0, 7.9, 8.0, 20.0, 60.0):
        assert eye_break_threshold(_rate) <= EYE_BREAK_AFTER, (
            f"眨眼率 {_rate} 把提醒**推后**了 —— 它是弱证据（依赖 EAR 阈值，"
            "比「坐了多久」脆弱得多），做成必要条件的话检测一失灵就永远不提醒")

    # 文案：测出来了就把数字摆出来，没测出来就别编一个
    _m = eye_break_message(45 * 60, 5.0)
    assert "45" in _m and "5" in _m, _m
    _m0 = eye_break_message(45 * 60, 0.0)
    assert "45" in _m0 and "次/分" not in _m0, \
        f"没测出眨眼率时文案里不该出现次数：{_m0!r}"

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
    # 学科名是实测漏得最狠的一类：用户真正看的标题长这样，旧关键词一个都不含。
    assert classify_app(
        "firefox.exe",
        "《高等数学》全程教学视频 2.0版【宋浩老师】_哔哩哔哩") == "work"
    assert classify_app("firefox.exe", "2 函数_哔哩哔哩") == "distract", \
        "「2 函数」这种标题本来就认不出来 —— 它只能落到中性，绝不能判走神"
    # 豁免不能滥用：不带学习关键词的娱乐内容照常判分心
    assert classify_app("chrome.exe", "【4K】舞蹈区精选 - 哔哩哔哩") == "distract"
    assert classify_app("chrome.exe", "王者荣耀 直播 - 哔哩哔哩") == "distract"

    # 状态机：每条分支都要走到
    base = dict(face_present=True, yaw=0.0, pitch=0.0, closed_for=0.0,
                idle_sec=0.0, app_kind="work", away_for=0.0)
    assert decide(**base) == "focused"
    assert decide(**{**base, "app_kind": "other"}) == "neutral"
    # ── 应用类型只抬高、不贬低 ──
    # 实测 bug：在 B 站看半小时高数课被判成走神，因为标题里带"哔哩哔哩"。
    # 娱乐应用现在只降到「中性」（中性算投入时长，见 ENGAGED）。
    assert decide(**{**base, "app_kind": "distract"}) == "neutral", \
        "娱乐应用又被直接判走神了 —— 用户在 B 站看高数课就是被这条冤枉的"
    # 走神只剩行为判据。第一条：来回切窗口（不看是哪个应用）。
    assert decide(**{**base, "switch_rate": DISTRACT_SWITCH_RATE}) == "distracted"
    assert decide(**{**base, "switch_rate": DISTRACT_SWITCH_RATE - 1}) == "focused"
    assert decide(**{**base, "app_kind": "work",
                     "switch_rate": DISTRACT_SWITCH_RATE}) == "distracted", \
        "来回切窗口的判据必须对工作应用同样生效 —— 它压根不看应用类型"
    # 通用护栏：只要脸在、正对屏幕、没在狂切窗口，**任何**应用都不能判走神。
    # 上一条断言钉的是那一个 bug，这条钉的是"不许再长出一个同类规则"。
    for _kind in ("work", "distract", "other", "", "WORK"):
        for _y, _p in ((0.0, 0.0), (YAW_TOL, PITCH_TOL), (-YAW_TOL, 15.8)):
            _st = decide(**{**base, "app_kind": _kind, "yaw": _y, "pitch": _p})
            assert _st != "distracted", (
                f"app_kind={_kind!r} 正对屏幕却被判走神（{_st}）—— 应用类型"
                "没有资格单独判走神（实测：B 站看高数课就是这么被冤枉的）")
    # 默认值必须是 0.0：老调用方（上面的单点断言）不该被新判据误伤
    assert decide(**{**base, "app_kind": "distract"}) == "neutral"
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
    # eye_rows 显式传：不传的话 build_html 会自己去读 focus.DB_PATH ——
    # 而自检跑到这里时那还指着**用户的真库**。自检不该碰它（只读也一样：
    # 结果会随真库内容变，本机绿、CI 红）。真实那条路在下面单独测。
    html = report.build_html(rows, eye_rows=(False, []))
    for needle in ("专注", "走神", "离开", "哔哩哔哩", "focus.py",
                   "<svg", "时间轴", "专注应用排行", "分心应用排行",
                   "应用使用记录", "占活跃", "时段", "视疲劳"):
        assert needle in html, f"报告缺少 {needle}"
    assert html.count("<html") == 1

    # ── 报告的「视疲劳」一节 ──
    #
    # 这里**走真实那条路**（build_html 内部自己调 load_eye() 去读库），
    # 而不是把数据直接喂进去 —— 最容易断的恰恰是那根接线
    # （`{_eye_stats(*eye_rows)}`），喂数据等于把它绕过去了。
    #
    # 三种库都要试，因为"列不存在"是**正常状态**而不是错误：报告可以跑在
    # 老库上（还没重启过新版本），而 ALTER TABLE 只发生在采集循环启动时。
    # 报告打不开是用户最能直接感知的故障。
    _keep_db = globals()["DB_PATH"]
    _eye_tmp = Path(tempfile.mkdtemp())
    try:
        # (a) 老库：有 samples 表、但没有 blink/rub 两列
        _old = _eye_tmp / "old.db"
        _c = sqlite3.connect(str(_old))
        _c.executescript("CREATE TABLE samples(ts REAL, state TEXT)")
        _c.commit()
        _c.close()
        globals()["DB_PATH"] = _old
        _h = report.build_html(rows)
        assert "还没开始记录" in _h, (
            "报告在老库（没有 blink/rub 两列）上没有降级成「还没开始记录」——"
            "要么报错了，要么显示了别的东西。这会让报告整页打不开")

        # (b) 空库：连 samples 表都没有（全新用户）
        globals()["DB_PATH"] = _eye_tmp / "nothing.db"
        assert "还没开始记录" in report.build_html(rows), \
            "报告在空库（连 samples 表都没有）上报错了"

        # (c) 新库 + 有数据：数字必须真的算出来
        _new = _eye_tmp / "new.db"
        _c = sqlite3.connect(str(_new))
        _c.executescript(SCHEMA)
        _base = 1789000000.0
        for _i in range(120):
            _c.execute(_SAMPLE_INSERT,
                       (_base + _i, "focused", "code.exe", "t", 0.0, 5.0, 0.30,
                        3.0, 1.0, 1, 0.0,
                        # 前 60 秒 blink=0（观测时长不够，表示"还不知道"），
                        # 不该被算进眨眼率里
                        0.0 if _i < 60 else (12.0 if _i < 110 else 5.0),
                        # rub 是"最近一分钟揉眼次数"的**滚动值**，
                        # 一次揉眼会在之后 60 条样本里都留下 1 —— 求和会放大
                        # 60 倍，所以只能按增量数事件。这里是 1 次。
                        1 if 30 <= _i < 90 else 0))
        _c.commit()
        _c.close()
        globals()["DB_PATH"] = _new
        _h = report.build_html(rows)
        # 眨眼率中位数：50 条 12.0 + 10 条 5.0 → 中位数 12
        assert ">12 次/分</div>" in _h, (
            "新库上有眨眼数据，报告里却没算出 12 次/分 —— 视疲劳那节没接上")
        # 低于 BLINK_LOW_RATE(8) 的 10 条 / 60 条 = 17%
        assert ">17%</div>" in _h, "「眨眼偏低的占比」没算对（应为 10/60）"
        # 揉眼按增量数：0→1 只有一次
        assert ">1 次</div>" in _h, (
            "揉眼次数不对。rub 是滚动 60 秒计数，**求和会放大 60 倍** ——"
            "这里必须按增量数事件")
        assert ">60 次</div>" not in _h, "揉眼次数被当成求和算了"
    finally:
        globals()["DB_PATH"] = _keep_db

    # ── 报告给出的行动建议必须**能照着做** ──
    #
    # 这条是补出来的：低相关横幅原来写"请先按「校准」一节调阈值"，而
    # 程序里根本没有校准功能（EAR 基线是自动学的，没有任何校准界面），
    # 文档的「校准」一节也只讲"什么时候该调"、不讲"在哪调"。
    # 用户照这句话去找 → 卡住（实测反馈原话："没找到校准交互"）。
    # 一句把人指到不存在的地方的建议，比什么都不说更浪费时间。
    assert "数据可信" in report.trust_banner(0.8, 40)
    assert "大致对得上" in report.trust_banner(0.5, 40)
    assert "尚未验证" in report.trust_banner(None, 3)
    _low = report.trust_banner(0.1, 40)
    assert "先别信" in _low, "低相关的横幅应该明确说「别信其他结论」"
    assert report._FIX_FIELD in _low, (
        f"低相关横幅没点名要改哪个设置项（{report._FIX_FIELD}）—— "
        "用户只能自己猜去哪调")

    # 点名的设置项必须在设置页真的存在（跨模块核对：横幅在 report，
    # 字段标签在 dashboard）。标签写错了就是另一句照做不了的话。
    try:
        import dashboard as _dash_chk
    except Exception:
        _dash_chk = None
    if _dash_chk is not None:
        _labels = {_lab for _grp, _flds in _dash_chk._FIELDS
                   for _k, _lab, _hint in _flds}
        assert report._FIX_FIELD in _labels, (
            f"横幅让用户去改「{report._FIX_FIELD}」，但设置页里没有这个字段。"
            f"现有标签：{sorted(_labels)}")

        # 设置页渲染的每个字段都必须真的存在于配置里 —— 否则用户改完点保存，
        # 那个值会被**静默丢掉**（apply_config 只认 _SCALARS / _LIST_KEYS 里的键），
        # 而界面上看不出任何异常：输入框还在、值也回显了，就是没生效。
        # 这是"指向不存在的地方"的镜像版本：界面指向一个配置里没有的键。
        # 反向也要钉：配置里有、界面改不到的键，用户只能去手改 config.json。
        _rendered = {_k for _g, _f in _dash_chk._FIELDS for _k, _l, _h in _f}
        _rendered_txt = {_k for _k, _l, _h in _dash_chk._TEXTAREAS}
        assert not (_rendered - set(_SCALARS)), (
            f"设置页有这些字段，但配置里没有：{sorted(_rendered - set(_SCALARS))}"
            " —— 用户改完保存会静默丢失")
        assert not (_rendered_txt - set(_LIST_KEYS)), (
            f"设置页有这些文本框，但配置里没有："
            f"{sorted(_rendered_txt - set(_LIST_KEYS))}")
        assert not (set(_SCALARS) - _rendered), (
            f"配置里有这些标量，但设置页改不到：{sorted(set(_SCALARS) - _rendered)}"
            " —— 不只是「用户只能手改 config.json」这么轻：_parse_form 会遍历"
            "_SCALARS 从表单取值，界面上没有输入框就取到空串，"
            "float('') 抛异常 → **整个设置页保存永远失败**，"
            "报的还是「XXX 不是有效数字：''」这种看不懂的错")
        assert not (set(_LIST_KEYS) - _rendered_txt), (
            f"配置里有这些列表，但设置页改不到："
            f"{sorted(set(_LIST_KEYS) - _rendered_txt)}")

        # ── 布尔开关走的是另一条路（checkbox），上面那张表管不到它们 ──
        # 单独钉一遍。漏渲染一个 checkbox 的后果是**这个开关永远是关的**：
        # _parse_form 靠 `key in form` 取值，页面上没有那个框就恒为 False ——
        # 而界面上你还看得见它、还勾得上、勾完还提示"已保存"。
        # 反过来说也成立：想打开的人永远打不开。
        # （AUTO_OPEN_PANEL 漏了的话，用户的原话会变成
        #  "设置里明明写着不自动弹，它还是弹"。）
        _page = _dash_chk._settings_page()
        for _flag in _FLAGS:
            assert f'name="{_flag}"' in _page, (
                f"配置开关 {_flag} 没有在设置页渲染出来 —— _parse_form 靠"
                " `key in form` 取值，没有那个 checkbox 就恒为 False，"
                "这个开关会**永远是关的**，而界面上完全看不出来")

        # 反向：_parse_form 必须真的能取到每一个开关。
        # 漏掉一个的话，用户取消勾选 → 表单里没有那个名字 → cfg 里也没有
        # 那个键 → apply_config 不认 → **取消勾选没有任何效果**，
        # 值还停在上一次的状态上。
        _cur_form = current_config()
        _form_cfg, _form_errs = _dash_chk._parse_form(
            {k: [str(_cur_form[k])] for k in _SCALARS})
        assert not _form_errs, _form_errs
        for _flag in _FLAGS:
            assert _flag in _form_cfg, (
                f"_parse_form 没有取到开关 {_flag} —— 用户取消勾选后那个键"
                "压根不进 cfg，apply_config 就不会碰它，"
                "于是**取消勾选没有任何效果**")
            assert _form_cfg[_flag] is False, (
                f"表单里没有 {_flag} 时应该解析成 False（checkbox 的语义），"
                f"实际 {_form_cfg[_flag]!r}")

    # 指的文档小节也必须真的有这个标题
    _tut = (ROOT / "使用教程.md").read_text(encoding="utf-8")
    assert f"## {report._FIX_DOC}" in _tut, (
        f"横幅让用户去读「使用教程 · {report._FIX_DOC}」，"
        "但文档里没有这一节 —— 又是一句指向不存在的地方的建议")

    # 时长全为 0 的数据不能把报告搞崩。
    #
    # 第三轮审查把"hour_avg 除零"列进"未能确认的项"，当时只加了个防御性守卫。
    # 现在确认了：**能触发，但触发点不在守卫的那个分母上**。
    # `hour_active_days` 的键一旦存在就至少有一个活跃日（它和 hour_active 是
    # 同一处代码同时写的），所以 len(days) 恒 ≥ 1；真正会变成 0 的是
    # `peak = max(hour_avg)` —— 某小时内所有非离开样本的 dur 都是 0 时，
    # hour_avg 全 0，算柱高就 0/0。day_active[d] 是同一类问题（算专注率时分母）。
    #
    # 采集出来的数据撞不上：写入侧有 `now - last_flush >= 1.0` 闸门，
    # 相邻样本至少差 1 秒，dur 必然 ≥ 1。但手工导入的数据能构造出来 ——
    # 所以照兜，并在这里钉死，免得将来有人把那两个 `or 1.0` 当冗余删掉。
    _zt = 1789000000.0
    zero_dur = [(_zt, "focused", "code.exe", "t", 0.0, 0.0, 0.3, 1.0, 1.0, 1, 0.0)
                for _ in range(10)]
    zero_dur.append((_zt, "away", "", "", 0.0, 0.0, 0.0, 0.0, 0.0, 0, 300.0))
    assert "<html" in report.build_html(zero_dur), \
        "时长全为 0 的数据把报告搞崩了（peak / day_active 的除零兜底失效）"

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
    # 视疲劳：提前提醒那一档大于常规档时，眨眼这条判据会**静默失效**
    # （永远取不到它），所以必须报错，而不是默默按常规档走。
    assert any("提前" in e for e in validate_config(
        {**snapshot, "EYE_BREAK_SOON": snapshot["EYE_BREAK_AFTER"] + 1})), \
        "「提前提醒的用眼时长」大于常规时长时必须报错"
    assert validate_config({**snapshot, "EYE_BREAK_SOON": 0.0}), \
        "提前档必须为正数（0 会让它在每一次结算都命中）"

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

    # ── AUTO_OPEN_PANEL：默认不许自动弹面板 ──
    # 这是用户报的"桌面一直弹「专注监视」窗口，打扰工作"的回归测试。
    #
    # _start_engine 会起真进程（DETACHED 的监视本体 + --wait-open），
    # 所以把 Popen 和"等它起来"都换掉，只记下它到底还想不想再起第二个。
    # 这条测的是**默认值**，也就是用户装完就生效的那份行为 ——
    # 光在常量那儿写 AUTO_OPEN_PANEL = False 是不够的：
    # 那个常量没人读的时候，它就是个注释。
    _spawned: list[list[str]] = []
    _keep_popen = subprocess.Popen
    _keep_waitrun = globals()["_wait_until_running"]
    _keep_auto = AUTO_OPEN_PANEL

    class _FakePopen:
        def __init__(self, cmd, *a, **kw):
            _spawned.append(list(cmd))

    try:
        subprocess.Popen = _FakePopen
        globals()["_wait_until_running"] = lambda _t: True

        apply_config({"AUTO_OPEN_PANEL": False})
        assert _start_engine(timeout=0.1) is True, (
            "关掉自动弹面板不该让启动失败 —— 起进程和弹窗口是两件事，"
            "把前者也一起关掉的话用户双击开关就没反应了")
        assert len(_spawned) == 1, (
            f"AUTO_OPEN_PANEL 关着，_start_engine 还是起了 {len(_spawned)} 个进程"
            f"（{_spawned}）—— 多出来的那个就是 --wait-open，"
            "它会等模型下完然后把面板窗口 show() 出来，正是用户抱怨的那个弹窗")
        assert not any("--wait-open" in _c for _c in _spawned), (
            "AUTO_OPEN_PANEL 关着，还是起了 --wait-open 子进程")

        _spawned.clear()
        apply_config({"AUTO_OPEN_PANEL": True})
        assert _start_engine(timeout=0.1) is True
        assert any("--wait-open" in _c for _c in _spawned), (
            "打开 AUTO_OPEN_PANEL 之后没起 --wait-open —— "
            "这个开关就成了个勾了也不起作用的装饰")
    finally:
        subprocess.Popen = _keep_popen
        globals()["_wait_until_running"] = _keep_waitrun
        apply_config({"AUTO_OPEN_PANEL": _keep_auto})

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

    # 低相关 / 负相关这两条**才是真正推着用户去动阈值**的结论，
    # 样本量提醒必须挂在这里 —— 原来只挂在上面两条"结论不错"的分支上，
    # 恰好把会引发动作的两条漏了（verdict 的 docstring 写的正是防这个）。
    # n=31 时半宽 0.37 > 0.25（该提醒），n=100 时 0.20（样本够了，不该再提）。
    assert "先别急着动阈值" in ratings.verdict(0.1, ratings.MIN_PAIRS + 1), \
        "低相关 + 刚过门槛时，必须劝住用户先别动阈值"
    assert "阈值需要重新调" in ratings.verdict(0.1, 100), \
        "样本够了却还是低相关，才该直接说阈值要调"
    assert "先别" not in ratings.verdict(0.1, 100), \
        "样本已经够了还说「先别急着」就是一句空话"
    assert "还可能变" in ratings.verdict(-0.5, ratings.MIN_PAIRS + 1)
    assert "建议先查阈值" in ratings.verdict(-0.5, 100)
    assert "还可能变" not in ratings.verdict(-0.5, 100), \
        "样本够了就不该再挂「结论还可能变」"

    # 措辞里不许出现「校准」——程序里没有这个功能（EAR 基线是自动学的），
    # 写了就是把人指到一个不存在的地方（实测反馈：用户说「没找到校准交互」）。
    # 扫全部 (r, n) 组合而不是抽查几条：以后任何分支想加回"重新校准"，
    # 都会在这里被拦下，不用指望谁记得住这条规则。
    for _r in (None, -1.0, -0.5, -0.4, -0.3, 0.0, 0.1, 0.39, 0.4, 0.5,
               0.69, 0.7, 0.8, 1.0):
        for _nn in (3, 4, 5, ratings.MIN_PAIRS, ratings.MIN_PAIRS + 1,
                    64, 65, 100, 500):
            _v = ratings.verdict(_r, _nn)
            assert "校准" not in _v, (
                f"verdict({_r}, {_nn}) 里出现了「校准」：{_v!r} —— "
                "程序没有校准功能，用户会照着去找（实测反馈过）")

    # ── 下面四段都用**临时目录里的库**，不再往仓库目录丢 `_selftest_*.db` ──
    #
    # 原来每段都是 `ROOT / "_selftest_xxx.db"` 这种固定名字，靠各自的 finally 删。
    # 实测踩到过：某次运行之后 `_selftest_seam.db` 留在了仓库目录里，下一次自检
    # 读到它 —— 库里已经有一条样本，再插一条就成了 2 条，
    # `assert len(report.load()) == 1` 于是失败。表现是**偶发失败**：30 次里红
    # 一次、下一次又好了；而 CI 每次要跑几十遍，迟早随机变红。
    #
    # 具体是哪一次运行没删掉，我没能复现出来。但这不重要 ——
    # **"依赖上一次干净退出"本身就是 bug**：路径固定 + 手工清理，任何一次异常
    # 退出都会留下污染源，而污染的表现是**另一次无关的测试失败**，查起来极绕。
    #
    # 临时目录一次解决两件事：路径每次都不同（不可能读到别人的数据），
    # 目录由 TemporaryDirectory 负责删（不用维护清理清单，异常退出也不留）。
    _st_td = tempfile.TemporaryDirectory()
    _st_tmp = Path(_st_td.name)

    # ── 落库的列名表与建表语句必须一致 ──
    # 这两处分开写，改一处忘另一处不会报错，只会把值塞进错误的列
    # （位置参数时代就是这个毛病，实测踩过）。所以钉死它们同源。
    _schema_cols = re.findall(
        r"(\w+)\s+(?:REAL|TEXT|INTEGER)",
        SCHEMA[SCHEMA.index("CREATE TABLE"):SCHEMA.index(");")])
    assert tuple(_schema_cols) == SAMPLE_COLS, (
        f"建表语句的列 {_schema_cols} 和落库用的列 {list(SAMPLE_COLS)} 不一致 —— "
        "插入会静默错位或直接报列数不匹配")

    # ── 老库要能原地补列 ──
    # SCHEMA 用的是 CREATE TABLE IF NOT EXISTS，对已存在的表**什么都不做**，
    # 所以用户手上那份十几万行的 focus.db 只能靠 ALTER 补。没有这一步的话，
    # 升级后要等采集线程第一次落库才炸，报的还是
    # 「table samples has no column named blink」——很难联想到"该迁移了"。
    _mig_db = _st_tmp / "old.db"
    _mc = sqlite3.connect(_mig_db)
    try:
        _mc.execute("CREATE TABLE samples("
                    "ts REAL, state TEXT, app TEXT, title TEXT,"
                    "yaw REAL, pitch REAL, ear REAL, tilt REAL, scale REAL,"
                    "present INTEGER, idle REAL)")     # 故意用旧版 11 列
        _mc.execute("INSERT INTO samples VALUES(1.0,'focused','a.exe','t',"
                    "0,0,0.3,0,1,1,0)")
        _mc.commit()
        assert ensure_columns(_mc) == ["blink", "rub"], "老表没补上后加的列"
        assert ensure_columns(_mc) == [], "补列必须幂等 —— 每次启动都会调一遍"
        _mc.execute(_SAMPLE_INSERT,
                    (2.0, "focused", "a.exe", "t", 0, 0, 0.3, 0, 1, 1, 0, 9.5, 3))
        _mc.commit()
        assert _mc.execute("SELECT COUNT(*) FROM samples").fetchone()[0] == 2, \
            "补列把老数据弄丢了"
        assert _mc.execute(
            "SELECT blink, rub FROM samples WHERE ts=2.0").fetchone() == (9.5, 3)
    finally:
        _mc.close()

    # ── report.load() 必须跟着 focus.DB_PATH 走 ──
    # 这是"多库对照"的唯一开关：ratings 一直是动态读的，report 原来读的是
    # `from focus import DB_PATH` 在 import 那一刻的副本，改了 focus.DB_PATH
    # 对它**完全不起作用** —— 而它自己的 docstring 恰好写着这个开关是有效的。
    # 后果不是报错而是**静默混库**：样本来自 A 库、评分来自 B 库，算出一个
    # 看起来合理、其实毫无意义的相关系数。（做真机验证时就是这么中招的。）
    _seam_db = _st_tmp / "seam.db"
    _keep_db = DB_PATH
    try:
        _sconn = open_db(_seam_db)
        try:
            _sconn.executescript(SCHEMA)
            _sconn.execute(_SAMPLE_INSERT,
                           (1.0, "focused", "code.exe", "t", 0.0, 0.0, 0.3,
                            1.0, 1.0, 1, 0.0, 0.0, 0))
            _sconn.commit()
        finally:
            _sconn.close()
        globals()["DB_PATH"] = _seam_db
        assert len(report.load()) == 1, (
            "改了 focus.DB_PATH，report.load() 却没跟着换库 —— "
            "它会继续读 import 时绑定的那份，于是样本和评分可能来自两个库，"
            "算出一个看起来合理、其实毫无意义的相关系数")
    finally:
        globals()["DB_PATH"] = _keep_db

    # ── 实时面板上同样不许出现「校准」──
    # 和上面同源：用户看到"校准中 37/60"会去找一个能点的校准按钮，
    # 而程序里没有这个功能。所以这条不是措辞偏好，是同一个缺陷的另一处。
    #
    # 直接渲染**真实的面板页面**来查，不读源码猜 —— 那行文案是内联在
    # _live_html() 里的，没有独立函数可以单测。样本数**故意少于
    # EAR_MIN_SAMPLES**：要命中的是"还在学"那一档，样本够了走的是
    # "已自适应"，那档本来就不含这个词，测了等于没测。
    try:
        import dashboard as _dash_live_mod
    except Exception:
        _dash_live_mod = None
    if _dash_live_mod is not None:
        _real_pref = DB_PATH
        _pdb = _st_tmp / "panel.db"
        try:
            _pc = open_db(_pdb)
            try:
                _pc.executescript(SCHEMA)
                _now = time.time()
                _pc.executemany(
                    _SAMPLE_INSERT,
                    [(_now - 60 + i, "focused", "code.exe", "t",
                      0.0, 0.0, 0.30, 1.0, 1.0, 1, 0.0, 0.0, 0)
                     for i in range(5)])
                _pc.commit()
            finally:
                _pc.close()
            globals()["DB_PATH"] = _pdb
            _panel = _dash_live_mod._live_html()
            assert "校准" not in _panel, (
                "实时面板上出现了「校准」—— 用户会去找一个不存在的校准按钮"
                "（实测反馈原话：「没找到校准交互」）")
            assert "学习中" in _panel, (
                "面板没写出基线还在学习 —— 用户不知道那个 37/60 是什么")
        finally:
            globals()["DB_PATH"] = _real_pref

    # 数据库往返。用临时库 —— 绝不能污染用户的真实 focus.db
    real_db = globals()["DB_PATH"]
    tmp_db = _st_tmp / "ratings.db"
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

    # ── 备份与保留策略 ──
    # backup_db / compact_db 是全项目唯一会碰用户数据文件的路径，必须有断言兜着：
    # "备份少拷了一截"和"没先备份就删"都是不可逆的事故，靠人工检查是查不出来的。
    real_db = globals()["DB_PATH"]
    tmp_db = _st_tmp / "compact.db"
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
            conn.execute(_SAMPLE_INSERT,
                         (now - 400 * 86400 + i, "focused", "code.exe", "t",
                          0.0, 0.0, 0.3, 1.0, 1.0, 1, 0.0, 0.0, 0))
        for i in range(2):
            conn.execute(_SAMPLE_INSERT,
                         (now - i, "focused", "code.exe", "t",
                          0.0, 0.0, 0.3, 1.0, 1.0, 1, 0.0, 0.0, 0))
        conn.commit()
        conn.close()
        assert _count(tmp_db) == 5

        # 备份出来的必须是**完整**的一份。WAL 下如果只拷 focus.db 主文件，
        # 还没 checkpoint 的样本会全部丢掉 —— 库能打开、能查，只是少一截。
        bk = backup_db(_st_tmp / "bk.db")
        assert bk.exists(), "备份文件没生成"
        assert _count(bk) == 5, f"备份不完整：{_count(bk)} != 5"

        # 预演阶段一行都不许少
        compact_db(90, assume_yes=False)
        assert _count(tmp_db) == 5, "预演阶段就删了数据"

        # 真删：旧的清掉、新的留着
        compact_db(90, assume_yes=True)
        assert _count(tmp_db) == 2, f"应剩 2 条新样本，实际 {_count(tmp_db)}"
        # 真删之前必须留下备份 —— 库所在目录里应该多出一个 backup 文件
        # （备份落在 DB_PATH 旁边，名字是 `<库名>-backup-<时间戳>.db`）
        assert list(_st_tmp.glob("compact-backup-*.db")), \
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
        empty_db = _st_tmp / "notable.db"
        sqlite3.connect(empty_db).close()
        globals()["DB_PATH"] = empty_db
        compact_db(90, assume_yes=True)           # 只该打印一句提示
    finally:
        # 不用手工清理了：_st_td 是临时目录，退出时整个删掉。
        # 早先这里是 `ROOT.glob("_selftest_*.db*")` 的清理清单 —— 清单漏一个
        # 就留下污染源，而 `-wal`/`-shm` 比主文件更容易被漏掉。
        globals()["DB_PATH"] = real_db

    # ── WAL：切不过去必须被**看见**，绝不能假定成功 ──
    #
    # 用户报的「自评分打不开：database is locked」根因就在这里。老代码是裸的
    # `sqlite3.connect()`，压根没开 WAL —— journal_mode=delete 下**读写互斥**，
    # 而采集线程攒批提交（每 SAMPLE_COMMIT_EVERY 条一次）会**连续持有写锁
    # 十几秒**，读端在第一条语句（连 sqlite_master 的存在性检查）上就撞锁。
    #
    # 但光"开了 WAL"不是重点，重点是**切失败时不能假装成功**。
    # `PRAGMA journal_mode=WAL` 有两种失败形态，都必须被抓住：
    #   ① 抛异常 —— 需要排他锁而拿不到时（`busy_timeout` 在这条 PRAGMA 上
    #      **不起作用**，实测 0.0 秒就抛）；
    #   ② **不抛异常，但模式也没变** —— 内存库就是这样（只支持 memory）。
    # 只测 ① 是不够的：`return True` 那种"假定成功"的写法照样能过 ①。
    with tempfile.TemporaryDirectory() as _wtd:
        _wp = Path(_wtd) / "focus.db"
        _keep_db = globals()["DB_PATH"]
        globals()["DB_PATH"] = _wp
        # 记着所有连接，统一在 finally 里关掉：断言失败时中途退出会漏句柄，
        # Windows 上文件锁着，临时目录清理会抛 PermissionError ——
        # 那会把真正的断言信息盖掉，看起来像"文件被占用"而不是"库不是 WAL"。
        _conns: list[sqlite3.Connection] = []

        def _at(p: Path | str, timeout: float = 5.0) -> sqlite3.Connection:
            c = sqlite3.connect(p, timeout=timeout)
            _conns.append(c)
            return c

        def _mode(p: Path) -> str:
            c = sqlite3.connect(p, timeout=5.0)
            try:
                return journal_mode(c)
            finally:
                c.close()

        def _reset_delete() -> None:
            """把库退回 delete 模式，好让下一条断言真的在测"切换"这件事。

            少了这一步，"open_db 之后是 WAL"会在库本来就是 WAL 时假绿 ——
            连"open_db 完全不做切换"都测不出来。
            """
            c = _at(_wp)
            c.execute("PRAGMA journal_mode=delete")
            c.commit()
            assert journal_mode(c) == "delete", journal_mode(c)
            c.close()
            _conns.remove(c)

        _keep_level = log.level
        try:
            # 这一段会**故意**制造"切 WAL 失败"，警告是预期的。不压住的话，
            # 一次全绿的自检里会冒出「切 WAL 失败（库仍是 delete）」——
            # 看日志的人会以为真出事了，而不是"这是被断言覆盖的那条路径"。
            log.setLevel(logging.CRITICAL)
            # 造一个 delete 模式的库 —— 正是用户那份的形态
            _c = _at(_wp)
            _c.execute("PRAGMA journal_mode=delete")
            _c.execute("CREATE TABLE samples(ts REAL)")
            _c.commit()
            assert journal_mode(_c) == "delete", journal_mode(_c)

            # ① 能切的时候要真的切过去，并且**读回**确认
            assert _switch_wal(_c) is True, "空库上切 WAL 应该成功"
            assert journal_mode(_c) == "wal", "PRAGMA 发出去了但库没变成 WAL"

            # ② open_db 之后库必须处于 WAL —— 不是"试一下就算了"
            _reset_delete()
            _o = open_db(_wp)
            try:
                assert journal_mode(_o) == "wal", \
                    "open_db 之后库不是 WAL —— 老代码就是这么留在 delete 模式的"
            finally:
                _o.close()

            # ③ ensure_wal（启动时那条路）同样要确认
            _reset_delete()
            assert ensure_wal(timeout=5.0) is True, "启动时应该能切到 WAL"
            assert _mode(_wp) == "wal", _mode(_wp)

            # ④ 失败形态①：拿不到排他锁时**必须如实返回 False**。
            #    busy_timeout 在这条 PRAGMA 上不起作用（实测 0.0 秒就抛），
            #    所以这是"采集线程正在写"时最真实的那种失败。
            _reset_delete()
            _blk = _at(_wp)
            _blk.execute("BEGIN IMMEDIATE")       # 占住写锁，模拟采集线程
            _blk.execute("INSERT INTO samples VALUES(1.0)")
            _probe = _at(_wp)
            assert _switch_wal(_probe) is False, \
                "拿不到排他锁却报告切成功 —— 库其实还在 delete 模式，" \
                "读照样会撞 database is locked"
            assert journal_mode(_probe) != "wal", journal_mode(_probe)
            _blk.rollback()

            # ⑤ 失败形态②：**不抛异常但模式也没变**。内存库就是这种
            #    （只支持 journal_mode=memory）。少了这一条，
            #    "return True 就算成功"的写法能一路绿过去 —— 而那正是
            #    老代码的形态：吞掉结果、假定成功。
            _mem = _at(":memory:")
            assert journal_mode(_mem) == "memory", journal_mode(_mem)
            assert _switch_wal(_mem) is False, \
                "PRAGMA 没抛异常就当切成功了 —— 内存库根本没变成 WAL"
            assert journal_mode(_mem) == "memory", journal_mode(_mem)
        finally:
            log.setLevel(_keep_level)
            for _cc in _conns:
                try:
                    _cc.close()
                except Exception:
                    pass
            globals()["DB_PATH"] = _keep_db

    # ── 面板的 500 必须把栈写进日志 ──
    #
    # 用户报的是"自评分打不开"：/rate 回了一个 500 页面，页面上只有一句
    # `database is locked`，而 focus.log 里**一个字都没有** —— 页面把异常字符串
    # 塞进 HTML 就完事了，栈压根没记。结果只能拿截图里那一句话去猜。
    #
    # 这里用真请求打一遍（真起 server、真发 HTTP、真走 CSRF 校验），确认：
    #   ① 故障确实变成 500（而不是把服务带崩、或静默回 200）
    #   ② 日志里有一条 ERROR
    #   ③ 那条日志**带 exc_info** —— 没有栈的日志等于没记
    # GET 和 POST 两条分支各打一遍：只修一条是最容易犯的错。
    _dash = None
    try:
        import dashboard as _dash
    except Exception as _imp_exc:               # 面板缺依赖不该让自检整体挂掉
        print(f"  (跳过面板日志自检：dashboard 导入失败 {_imp_exc!r})")

    if _dash is not None:
        # ── do_POST 必须**先读请求体、再判来源** ──
        #
        # 顺序反过来会踩一个只在 Windows 上、而且**偶发**的坑：被拒的 POST
        # （跨源 Origin、Host 不对）如果带着一段**没人读的请求体**就关连接，
        # 系统会回一个 RST，客户端读 403 响应的中途直接炸：
        #     ConnectionAbortedError: [WinError 10053]
        # 实测探针：body 400 KB 时 200 次里炸 23 次，body 只有几十字节时
        # 200 次一次不炸 —— 所以它在自检里表现为**约 1% 的偶发红**，
        # 看起来像网络抖动，而 CI 每次要跑几十遍自检，迟早随机变红。
        #
        # 行为上抓不稳（要撞运气，撞不到就是假绿），所以这里钉**顺序**这个
        # 确定性的事实。放在这段 HTTP 测试**之前**：断言要跑在会偶发崩溃的
        # 代码前面，否则变异可能先崩在 ConnectionAbortedError 上，
        # 变异闸门只会打一个 `??`（不是被断言抓住的）。
        _dash_src = (ROOT / "dashboard.py").read_text(encoding="utf-8")
        _post_src = _dash_src[_dash_src.index("def do_POST"):]
        assert (_post_src.index("self.rfile.read")
                < _post_src.index("self._guard()")), (
            "do_POST 先判来源、后读请求体 —— 被拒的 POST（跨源 / Host 不对）"
            "会带着没读完的请求体关连接，Windows 回 RST，客户端读 403 响应的"
            "中途就炸（ConnectionAbortedError 10053，自检约 1% 偶发变红）")

        import http.server as _httpsrv
        import urllib.error as _urlerr
        import urllib.parse as _urlparse

        # 端口顺延必须**真的**会顺延。不能靠 bind 失败来判断"端口被占用"：
        # ThreadingHTTPServer 的类默认值是 allow_reuse_address = 1
        # → 会设 SO_REUSEADDR，而 Windows 的 SO_REUSEADDR 允许**两个进程
        # 同时绑住同一个 127.0.0.1:端口、两边都成功**（实测）。
        # 后果不是"谁接走"这么轻：后绑的那个一个请求都收不到，是个幽灵服务，
        # 而顺延循环因为 bind 永远不报错，成了死代码 —— 端口被占着也照绑不误。
        # 所以这里真起一个监听，看 _make_server 会不会绕开它。
        import dashboard as _dashmod
        _hold = _httpsrv.ThreadingHTTPServer(("127.0.0.1", 0), _dashmod._Handler)
        _hold_port = _hold.server_address[1]
        threading.Thread(target=_hold.serve_forever, daemon=True).start()
        try:
            _shifted = _dashmod._make_server(_hold_port)
            try:
                assert _shifted.server_address[1] != _hold_port, (
                    f"127.0.0.1:{_hold_port} 上已经有服务在监听，_make_server "
                    "却还是绑了同一个端口 —— 顺延是死代码：它以为自己拥有"
                    "这个端口，实际一个请求都收不到（连接全被先绑的接走），"
                    "面板会打到别人身上")
            finally:
                _shifted.server_close()
        finally:
            _hold.shutdown()
            _hold.server_close()

        class _Capture(logging.Handler):
            def __init__(self) -> None:
                super().__init__()
                self.records: list[logging.LogRecord] = []

            def emit(self, record: logging.LogRecord) -> None:
                self.records.append(record)

        _cap = _Capture()
        _dlog = logging.getLogger("focus.dashboard")
        _root = logging.getLogger()
        _keep_handlers = _root.handlers[:]
        _keep_dlvl = _dlog.level
        # 临时掐断根日志的出口：注入的故障栈不该被写进用户真正的 focus.log。
        # 顺带让 _ensure_log 认为"已经配好了"，不去碰真实文件。
        _root.handlers = [logging.NullHandler()]
        _dlog.addHandler(_cap)
        _dlog.setLevel(logging.DEBUG)
        _srv = None
        _orig_rate = _dash._rate_page
        _orig_save = None
        try:
            def _boom(*_a, **_k):
                raise RuntimeError("自检注入的故障")

            _dash._rate_page = _boom
            _srv = _httpsrv.ThreadingHTTPServer(("127.0.0.1", 0), _dash._Handler)
            _srv.daemon_threads = True
            _port = _srv.server_address[1]
            threading.Thread(target=_srv.serve_forever, daemon=True).start()

            def _hit(path: str, data: bytes | None = None) -> tuple[int, str]:
                """真发一个请求，返回 (状态码, 页面文本)。500 也要拿得到 body。"""
                try:
                    with urllib.request.urlopen(
                            f"http://127.0.0.1:{_port}{path}",
                            data=data, timeout=15) as _r:
                        return _r.status, _r.read().decode("utf-8")
                except _urlerr.HTTPError as _he:
                    return _he.code, _he.read().decode("utf-8")

            def _check(what: str, code: int, body: str) -> None:
                assert code == 500, \
                    f"{what}：注入故障后应该回 500，实际 {code}"
                assert "500" in body, f"{what}：500 的页面内容不对"
                _recs = _cap.records
                assert [r for r in _recs if r.levelno >= logging.ERROR], \
                    f"{what}：500 没写日志 —— 页面又把原因吞掉了（只能靠截图猜）"
                assert any(r.exc_info for r in _recs), \
                    f"{what}：记了日志但没带 exc_info —— 没有栈，等于还是查不出原因"
                assert any(r.exc_info and "RuntimeError" in str(r.exc_info[0])
                           for r in _recs), \
                    f"{what}：日志里的栈不是被注入的那个异常"

            # ① GET 分支
            _check("GET /rate", *_hit("/rate"))

            # ② POST 分支。CSRF 是真校验（compare_digest），所以得拿本次进程
            #    生成的那个 token —— 这也顺带证明"500 发生在校验之后"，
            #    不是被 403 挡掉后误判成成功。
            _cap.records.clear()
            import ratings as _ratings
            _orig_save = _ratings.save
            _ratings.save = _boom
            _form = _urlparse.urlencode({
                "csrf": _dash.CSRF_TOKEN, "block_start": "1", "score": "3",
            }).encode("utf-8")
            _check("POST /rate", *_hit("/rate", data=_form))

            # ③ /show-panel：跨进程"请监视进程显示面板"这条路的**服务端**那一半。
            #
            # 为什么必须真打一次：客户端那半边自检已经验过"请求发到了
            # /show-panel"（用假的 urlopen），注册那行也有结构断言 —— 但**路由
            # 本身**在 dashboard.do_GET 里，中间还隔着"路径比对 → 起线程 →
            # 调注入的实现"三步。路由名打错、或者忘了起线程，客户端照样收到
            # 200、子进程日志里一切正常，而窗口永远不出现 —— 用户看到的就是
            # "双击开关没反应"，正好是这条链路要修的那个症状。
            # 上面那两条断言都抓不到它，所以这里把中间那段真跑一遍。
            _shown: list[int] = []
            _keep_shower = _dash._panel_shower
            _cap.records.clear()          # 只留这次请求的记录，下面要按它断言
            try:
                _dash._panel_shower = lambda: _shown.append(1)
                _sp_code, _sp_body = _hit("/show-panel")
                assert _sp_code == 200, \
                    f"/show-panel 应该回 200（客户端只认这个），实际 {_sp_code}"
                assert _sp_body.strip() == "ok", \
                    f"/show-panel 的响应体应该是 ok，实际 {_sp_body[:80]!r}"
                # 注入的实现是**另起线程**调的，给它一点时间落地
                for _ in range(100):
                    if _shown:
                        break
                    time.sleep(0.02)
                assert _shown == [1], (
                    "/show-panel 回了 200，但**没有调用**注入的「显示面板」实现"
                    " —— 子进程会以为一切正常，而监视进程的窗口永远不出现"
                    "（用户看到的就是「双击开关没反应」）")
                # 服务端还要**自己**留一条日志。
                #
                # 客户端那条 `已请求监视进程显示面板` 是另一个进程写的：它压日志、
                # 或者发请求的压根不是我们的程序，服务端这边就一片安静 —— 而
                # 窗口正在往最顶层弹。实测踩过：自检里一处裸奔的 wait_and_open
                # 打到了用户正在跑的实例上，focus.log 里一条都没有，最后靠
                # EnumWindows 反复试才定位到。有这条，下次就是一眼的事。
                assert any(r.levelno == logging.INFO
                           and "/show-panel" in r.getMessage()
                           for r in _cap.records), (
                    "/show-panel 服务端没写日志 —— 窗口自己弹出来了也查不出是谁弹的"
                    "（客户端那条在另一个进程里，可能被压掉或压根不是我们发的）")
            finally:
                _dash._panel_shower = _keep_shower

            # _ensure_log 的补配分支：面板被**单独**跑起来（python dashboard.py）
            # 时没人调过 setup_log，而 pythonw 没有 stderr —— 不补就什么都留不下。
            # 把 setup_log 换掉再验，避免真去动用户的 focus.log。
            #
            # 注意这里不能用 `focus.setup_log`：`--selftest` 下本文件是 __main__，
            # 而 dashboard 里的 `focus` 是**另一个**模块对象（sys.modules["focus"]，
            # 由 dashboard 的 `import focus` 产生）。补丁要打在它身上才生效。
            _fmod = sys.modules.get("focus")
            assert _fmod is not None, "dashboard 导入后 sys.modules 里应该有 focus"
            _calls: list[int] = []
            _orig_setup = _fmod.setup_log
            _keep_ready = _dash._log_ready
            try:
                _fmod.setup_log = lambda *a, **k: _calls.append(1)
                _root.handlers = []
                _dash._log_ready = False
                _dash._ensure_log()
                assert _calls, \
                    "根日志没有出口时 _ensure_log 没补配 —— 单独跑面板时栈会消失"
            finally:
                _fmod.setup_log = _orig_setup
                _dash._log_ready = _keep_ready
        finally:
            _dash._rate_page = _orig_rate
            if _orig_save is not None:
                _ratings.save = _orig_save
            if _srv is not None:
                _srv.shutdown()
                _srv.server_close()
            _dlog.removeHandler(_cap)
            _dlog.setLevel(_keep_dlvl)
            _root.handlers = _keep_handlers

        # ③ 守卫必须认得 WebView2 的**表单 POST**。
        #    实测（Edge 153 / WebView2）同源表单 POST 送的是
        #        Origin: null    Sec-Fetch-Site: same-origin
        #    而不是 Origin: http://127.0.0.1:端口。旧写法把"scheme 不是
        #    http/https"一律当跨源拒掉，于是应用窗口里**所有**表单
        #    （自述评分、跳过、设置保存）点下去都只得到一张 403 页面。
        #
        #    上面两段 500 自检永远碰不到这条路径：urllib 默认**不发** Origin，
        #    所以"拿 urlopen 打一个 POST"和真实客户端长得一点都不像。
        #    要按真客户端的头来发，才测得出来。
        assert _dash._origin_ok("null", 8787), \
            "Origin: null 是 WebView2 同源表单 POST 的正常值，必须放行"
        assert _dash._origin_ok("", 8787), "没有 Origin 必须放行"
        assert _dash._origin_ok("NULL", 8787), "大小写不该改变结论"
        assert _dash._origin_ok("http://127.0.0.1:8787", 8787), "本机同源放行"
        assert _dash._origin_ok("http://localhost:8787", 8787), "localhost 放行"
        assert not _dash._origin_ok("http://evil.example", 8787), \
            "跨源 Origin 必须拒 —— 放行 null 不能顺手把防线拆了"
        assert not _dash._origin_ok("http://127.0.0.1:9999", 8787), \
            "回环但不是本面板的端口，要拒"
        assert not _dash._origin_ok("file:///C:/x.html", 8787), \
            "file:// 带主机路径，不是不透明来源，照旧拒"

        _srv2 = _httpsrv.ThreadingHTTPServer(("127.0.0.1", 0), _dash._Handler)
        _srv2.daemon_threads = True
        _port2 = _srv2.server_address[1]
        threading.Thread(target=_srv2.serve_forever, daemon=True).start()
        _dlog.addHandler(_cap)
        _dlog.setLevel(logging.DEBUG)
        _keep_ready2 = _dash._log_ready
        _root.handlers = [logging.NullHandler()]
        try:
            def _hit2(path: str, data: bytes | None = None,
                      origin: str | None = None,
                      site: str | None = None) -> tuple[int, str]:
                """按真实浏览器的头发请求 —— Origin 由调用方指定。"""
                hdrs = {}
                if origin is not None:
                    hdrs["Origin"] = origin
                if site is not None:
                    hdrs["Sec-Fetch-Site"] = site
                    hdrs["Sec-Fetch-Mode"] = "navigate"
                    hdrs["Sec-Fetch-Dest"] = "document"
                req = urllib.request.Request(
                    f"http://127.0.0.1:{_port2}{path}", data=data, headers=hdrs)
                try:
                    with urllib.request.urlopen(req, timeout=15) as r:
                        return r.status, r.read().decode("utf-8")
                except _urlerr.HTTPError as he:
                    return he.code, he.read().decode("utf-8")

            # 窗口首次加载页面：无 Origin（Sec-Fetch-Site: none）
            code, _b = _hit2("/settings")
            assert code == 200, f"窗口首次加载页面被拒：{code}"

            # 报告页必须带导航。它自己就会给出「去设置调阈值」的建议，
            # 而页面上原来一个入口都没有 —— 用户读完只能关窗口。
            # 反过来，导出成 report.html 时**不该**有导航：独立文件里是死链。
            code, body = _hit2("/report")
            assert code == 200, f"报告页打不开：{code}"
            assert '<div class="nav">' in body and 'href="/settings"' in body, (
                "面板里的报告页没有导航 —— 用户读到「去设置调阈值」也无处可点")
            assert '<div class="nav">' not in report.build_html([]), \
                "导出的 report.html 不该有导航：独立文件里的链接点不动"
            assert '<div class="nav">' not in report.build_html(
                [(1789000000.0, "focused", "code.exe", "t",
                  0.0, 0.0, 0.3, 1.0, 1.0, 1, 0.0)]), \
                "导出的 report.html 不该有导航：独立文件里的链接点不动"

            # WebView2 的评分提交：Origin: null 要过守卫，落到业务校验（400）
            _form0 = _urlparse.urlencode({
                "csrf": _dash.CSRF_TOKEN, "block_start": "0", "score": "3",
            }).encode("utf-8")
            code, body = _hit2("/rate", data=_form0, origin="null",
                               site="same-origin")
            assert code != 403 and "已拒绝" not in body, (
                f"WebView2 的表单 POST（Origin: null）被守卫拒了：{code} —— "
                "应用窗口里的自述评分/跳过/设置保存会全部打不开")
            assert code == 400, (
                "Origin: null 的表单应该过守卫、过 token 校验，落到 "
                f"block_start 的业务校验（400），实际 {code}")

            # 跨源来源照旧拒
            code, body = _hit2("/rate", data=_form0,
                               origin="http://evil.example", site="cross-site")
            assert code == 403 and "来源" in body, "跨源 Origin 必须继续拒"
            code, _b = _hit2("/settings", origin="http://evil.example")
            assert code == 403, "跨源页面不能读到面板内容"

            # 不透明来源不是免死金牌：token 照旧要校验
            code, body = _hit2("/rate", data=_urlparse.urlencode({
                "csrf": "x" * 8, "block_start": "0", "score": "3",
            }).encode("utf-8"), origin="null", site="same-origin")
            assert code == 403 and "令牌" in body, (
                "Origin: null + 错 token 必须拒 —— POST 的真防线是 token，"
                f"不是来源（实际 {code}）")

            # DNS rebinding 那条线：Host 不是回环地址一律拒
            import http.client as _httpc
            _conn = _httpc.HTTPConnection("127.0.0.1", _port2, timeout=15)
            try:
                _conn.request("GET", "/settings",
                              headers={"Host": "evil.example"})
                _hostcode = _conn.getresponse().status
            finally:
                _conn.close()
            assert _hostcode == 403, \
                "Host 不是回环地址必须拒（DNS rebinding 防线）"

            # 403 必须留下现场：被拒的 Origin 值要进日志，否则用户只能靠截图
            _cap.records.clear()
            _hit2("/settings", origin="http://evil.example")
            _warns = [r for r in _cap.records if r.levelno >= logging.WARNING]
            assert _warns, "403 没写日志 —— 用户只能拿一张截图来问，没法自查"
            assert any("evil.example" in r.getMessage() for r in _warns), (
                "403 的日志里没带上被拒的 Origin 值，等于没写")
        finally:
            _dash._log_ready = _keep_ready2
            _dlog.removeHandler(_cap)
            _dlog.setLevel(_keep_dlvl)
            _root.handlers = _keep_handlers
            _srv2.shutdown()
            _srv2.server_close()

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

    # ── 「重新启动自己」的命令行必须区分源码运行和冻结成 exe ──
    #
    # 这一条是实机才能暴露的那种：`getattr(sys, "frozen", False)` 在开发机上
    # 永远是 False，所以"发布包里入口变成 exe 自己"那条分支根本没人跑过。
    # 写死 .venv/Scripts/pythonw.exe 的后果是：发布包里既没有 .venv 也没有
    # focus.py，subprocess.Popen 抛 FileNotFoundError，而 exe 是 --windowed、
    # 没有控制台 —— 用户双击桌面开关，**什么都没发生，也没有任何报错**。
    _frozen = relaunch_cmd(frozen=True)
    assert _frozen == [sys.executable], \
        f"冻结态的启动命令应该是 exe 自己，实际是 {_frozen}"
    # 断言"不指向 PYW / focus.py"，而不是"不含 .venv" —— 开发机上
    # sys.executable 本来就在 .venv 里，那样写会在自检里自己误报。
    assert str(PYW) not in _frozen, \
        f"冻结态不该再指向 {PYW.name}（发布包里没有 .venv）：{_frozen}"
    assert not any(p.endswith("focus.py") for p in _frozen), \
        f"冻结态不该再传 focus.py（发布包里没有它）：{_frozen}"
    _dev = relaunch_cmd(frozen=False)
    assert _dev[0].endswith("pythonw.exe"), \
        f"源码运行要用 pythonw（不弹控制台），实际是 {_dev[0]}"
    assert _dev[-1].endswith("focus.py"), \
        f"源码运行的入口是 focus.py，实际是 {_dev[-1]}"
    # 源码形态下 PYW 必须真的存在，否则开发机上双击开关就是坏的。
    #
    # 但**只在项目确实用了本地 .venv 时**才检查：CI 是用 actions/setup-python
    # 装全局依赖的，根本没有 .venv —— 无条件断言会在 CI 上红，而本机
    # （有 .venv）永远复现不了。这条是实测踩出来的：第一版就是这么红的。
    # 断言不能编码"开发机的目录布局"这种环境假设。
    if not getattr(sys, "frozen", False) and (ROOT / ".venv").exists():
        assert PYW.exists(), f"找不到 {PYW} —— 开发机上桌面开关会失效"

    # ── wait_and_open：服务等不到要退出、显示面板要**交给监视进程** ──
    #
    # 原来这两件事挤在同一个 try 里：开窗口失败会被当成"服务还没起来"，
    # 于是一路重试到 120 秒超时、最后一声不吭地退出。发布包漏了 pywebview
    # 时就是这个表现。下面几条把这几条路径分别钉住。
    #
    # 这一段必须**压住日志**：wait_and_open 第一件事就是 setup_log()，于是下面
    # 这些"故意制造失败"的调用会把假的 ERROR 写进用户真正的 focus.log。
    # 实测踩到过：用户日志里躺着几条这样的 ERROR，看起来像"窗口层真的坏了"，
    # 其实是自检自己造的 —— 排查时会被带偏。
    #
    # **只调 setLevel 不够。** 这一版就栽在这上面：级别是"谁改谁还"的全局状态，
    # `_run_wa` 的 finally 里顺手把它还成了**进保护前**的值，于是第一次调用一
    # 结束闸门就开了，紧接着那次故意失败的调用正好在裸奔 —— 用户日志里
    # 10:17:58 那三条假 ERROR 就是这么来的。
    # 所以除了设级别，还在**日志出口**上挂一个探针：这一段跑完断言它一条都没
    # 收到。级别被谁改回去都逃不过它。（setup_log() 用的是
    # `basicConfig(force=True)`，它只清理 root 上的 handler，挂在 focus 这个
    # logger 上的探针不会被拆掉。）
    _leaked: list[str] = []

    class _LeakProbe(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            _leaked.append(f"{record.levelname} {record.getMessage()}")

    _probe = _LeakProbe()
    log.addHandler(_probe)
    _keep_lvl_wa = log.level
    log.setLevel(logging.CRITICAL)

    # ── 服务等不到时必须及时返回，而且**绝不能碰真网络** ──
    #
    # ⚠️ 这条断言曾经一边绿着、一边把用户正在运行的那个窗口弹到最前面。
    #
    # 它原来是裸奔的：用**真的 URL_PATH**（用户机器上就是运行中的应用写下的
    # `http://127.0.0.1:8787/`）+ **真的 urlopen**。于是"服务等不到"这个前提
    # 在应用正在运行的机器上根本不成立 —— 服务就在那儿，wait_and_open 会一路
    # 走到底、请求 `/show-panel`，把那个窗口 `show()` 出来。而断言只看
    # "有没有超过 10 秒"，照样是绿的：它测的根本不是它想测的东西。
    #
    # 副作用正好是用户最烦的那件事。用户的原话是
    # 「每次让你改一点东西的时候就是这样一直弹出这个在最顶层」——
    # 因为"改一点东西"就要跑一遍自检，而**每跑一遍自检就弹一次他的窗口**。
    # 实测（把窗口点 × 收起来，跑一次 `--selftest`，它立刻又可见了；再用
    # urlopen 探针拦，看到的就是 `http://127.0.0.1:8787/` 和
    # `.../show-panel` 这两条真实请求）。
    #
    # 所以现在两件事一起钉：地址指到**没人监听的端口**，urlopen 换成必定失败
    # 的假货，并且把"它到底请求过谁"记下来 —— 只要出现任何一个不是那个死地址
    # 的请求，就说明它又在碰真东西了。
    global URL_PATH          # 下面几段都要临时改它，声明必须在**所有使用之前**
    _real_urlopen = urllib.request.urlopen
    _dead_td = tempfile.TemporaryDirectory()
    _dead_url = Path(_dead_td.name) / "focus.url"
    _dead_url.write_text("http://127.0.0.1:1/", encoding="utf-8")
    _asked_real: list[str] = []

    def _refuse_urlopen(u, *a, **k):
        _asked_real.append(str(u))
        raise OSError("自检：不许碰真网络（会请求到正在运行的那个实例上）")

    _keep_urlpath_early = URL_PATH
    _t0 = time.time()
    try:
        globals()["URL_PATH"] = _dead_url
        urllib.request.urlopen = _refuse_urlopen
        wait_and_open(timeout=0.1)      # 服务起不来 → 必须很快返回，不能卡住
    finally:
        urllib.request.urlopen = _real_urlopen
        globals()["URL_PATH"] = _keep_urlpath_early
    assert time.time() - _t0 < 10.0, "服务等不到时必须及时返回，不能一直转"
    assert _asked_real and all(u.startswith("http://127.0.0.1:1/")
                               for u in _asked_real), (
        f"wait_and_open 请求了 {_asked_real} —— 自检里只许敲那个死地址。"
        "打到真地址上就是打到了**用户正在运行的那个实例**，/show-panel 会把"
        "他的窗口弹到最前面（用户报的「每次让你改点东西就弹窗」就是这么来的）")

    _real_open = webbrowser.open
    _win_mod = sys.modules.get("window", "__absent__")
    _opened: list[str] = []
    _asked: list[str] = []

    # 面板地址必须是**监视进程写下的实际值**，不能猜端口。
    # 8787 被占用时服务会顺延到 8788…8806（dashboard._make_server），而
    # wait_and_open 跑在另一个进程里、拿不到 base_url()。硬编码默认端口的
    # 写法在那种机器上会让子进程一直敲一个没人监听的地址、等满 120 秒然后
    # 放弃 —— 用户看到的还是"双击了没反应"。
    # 下面把 URL_PATH 指到临时文件，并且**故意用非默认端口**：这样
    # "到底有没有读文件"就成了一个能被变异抓到的差别，而不是"反正都通"。
    # （`global URL_PATH` 已经在上面那段声明过了，这里不要再写一遍 ——
    #  写在"使用之后"会直接 SyntaxError。）
    _keep_urlpath = URL_PATH
    _wa_td = tempfile.TemporaryDirectory()
    _wa_url = Path(_wa_td.name) / "focus.url"
    URL_PATH = _wa_url
    try:
        from dashboard import DEFAULT_PORT as _dport
    except Exception:
        _dport = 8787

    class _FakeResponse:
        """要能当上下文管理器用、还要有 read() —— wait_and_open 会把响应体
        读完再关（不读完就关，会在服务端留下 ConnectionAbortedError）。"""

        def read(self, *a, **k):
            return b""

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def close(self):
            pass

    def _run_wa(fail_show: bool,
                url_text: str = "http://127.0.0.1:8801/") -> None:
        """跑一次 wait_and_open，记下"请求了哪些地址、开了哪些网页"。

        每次都先把 URL_PATH 写成**非默认端口**的地址：这样"wait_and_open
        有没有真的去读监视进程写下的地址"就成了一个可观测的差别。
        url_text 传空串表示"文件还不存在"（监视进程刚起来那种时刻）。
        """
        def _fake_urlopen(u, *a, **k):
            _asked.append(u)
            if fail_show and u.endswith("/show-panel"):
                raise OSError("404")      # 模拟老版本进程没有这个路由
            return _FakeResponse()

        try:
            urllib.request.urlopen = _fake_urlopen
            # 让 `import window` 失败（等价于发布包里漏了 pywebview）：
            # sys.modules 里放 None 会让 import 直接抛 ImportError。
            sys.modules["window"] = None
            webbrowser.open = lambda u, *a, **k: _opened.append(u)
            if url_text:
                _wa_url.write_text(url_text, encoding="utf-8")
            else:
                _wa_url.unlink(missing_ok=True)
            wait_and_open(timeout=5.0)
        finally:
            urllib.request.urlopen = _real_urlopen
            webbrowser.open = _real_open
            # **不要在这里还日志级别。** 闸门是这一段整体的，由这一段末尾统一还。
            # 在这里还，还的是"进保护前"的值 —— 第一次调用一结束闸门就开了，
            # 后面那次故意失败的调用就把假 ERROR 写进了用户真正的 focus.log。
            if _win_mod == "__absent__":
                sys.modules.pop("window", None)
            else:
                sys.modules["window"] = _win_mod

    # 正常路径：请求必须发给**监视进程**，而且成功时**绝不能**再开浏览器。
    # 后者正是用户报的「双击开关不断弹出新网页」—— 子进程里没有窗口，
    # 它自己调 open_page() 每次都退到浏览器。
    _run_wa(fail_show=False)
    assert any(u.endswith("/show-panel") for u in _asked), (
        "wait_and_open 没有请监视进程显示面板 —— 它会自己调 open_page()，"
        "而那个子进程里没有窗口，每次都会退到浏览器弹一个新网页")
    assert not _opened, (
        f"监视进程已经接管了显示面板，wait_and_open 还是自己开了浏览器："
        f"{_opened} —— 用户看到的就是「双击开关不断弹出新网页」")
    # 而且必须是**监视进程写下的那个地址**（上面故意用了非默认端口 8801）。
    # 硬编码 DEFAULT_PORT 的写法在"8787 被占用、服务顺延到 8788"的机器上
    # 会让子进程一直敲一个没人监听的地址，等满 120 秒然后放弃。
    assert _asked and _asked[0].startswith("http://127.0.0.1:8801/"), (
        f"wait_and_open 没用监视进程写下的面板地址，实际请求了 {_asked[:2]} —— "
        "8787 被占用时服务会顺延端口，硬编码默认端口会让子进程等一个"
        "没人监听的地址，等满 120 秒然后放弃（用户看到的是「双击了没反应」）")

    # 兜底路径：老版本监视进程没有 /show-panel 路由（用户刚更新了开关、
    # 后台进程还是旧的）→ 必须退到本进程打开，哪怕落在浏览器里。
    _opened.clear()
    _asked.clear()
    _run_wa(fail_show=True)
    assert _opened, (
        "连监视进程都请不动时，wait_and_open 必须退到系统浏览器 —— "
        "否则用户看到的是「双击了没反应」")

    # ── 地址文件缺失 / 写得不规范时，取地址的两个函数各自要兜住 ──
    #
    # ① 监视进程刚起来、地址还没落盘（PID 文件比地址先写）→ 退回默认端口，
    #    不能抛异常，也不能拿空串去请求。调用方是轮询，下一轮就读到真值了。
    _wa_url.unlink(missing_ok=True)
    _fb = _persisted_panel_url()
    assert _fb.endswith("/"), f"面板地址必须以 / 结尾，实际 {_fb!r}"
    assert f":{_dport}/" in _fb, \
        f"地址文件缺失时应该退回默认端口，实际 {_fb!r}"

    # ② 文件里没有结尾斜杠时要补上 —— 否则拼出来的是 `...:8801show-panel`，
    #    服务端只认 `/show-panel`，会静默 404（看起来又像"窗口层坏了"）。
    _wa_url.write_text("http://127.0.0.1:8801", encoding="utf-8")
    assert _persisted_panel_url() == "http://127.0.0.1:8801/", (
        f"地址没有结尾斜杠时没补上，实际 {_persisted_panel_url()!r}"
        " —— 拼出来的会是 ...:8801show-panel，服务端只会回 404")

    # ③ _panel_url（托盘菜单 / 浏览器兜底那条路）也必须认 URL_PATH：
    #    子进程里 `dashboard._server` 是 None，`base_url()` 只会退回默认端口。
    assert _panel_url("/rate") == "http://127.0.0.1:8801/rate", (
        f"_panel_url 没认监视进程写下的地址，实际 {_panel_url('/rate')!r}"
        " —— 端口顺延过时，托盘打开 / 浏览器兜底都会落到一个连不上的地址")

    # ④ **子进程取地址时绝不能把服务起起来。** `_panel_url` 会调
    #    `serve_background`（那是它的正常语义），子进程里 `_server` 是 None，
    #    于是它真的会在子进程里另起一个面板服务 —— 之后子进程是在自己回
    #    自己的请求：/show-panel 回 200 ok 却什么都不做，窗口永远不出现，
    #    **而日志里一切正常**。所以 wait_and_open 走的是只读的那个。
    if _dash is not None:
        _sb_calls: list[int] = []
        _orig_sb = _dash.serve_background
        try:
            _dash.serve_background = (
                lambda *a, **k: _sb_calls.append(1) or "http://127.0.0.1:9/")
            _wa_url.unlink(missing_ok=True)
            _persisted_panel_url()
            assert not _sb_calls, (
                "子进程取面板地址时把服务**又起了一遍** —— 它会自己回自己的"
                "请求：/show-panel 回 200 ok 却什么都不做，窗口永远不出现，"
                "而且日志里一切正常（这种静默失败最难查）")
        finally:
            _dash.serve_background = _orig_sb

    # 地址文件在上面那条用例里被删掉了，这里写回来 —— 后面几个用例
    # （show_panel 的浏览器兜底、open_page 的浏览器兜底）都会走
    # `_panel_url()`，而它在**地址文件缺失**时会真的起一个面板服务
    # （那是它的正常语义）。自检不该顺手占一个端口；更要紧的是"占哪个端口"
    # 取决于用户机器上 8787 有没有被应用自己占着 —— 断言跟着环境变，
    # 是最难查的那类偶发失败。
    _wa_url.write_text("http://127.0.0.1:8801", encoding="utf-8")

    # ── show_panel：窗口层在、GUI 却没起来 → 仍然要开出一个页面来 ──
    #
    # 这条钉两件事：
    #   1. 必须**真的等**窗口对象。超时传 0（或者干脆不等）等于不等 ——
    #      那时 _window 还是 None，open_page 会走"窗口层不可用"的兜底弹一个
    #      浏览器网页，正是要修的那个 bug。所以这里验的是"等的时候带了
    #      正的超时"，不是"等到了"。
    #   2. 等不到之后不能干耗着，还是得把面板开出来。
    #
    # 浏览器兜底本身在 window.open_page 里面（那边才看得到 _window），
    # 这里用一个假的 window 模块顶替，所以断言落在"交给它了"这一层。
    _win_mod3 = sys.modules.get("window", "__absent__")
    _real_open3 = webbrowser.open
    _fake_win = type(sys)("window")
    _fake_calls: list[str] = []
    _wa_timeouts: list[float] = []

    def _fake_wait(timeout=20.0):
        _wa_timeouts.append(timeout)
        return False                    # 模拟 GUI 循环一直没起来

    _fake_win.open_page = lambda page="panel": _fake_calls.append(page)
    _fake_win.wait_until_ready = _fake_wait
    try:
        sys.modules["window"] = _fake_win
        show_panel(timeout=0.1)
    finally:
        if _win_mod3 == "__absent__":
            sys.modules.pop("window", None)
        else:
            sys.modules["window"] = _win_mod3
    assert _wa_timeouts and _wa_timeouts[0] > 0, (
        f"show_panel 没有真的去等窗口对象（收到的超时是 {_wa_timeouts}）—— "
        "_window 还没建出来就 show，open_page 会走「窗口层不可用」的兜底"
        "弹一个浏览器网页，正是要修的那个 bug")
    assert _fake_calls == ["panel"], (
        f"窗口层在、但 GUI 一直没起来时，show_panel 还是得把面板开出来"
        f"（实际调了 {_fake_calls}）—— 否则双击开关之后什么都不出现")

    # 窗口层整个加载不了（发布包漏了 pywebview）→ 必须退到系统浏览器
    _opened3: list[str] = []
    try:
        sys.modules["window"] = None
        webbrowser.open = lambda u, *a, **k: _opened3.append(u)
        show_panel(timeout=0.1)
    finally:
        webbrowser.open = _real_open3
        if _win_mod3 == "__absent__":
            sys.modules.pop("window", None)
        else:
            sys.modules["window"] = _win_mod3
    assert _opened3, (
        "窗口层整个加载不了时，show_panel 必须退到系统浏览器 —— "
        "否则用户双击开关之后什么都不出现")

    # ── open_page：窗口层坏了要退到浏览器，**而且托盘菜单必须都走它** ──
    #
    # 这条是实测踩出来的。用户的 .venv 里**没有 pywebview** —— 他更新前那版
    # 是用系统浏览器开面板的，压根不需要这个依赖；而更新后的 run_tray 最后
    # 一步是裸的 `import window; window.start_main()`，异常一路穿到 main() 的
    # `except BaseException` 再 `raise`，**进程启动两秒后就没了**。
    # 采集线程和托盘都不依赖窗口层，凭什么让它们陪葬。
    _win_mod2 = sys.modules.get("window", "__absent__")
    _real_open2 = webbrowser.open
    _urls: list[str] = []
    try:
        sys.modules["window"] = None          # import 直接抛 ImportError
        webbrowser.open = lambda u, *a, **k: _urls.append(u)
        # 注入的故障栈是预期的，日志级别由这一段开头那道闸门管着，这里不用再压
        open_page("rate")
        assert _urls, "窗口层不可用时 open_page 必须退到系统浏览器"
        assert _urls[-1] == "http://127.0.0.1:8801/rate", (
            f"退到浏览器时地址错了：{_urls[-1]} —— 要么页面路由不对"
            "（rate 不该落到首页），要么没认监视进程写下的端口")
    finally:
        webbrowser.open = _real_open2
        if _win_mod2 == "__absent__":
            sys.modules.pop("window", None)
        else:
            sys.modules["window"] = _win_mod2

    # 结账：上面那一串**不该真的起一个面板服务**。
    #
    # 这是个很容易被"简化"掉的性质：把 `serve_background` 从 `_panel_url` 的
    # `if not base:` 分支里提到外面，逻辑上像是"顺手把服务备好"，测试也照样
    # 全绿 —— 但自检从此会真的占一个端口，而且占哪个端口取决于用户机器上
    # 8787 有没有被应用自己占着。断言跟着环境变，是最难查的那类偶发。
    URL_PATH = _keep_urlpath
    assert _dash is None or _dash._server is None, (
        "自检过程中真的起了一个面板服务 —— 它不该有这个副作用："
        "用户正开着应用时 8787 被占、服务会顺延，断言就跟着环境变了")

    # 这一段到此结束：撤掉探针、还回日志级别，然后**结账**。
    #
    # 这条断言是这次真被咬出来的：自检往用户真正的 focus.log 里写了三条假
    # ERROR（"请求监视进程显示面板失败"、"应用窗口打不开"、"已改用系统浏览器
    # 打开"），看起来就像窗口层真的坏了。它钉的是**结果**（日志里到底漏没漏），
    # 不是"级别变量等于几"，所以以后怎么重构闸门都还成立。
    log.removeHandler(_probe)
    log.setLevel(_keep_lvl_wa)
    assert not _leaked, (
        f"自检把日志写进了用户真正的 focus.log：{_leaked} —— "
        "这些是自检**故意制造**的失败（模拟发布包漏了 pywebview、监视进程还是"
        "旧版本），混进真日志里看起来就像窗口层真的坏了，排查时会被带偏。"
        "多半是有人把这一段的日志闸门提前还回去了 —— 见这一段开头的注释。")

    # ── 桌面快捷方式的目标必须来自 relaunch_cmd() ──
    #
    # 桌面开关是不是死的，全看这一条：写死 .venv/Scripts/pythonw.exe 的话，
    # 发布包里 TargetPath 指向一个不存在的文件，双击毫无反应、也没有报错。
    # _write_lnk() 本身只是拼一个 PowerShell 字符串，所以可以拦下 subprocess
    # 把那段命令抓出来验，不用真的建快捷方式、也不碰用户的桌面。
    #
    # 但**必须把 _desktop() 指到临时目录**：改成"先写临时文件再 os.replace"
    # 之后，这个函数会真的落盘 —— 不指走的话，自检会往用户桌面上写一个
    # 临时 .lnk 并把它改名成「专注监视.lnk」，等于悄悄覆盖/创建他的快捷方式。
    # 上一版只拼字符串不落盘，所以没有这个副作用，改完就有了。
    global _desktop
    _real_desktop = _desktop
    _real_run = subprocess.run
    _td = tempfile.TemporaryDirectory()
    _desk = Path(_td.name)
    _tmp_lnk = _desk / f"{LNK_NAME}.new.lnk"
    _ps_seen: list[str] = []

    def _fake_run(cmd, *a, **k):
        if cmd and cmd[0] == "powershell":
            _ps_seen.append(" ".join(map(str, cmd)))
            # 假装 PowerShell 真的把临时文件写出来了 —— _write_lnk 靠它
            # 判断成功，然后 os.replace 到正式路径。
            _tmp_lnk.write_bytes(b"lnk")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return _real_run(cmd, *a, **k)

    try:
        subprocess.run = _fake_run
        _desktop = lambda: _desk
        _write_lnk(Path("dummy.ico"))                      # 默认 = relaunch_cmd()
        _write_lnk(Path("dummy.ico"),
                   cmd=[r"C:\sentinel\focus-monitor.exe", "--sentinel-arg"])

        assert len(_ps_seen) == 2, \
            f"_write_lnk 没有调用 powershell（{len(_ps_seen)} 次）"

        # ① 默认那条：目标必须就是 relaunch_cmd() 的入口，且带 --toggle
        _default_ps = _ps_seen[0]
        assert "--toggle" in _default_ps, \
            "快捷方式没带 --toggle —— 双击只会去起第二个实例，被单例挡住"
        assert relaunch_cmd()[0] in _default_ps, \
            "快捷方式的目标不是 relaunch_cmd() 的入口"

        # ② 哨兵那条：传进来的命令必须被**原样采纳**。
        #    这才是"目标不是写死的"的证明 —— 写死 .venv/Scripts/pythonw.exe 的
        #    旧写法同样含 pythonw.exe，光看①根本区分不出来。
        _sentinel_ps = _ps_seen[1]
        assert r"C:\sentinel\focus-monitor.exe" in _sentinel_ps, \
            "快捷方式的目标是写死的，没有采用传入的命令 —— " \
            "发布包里 TargetPath 会指向不存在的文件，桌面开关直接是死的"
        assert "--sentinel-arg" in _sentinel_ps, \
            "传入命令的额外参数没被写进快捷方式"

        # ③ 落盘走的是"临时文件 + 原子改名"：写的是临时路径，最终只留下正式文件。
        #    这个函数现在会被程序运行时反复调用（启动/看门狗/退出兜底），
        #    直接覆盖 Save() 中途失败会留下 0 字节的 .lnk，双击没反应也删不掉。
        assert str(_tmp_lnk) in _ps_seen[1], \
            "PowerShell 写的不是临时文件 —— 覆盖写不是原子的"
        assert (_desk / LNK_NAME).exists(), "快捷方式没真的落到目标路径上"
        assert not _tmp_lnk.exists(), \
            "临时 .lnk 没被 os.replace 收走，桌面上会留下一个残留文件"
    finally:
        subprocess.run = _real_run
        _desktop = _real_desktop
        _td.cleanup()

    # ── 桌面开关的图标必须反映「程序在不在跑」 ──
    #
    # 用户报的 bug：待机/断网之后再双击开关，图标是绿的，但程序根本没起来。
    # 根因是 toggle() 先写绿点、再起进程，之后再也不回头看。所以这组断言钉的是
    # **图标只由观测到的状态决定，不由"用户按过开关"决定**。
    assert shortcut_state(running=False, stale=False) == "off"
    assert shortcut_state(running=False, stale=True) == "off", \
        "进程都没了还显示黄叹号 —— 没在跑就该一律灰杠"
    assert shortcut_state(running=True, stale=False) == "on"
    assert shortcut_state(running=True, stale=True) == "alert", \
        "在跑但没数据必须变黄叹号，否则和健康状态看不出区别"

    # 三个状态必须是**三个不同的文件**：.lnk 的 IconLocation 存的是路径，
    # Windows 按路径缓存图标 —— 同一个路径只换内容，桌面往往不刷新，
    # 于是"图标改了、用户看到的还是绿的"。旧的 ensure_icons() 就是这种写法。
    _icons = ensure_icons()
    assert set(_icons) == set(_ICON_STATES), _icons
    assert len({p.read_bytes() for p in _icons.values()}) == 3, \
        "三个状态的图标内容一模一样 —— 桌面开关分不出在跑/没在跑"

    # 两条都容易写反的规则
    assert not should_sync_shortcut(False, "on", None, True), \
        "快捷方式不存在时不该替用户凭空建一个（不是所有人都要桌面开关）"
    assert not should_sync_shortcut(True, "on", "on", False), \
        "状态没变不该重写 —— 看门狗每 30 秒调一次，每次写都白起一次 PowerShell"
    assert should_sync_shortcut(True, "alert", "on", False), "状态变了必须重写"
    assert should_sync_shortcut(True, "on", "on", True), "force 必须能强制写"

    # 「起没起来」要真的去等、去确认。probe 可注入，否则"失败"那一支在开发机上
    # 永远没人跑过：本机真开着监视时 running_pid() 恒为真。
    assert _wait_until_running(timeout=1.0, probe=lambda: 1234) is True, \
        "进程起来了却判定失败 —— 绿点会永远写不出来"
    _t0 = time.time()
    assert _wait_until_running(timeout=0.3, probe=lambda: None) is False, \
        "进程根本没起来却判定成功 —— 正是用户报的那个 bug"
    assert time.time() - _t0 < 5.0, "确认失败也不该拖很久"

    # ── 结构性断言：采集循环里的「视觉链路」必须还连着 ──
    #
    # 来自一次真实事故：采集循环重构时，把 vision.read_face / read_posture 和
    # 四个缓冲区的填充**整段删掉了，消费端却留着**。于是 face_buf / tilt_buf /
    # _ear_hist 永远为空 → present 恒为 False → decide() 只会返回
    # distracted / away —— 专注 / 中性 / 伏案 / 疲劳四种状态一个都出不来。
    #
    # 当时自检**全绿**。因为它测的是 decide() 和 ear_threshold() 本身，
    # 而这两个函数的逻辑确实没问题，坏的是"根本没人喂数据给它"。
    # 纯逻辑断言对"接线断了"这一类故障是盲的，所以补一条只验证接线、
    # 不验证行为的断言 —— 这正是当时丢掉的那个东西。
    #
    # 只扫 def selftest 之前的源码：否则下面这些针脚字符串会自己命中自己
    # （它们本身就是本函数的字面量），断言就变成永远为真的摆设。
    _self_src = ROOT / "focus.py"
    if _self_src.exists():                 # 冻结成 exe 后源码不在旁边，跳过
        _src = _self_src.read_text(encoding="utf-8")

        # 顶层不许有重名定义 —— Python **不报错**，后面的会静默覆盖前面的。
        #
        # 实测踩到过（就在这一轮）：想加一个"读监视进程写下的面板地址"的函数，
        # 起名 `_panel_url` —— 而这个名字**早就有了**（open_page 一直在用，
        # 还带一个 path 参数）。新定义把它整个覆盖掉，于是 open_page 传参时
        # 报 TypeError，而报错信息出现在完全无关的地方（"浏览器兜底打不开
        # 页面"），要绕一圈才查得到。用 ast 数一遍只要几行。
        #
        # 只看模块顶层（tree.body）：嵌套函数同名是正常的（自检里一堆辅助函数）。
        _names = [n.name for n in ast.parse(_src).body
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                    ast.ClassDef))]
        _dupes = sorted({n for n in _names if _names.count(n) > 1})
        assert not _dupes, (
            f"顶层有重名定义：{_dupes} —— Python 不报错，后面的会**静默覆盖**"
            "前面的，症状会出现在完全无关的地方（实测踩到过）")

        _loop_src = _src[:_src.index("def selftest")]
        for _needle in (
            "vision.read_face(proc)", "vision.read_posture(proc)",
            "face_buf.append(got)", 'tilt_buf.append(pose["tilt"])',
            'scale_hist.append(got["scale"])', 'self._ear_hist.append(got["ear"])',
            'closed_since = (None if got["ear"] >= self._ear_thr',
            # 丢脸超时作废闭眼计时。少了这一条不会报错，只会让人"闭眼 2 秒、
            # 转头出画 3 秒、回来还闭着眼"时一回来就被记一条「疲劳」。
            "now - last_face_seen > FACE_LOSS_GRACE",
            "last_face_seen = now",
            # 眨眼统计的喂数据也必须在这里。掉了不会报错：眨眼率恒为 0，
            # 而那被解释成"还不知道"，于是**永远不提醒**，静默失效。
            "blinks.feed(now, got[\"ear\"], self._ear_thr",
            "rubs.feed(now, pose[\"hand_eye\"])",
        ):
            assert _needle in _loop_src, \
                f"采集循环的视觉链路断了：找不到 {_needle!r}" \
                "（生产端被删了、消费端还在，会静默只产出走神/离开）"

        # 「眼睛该歇会儿了」的触发链路。这一条和上面同一类故障，但更难发现：
        # 眨眼率照样在算、面板照样显示数字、库里照样落 blink/rub —— 只有那个
        # if 不接 on_eye_break 的时候，**永远不弹提醒**，没有任何报错。
        # 而用户要的就是那一条提醒，等于整个功能静默消失。
        #
        # 三条一起钉：触发本身、最小间隔（少了它每秒弹一次，用户会把提醒
        # 整个关掉，等于没有）、面板读的实时状态（少了它面板永远显示"还没数据"）。
        # 行为测试在 smoke_pipeline.py 的场景④ —— 那里真跑一遍采集循环，
        # 确认回调真的被调用过。
        for _needle in (
            "self._announce_eye_break(eye_run, blink_rate)",
            "now - last_eye_remind >= EYE_REMIND_EVERY",
            '_runtime["eye"] = {"run": eye_run, "blink": blink_rate',
        ):
            assert _needle in _loop_src, \
                f"「眼睛该歇会儿了」的链路断了：找不到 {_needle!r}" \
                "（眨眼率照算、面板照显示，但永远不弹提醒 —— 静默失效）"

        # 光"还在"不够 —— 那句复位必须待在"重新开始观察"那一侧，也就是
        # 读脸**之前**。放进"丢脸"那一支同样能让丢脸场景变绿，但暂停那条
        # 路径在循环开头就 continue 了，根本走不到那一支：恢复时陈旧的
        # closed_since 会直接命中，人一恢复就被记一条「疲劳」。
        # 这个顺序约束没法用"子串在不在"表达，只能比下标。
        # 对应的行为测试是 smoke_pipeline.py 的场景③。
        assert (_loop_src.index("now - last_face_seen > FACE_LOSS_GRACE")
                < _loop_src.index("vision.read_face(proc)")), \
            "闭眼计时的复位跑到读脸之后了 —— 暂停后恢复时它不会执行，" \
            "陈旧的起点会让「疲劳」在恢复的瞬间就误报"

        # "来回切窗口"这个信号必须真的接上。它和上面那段视觉链路是同一类故障：
        # 生产端（统计切换次数）被删了、消费端（decide 的 switch_rate）还在，
        # 不会有任何报错 —— switch_rate 恒为 0.0，那条判据静默失效，而自检
        # 全绿（它测的是 decide 本身，测不出"输入根本没接上"）。所以两头都钉。
        for _needle in (
            "switch_at.append(now)",
            "switch_at.popleft()",
            "switch_rate = len(switch_at)",
            "switch_rate=switch_rate)",
        ):
            assert _needle in _loop_src, \
                f"「来回切窗口」的信号断了：找不到 {_needle!r}" \
                "（统计端被删了、decide 的 switch_rate 会退回默认值 0.0，" \
                "那条判据静默失效而自检全绿）"

        # 桌面开关：绿点必须是**确认启动成功**的结果，不能是"用户按了开关"。
        #
        # 这条行为上测不到（toggle 要真的起进程、真的弹窗，还不能真的动用户的
        # 桌面），只能钉顺序：确认动作 _start_engine 必须排在写绿点之前，
        # 而且写绿点只能有一处。用户报的"待机/断网后双击开关，图标是绿的
        # 但程序没在跑"就是旧写法（先写绿点、再起进程）的直接后果。
        _toggle_src = _src[_src.index("def toggle() -> None:")
                           :_src.index("def install_shortcut")]
        assert _toggle_src.count("running=True") == 1, \
            "toggle() 里有多处写绿点 —— 绿点只该出现在确认启动成功那一处"
        assert _toggle_src.index("_start_engine") < _toggle_src.index("running=True"), \
            "toggle() 在确认进程起来之前就写了绿点 —— 正是用户报的那个 bug"
        assert _toggle_src.count("sync_shortcut_icon") >= 3, \
            "toggle() 的三条出路（关掉 / 起成功 / 起失败）都要同步图标"

        # WAL 迁移必须在**采集线程起来之前**，也必须在面板开始接请求之前。
        #
        # 这条只能钉顺序，行为测不到（main 要起托盘和窗口）。理由：切
        # journal_mode 需要排他锁，而 busy_timeout 在这条 PRAGMA 上不起作用；
        # 采集线程攒批提交会连续持有写锁十几秒，那时候再切必然失败，库就留在
        # "读写互斥"的 delete 模式 —— 用户看到的正是「自评分打不开：
        # database is locked」。启动时没有竞争，一次就成。
        #
        # 用 rindex 而不是 index：**本函数里就写着 "def main() -> None:" 这个
        # 字面量**，用 index 会命中自己，切出一段从 selftest 中间开始的源码 ——
        # 那样下面几条断言会互相命中，全部变成永远为真的摆设（这个坑上面的
        # 注释警告过一次，我还是先踩了一遍）。真正的定义在 selftest 之后。
        #
        # 后来又踩了第二次，而且更隐蔽：新加一条"main 里必须有 X"的断言时
        # 顺手又写了一遍 `_main_src = _src[_src.index(...):]`，把这里正确的
        # 那份**覆盖掉**了 —— 于是那条新断言自己命中自己的字面量，恒为真。
        # 变异测试当场抓到了它（"这条断言是摆设"）。所以：**这段切片只能有
        # 一份，下面所有用到 _main_src 的断言都复用它，别再自己切一次。**
        _main_src = _src[_src.rindex("def main() -> None:"):]
        assert "def selftest" not in _main_src, \
            "切片起点取错了（命中了自检里的字面量），下面的顺序断言会全部失效"
        assert "ensure_wal()" in _main_src, \
            "main() 里没调用 ensure_wal() —— 库可能一直留在 delete 模式"
        assert (_main_src.index("ensure_wal()")
                < _main_src.index("mon.start()")), \
            "ensure_wal() 跑到采集线程后面了 —— 那时切不动 WAL"
        assert (_main_src.index("ensure_wal()")
                < _main_src.index("serve_background")), \
            "ensure_wal() 跑到面板之后了 —— 面板随时可能开始接请求并撞锁"
        # 而且必须在 first_run 之后：ensure_wal 会把 focus.db 建出来，
        # 放前面的话"首次运行"永远判不出来，新手第一次启动看不到面板。
        assert (_main_src.index("first_run = not DB_PATH.exists()")
                < _main_src.index("ensure_wal()")), \
            "ensure_wal() 跑到 first_run 之前了 —— 首次运行的面板不会再弹"

        # 监视进程必须把"显示面板"的实现注册给面板服务。漏了这一步，
        # --wait-open 子进程发来的 /show-panel 会回 ok 但什么都不做 ——
        # 用户双击开关后窗口永远不出现，而日志里一切正常（子进程那边看到的
        # 是 200）。这属于"两个进程之间的接线"，要起真进程才测得到。
        #
        # 注意这里**复用上面的 _main_src**，不再自己切一次 —— 见上面那段
        # 注释：自己切过一次，结果断言恒为真，被变异测试抓出来了。
        assert "dashboard.set_panel_shower(show_panel)" in _main_src, \
            "监视进程没把「显示面板」注册给面板服务 —— 子进程发来的 " \
            "/show-panel 会回 ok 但什么都不做，窗口永远不出现"

        # 面板的**实际地址**必须落盘给子进程读（见 URL_PATH / _panel_url）。
        #
        # 端口在 8787 被占用时会顺延到 8788…8806，而子进程是另一个进程、
        # 拿不到 dashboard.base_url()。不写这个文件，子进程就只能猜端口，
        # 顺延过的那次就会一直敲一个没人监听的地址、等满 120 秒然后放弃 ——
        # 用户看到的是"双击了没反应"，日志里只有一句"等面板就绪超时"。
        assert "URL_PATH.write_text(_panel" in _main_src, \
            "main() 没把面板的实际地址落盘 —— --wait-open 子进程只能猜端口，" \
            "端口顺延过（8787 被占用）时它会一直等一个没人监听的地址"
        assert (_main_src.index("serve_background")
                < _main_src.index("URL_PATH.write_text(_panel")), \
            "地址是在 serve_background 之前写的 —— 那时还不知道实际端口，写的是错的"

        # 采集线程的退出清理必须在 **finally** 里。
        #
        # 这条也是行为测不到的（要构造"采集循环意外死掉"才能验证），只能钉结构。
        # 但它对应的事故很实在：用户日志里 Thread-2 死在 `conn.commit()` 上，
        # 连接没关、写事务还开着，**接下来 1 小时 49 分每一条读都撞同一个锁**
        # —— 面板、报告、自评分全部 500，而进程还活着、桌面图标也还在，
        # 看起来"在跑"。清理一旦不在 finally 里，任何没接住的异常（摄像头 read、
        # cv2.resize、重开摄像头都在循环体的 try 之外）都能重现这个状态。
        _run_src = _src[_src.index("    def run(self) -> None:")
                        :_src.index("    def stop(self) -> None:")]
        assert "        try:\n            while self.running:" in _run_src, \
            "采集循环没被 try 包住 —— 线程意外死掉时 finally 不会执行"
        assert "        finally:\n" in _run_src, \
            "采集循环没有 finally —— 写连接不会关，库会被锁死"
        _fin = _run_src.index("        finally:\n")
        assert "conn.close()" in _run_src[_fin:], \
            "finally 里没有 conn.close() —— 库会被锁死"
        assert "cap.release()" in _run_src[_fin:], \
            "finally 里没有释放摄像头 —— 指示灯会一直亮着"

        # 窗口层加载不了**不能把整个进程带走**。
        #
        # 这条行为上很难测（要真的把 pywebview 从环境里拿掉），只能钉结构。
        # 但它对应的事故是实测的：用户的 .venv 里没有 pywebview（更新前那版
        # 用浏览器开面板，不需要这个依赖），而 run_tray 最后一步是裸的
        # `import window; window.start_main()` —— 异常穿到 main() 的
        # `except BaseException` 再 raise，**进程启动两秒后就没了**，
        # pythonw 还没有 stderr，用户只看到托盘图标闪一下。
        # 采集和托盘都不依赖窗口层，凭什么陪葬。
        _tray_src = _src[_src.index("def run_tray(mon: Monitor) -> None:")
                         :_src.index("# ══════════════════════ 开 / 关 / 桌面开关")]
        assert "    try:\n        import window\n" in _tray_src, \
            "run_tray 裸 import window —— 窗口层一坏，整个程序跟着退出"
        assert "_quit.wait()" in _tray_src, \
            "没有窗口层时主线程必须守着（_quit.wait()）—— 直接返回的话 main() " \
            "会走到结尾，而托盘和采集都是 daemon 线程，会被一起拔掉"
        # 托盘菜单不许再各写各的 `import window`：窗口层一坏，那些菜单项
        # 全变成"点了没反应"，而它们是用户唯一的入口（发布包没有控制台）。
        assert "window.open_page" not in _tray_src, \
            "托盘菜单绕过了 open_page() —— 窗口层坏掉时那些菜单项会失灵"
        assert _tray_src.count("open_page(") >= 4, \
            "托盘里打开页面的入口都要走 open_page()（面板/评分/报告/失败提示）"

        # 到点提醒打分的气泡要受 RATE_REMIND 管。
        #
        # 行为测不到（要起真托盘，还要等一个 30 分钟的块走完），只能钉结构 ——
        # 但它对应的是"用户嫌吵"这件事：门一旦丢了，设置页那个开关还看得见、
        # 还勾得上，勾了没用，气泡照弹。**这正是这个项目反复踩的那一类**。
        #
        # 断言写成"门 + return"**一整段**，而不是只找 `if not RATE_REMIND:`
        # 那半句：只找半句的话，把 `return` 换成 `pass`（门成了摆设、
        # 气泡照弹）照样是绿的。下面两条变异分别打这两半。
        #
        # 切片要**只包 on_block_end 这一支**：在整个 `_tray_src` 里找的话，
        # 把门挪到 on_eye_break（管错地方了）也照样绿。
        _obe_src = _tray_src[_tray_src.index("def on_block_end("):
                             _tray_src.index("mon.on_block_end = on_block_end")]
        assert "if not RATE_REMIND:\n            return" in _obe_src, (
            "托盘里的「到点提醒打分」没接 RATE_REMIND，或者门是空的（没有 return）"
            " —— 设置页那个开关会变成勾了没用的装饰，用户关不掉那个气泡")

        # 「眼睛该歇会儿了」的气泡同理 —— 同一类毛病，另一条气泡。
        # 切片同样只包 on_eye_break 这一支：两条门要是**互换了位置**
        # （眼睛的门写进 on_block_end、打分的门写进 on_eye_break），
        # 上面那条断言和这条会各红一个，跑不掉。
        _oeb_src = _tray_src[_tray_src.index("def on_eye_break("):
                             _tray_src.index("mon.on_eye_break = on_eye_break")]
        assert "if not EYE_BREAK_REMIND:\n            return" in _oeb_src, (
            "托盘里的「眼睛该歇会儿了」没接 EYE_BREAK_REMIND，或者门是空的"
            "（没有 return）—— 设置页那个开关会变成勾了没用的装饰，"
            "用户关不掉那个气泡")

        # 面板两条 500 分支都得既补日志配置、又记栈。
        #
        # 行为测试只打得到 GET 那条（POST 要 CSRF token、要真提交表单），
        # 所以 POST 这条只能钉结构 —— 别让下一个人只改了 GET 就以为完事。
        #
        # 这里**不能数总数**：`_ensure_log()` 现在有三处（403 拒绝路径也补了
        # 一次），数 `>= 2` 的话，删掉一条 500 分支的调用还剩两处，断言照样绿
        # ——变异测试抓出来的就是这个。改成钉住"每个 500 分支的 log.exception
        # 往上第一个非注释行必须是 _ensure_log()"，位置错了就红。
        _dpath = _self_src.parent / "dashboard.py"
        if _dpath.exists():
            _dtxt = _dpath.read_text(encoding="utf-8")
            _dlines = _dtxt.splitlines()
            _branches = 0
            for _i, _ln in enumerate(_dlines):
                if not _ln.strip().startswith('log.exception("面板处理 '):
                    continue
                _branches += 1
                for _j in range(_i - 1, -1, -1):
                    _up = _dlines[_j].strip()
                    if _up and not _up.startswith("#"):
                        assert _up == "_ensure_log()", (
                            f"面板的 500 分支（{_ln.strip()[:44]}…）往上第一个"
                            f"非注释行是 {_up!r}，不是 _ensure_log() —— "
                            "单独跑面板时栈会直接消失")
                        break
            assert _branches >= 2, \
                f"面板只有 {_branches} 条 500 分支的 log.exception —— " \
                "GET 和 POST 都得记栈，否则页面会把原因吞掉"
            assert _dtxt.count("log.exception(") >= 2, \
                "面板的 500 分支没记栈 —— 页面会把原因吞掉，日志里查不到"

    # 版本号有两处副本，必须一致 —— __version__ 正上方那行注释就是这么写的。
    # 但注释看得见、没人会去看：上游 aaa2c0a「v0.2.3: 版本号跟进」就只改了
    # pyproject.toml，漏掉 __version__，于是 `focus.py --version` 报 0.2.2、
    # 而元数据是 0.2.3 —— 打包成 exe 后 bug 报告唯一能问到的版本信息就是这一行，
    # 用户报的版本和实际跑的代码对不上，排查从错误前提开始。
    # 所以这里加一条断言，把"必须一致"从注释变成会失败的检查。
    # 冻结成 exe 后 pyproject.toml 不会跟着打包，找不到就跳过（避免误报）。
    pyproject = ROOT / "pyproject.toml"
    if pyproject.exists():
        try:
            with pyproject.open("rb") as f:
                meta_version = tomllib.load(f)["project"]["version"]
        except tomllib.TOMLDecodeError as exc:
            # 上游真出过这个：aaa2c0a 把没解决的冲突标记一起提交了，
            # pyproject.toml 根本不是合法 TOML（pip install . 直接失败）。
            # 而发布脚本当时用正则读版本号，恰好还能读到数字，
            # 于是"打包能出包"的假象让这个问题一直活了下来。
            raise AssertionError(
                f"pyproject.toml 不是合法 TOML：{exc}"
                "（先看有没有未解决的冲突标记 <<<<<<< / ======= / >>>>>>>）"
            ) from None
        assert meta_version == __version__, \
            f"版本号不一致：pyproject.toml={meta_version}，focus.py={__version__}"

        # 第三处副本：uv.lock。这条是**实测撞出来的**，不是假想 ——
        # 仓库里提交的 uv.lock 一直写着 `focus-monitor version = "0.2.1"`，
        # 而 pyproject.toml 早就是 0.2.3 了（0.2.3 那次只改了 pyproject，
        # 没人记得 lock 里也有一份）。后果很实在：
        #   - `uv sync --extra build` 要先重新 lock 才知道 build extra 存在，
        #     发布脚本提示的那条命令在离线/CI 下会失败；
        #   - lock 里没有 pyinstaller 那一串（altgraph/macholib/pefile…）。
        # 锁文件是**生成物**，但既然提交进仓库了，就该有人检查它跟 pyproject
        # 对不对得上 —— 否则它只是一个会腐烂的副本。
        # 冻结成 exe 后 uv.lock 不会跟着打包，找不到就跳过。
        uvlock = ROOT / "uv.lock"
        if uvlock.exists():
            # 不用正则，省得为这一处引入 import re —— 锁文件这个片段是固定形状的
            _txt = uvlock.read_text(encoding="utf-8")
            _key = 'name = "focus-monitor"\nversion = "'
            _i = _txt.find(_key)
            assert _i >= 0, \
                "uv.lock 里找不到 focus-monitor 的版本号（锁文件格式变了？）"
            _lock_ver = _txt[_i + len(_key):].split('"', 1)[0]
            assert _lock_ver == meta_version, \
                (f"uv.lock 里的版本号是 {_lock_ver}，pyproject.toml 是 "
                 f"{meta_version} —— 改了版本号记得跑一次 `uv lock`")

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
    #
    # 但它**受 AUTO_OPEN_PANEL 管**（默认关）。不这么做的话，那个开关就是个
    # 半真的开关：写着"不自动弹"，删掉 focus.db 或换台机器就冷不丁弹一个。
    # 首次运行的"像没反应"另有托盘气泡兜着（run_tray 的 on_progress），
    # 所以关掉它并不会让新手对着空气发呆。
    first_run = not DB_PATH.exists()

    # 把库切到 WAL —— **必须在采集线程起来之前**，也必须在面板开始接请求之前。
    # 这时没有别的连接持锁，一次就成；一旦晚了，采集线程攒批提交会连续持有
    # 写锁十几秒，切模式基本切不动，库会一直留在"读写互斥"的 delete 模式，
    # 面板和报告就会报 database is locked。
    #
    # 位置也讲究：必须放在 first_run 之后 —— ensure_wal 会把 focus.db 建出来，
    # 放在前面的话"首次运行"就永远判不出来，新手第一次启动看不到面板。
    ensure_wal()

    # 面板随监视一起起，这样桌面快捷方式随时点得开，不用先去托盘菜单。
    # 只监听 127.0.0.1；起不来也不影响采集，所以异常只记日志。
    try:
        import dashboard
        # 把"显示面板"的实现交给面板服务。桌面开关 fork 出来的 --wait-open
        # 子进程碰不到本进程的窗口，只能发个请求过来让**我们**自己 show()
        # —— 见 wait_and_open 的注释：不这么做，它每次都弹一个浏览器网页。
        dashboard.set_panel_shower(show_panel)
        _panel = dashboard.serve_background(open_browser=False)
        log.info("实时面板: %s", _panel)
        # 把**实际**地址落盘给 --wait-open 子进程用（见 URL_PATH 的注释）。
        # 端口可能不是默认那个，所以必须写实际值，不能让它去猜。
        try:
            URL_PATH.write_text(_panel, encoding="utf-8")
        except OSError:
            log.exception("写面板地址失败 —— 桌面开关可能等不到面板"
                          "（端口顺延过时尤其会）")
    except Exception:
        log.exception("实时面板启动失败，监视继续")

    mon = Monitor(camera=args.camera)
    mon.start()

    if first_run and AUTO_OPEN_PANEL:
        # 模型要下十几秒、窗口层又要等主线程的 GUI 循环起来，所以丢到子线程，
        # 别卡住托盘启动。show_panel 自己会等窗口对象建出来（等不到就降级到
        # 系统浏览器，见该函数）。
        def _show_first_run() -> None:
            time.sleep(3.0)
            try:
                show_panel()
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
        # 地址也要清掉：端口是这次运行实际用的那个，留着下次的桌面开关
        # 会先读到旧端口（多轮之后才收敛，白等）。
        URL_PATH.unlink(missing_ok=True)
        # 退出兜底：进程都走了，桌面开关不能还挂着绿点。走的是同一条
        # 观测逻辑（running=False → 灰杠），所以退出路径不止这一条也没关系。
        sync_shortcut_icon(force=True, running=False)
        log.info("专注监视已退出")
    print("已退出，运行 `uv run focus.py --report` 看报告")


if __name__ == "__main__":
    main()
