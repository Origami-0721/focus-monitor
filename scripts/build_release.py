# -*- coding: utf-8 -*-
"""构建发布包 (Windows)：
onedir 传统布局 + 剔除 matplotlib 全家 + 打补丁剔除 mediapipe 鼠标绘图硬依赖。

输出:
  dist/focus-monitor/                    # 软件文件夹(直接可跑)
      focus-monitor.exe                  # 双击即用
      <运行时自动生成的 focus.db / report.html / focus.log 都在这层>
  dist/focus-monitor-<版本>-win64.zip    # 真正发给用户的那个文件
  dist/focus-monitor-<版本>-win64.zip.sha256

用法: .venv\\Scripts\\python.exe scripts\\build_release.py

为什么 onedir 而不是 onefile：onefile 每次启动都要把上百 MB 解压到临时目录，
MediaPipe 模型加载本来就慢，叠加起来冷启动要几十秒，用户会以为程序卡死了。
onedir 快一个量级，代价只是多打一个 zip —— 由本脚本自动完成。
"""

import hashlib
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv"
SITE = VENV / "Lib" / "site-packages"
PYINST = VENV / "Scripts" / "pyinstaller.exe"
DIST = ROOT / "dist"
APP = DIST / "focus-monitor"

if not PYINST.exists():
    sys.exit("找不到 pyinstaller。先执行：uv sync --extra build，"
             "或者 pip install pyinstaller")


def version() -> str:
    """从 pyproject.toml 读版本号 —— 单一事实来源。

    不额外维护一份版本常量：手写两处迟早会不一致，
    发出来的包名和 exe 里报的版本对不上，排查问题时会很痛苦。
    """
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.M)
    if not m:
        sys.exit("pyproject.toml 里找不到 version")
    return m.group(1)


OLD = "import matplotlib.pyplot as plt\n"
NEW = ("try:\n"
       "    import matplotlib.pyplot as plt\n"
       "except ImportError:\n"
       "    plt = None\n")


def patch_drawing_utils() -> None:
    """mediapipe 的绘图工具顶层 import matplotlib —— 我们从不调用它的绘图函数。
    打成 try/except 后，打包时就能安全把 matplotlib 全家 exclude 掉。

    幂等：先把历史遗留的旧补丁痕迹（嵌套 try/except/plt=None 行）清掉，
    再打一遍；重复执行多少次结果都一样，不会叠出 IndentationError。
    """
    path = SITE / "mediapipe" / "tasks" / "python" / "vision" / "drawing_utils.py"
    if not path.exists():
        print(f"[patch] {path} 不存在, 跳过")
        return
    text = path.read_text(encoding="utf-8")
    # 1) 清理旧补丁痕迹（可能多层嵌套），回到未补丁形态
    text = re.sub(r"^\s*try:\s*$|^\s*except ImportError:\s*$|^\s*plt = None\s*$",
                  "", text, flags=re.M)
    text = re.sub(r"^ {4}(import matplotlib\.pyplot as plt)$", r"\1", text, flags=re.M)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 2) 打补丁
    if OLD not in text:
        print(f"[patch] {path.name}: matplotlib import 未找到, 跳过")
        return
    path.write_text(text.replace(OLD, NEW), encoding="utf-8")
    print(f"[patch] {path.name}: matplotlib 顶格 import 已降级为 try/except")


def build() -> None:
    args = [
        str(PYINST),
        "--clean", "--noconfirm",
        "--onedir", "--windowed",
        "--name", "focus-monitor",
        "--icon", str(ROOT / "icons" / "on.ico"),
        "--add-data", f"{ROOT / 'icons'};icons",
        # mediapipe 官方没有 hook, collect-all 补足 (剔除前先打过补丁)
        "--collect-all", "mediapipe",
        # matplotlib 全家: 无人使用, 纯打包负担 (mediapipe 绘图工具只会在被调用时才需要)
        "--exclude-module", "matplotlib",
        "--exclude-module", "mpl_toolkits",
        "--exclude-module", "fontTools",
        "--exclude-module", "kiwisolver",
        "--exclude-module", "contourpy",
        "--exclude-module", "cycler",
        "--exclude-module", "dateutil",
        "--exclude-module", "python-dateutil",
        str(ROOT / "focus.py"),
    ]
    r = subprocess.run(args, cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"pyinstaller 打包失败（退出码 {r.returncode}）")


def clean_dist() -> None:
    shutil.rmtree(APP, ignore_errors=True)


def make_zip(ver: str) -> Path:
    """打成 zip，并顺手确认里面没有任何本机数据。

    第一层目录名就是 focus-monitor/ —— 用户解压后直接进文件夹双击 exe，
    不会散落一桌文件。
    """
    # 绝不能进发布包的：本机隐私数据 + 运行时产物。
    # 构建目录不该有这些，但万一上次本地调试留下的没清掉就会被打进去。
    BAD_NAMES = {"focus.db", "focus.db-wal", "focus.db-shm", "focus.log",
                 "report.html"}
    zip_path = DIST / f"focus-monitor-{ver}-win64.zip"
    zip_path.unlink(missing_ok=True)

    n = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(APP.rglob("*")):
            if p.is_dir():
                continue
            name = p.name.lower()
            if name in BAD_NAMES or name.endswith(".csv"):
                sys.exit(f"发现本机数据混进构建目录，中止打包: {p}")
            z.write(p, Path("focus-monitor") / p.relative_to(APP))
            n += 1

    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    (zip_path.with_suffix(".zip.sha256")).write_text(
        f"{digest}  {zip_path.name}\n", encoding="utf-8")
    print(f"[zip] {zip_path.name}: {n} 个文件, "
          f"{zip_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"[sha256] {digest}")
    return zip_path


if __name__ == "__main__":
    ver = version()
    print(f"[build] focus-monitor {ver}")
    patch_drawing_utils()
    clean_dist()
    build()
    z = make_zip(ver)
    print(f"[done] dist/focus-monitor/focus-monitor.exe 就位")
    print(f"[done] 发给用户的是 {z.name}（把 .sha256 一起贴上）")
