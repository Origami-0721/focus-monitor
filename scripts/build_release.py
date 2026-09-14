# -*- coding: utf-8 -*-
"""构建发布包 (Windows)：
onedir 传统布局 + 剔除 matplotlib 全家 + 打补丁剔除 mediapipe 鼠标绘图硬依赖。

输出:
  dist/focus-monitor/           # 软件文件夹(发布 zip 的根目录第一层就是它)
      focus-monitor.exe         # 双击即用
      <运行时自动生成的 focus.db / report.html / focus.log 都在这层>

用法: .venv\\Scripts\\python.exe scripts\\build_release.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv"
SITE = VENV / "Lib" / "site-packages"
PYINST = VENV / "Scripts" / "pyinstaller.exe"

if not PYINST.exists():
    sys.exit("pyinstaller not found; run: uv sync --extra build 或 pip install pyinstaller")

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
    import re
    path = SITE / "mediapipe" / "tasks" / "python" / "vision" / "drawing_utils.py"
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
    text = text.replace(OLD, NEW)
    path.write_text(text, encoding="utf-8")
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
        sys.exit(f"pyinstaller failed ({r.returncode})")

def clean_dist() -> None:
    shutil.rmtree(ROOT / "dist" / "focus-monitor", ignore_errors=True)

if __name__ == "__main__":
    patch_drawing_utils()
    clean_dist()
    build()
    print("[done] dist/focus-monitor/focus-monitor.exe 就位")