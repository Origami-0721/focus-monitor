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
import tomllib
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

    用 tomllib 而不是正则：正则只关心"有没有 `version = "` 这一行"，
    文件里混进未解决的冲突标记（`<<<<<<<` / `>>>>>>>`）它照样能读出数字 ——
    实测上游真发生过一次，pyproject.toml 根本不是合法 TOML，
    而发布脚本一声不吭地出了包，直到 `pip install .` 才炸。
    解析器会直接报错，正则不会。
    """
    path = ROOT / "pyproject.toml"
    try:
        with path.open("rb") as f:
            return tomllib.load(f)["project"]["version"]
    except FileNotFoundError:
        sys.exit(f"找不到 {path}")
    except tomllib.TOMLDecodeError as exc:
        sys.exit(f"{path.name} 不是合法 TOML：{exc}\n"
                 "（先看有没有未解决的冲突标记 <<<<<<< / ======= / >>>>>>>）")
    except KeyError:
        sys.exit(f"{path.name} 里找不到 project.version")


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
        # pywebview 的 js 桥接 + WebView2 的 DLL 必须显式收进来。
        #
        # **包名是 `webview`，不是 `pywebview`。** `pywebview` 是发行包名
        # （PyPI 上那个、`pyproject.toml` 里写那个），导入名才叫 `webview`
        # —— 磁盘上就是 `site-packages/webview/`。
        #
        # 这里原来写的是 `--collect-all pywebview`：PyInstaller 找不到那个模块，
        # 只在构建输出里打一行
        #     WARNING: collect_data_files - skipping data collection for module
        #              'pywebview' as it is not a package.
        # 然后**什么都不收**（实测 `collect_all('pywebview')` → datas=0、
        # binaries=0、hidden=0），而构建照样"成功"。也就是说这道防线
        # 从来没生效过，而它的注释还写着"漏了会表现为窗口一片空白"。
        #
        # 当时为什么没出事：pywebview **自带**一个 hook
        # （`webview/__pyinstaller/hook-webview.py`，靠 dist-info 里的
        # `[pyinstaller40] hook-dirs` 入口点被发现），它收的正好是
        # `webview/js` 和 `webview/lib` —— 一直是**另一个机制**在顶。
        # 不能靠它：入口点没被注册（包是被拷进来的时候）就静默失效，
        # 症状是"窗口能创建、页面里 `pywebview.api` 是空的"= 面板一片空白，
        # 而且没有任何报错。
        #
        # 用 `--collect-data` 而不是 `--collect-all`：要的只是**数据文件**
        # （js + lib 里那几个 DLL）。平台模块靠静态分析就收得到 ——
        # `webview/guilib.py` 里那些 `import webview.platforms.xxx` 虽然写在
        # 嵌套函数里，但都是**静态** import，字节码分析扫得到；
        # 而 `--collect-all` 会把 cocoa/gtk/qt/cef/android 一起塞成
        # hiddenimports，白拉一堆根本没装的平台（构建时刷一屏 WARNING）。
        "--collect-data", "webview",
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


# 构建产物里**必须**存在这些文件。按文件名找（`rglob`），不写路径 ——
# PyInstaller 6.x 把数据文件放在 `_internal/` 下，以后还可能再变，
# 钉死路径的断言会在升级时莫名其妙地红。
_MUST_BE_BUNDLED = (
    # pywebview 的 js 桥接：页面里的 window.pywebview.api 靠它注入。
    # 缺了 = 面板能开但一片空白。
    "api.js", "customize.js", "finish.js", "state.js",
    # WebView2 的 .NET 封装（edgechromium 后端）
    "Microsoft.Web.WebView2.Core.dll",
    "Microsoft.Web.WebView2.WinForms.dll",
    # pythonnet 的运行时 —— **所有** Windows 后端（winforms / edgechromium /
    # mshtml）都要 `import clr`，缺了窗口根本建不起来。
    "Python.Runtime.dll",
)


def missing_bundle_assets(root: Path) -> list[str]:
    """`root` 里缺哪些必须的运行资源。**纯函数** —— 抽出来是为了能被单独验：
    "这个断言到底会不会红"不能靠"构建通过"来证明（构建通过只说明资源在，
    证明不了缺了它会拦）。实测拿一个空目录喂进来，7 个名字全在返回里。
    """
    got = {p.name for p in root.rglob("*") if p.is_file()}
    return [w for w in _MUST_BE_BUNDLED if w not in got]


def verify_bundle() -> None:
    """构建产物里必须真的有 pywebview / WebView2 / pythonnet 的运行资源。

    **为什么在"产物"上验，而不是相信命令行旗标。** 旗标写错了 PyInstaller
    只会打一行 WARNING，构建照样"成功"—— `--collect-all pywebview` 就是
    这么空了整整几轮（包名该是 `webview`），而它的注释还在承诺"漏了会
    表现为窗口一片空白"。产物里有没有，才是用户会不会看到一片空白的分界。

    注意这里管的是"窗口内容不对"和"窗口建不起来"。**窗口层的整体兜底**是
    另一条路：`focus.wait_and_open()` 在 `import window` 失败时会退到系统
    浏览器，并把原因写进 `focus.log` —— 所以这一条断言失败时，发布包不是
    "完全不能用"，而是"面板窗口没了、只剩浏览器"，同样得拦住。
    """
    missing = missing_bundle_assets(APP)
    if missing:
        sys.exit(
            "构建产物里缺这些运行资源：" + "、".join(missing) + "\n"
            "打出来的包会表现为「面板一片空白」（窗口能开，但页面里的\n"
            "pywebview.api 是空的），或者窗口根本建不起来 —— 两种情况都\n"
            "没有任何报错，用户只会觉得「这功能坏了」。\n"
            "先检查 build() 里的 --collect-data webview（**包名是 webview，\n"
            "不是 pywebview**），以及 pythonnet 有没有装。")
    print(f"[verify] pywebview / WebView2 / pythonnet 的资源都在产物里"
          f"（查了 {len(_MUST_BE_BUNDLED)} 个文件）")


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
    # 先验产物再打包：缺了资源就不要出 zip，否则坏包直接发出去。
    verify_bundle()
    z = make_zip(ver)
    print(f"[done] dist/focus-monitor/focus-monitor.exe 就位")
    print(f"[done] 发给用户的是 {z.name}（把 .sha256 一起贴上）")
