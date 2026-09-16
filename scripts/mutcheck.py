"""变异测试：把每条关键断言对应的实现改坏，确认自检真的会红。

**为什么值得有这个脚本。** 这个项目没有单元测试，`focus.py --selftest` 就是
全部防线。而一条**永远不会失败**的断言和一个没有断言是同一件事 —— 而且更糟：
它让人以为这块已经被覆盖了。这个坑在本项目里反复出现：

- `assert str(PYW) in _lnk`：写死路径的旧写法同样含 `PYW`，断言永远为真；
- `assert PYW.exists()`：在开发机（有 `.venv`）恒为真，到 CI 就红；
- 「绿点为什么不见了」那次，自检全绿，坏的是"根本没人喂数据给逻辑"。

所以每条断言都要有一个对应的"改坏"版本，跑一遍确认它会红。
本脚本自动做这件事：复制一份工程 → 注入一处变异 → 跑 `--selftest`
→ 必须失败。全绿就说明那条断言是摆设。

用法（要能 import cv2/numpy/pillow，所以用装了依赖的解释器跑）：

    uv run python scripts/mutcheck.py

退出码 0 = 所有变异都被抓住；1 = 有变异逃逸；2 = 基线自己就跑不过。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 复制时排除的东西分两类：
#   1. 大的 / 环境相关的（.venv、models）—— 复制它们纯属浪费；
#   2. **个人数据和它的派生物**（focus.db*、focus.log*、icons、_selftest*.db*）。
#      第二类必须排掉：在 CI 里前面的步骤已经跑过自检，目录里躺着 focus.db，
#      带进副本会让自检的数据库那几步跑在"已有历史"上 —— 那是一种谁也没
#      在测的第三种环境，本机更是永远复现不了（本机 focus.db 内容不一样）。
SKIP = (".venv", "__pycache__", ".git", "icons", "models", "build", "dist",
        "focus.db", "focus.db-*", "focus.log", "focus.log.*", "focus.pid",
        "focus-backup-*.db*", "_selftest*.db*", "report.html", "focus-*.csv")

# (说明, 原片段, 改坏后的片段)
# 原片段必须在源码里**只出现一次**，否则拒绝变异（改错了地方会给出假结果）。
MUTATIONS: list[tuple[str, str, str]] = [
    (
        "shortcut_state 忽略 running（进程没了也显示绿/黄）",
        '    if not running:\n        return "off"\n',
        '    if False:\n        return "off"\n',
    ),
    (
        "shortcut_state 忽略 stale（在跑但没数据看不出来）",
        '    return "alert" if stale else "on"\n',
        '    return "on"\n',
    ),
    (
        "should_sync_shortcut 丢掉 lnk_exists（替所有用户凭空建快捷方式）",
        "    return lnk_exists and (force or state != last_state)\n",
        "    return force or state != last_state\n",
    ),
    (
        "should_sync_shortcut 丢掉 last_state（看门狗每 30 秒白起一次 PowerShell）",
        "    return lnk_exists and (force or state != last_state)\n",
        "    return lnk_exists\n",
    ),
    (
        "ensure_icons 三个状态共用一个文件（Windows 按路径缓存图标，桌面不刷新）",
        '        p = ICON_DIR / f"{state}.ico"\n',
        '        p = ICON_DIR / "state.ico"\n',
    ),
    (
        "_wait_until_running 无条件返回 True（没起来也当起来了）",
        "        if probe() is not None:\n",
        "        if True:\n",
    ),
    (
        "_write_lnk 直接覆盖写正式文件（不是原子改名）",
        "    ) % (tmp, cmd[0], args, ROOT, icon)\n",
        "    ) % (lnk, cmd[0], args, ROOT, icon)\n",
    ),
    (
        "toggle 先写绿点、再起进程（用户报的那个 bug 本身）",
        '    if _start_engine():\n'
        '        sync_shortcut_icon(force=True, running=True)\n'
        '        print("已启动专注监视，面板就绪后会自动打开。")\n',
        '    sync_shortcut_icon(force=True, running=True)\n'
        '    if _start_engine():\n'
        '        print("已启动专注监视，面板就绪后会自动打开。")\n',
    ),
]


def copy_project(dst: Path) -> Path:
    shutil.copytree(ROOT, dst, ignore=shutil.ignore_patterns(*SKIP))
    return dst


def run_selftest(tree: Path) -> tuple[bool, str]:
    """在 tree 里跑自检。cp1252 是照抄 CI 的环境变量 —— 中文提示在西文
    代码页下抛 UnicodeEncodeError 这个坑，本机的 GBK 控制台永远复现不了。"""
    r = subprocess.run([sys.executable, "focus.py", "--selftest"],
                       cwd=str(tree), capture_output=True, text=True,
                       timeout=300,
                       env={**os.environ, "PYTHONIOENCODING": "cp1252"})
    return r.returncode == 0, (r.stdout or "") + (r.stderr or "")


def main() -> int:
    # 复用 focus 自己的控制台兜底。CI 把 PYTHONIOENCODING 钉在 cp1252 上，
    # 这个脚本要打中文 —— 不加这一步，它会**崩在最后一行**（前面的 OK 全都
    # 打出来了，光看日志根本猜不到是编码问题），而且只把这一条 CI 步骤判红，
    # 看起来像"变异测试发现了问题"，其实是它自己先炸了。
    sys.path.insert(0, str(ROOT))
    import focus
    focus.use_safe_console()

    with tempfile.TemporaryDirectory() as td:
        base = copy_project(Path(td) / "base")
        ok, out = run_selftest(base)
        if not ok:
            print("基线（未变异）就跑不过自检，先修这个：\n" + out[-3000:])
            return 2
    print(f"基线（未变异）: 自检通过，共 {len(MUTATIONS)} 条变异待验\n")

    escaped: list[str] = []
    for name, old, new in MUTATIONS:
        with tempfile.TemporaryDirectory() as td:
            tree = copy_project(Path(td) / "m")
            f = tree / "focus.py"
            src = f.read_text(encoding="utf-8")
            if src.count(old) != 1:
                print(f"?? {name}\n     原片段出现 {src.count(old)} 次，无法变异")
                escaped.append(name)
                continue
            f.write_text(src.replace(old, new), encoding="utf-8")
            ok, out = run_selftest(tree)
        if ok:
            print(f"XX {name}\n     自检仍然全绿 —— 这条断言是摆设")
            escaped.append(name)
        else:
            last = [ln for ln in out.strip().splitlines() if ln.strip()][-1:]
            print(f"OK {name}\n     -> {last[0][:140] if last else '(无输出)'}")

    print()
    if escaped:
        print(f"有 {len(escaped)} 条变异逃逸了，对应的断言需要重写：")
        for n in escaped:
            print("  -", n)
        return 1
    print(f"全部 {len(MUTATIONS)} 条变异都被断言抓住")
    return 0


if __name__ == "__main__":
    sys.exit(main())
