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

# (说明, 原片段, 改坏后的片段[, 目标文件])
# 原片段必须在源码里**只出现一次**，否则拒绝变异（改错了地方会给出假结果）。
# 目标文件默认为 focus.py；要变异别的文件就补第 4 项（相对工程根）。
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
    (
        "_switch_wal 无条件返回 True（假定切成功，其实库还在 delete 模式）",
        '    return journal_mode(conn) == "wal"\n',
        "    return True\n",
    ),
    (
        "open_db 不再验证/切换 WAL（老代码的形态：裸 connect）",
        '    if journal_mode(conn) != "wal":\n',
        "    if False:\n",
    ),
    (
        "main() 不再做启动时的 WAL 迁移（采集起来后切不动）",
        "    ensure_wal()\n",
        "",
    ),
    (
        "采集循环的清理不在 finally 里（线程一死就把库锁死）",
        "        finally:\n            # 无论怎么退出都要**关掉写连接**",
        "        except BaseException:\n"
        "            raise\n"
        "        if True:\n"
        "            # 无论怎么退出都要**关掉写连接**",
    ),
    (
        "面板的 500 用 log.error 代替 log.exception（记了日志，但没栈）",
        '            log.exception("面板处理 GET %s 出错", raw_path)\n',
        '            log.error("面板处理 GET %s 出错", raw_path)\n',
        "dashboard.py",
    ),
    (
        "面板的 500 分支忘了补日志配置（单独跑面板时栈直接消失）",
        "        except Exception as exc:                       # 单次请求出错不该带崩服务\n"
        "            _ensure_log()\n",
        "        except Exception as exc:                       # 单次请求出错不该带崩服务\n",
        "dashboard.py",
    ),
    (
        "面板 POST 的 500 分支不记栈（只改了 GET 那条）",
        '            log.exception("面板处理 POST %s 出错", path)\n',
        "",
        "dashboard.py",
    ),
    (
        "run_tray 裸 import window（窗口层一坏，整个程序跟着退出）",
        '    try:\n'
        '        import window\n'
        '    except Exception:\n'
        '        log.exception("窗口层加载不了，面板/报告将改用系统浏览器；"\n'
        '                      "记录与托盘继续运行")\n'
        '        window = None\n',
        '    import window\n',
    ),
    (
        "没有窗口层时主线程直接返回（daemon 的托盘和采集被一起拔掉）",
        '        _quit.wait()\n        return\n',
        '        return\n',
    ),
    (
        "open_page 不再退到系统浏览器（窗口层坏了就「点了没反应」）",
        '    try:\n'
        '        webbrowser.open(url)\n'
        '        log.info("已改用系统浏览器打开：%s", url)\n'
        '    except Exception:\n'
        '        log.exception("系统浏览器也打不开：%s", url)\n',
        '    if False:\n'
        '        webbrowser.open(url)\n',
    ),
    (
        "托盘菜单绕过 open_page 自己 import window（窗口层坏掉就失灵）",
        '    def on_rate(icon, _item):\n        open_page("rate")\n',
        '    def on_rate(icon, _item):\n'
        '        import window\n'
        '        window.open_page("rate")\n',
    ),
    (
        "open_page 把页面路由丢了（评分/报告全落到首页）",
        '_PAGE_PATHS = {"panel": "/", "rate": "/rate", "report": "/report"}\n',
        '_PAGE_PATHS = {}\n',
    ),
    (
        "uv.lock 的版本号没跟上 pyproject（真实的旧状态：0.2.1 vs 0.2.3）",
        'name = "focus-monitor"\nversion = "0.2.3"\n',
        'name = "focus-monitor"\nversion = "0.2.1"\n',
        "uv.lock",
    ),
    (
        "把 Origin: null 当跨源拒（真实的旧状态：窗口里所有表单都 403）",
        '    if origin.strip().lower() == "null":     # 不透明来源，见上\n'
        '        return True\n',
        '    if False:                                # 变异：null 当跨源拒\n'
        '        return True\n',
        "dashboard.py",
    ),
    (
        "403 的现场只记 debug（等于没记：用户还是只能给一张截图）",
        '        log.warning(\n'
        '            "拒绝 %s %s：%s（Host=%r Origin/Referer=%r Sec-Fetch-Site=%r）",\n',
        '        log.debug(\n'
        '            "拒绝 %s %s：%s（Host=%r Origin/Referer=%r Sec-Fetch-Site=%r）",\n',
        "dashboard.py",
    ),
    (
        "低相关横幅退回「请先按「校准」一节调阈值」（指到不存在的地方）",
        '            f"要调阈值就去 <b>{_FIX_PATH}</b>（顺序：先 EAR、再姿态角），"\n',
        '            "请先按「校准」一节调阈值。"\n',
        "report.py",
    ),
    (
        "面板不把导航传给报告页（用户读到「去设置调阈值」却无处可点）",
        "build_html(load(), nav_html=_NAV)",
        "build_html(load())",
        "dashboard.py",
    ),
    (
        "低相关 + 样本不足时不再劝住用户（直接推着他去动阈值）",
        '                + ("；但样本量偏少，先别急着动阈值，再攒一些评分看看"\n',
        '                + ("，阈值需要重新调"\n',
        "ratings.py",
    ),
    (
        "结论里退回「阈值需要重新校准」（又把人指去找一个不存在的功能）",
        '                   if half > 0.25 else f"，阈值需要重新调{_WHERE}"))\n',
        '                   if half > 0.25 else "，阈值需要重新校准"))\n',
        "ratings.py",
    ),
    (
        "「中等相关」那档也用上「校准」这个不存在的功能名",
        '        return f"中等相关 —— 大方向对得上，但阈值还有调的空间{loose}"\n',
        '        return f"中等相关 —— 大方向对得上，但阈值需要重新校准{loose}"\n',
        "ratings.py",
    ),
    (
        "面板上退回「校准中 N/60」（又让人去找一个不存在的校准按钮）",
        "        base_note = f'睁眼基线学习中 {len(ears)}/{focus.EAR_MIN_SAMPLES}'\n",
        "        base_note = f'校准中 {len(ears)}/{focus.EAR_MIN_SAMPLES}'\n",
        "dashboard.py",
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
    for mut in MUTATIONS:
        name, old, new = mut[0], mut[1], mut[2]
        rel = mut[3] if len(mut) > 3 else "focus.py"
        with tempfile.TemporaryDirectory() as td:
            tree = copy_project(Path(td) / "m")
            f = tree / rel
            src = f.read_text(encoding="utf-8")
            if src.count(old) != 1:
                print(f"?? {name}\n     在 {rel} 里出现 {src.count(old)} 次，无法变异")
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
