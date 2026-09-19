"""冒烟变异测试：确认 smoke_pipeline.py 场景①/④/⑤ 的断言真的会失败。

**为什么单独有这个脚本，而不是并进 `mutcheck.py`。**
`mutcheck.py` 跑的是 `focus.py --selftest`，它覆盖不到"采集循环里那个 if
有没有把 `on_eye_break` 接上"—— 那件事**唯一**的行为测试是
`smoke_pipeline.py` 的场景④。而一条永远不会失败的断言比没有断言更糟：
它让人以为这块已经覆盖了。

所以场景④ 的每条断言也要有一个"改坏"版本。区别只在于**代价**：
自检是纯逻辑、零点几秒，可以跑 83 遍；场景④ 要真跑一遍采集循环，
所以这里只跑相关场景、只放针对它们的几条变异。

**为什么还是进了 CI。** 它只保护 9 条变异，看着边际收益不大 —— 但
`mutcheck.py` 保护的 `--selftest` 也从来不跑采集循环，"提醒到底会不会弹"
这件事在整个项目里只有场景④ 在测。一条永远不会失败的安全网和一条永远不会
失败的断言是同一个毛病，而后者正是这个项目反复踩的坑（见 mutcheck.py 开头）。
既然对自检愿意花三分钟，对这个就该花一分半。

**场景① 是后来补的。** 它那几条断言（视觉链路真的被调用过、真的落了样本、
正对屏幕的脸判得出「专注」、ear 真的写进了库）一条变异都没有 —— 而它保护的
正是"视觉链路断了"那类**真实发生过**的事故（采集循环重构时把 read_face 和
缓冲区填充整段删了，自检全绿，因为自检测的是 decide() 本身）。
所以这个脚本现在**按变异各自指定"打哪个文件、跑哪个场景"**：每条变异自带
目标文件和场景，谁也不替谁白等。目标文件不一定等于场景所在的那个文件 ——
比如场景① 的两条变异，一条打 `smoke_pipeline.py`（假视觉转头 60 度），
另一条打 `focus.py`（结算节拍），因为要证明的断言在两边各有一半。

**第一次打脸：这个脚本本身就是被自己打脸打出来的。** 场景④ 里有个"反例"
（把阈值配成 1e9，本该一次都不弹）。写完看着很稳妥，直到拿"阈值写死成
0.4 秒"这条变异去试 —— A 段照样过（该弹还弹）、B 段照样过（第一次仍在
1.0 秒，不小于 0.5），**只有反例抓得住它**。也就是说：在造出那条变异之前，
那个反例从来没起过作用。

**第二次打脸（CI #36，3.12 那个 job）。** 同一个提交，3.11 全绿、3.12 红，
红的正是上面那条"只有反例抓得住"的第 5 条变异 —— 它**逃逸**了（XX）。
根因不在变异里，在反例自己身上：反例的时长原来是 3.0 秒，注释还写着
"够跑到第二次结算就行"。而采集循环**第一次结算的 `eye_run` 恒为 0**
（`eyes_since` 是在那一拍才被设成 `now` 的），所以"弹"至少得跑到**第二次**
结算 —— 本机节拍 1.0 秒时 3.0 秒刚好只弹 1 次、勉强抓住；CI 的 runner 稍慢，
只跑到一次结算，就一次都不弹了。实测弹的次数与时长是 `fired ≈ duration - 2`：
1.0/1.5/2.0 秒 → 0 次，2.5 秒起才有 1 次。现在反例跑 6.0 秒（本机 4~5 次）。

**教训一：反例的"安静时长"不能抠到只剩一次余量。** 它越"该安静"，
就越容易在慢机器上退化成"什么都不发生"—— 于是它守的那条变异悄悄逃逸，
而日志上看起来一切正常（这里至少还能靠"只在一边红"察觉）。
改这里的时间常量时，先问一句：**慢一倍还够吗？**

**第三次打脸（CI #40，还是 3.12 那个 job）—— 这次是闸门自己踩了产品代码的坑。**
当时场景① 那条变异是"采集窗口砍到 0.5 秒"，本该**立刻**断言失败（只落 0 条样本），
却把子进程挂满 300 秒超时。根因不在变异身上：`Monitor.run()` 是做完初始化
才 `self.running = True`，而 `stop()` 只置 `running = False` —— 停止请求
落在初始化窗口里就会被**覆盖掉**，`while self.running` 永远为真。
0.5 秒比初始化还短，正好落进那个窗口；3.11 的 runner 快一点、初始化先跑完，
所以又只有一边红。修法在产品侧（`focus.py` 加 `_stop_requested`），
同时补了场景⑤ 和它自己的两条变异。

顺带查出**那条变异本身的标签是假的**：它写着"样本数断言必须红"，实测被抓住的
却是 `采集循环从没调用过 read_posture` —— `POSE_FPS = 2`，第一次姿态采样正好在
t≈0.5，和 0.5 秒的停表撞在同一刻，谁先跑赢是掷硬币。也就是说 `len(rows) >= 2`
从来没被验证过。现在换成"把结算节拍拉长到 60 秒"（打机制，不碰时序常量），
那条断言才真的有了牙。

**教训二：子进程超时不是一种结论，是"没法判断"。** 把它当成"逃逸了"或
"抓住了"都会得出相反的结论；让它变成一条未处理的栈，则只会在 CI 上留下
"这一步超时了"，连哪个场景都不知道（#40 就是查了半天才定位到场景①）。
现在超时转成带场景名的 `ScenarioHang`，单独算一类（`!!`）。
另外，**测试脚手架也会踩产品代码的坑**：这里的 `_run` 把采集循环跑在调用
线程上，同一个缺陷在产品上只是"退出卡 3 秒"，在这里就是无限挂死。

用法（要能 import numpy/cv2，所以用装了依赖的解释器跑）：

    uv run python scripts/smoke_mutcheck.py

退出码 0 = 所有变异都被对应场景的断言抓住；1 = 有变异逃逸 / 不合法 / 挂死；
2 = 基线自己就跑不过。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 与 mutcheck.py 保持一致：个人数据、环境、大的东西都不复制。
SKIP = (".venv", "__pycache__", ".git", "icons", "models", "build", "dist",
        "focus.db", "focus.db-*", "focus.log", "focus.log.*", "focus.pid",
        "focus.url", "focus-backup-*.db*", "_selftest*.db*", "report.html",
        "focus-*.csv", "config.json")

# (说明, 原片段, 改坏后的片段, 目标文件, 跑哪个场景)
# 原片段必须在**目标文件**里**只出现一次**，否则拒绝变异。
#
# 为什么多了"目标文件 / 场景"两列：这个脚本一开始只管场景④，所以每条变异都
# 打在 focus.py 上、都跑同一个场景。但**场景① 的断言一条变异都没有** ——
# 而它保护的正是"视觉链路断了"那类真实事故（采集循环重构时把 read_face 和
# 缓冲区填充整段删了，自检全绿，因为自检测的是 decide() 本身）。
# 在这个项目里，没被变异验证过的断言和永远不会失败的断言是同等待遇。
#
# 场景这一列**别图省事统一填 ④**：打场景① 的变异跟跑场景④ 是白等二十秒，
# 还会把"到底哪条断言抓的"搅浑。
MUTATIONS: list[tuple[str, str, str, str, str]] = [
    (
        "提醒的触发点不接（眨眼率照算、面板照显示，就是永远不弹窗）",
        "                            self._announce_eye_break(eye_run, blink_rate)\n",
        "                            pass\n",
        "focus.py", "④",
    ),
    (
        "最小间隔被删掉（每个结算时刻都弹，用户只能把提醒整个关掉）",
        "                        if (eye_run >= eye_break_threshold(blink_rate)\n"
        "                                and now - last_eye_remind >= EYE_REMIND_EVERY):\n",
        "                        if (eye_run >= eye_break_threshold(blink_rate)):\n",
        "focus.py", "④",
    ),
    (
        "提醒无条件触发（绕过用眼时长判定）",
        "                        if (eye_run >= eye_break_threshold(blink_rate)\n"
        "                                and now - last_eye_remind >= EYE_REMIND_EVERY):\n",
        "                        if (now - last_eye_remind >= EYE_REMIND_EVERY):\n",
        "focus.py", "④",
    ),
    (
        "用眼时长判据被去掉（眨眼率那条还在，但阈值变得不可达）",
        "    if 0 < blink_rate < BLINK_LOW_RATE:\n        return EYE_BREAK_SOON\n"
        "    return EYE_BREAK_AFTER\n",
        "    return 1e9\n",
        "focus.py", "④",
    ),
    (
        # 这条是"反例 C 有没有牙齿"的专属试金石，见文件开头那段。
        "阈值写死成 0.4 秒（不读配置里的那个值）",
        "    return EYE_BREAK_AFTER\n",
        "    return 0.4\n",
        "focus.py", "④",
    ),
    # ── 场景①：采集循环真的跑起来了吗（"视觉链路断了"那条防线）──
    (
        # 结算节拍拉长到 60 秒 → 6 秒窗口里一拍都不结算 → 一条样本都不落，
        # `len(rows) >= 2` 必须红。
        #
        # **为什么打"节拍"而不是把 `DURATION` 改短**：改短是碰运气。
        # 原来这条写的是"窗口砍到 0.5 秒"，实测被抓住的却是
        # `采集循环从没调用过 read_posture` —— 因为 `POSE_FPS = 2`，第一次
        # 姿态采样正好在 t≈0.5，和 0.5 秒的停表**撞在同一刻**：谁先跑赢是
        # 掷硬币，而"样本数不够"那条断言反而永远轮不到。
        # （0.5 秒还比采集循环的初始化短，顺带把 CI #40 那个挂死也引出来了。）
        # 打节拍则完全不依赖时序：人脸 10Hz、姿态 2Hz 照常被调用，
        # 只有结算那一段进不去 —— 于是第一条炸的必然是"样本数不够"。
        #
        # 片段用**机制**而不是常量值，也是跟调参解耦：以后改 DURATION
        # 或结算节拍都不用跟着改这里。
        "结算节拍被拉长到 60 秒（一拍都不结算，样本数断言必须红）",
        "                    if now - last_flush >= 1.0:\n",
        "                    if now - last_flush >= 60.0:\n",
        "focus.py", "①",
    ),
    (
        # 假视觉转头 60 度 → 判不出「专注」→ `"focused" in dist` 必须红。
        # 这一条盯的是"视觉结果真的流进了 decide()"：接线断了就是这个症状。
        # 片段唯一：另外两个假视觉的 ear 是变量，只有这一个写死 0.30。
        "场景①的假视觉转头 60 度（视觉结果没流进 decide，专注断言必须红）",
        '        return {"yaw": 0.0, "pitch": 5.0, "ear": 0.30, "scale": 1.0}\n',
        '        return {"yaw": 60.0, "pitch": 5.0, "ear": 0.30, "scale": 1.0}\n',
        "smoke_pipeline.py", "①",
    ),
    # ── 场景⑤：停表不能被初始化吃掉（"循环停不停得下来"那条防线）──
    #
    # 这一组的两条各打一半：**标志要写（stop），也要读（run）**。
    # 少了任何一半，`running = False` 都会被初始化末尾的 `running = True`
    # 覆盖掉。它们失效时的症状不是"断言失败"而是**挂死**，靠场景⑤自己那个
    # 5 秒闹钟抓住 —— 所以这两条变异也顺便证明那个闹钟不是摆设。
    (
        "run() 把初始化期间到的停止请求覆盖掉（循环永远退不出来）",
        "        if self._stop_requested:\n",
        "        if False:\n",
        "focus.py", "⑤",
    ),
    (
        "stop() 不记「停止请求」这一笔（running=False 照样被初始化覆盖）",
        "        self._stop_requested = True\n",
        "        pass\n",
        "focus.py", "⑤",
    ),
]

# 每个场景一条"只跑它"的命令。都不跑整条冒烟 —— 其余场景和这里的变异无关，
# 白等二十秒，而**代价高的闸门最终会被绕过**。
SCENARIOS: dict[str, str] = {
    "①": ("import smoke_pipeline as sp; sp.focus.use_safe_console(); "
          "print('场景①:', sp._check_constant_input())"),
    "④": ("import smoke_pipeline as sp; sp.focus.use_safe_console(); "
          "print('场景④:', sp._check_eye_break_reminder())"),
    "⑤": ("import smoke_pipeline as sp; sp.focus.use_safe_console(); "
          "print('场景⑤:', sp._check_stop_is_not_lost())"),
}

# 子进程的兜底超时。**这不是预算，是安全网**：各场景自己跑 1~8 秒，
# 场景⑤ 还自带一个 5 秒闹钟（挂死会在那里变成 AssertionError）。
# 留 300 秒只为了"解释器整个卡住"这种最坏情况 —— 那种时候我们要的是一句
# 说得清的错误，而不是一条未处理的栈（CI #40 查了半天就是吃了这个亏）。
SCENARIO_TIMEOUT = 300


class ScenarioHang(RuntimeError):
    """子进程没在 SCENARIO_TIMEOUT 内退出。

    既不是"断言抓住了"也不是"断言没抓住"，所以**不能**混进任何一种结论 ——
    当成后者会得出"这条变异有效"的错误判断。
    """


def copy_project(dst: Path) -> Path:
    shutil.copytree(ROOT, dst, ignore=shutil.ignore_patterns(*SKIP))
    return dst


def run_scenario(tree: Path, key: str) -> tuple[int, str]:
    """在 tree 里只跑某个场景（见 SCENARIOS）。

    编码那一段照抄 mutcheck.py 的教训：不显式指定就是拿父进程 locale 去解码，
    CI 的西文代码页下会抛 UnicodeDecodeError，线程静默死掉、输出变成空字符串，
    于是"拿不到输出"被误判成"抓住了"。

    超时单独转成 `ScenarioHang`，不让它变成一条未处理的栈：那样 CI 上只能看到
    "这一步超时了"，既不知道哪个场景、也不知道哪条变异，更看不出"挂死"和
    "断言抓住了"是两回事（CI #40 就是这么查的）。
    """
    try:
        r = subprocess.run([sys.executable, "-c", SCENARIOS[key]], cwd=str(tree),
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=SCENARIO_TIMEOUT,
                           env={**os.environ, "PYTHONIOENCODING": "cp1252"})
    except subprocess.TimeoutExpired:
        raise ScenarioHang(
            f"场景{key} 的子进程跑了 {SCENARIO_TIMEOUT} 秒还没退出 —— "
            "先查这个变异是不是让采集循环停不下来"
            "（`Monitor.stop()` 被 `self.running = True` 覆盖就是这个症状）") from None
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def main() -> int:
    # 复用 focus 自己的控制台兜底。CI 把 PYTHONIOENCODING 钉在 cp1252 上，
    # 这个脚本要打中文 —— 不加这一步，它会**崩在最后一行**（前面的"抓住了"
    # 全都打出来了，光看日志根本猜不到是编码问题），而且只把这一条 CI 步骤
    # 判红，看起来像"场景④ 的断言出问题了"，其实是它自己先炸了。
    sys.path.insert(0, str(ROOT))
    import focus
    focus.use_safe_console()

    tmp = Path(tempfile.mkdtemp(prefix="smoke-mutcheck-"))
    tree = copy_project(tmp / "fm")

    # 每个目标文件读一份"未变异"的原文，变异完立刻还原。
    pristine: dict[str, str] = {}
    for _f in sorted({t for *_x, t, _s in MUTATIONS}):
        pristine[_f] = (tree / _f).read_text(encoding="utf-8")

    # 基线：**每个用到的场景**都得先自己跑得过。否则"变异后红了"可能只是
    # 那个场景本来就红 —— 那这条变异什么都没证明。
    for key in sorted({s for *_x, s in MUTATIONS}):
        try:
            code, out = run_scenario(tree, key)
        except ScenarioHang as exc:
            print(f"基线（未变异）场景{key} 就挂死了：{exc}")
            return 2
        if code != 0 or not out.strip():
            print(f"基线（未变异）场景{key} 就跑不过："
                  f"退出码 {code}\n{out[-2000:]}")
            return 2
        print(f"基线（未变异）: 场景{key} 通过")
    print(f"共 {len(MUTATIONS)} 条变异待验\n")

    escaped: list[str] = []
    invalid: list[str] = []
    hung: list[str] = []
    for i, (label, old, new, target, key) in enumerate(MUTATIONS, 1):
        n = pristine[target].count(old)
        if n != 1:
            invalid.append(label)
            print(f"[{i}] ?? 原片段在 {target} 里出现 {n} 次"
                  f"（要求 1 次）：{label}")
            continue
        (tree / target).write_text(pristine[target].replace(old, new),
                                   encoding="utf-8")
        try:
            code, out = run_scenario(tree, key)
        except ScenarioHang as exc:
            # 还原要放在判之前：挂死的变异最容易让人以为"文件本来就是坏的"。
            (tree / target).write_text(pristine[target], encoding="utf-8")
            hung.append(label)
            print(f"[{i}] !! 场景{key} 挂死了（既不是抓住、也不是逃逸）：{label}")
            print(f"      {exc}")
            continue
        (tree / target).write_text(pristine[target], encoding="utf-8")

        if code == 0:
            escaped.append(label)
            print(f"[{i}] XX 场景{key} 仍然全绿，这条断言是摆设：{label}")
        elif "AssertionError" not in out:
            invalid.append(label)
            print(f"[{i}] ?? 是崩了/语法错了，不是被断言抓住：{label}")
            print("      " + (out.strip().splitlines() or ["(无输出)"])[-1][:160])
        else:
            last = (out.strip().splitlines() or ["(无输出)"])[-1]
            print(f"[{i}] 抓住了  场景{key}  {label}")
            print(f"      {last[:160]}")

    print()
    if escaped or invalid or hung:
        if escaped:
            print(f"有 {len(escaped)} 条变异没被抓住：")
            for label in escaped:
                print(f"  - {label}")
        if invalid:
            print(f"有 {len(invalid)} 条变异本身不合法：")
            for label in invalid:
                print(f"  - {label}")
        if hung:
            print(f"有 {len(hung)} 条变异让子进程挂死了"
                  f"（要先查清楚，不能当成「抓住了」）：")
            for label in hung:
                print(f"  - {label}")
        return 1
    print(f"全部 {len(MUTATIONS)} 条变异都被对应场景的断言抓住")
    return 0


if __name__ == "__main__":
    sys.exit(main())
