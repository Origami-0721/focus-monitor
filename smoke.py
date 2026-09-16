# -*- coding: utf-8 -*-
"""冒烟测试：确认每个页面都能真的渲染出来。

和 `focus.py --selftest` 的分工：自检覆盖状态机、阈值、时间分带这些**纯逻辑**；
它也会调用 report.build_html，但从不碰 dashboard 的页面函数。而"模板里少个括号"
这类错误纯逻辑断言是看不出来的 —— 用户点开面板第一眼就崩。所以这里把每个页面
真的渲染一遍。

为什么是独立文件，而不是 workflow 里的 `python -c "..."`：
那个内联写法**从加进去那天起就是坏的**，只是每次都因为前面的步骤先失败而被
skip，所以一直没人发现。原因是 YAML 会把续行折叠成一行、并且**保留行首缩进**，
于是 python -c 收到的程序以空格开头，直接 `IndentationError: unexpected indent`。
放进文件里就没有这层转义/折叠的雷，本地也能直接跑。

用法：
    python smoke.py
"""

from __future__ import annotations

import sys

import dashboard
import focus
import report


def main() -> int:
    # 必须在任何输出之前。CI 的 Windows runner 是西文代码页，中文提示会抛
    # UnicodeEncodeError（详见 focus.use_safe_console）。
    focus.use_safe_console()

    # 报告：整页由 build_html 一次产出。
    html = report.build_html([])
    assert "<html" in html, "报告渲染失败"
    assert "focus.db" not in html, "报告里不该出现数据库路径"

    # 面板：页面函数只返回 <body> 片段，整页由 _shell() 拼起来 ——
    # 两边都要测，否则"_rate_page 没抛异常"其实什么也没验证。
    for name, body in (("自述评分页", dashboard._rate_page()),
                       ("设置页", dashboard._settings_page())):
        assert body.strip(), f"{name}渲染成了空字符串"
        assert "<html" in dashboard._shell(body), f"{name}拼不成完整 HTML"

    print("冒烟测试通过 - 所有页面都能渲染")
    return 0


if __name__ == "__main__":
    sys.exit(main())
