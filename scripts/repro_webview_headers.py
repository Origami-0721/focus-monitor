"""抓应用窗口（WebView2）真正发出去的请求头。

**为什么要单独有个工具。** 窗口里的页面和浏览器里的页面，看起来是同一个
页面，发出去的请求头却不一样 —— 而这类差异**用 urllib/curl 复现不出来**，
因为那些客户端根本不发 Origin。本项目的真实案例：

    用户在窗口里点完自述评分 → 弹出一张「403 请求来源不是本机面板」。
    抓下来才发现 Edge/WebView2 对**同源表单 POST** 送的是
        Origin: null    Sec-Fetch-Site: same-origin
    而守卫当时把"scheme 不是 http/https"一律当跨源拒掉
    （urlsplit("null").scheme == ""），于是窗口里所有表单全 403。

自检里那两段 HTTP 测试碰不到这条路径，就是因为 `urllib.request.urlopen`
默认**不发 Origin** —— 它和真实客户端长得一点都不像。

用法（要装了 pywebview 的解释器；Windows 才有 WebView2）：

    uv run python scripts/repro_webview_headers.py

它会：起一个**独立端口**的临时面板 → 开一个隐藏窗口加载 /rate →
在页面里注入一个真表单并 submit（真导航 POST）→ 把服务端收到的
Host/Origin/Referer/Sec-Fetch-* 全量打出来。

两个必须知道的坑：

1. **端口不能跟正在跑的实例撞。** Windows 的 SO_REUSEADDR 允许两个进程
   绑同一个 127.0.0.1:8787，连接会被先绑的那个接走 —— 探针会"什么都抓不到"，
   看起来像窗口没发请求，其实是请求被用户那个实例接走了。所以固定用 8799。
2. **数据库指向副本。** 表单真的会提交，别写进用户的 focus.db。
   用 block_start=0（会被业务校验挡成 400，不落库）也是出于同样的考虑。
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

PORT = 8799                     # 见上文坑 1：别用 DEFAULT_PORT

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:               # 控制台不支持就随它去
    pass

# ── 数据库指到副本：表单提交是真的，别污染用户的数据 ────────────────
_tmp = Path(tempfile.mkdtemp(prefix="fmprobe-"))
if (ROOT / "focus.db").exists():
    shutil.copy2(ROOT / "focus.db", _tmp / "focus.db")

import focus                                     # noqa: E402
focus.DB_PATH = _tmp / "focus.db"
import dashboard                                 # noqa: E402

CAPTURED: list[dict] = []
_lock = threading.Lock()

_KEYS = ("Host", "Origin", "Referer", "User-Agent", "Sec-Fetch-Site",
         "Sec-Fetch-Mode", "Sec-Fetch-Dest", "Content-Type")

_orig_guard = dashboard._Handler._guard
_orig_send = dashboard._Handler._send


def _guard(self):
    rec = {k: v for k, v in
           (("method", self.command), ("path", self.path))
           if v is not None}
    for k in _KEYS:
        v = self.headers.get(k)
        if v is not None:
            rec[k] = v
    with _lock:
        CAPTURED.append(rec)
    ok = _orig_guard(self)
    rec["guard_ok"] = ok
    return ok


def _send(self, body, code=200):
    with _lock:
        if CAPTURED:
            CAPTURED[-1]["resp"] = code
            CAPTURED[-1]["resp_head"] = body[:70].replace("\n", " ")
    return _orig_send(self, body, code)


dashboard._Handler._guard = _guard
dashboard._Handler._send = _send

try:
    import webview
except ImportError:
    raise SystemExit("这个脚本要 pywebview（uv sync 一下），且只在 Windows 上有意义")

url = dashboard.serve_background(port=PORT, open_browser=False)
assert dashboard.base_url() == f"http://127.0.0.1:{PORT}/", \
    f"端口没拿到 {PORT}（被占了？），换个端口再跑：{dashboard.base_url()}"
print("[probe] server =", url)
print("[probe] DB_PATH =", focus.DB_PATH)

_win = webview.create_window("probe", f"http://127.0.0.1:{PORT}/rate",
                             width=980, height=720, hidden=True)


def work() -> None:
    time.sleep(6)
    print("[probe] location =", _win.evaluate_js("location.href"))
    # 注入真表单并 submit —— 和用户点按钮是同一条路径（真导航 POST）。
    # block_start=0 会在守卫之后被业务校验挡成 400，不落库。
    token = json.dumps(dashboard.CSRF_TOKEN)
    js = (
        "(function(){"
        f"var t={token};"
        "var f=document.createElement('form');"
        "f.method='POST';f.action='/rate';"
        "function h(n,v){var i=document.createElement('input');"
        "i.type='hidden';i.name=n;i.value=v;f.appendChild(i);}"
        "h('csrf',t);h('block_start','0');h('score','3');"
        "document.body.appendChild(f);f.submit();return 'submitted';})()"
    )
    try:
        print("[probe] 注入表单提交 ->", _win.evaluate_js(js))
    except Exception as exc:                    # noqa: BLE001
        print("[probe] submit 失败:", exc)
    time.sleep(6)
    print("===CAPTURED===")
    print(json.dumps(CAPTURED, indent=1, ensure_ascii=False))
    print("===END===")
    sys.stdout.flush()
    os._exit(0)                                 # GUI 循环不会自己退，硬退


webview.start(work)
