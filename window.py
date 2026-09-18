"""应用窗口层 —— 用系统 WebView2 承载本地面板，替代浏览器。

架构约束（pywebview 6 强制 GUI 循环跑主线程）：
    主线程  = 本模块的 start_main()，常驻隐藏窗口的 GUI 循环；
    子线程  = pystray 托盘（focus.run_tray 里启动）；
    托盘点击 → open_page()：显示窗口并导航到对应页面。

关闭窗口 = 隐藏到托盘（后台监视继续），真的退出只有托盘「退出」。
页面路由全部来自 dashboard 的本地 HTTP 服务（只绑 127.0.0.1）：
    /        实时面板
    /rate    自述评分
    /report  完整报告
"""

from __future__ import annotations

import threading
import time
import webbrowser

import dashboard
import webview

_TITLE = "专注监视"
_SIZE = (980, 720)
_MIN = (720, 520)

_pages = {"panel": "/", "rate": "/rate", "report": "/report"}

_started = False
_window: "webview.Window | None" = None
_lock = threading.Lock()
# 「退出」放行开关：托盘退出时置 True，让 closing 事件放行关闭。
# 否则点 X 的隐藏逻辑会把程序化关闭也拦下来 —— closing 返回 False
# 会被 pywebview 当作 Cancel，窗口关不掉，GUI 循环永远挂着，进程残留。
_quitting = False


def _page_url(page: str) -> str:
    """面板页地址。dashboard 被占用端口时端口会顺延，必须经 base_url 现算。"""
    return dashboard.base_url().rstrip("/") + _pages.get(page, "/")


def ensure_server() -> None:
    """窗口渲染的就是 dashboard 的页面，服务没起时先拉起来（不开浏览器）。"""
    if dashboard._server is None:
        dashboard.serve_background(open_browser=False)


def start_main() -> None:
    """主线程入口（focus.run_tray 调用）：常驻 GUI 循环，随进程退出。

    pywebview 硬性要求 start() 在主线程；托盘在子线程（pystray 消息
    循环不受此限制）。窗口以隐藏方式创建，托盘点击时才 show。
    """
    global _started
    # 隐藏窗口的 GUI 循环：一次进程一个，重复调用只返回。
    if _started:
        return
    _started = True
    url = _page_url("panel")
    page = webview.create_window(
        _TITLE, url, width=_SIZE[0], height=_SIZE[1],
        min_size=_MIN, background_color="#f3f3f3", hidden=True)
    global _window
    _window = page
    page.events.closing += _on_closing
    webview.start()


def wait_until_ready(timeout: float = 20.0) -> bool:
    """等 GUI 窗口对象建出来（`start_main` 在**主线程**上创建它）。

    为什么要等：面板服务在 `run_tray` 之前就起来了，所以"请把面板显示出来"
    这个请求可能比 GUI 循环更早到。那时 `_window` 还是 None，`open_page`
    会走"窗口层不可用"的兜底 —— 用系统浏览器弹一个新网页。那正是要修的
    bug（用户报的"不断开新网页弹出面板"），所以这里必须等一下，而不是
    一看到 None 就当成"窗口层坏了"。

    返回 False 表示超时（GUI 循环没跑起来）。**别在这里替调用方兜底** ——
    降级成浏览器是调用方的决定，它才知道那时还有没有别的路可走。
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _window is not None:
            return True
        time.sleep(0.1)
    return False


def open_page(page: str = "panel") -> None:
    """显示（或聚焦并导航到）应用窗口。

    GUI 未启动（比如只跑命令行工具）时回退到系统浏览器，功能不丢。
    """
    url = _page_url(page)
    ensure_server()
    with _lock:
        if _window is None:
            webbrowser.open(url)
            return
        try:
            _window.load_url(url)
            _window.restore()
            _window.show()
        except Exception:
            webbrowser.open(url)


def quit_app() -> None:
    """托盘「退出」真正退干净：放行关闭 → 销毁窗口 → GUI 循环返回 → 进程退出。

    close 只差 on_quit 里 icon.stop() 不够：GUI 循环在主线程，窗口不销毁
    app.Run() 永不返回，进程就留在任务管理器里。
    """
    global _quitting
    _quitting = True
    try:
        if _window is not None:
            _window.destroy()
    except Exception:
        pass


def _on_closing() -> bool:
    """点 X = 隐藏到托盘，不是退出 —— 监视器本来就应该常驻。

    pywebview 的 closing 事件返回 False 即取消关闭；真的退出时
    quit_app() 已把 _quitting 置 True，这里放行返回 True。
    """
    if _quitting:
        return True
    try:
        if _window is not None:
            _window.hide()
    except Exception:
        pass
    return False