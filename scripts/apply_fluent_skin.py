# -*- coding: utf-8 -*-
"""把 dashboard._PAGE 的皮肤替换为 Win10 Fluent 风格，随系统深浅色切换。

实现方式：
  :root 定义一套 CSS 变量（浅色为默认），
  @media (prefers-color-scheme: dark) 覆盖为深色变量；
  WebView2（Chromium）会自动跟随 Windows 深浅色主题实时变。

只动 <style> 块内内容；HTML 结构与类名、JS 轮询逻辑完全不动，
页面功能零变化。幂等：已换过皮时会直接退出。
"""
import re
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DASH = ROOT / "dashboard.py"
text = DASH.read_text(encoding="utf-8")

MARK = "/* fluent-skin */"
if MARK in text:
    print("[skin] 已经是 Fluent 皮肤，跳过")
    sys.exit(0)

NEW_CSS = """<style>
  /* fluent-skin：Win10 Fluent 皮肤，跟随系统深浅色。
     浅色为默认变量，深色在 prefere-dark 里整体覆盖。 */
  :root {
    color-scheme: light dark;
    --bg:#f3f3f3; --fg:#1b1b1b; --h2:#555; --muted:#666; --faint:#777;
    --card:#fff; --line:#e1e1e1; --input-line:#8a8a8a; --input-hover-line:#323130;
    --accent:#0078d7; --accent-hover:#106ebe; --accent-soft:#e5f1fb;
    --ok:#107c10; --ok-soft:#dff6dd;
    --warn:#d83b01; --warn-soft:#fff4ce; --warn-line:#fce100;
    --pause:#744da9; --code-bg:#f0f0f0;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg:#202020; --fg:#f3f3f3; --h2:#9d9d9d; --muted:#a5a5a5; --faint:#8f8f8f;
      --card:#2b2b2b; --line:#3b3b3b; --input-line:#5d5d5d; --input-hover-line:#999;
      --accent:#60cdfe; --accent-hover:#7ad1fe; --accent-soft:#1f3b4f;
      --ok:#6ccb5f; --ok-soft:#14371a;
      --warn:#f9b25e; --warn-soft:#4a2e0a; --warn-line:#8a5a00;
      --pause:#c0a5e8; --code-bg:#333;
    }
  }
  * { box-sizing: border-box; }
  body { margin:0; padding:26px 20px 50px; background:var(--bg); color:var(--fg);
        font:14px/1.6 "Segoe UI",system-ui,-apple-system,"Microsoft YaHei",sans-serif; }
  .wrap { max-width:920px; margin:0 auto; }
  h1 { font-size:22px; margin:0 0 4px; font-weight:600; }
  h2 { font-size:14px; margin:28px 0 12px; color:var(--h2); font-weight:600;
       letter-spacing:.02em; }
  .top { display:flex; justify-content:space-between; align-items:center;
        flex-wrap:wrap; gap:10px; margin-bottom:22px; }
  .now { display:flex; align-items:center; gap:12px; }
  .dot { width:18px; height:18px; border-radius:50%; }
  .st { font-size:30px; font-weight:600; }
  .health { font-size:13px; color:var(--muted); }
  .health a { color:var(--accent); text-decoration:none; }
  .health a:hover { text-decoration:underline; }
  .ok { color:var(--ok); }
  .warn { color:var(--warn); font-weight:600; }
  .pause { color:var(--pause); font-weight:600; }
  .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:14px 16px; }
  .card .k { color:var(--muted); font-size:13px; margin-bottom:4px; }
  .big { font-size:26px; font-weight:600; font-variant-numeric:tabular-nums; }
  .sub { font-size:12px; color:var(--muted); margin-top:4px; }
  .pbar { background:var(--line); height:6px; border-radius:3px; overflow:hidden;
          margin:8px 0 4px; }
  .pbar i { display:block; height:100%; border-radius:3px; }
  .tape { display:flex; flex-wrap:wrap; gap:1px; background:var(--card);
          border:1px solid var(--line); border-radius:8px; padding:10px; min-height:34px;
          align-items:center; }
  .seg { width:9px; height:22px; border-radius:2px; }
  .hbars { display:flex; gap:6px; align-items:flex-end; background:var(--card);
           border:1px solid var(--line); border-radius:8px; padding:12px; min-height:100px; }
  .hbar { flex:1; display:flex; flex-direction:column; align-items:center; gap:4px; }
  .hbar span { font-size:10px; color:var(--muted); }
  .hbg { width:100%; height:70px; display:flex; flex-direction:column;
         justify-content:flex-end; align-items:center; }
  .hbg i { display:block; width:100%; background:var(--accent); border-radius:2px 2px 0 0; }
  .hbg b { display:block; width:100%; background:var(--ok); border-radius:0; margin-top:-1px; }
  .muted { color:var(--faint); }
  .note { color:var(--muted); font-size:12px; margin-top:12px; }
  code { background:var(--code-bg); border:1px solid var(--line); padding:1px 5px; border-radius:4px;
         font-size:13px; }
  .nav { display:flex; gap:18px; margin-bottom:16px; padding:10px 14px;
         background:var(--card); border:1px solid var(--line); border-radius:8px; }
  .nav a { color:var(--fg); text-decoration:none; font-size:14px; padding:4px 2px;
           opacity:.75; }
  .nav a:hover { color:var(--accent); opacity:1; }
  .fields { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr));
            gap:12px; }
  .fld { display:flex; flex-direction:column; gap:5px; background:var(--card);
         border:1px solid var(--line); border-radius:8px; padding:12px 14px; }
  .fld .fl { font-size:13px; color:var(--fg); }
  .fld .hint { font-size:11px; color:var(--muted); line-height:1.5; }
  .fld.chk { flex-direction:row; align-items:flex-start; gap:10px; flex-wrap:wrap; }
  .fld.chk .fl { font-weight:600; }
  .fld.chk .hint { flex-basis:100%; }
  input[type=number] { background:var(--card); border:1px solid var(--input-line); color:var(--fg);
        border-radius:4px; padding:6px 10px; font:inherit; font-size:14px;
        width:100%; min-height:32px; }
  input[type=number]:hover, textarea:hover { border-color:var(--input-hover-line); }
  input[type=number]:focus, textarea:focus { outline:none; border:2px solid var(--accent);
        padding:5px 9px; }
  input[type=checkbox] { width:17px; height:17px; accent-color:var(--accent); margin-top:2px; }
  textarea { width:100%; background:var(--card); border:1px solid var(--input-line); color:var(--fg);
        border-radius:4px; padding:10px 12px; font:13px/1.6 ui-monospace,
        Consolas,"Cascadia Mono",monospace; resize:vertical; }
  .acts { display:flex; gap:12px; margin-top:26px; }
  button { background:var(--accent); color:#fff; border:1px solid var(--accent); border-radius:4px;
        padding:6px 20px; font:inherit; font-size:14px; font-weight:600;
        cursor:pointer; min-height:32px; }
  button:hover { background:var(--accent-hover); border-color:var(--accent-hover); }
  button:focus { outline:none; border-color:var(--bg); box-shadow:0 0 0 2px var(--accent); }
  button.ghost { background:var(--card); color:var(--fg); border:1px solid var(--input-line);
        font-weight:400; }
  button.ghost:hover { background:var(--bg); border-color:var(--input-hover-line); }
  .alert { border-left:4px solid; border-radius:4px; padding:12px 16px;
           margin-bottom:20px; font-size:14px; }
  .alert ul { margin:8px 0 0; padding-left:20px; }
  .alert.good { background:var(--ok-soft); border-color:var(--ok); color:var(--ok); }
  .alert.bad { background:var(--warn-soft); border-color:var(--warn-line); color:var(--warn); }
  .rlist { display:flex; flex-direction:column; gap:8px; }
  .rrow { display:flex; align-items:center; gap:16px; background:var(--card);
          border:1px solid var(--line); border-radius:8px; padding:10px 16px; }
  .rleft { display:flex; flex-direction:column; gap:3px; flex:1; min-width:0; }
  .rt { font-size:15px; color:var(--fg); font-variant-numeric:tabular-nums; }
  .rapps { font-size:12px; color:var(--muted); white-space:nowrap; overflow:hidden;
           text-overflow:ellipsis; }
  .rb { display:flex; gap:6px; }
  .rb button { background:var(--card); color:var(--fg); border:1px solid var(--input-line);
        border-radius:4px; width:40px; padding:6px 0; font-size:14px;
        font-weight:600; cursor:pointer; min-height:32px; }
  .rb button:hover { background:var(--accent-soft); color:var(--accent); border-color:var(--accent); }
  /* 不固定列宽的话三列会塌在一起：<th> 浏览器默认居中，加上列宽自动分配，
     表头会挤成一坨（"自评备注" 连成一个词）。 */
  .hist th, .hist td { text-align:left; }
  .hist th:nth-child(1), .hist td:nth-child(1) { width:230px; }
  .hist th:nth-child(2), .hist td:nth-child(2) { width:80px; text-align:right; }
</style>"""

start = text.index("<style>")
end = text.index("</style>")
text = text[:start] + NEW_CSS + text[end:]
DASH.write_text(text, encoding="utf-8")
print("[skin] dashboard Fluent 双主题皮肤已替换")