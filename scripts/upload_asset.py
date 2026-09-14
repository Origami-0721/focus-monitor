"""把构建好的 exe 作为资产上传到 GitHub Release。

token 只在进程内使用：从 `git credential fill` 读取，从不打印。
用法: python scripts/upload_asset.py <release_id> <exe路径> <资产名>
"""
import json
import subprocess
import sys
import urllib.request

REPO = "Origami-0721/focus-monitor"


def get_token() -> str:
    inp = "protocol=https\nhost=github.com\n\n"
    out = subprocess.run(
        ["git", "credential", "fill"], input=inp, capture_output=True, text=True, cwd=r"C:\Users\31866\focus-monitor"
    ).stdout
    for line in out.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1]
    raise SystemExit("no token")


def main() -> None:
    release_id, exe_path, asset_name = sys.argv[1], sys.argv[2], sys.argv[3]
    token = get_token()

    with open(exe_path, "rb") as f:
        data = f.read()
    url = f"https://uploads.github.com/repos/{REPO}/releases/{release_id}/assets?name={asset_name}"
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/octet-stream",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            info = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode('utf-8')[:500]}")
        raise SystemExit(1)
    print(f"asset {info.get('name')} {info.get('size')} bytes -> {info.get('browser_download_url')}")


if __name__ == "__main__":
    main()