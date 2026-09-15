"""把构建好的发布包作为资产上传到 GitHub Release。

token 只在进程内使用：从 `git credential fill` 读取，从不打印。
用法: python scripts/upload_asset.py <release_id|tag> <文件路径> <资产名>

注意：这个脚本只有仓库维护者用得上，普通用户跑不到它。
"""
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = "Origami-0721/focus-monitor"
ROOT = Path(__file__).resolve().parent.parent

# 绝不能上传到 Release 的东西：库里是逐秒的窗口标题历史。
FORBIDDEN_SUFFIXES = (".db", ".db-wal", ".db-shm", ".csv", ".log")


def get_token() -> str:
    """从 git 凭据助手取 token。

    cwd 必须是仓库目录 —— 凭据助手是按仓库路径匹配配置的，
    硬编码某台机器的绝对路径（原来是 C:\\Users\\31866\\...）在别人
    的机器和 CI 上都直接失效，等于脚本只能一个人跑。
    """
    inp = "protocol=https\nhost=github.com\n\n"
    out = subprocess.run(
        ["git", "credential", "fill"], input=inp, capture_output=True, text=True,
        cwd=str(ROOT),
    ).stdout
    for line in out.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1]
    raise SystemExit(
        "拿不到 token：`git credential fill` 没返回 password=。"
        "先确认 git 凭据助手已配置，或改用 gh auth login。")


def resolve_id(token: str, tag_or_id: str) -> str:
    """上传端点要数字 id；传 tag 名（v0.2.2）也能用，先查一次再传。"""
    if tag_or_id.isdigit():
        return tag_or_id
    req = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/releases/tags/{tag_or_id}",
        headers={"Authorization": f"Bearer {token}",
                 "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))["id"]


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("用法: upload_asset.py <release_id|tag> <文件路径> <资产名>")
    release_id, path, asset_name = sys.argv[1], sys.argv[2], sys.argv[3]

    # 上传是不可撤销的公开动作。含个人数据的文件在这里就拦死 ——
    # 免得手滑把 focus.db 传成所有人可下载的附件。
    if Path(path).name.lower().endswith(FORBIDDEN_SUFFIXES):
        raise SystemExit(
            f"拒绝上传 {path}：这个后缀是本地隐私数据（含窗口标题），不能发到 Release。")

    token = get_token()
    release_id = resolve_id(token, release_id)

    with open(path, "rb") as f:
        data = f.read()
    url = (f"https://uploads.github.com/repos/{REPO}/releases/"
           f"{release_id}/assets?name={asset_name}")
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
    print(f"已上传 {info.get('name')}（{info.get('size')} 字节）-> "
          f"{info.get('browser_download_url')}")


if __name__ == "__main__":
    main()
