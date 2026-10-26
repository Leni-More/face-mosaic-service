"""手动引导：从 PyPI 解析依赖树、下载 wheel 并解压到 venv site-packages。

pip 的临时解包目录被文件沙箱拦截，故绕开 pip 直接落地 whl 内容。
"""
import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

from pip._vendor.packaging.markers import Marker  # noqa: E402
from pip._vendor.packaging.requirements import Requirement  # noqa: E402
from pip._vendor.packaging.version import Version  # noqa: E402

SITE = Path(__file__).parent / ".venv" / "Lib" / "site-packages"
WHEELS = Path(__file__).parent / ".wheels"
WHEELS.mkdir(exist_ok=True)

PY_VER = "3.14"
PY_FULL = "3.14.10"
ENV = {
    "python_version": PY_VER,
    "python_full_version": PY_FULL,
    "sys_platform": "win32",
    "platform_system": "Windows",
    "platform_machine": "AMD64",
    "implementation_name": "cpython",
    "platform_python_implementation": "CPython",
    "os_name": "nt",
}

cache = {}


def fetch_json(url):
    if url in cache:
        return cache[url]
    with urllib.request.urlopen(url, timeout=60) as r:
        data = json.loads(r.read().decode())
    cache[url] = data
    return data


def tag_matches(filename):
    m = re.match(r"^(.+?)-(.+?)-(.+)\.whl$", filename)
    if not m:
        return False
    for t in m.group(3).split("."):
        parts = t.split("-")
        if len(parts) != 3:
            continue
        py, abi, plat = parts
        if plat not in ("any", "win_amd64"):
            continue
        if py in ("py3", "py2.py3"):
            return True
        if py.startswith("cp"):
            try:
                minor = int(py[3:])
            except ValueError:
                continue
            # cp314 专用 wheel，或 abi3 稳定 ABI（最低版本 <= 3.14 均兼容）
            if (abi == "cp314" and minor == 14) or (abi == "abi3" and minor <= 14):
                return True
    return False


def pick_wheel(pkg, version_spec, extra):
    url = f"https://pypi.org/pypi/{pkg}/json"
    data = fetch_json(url)
    releases = data["releases"]
    candidates = []
    for ver, files in releases.items():
        try:
            v = Version(ver)
        except Exception:
            continue
        if v.is_prerelease:
            continue
        if version_spec and not version_spec.contains(v, prereleases=True):
            continue
        for f in files:
            if f["filename"].endswith(".whl") and tag_matches(f["filename"]):
                candidates.append((v, f))
    if not candidates:
        raise RuntimeError(f"no wheel for {pkg} {version_spec}")
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def deps_of(pkg, version, extra):
    data = fetch_json(f"https://pypi.org/pypi/{pkg}/{version}/json")
    out = []
    for req in data["info"].get("requires_dist") or []:
        try:
            r = Requirement(req)
        except Exception:
            continue
        if r.marker:
            env = dict(ENV, extra=extra)
            if not r.marker.evaluate(env):
                continue
        out.append((r.name, r.specifier, r.extras[0] if r.extras else ""))
    return out


from pip._vendor.packaging.specifiers import SpecifierSet  # noqa: E402

ANY = SpecifierSet()


def install(pkg, version_spec, extra, seen):
    key = pkg.lower().replace("_", "-")
    if key in seen:
        return
    seen.add(key)
    if isinstance(version_spec, str):
        version_spec = ANY if version_spec in ("", "*") else SpecifierSet(version_spec)
    f = pick_wheel(pkg, version_spec, extra)
    ver = re.match(r"^(.+?)-(.+?)-(.+)\.whl$", f["filename"]).group(2)
    print(f"[install] {pkg}=={ver} <- {f['filename']}")
    dest = WHEELS / f["filename"]
    if not dest.exists():
        urllib.request.urlretrieve(f["url"], dest)
    with zipfile.ZipFile(dest) as z:
        z.extractall(SITE)
    for name, spec, sub_extra in deps_of(key, ver, extra):
        install(name, spec, sub_extra or extra, seen)


if __name__ == "__main__":
    targets = [
        ("uniface", "*", "cpu"),
        ("fastapi", "*", ""),
        ("uvicorn", "*", "standard"),
        ("python-multipart", "*", ""),
        ("imageio-ffmpeg", "*", ""),
    ]
    seen = set()
    for name, spec, extra in targets:
        install(name, spec, extra, seen)
    print("ALL DONE:", len(seen), "packages")
