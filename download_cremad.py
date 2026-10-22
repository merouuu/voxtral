"""Download CREMA-D (AudioWAV + VideoDemographics.csv) from the original GitHub repo, without git-lfs."""
import json, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "CheyneyComputerScience/CREMA-D"
root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/CREMA-D")

tree = json.load(urllib.request.urlopen(f"https://api.github.com/repos/{REPO}/git/trees/master?recursive=1"))["tree"]
files = [t["path"] for t in tree if t["path"] == "VideoDemographics.csv"
         or (t["path"].startswith("AudioWAV/") and t["path"].endswith(".wav"))]

def fetch(path):
    dst = root / path
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        host = "media.githubusercontent.com/media" if path.endswith(".wav") else "raw.githubusercontent.com"  # wav are in LFS
        urllib.request.urlretrieve(f"https://{host}/{REPO}/master/{path}", dst)

with ThreadPoolExecutor(16) as ex:
    list(ex.map(fetch, files))
print(len(files), "files in", root)
