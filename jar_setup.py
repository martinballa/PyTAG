"""Download TAG.jar from a PyTAG GitHub release into pytag/jars/."""
import argparse
import os
import urllib.request

REPO = "martinballa/PyTAG"
DEFAULT_TAG = "v0.2"
DEST = "pytag/jars/TAG.jar"


def _progress(count, block_size, total):
    pct = min(count * block_size * 100 // total, 100)
    bar = "#" * (pct // 2)
    print(f"\r  [{bar:<50}] {pct:3d}%", end="", flush=True)


def download(tag: str, dest: str):
    url = f"https://github.com/{REPO}/releases/download/{tag}/TAG.jar"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"Downloading TAG.jar from {url}")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        print()  # newline after progress bar
    except urllib.error.HTTPError as e:
        raise SystemExit(f"Download failed ({e.code}): {url}") from e
    size_mb = os.path.getsize(dest) / 1e6
    print(f"Saved {dest} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag",
        default=DEFAULT_TAG,
        help=f"PyTAG release tag to download from (default: {DEFAULT_TAG})",
    )
    parser.add_argument(
        "--dest",
        default=DEST,
        help=f"Destination path for TAG.jar (default: {DEST})",
    )
    args = parser.parse_args()
    download(args.tag, args.dest)
