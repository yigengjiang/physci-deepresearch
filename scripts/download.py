"""Download PhySciBench from HuggingFace into the local data directory.

Pulls `physcibench.json` (the ~1 MB scorable metadata — always) and, with
`--with-files`, the ~485 MB `files/` directory of third-party source material.
The scorer is fully functional on `physcibench.json` alone.

    python scripts/download.py                 # json only
    python scripts/download.py --with-files     # json + files/ (~485 MB)

This script performs a NETWORK download; it is authored for users to run, not
executed during the release build.
"""

import argparse
import pathlib

REPO_ID = "littletreee/PhySciBench"
REPO_TYPE = "dataset"
DEFAULT_LOCAL_DIR = "PhySciBench"


def download(local_dir: str, with_files: bool) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    local_path = pathlib.Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)

    if with_files:
        # Full snapshot: physcibench.json + files/
        path = snapshot_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            local_dir=local_dir,
        )
        print(f"Downloaded full dataset (incl. files/) to: {path}")
    else:
        # JSON-only: enough to score every text-based type.
        path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename="physcibench.json",
            local_dir=local_dir,
        )
        print(f"Downloaded physcibench.json to: {path}")
        print("Tip: pass --with-files to also fetch the ~485 MB files/ directory.")

    json_path = local_path / "physcibench.json"
    if json_path.exists():
        print(f"physcibench.json present: {json_path} ({json_path.stat().st_size} bytes)")
    files_dir = local_path / "files"
    if files_dir.is_dir():
        n = sum(1 for _ in files_dir.iterdir())
        print(f"files/ present: {files_dir} ({n} entries)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PhySciBench from HuggingFace")
    parser.add_argument("--local-dir", default=DEFAULT_LOCAL_DIR, help=f"Target directory (default: {DEFAULT_LOCAL_DIR})")
    parser.add_argument("--with-files", action="store_true", help="Also download the ~485 MB files/ directory")
    args = parser.parse_args()
    download(args.local_dir, args.with_files)


if __name__ == "__main__":
    main()
