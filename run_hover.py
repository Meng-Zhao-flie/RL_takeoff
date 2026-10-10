"""Run the installed hover release from the project root. Output is English."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    receipt = json.loads((root / "CURRENT_HOVER_RELEASE.json").read_text())
    active = root / receipt["active_project"]
    active_archive = active / receipt.get(
        "active_flash_archive", "firmware/simple_hover_trial/build/firmware.zip"
    )
    if not active_archive.resolve().is_relative_to(active.resolve()):
        print("Blocked: Active firmware archive is outside the installed release.", file=sys.stderr)
        return 2
    archives = (
        root / receipt["root_flash_archive"],
        active_archive,
    )
    for archive in archives:
        if not archive.is_file() or hashlib.sha256(archive.read_bytes()).hexdigest() != receipt["firmware_zip_sha256"]:
            print("Blocked: Installed hover firmware does not match the active release.", file=sys.stderr)
            return 2
    args = sys.argv[1:]
    for i, value in enumerate(args):
        if value == "--config" and i + 1 < len(args):
            args[i + 1] = str(Path(args[i + 1]).resolve())
        elif value.startswith("--config="):
            args[i] = "--config=" + str(Path(value.split("=", 1)[1]).resolve())
    return subprocess.call([sys.executable, str(active / "run_hover.py"), *args], cwd=active)


if __name__ == "__main__":
    raise SystemExit(main())
