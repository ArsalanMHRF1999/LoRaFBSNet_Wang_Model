
import os
import sys
import runpy
import subprocess
import base64
from pathlib import Path

REPO_URL = 'https://github.com/ArsalanMHRF1999/LoRaFBSNet_Wang_Model.git'
BRANCH = 'main'

ENTRYPOINT = 'main.py'
REQUIREMENTS = 'requirements.txt'

WORK_DIR = Path("/kaggle/working")
REPO_DIR = WORK_DIR / "LoRaFBsnet_Wang"


def main():

    print("Cloning project from GitHub...")

    if REPO_DIR.exists():
        import shutil
        shutil.rmtree(REPO_DIR)

    subprocess.run(
        [
            "git", "clone",
            "--depth", "1",
            "--branch", BRANCH,
            REPO_URL,
            str(REPO_DIR)
        ],
        check=True
    )

    # Install requirements
    requirements = REPO_DIR / "requirements.txt"

    if requirements.exists():
        subprocess.run(
            [
                sys.executable,
                "-m", "pip",
                "install",
                "-r", str(requirements)
            ],
            check=True
        )

    # Execute ROOT main.py
    project_main = REPO_DIR / "main.py"

    if not project_main.exists():
        raise FileNotFoundError(project_main)

    print("\nExecuting ROOT main.py...\n")

    os.chdir(REPO_DIR)
    sys.path.insert(0, str(REPO_DIR))

    runpy.run_path(
        str(project_main),
        run_name="__main__"
    )

if __name__ == "__main__":
    main()