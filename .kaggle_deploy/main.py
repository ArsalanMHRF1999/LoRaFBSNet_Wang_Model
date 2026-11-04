
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


def execute(command, env=None):
    print("\n>>>", " ".join(command), flush=True)

    subprocess.run(
        command,
        check=True,
        env=env
    )


def main():

    print("=" * 60)
    print("LoRaFB-SNet | Kaggle GPU Deployment")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Retrieve GitHub credentials if repository is private
    # --------------------------------------------------------

    clone_env = os.environ.copy()

    try:
        from kaggle_secrets import UserSecretsClient

        print("Retrieving GitHub token...")

        token = UserSecretsClient().get_secret(
            "GITHUB_TOKEN"
        )

        if not token:
            raise RuntimeError(
                "GITHUB_TOKEN is empty or unavailable. "
                "Check Kaggle Secrets permissions."
            )

        auth = base64.b64encode(
            f"x-access-token:{token}".encode()
        ).decode()

        clone_env = os.environ.copy()

        clone_env["GIT_CONFIG_COUNT"] = "1"
        clone_env["GIT_CONFIG_KEY_0"] = "http.extraheader"
        clone_env["GIT_CONFIG_VALUE_0"] = (
            "AUTHORIZATION: basic " + auth
        )

        print("GitHub authentication configured.")

    except Exception:
        print("No GitHub token available; trying public access.")

    # --------------------------------------------------------
    # 2. Clone latest GitHub repository
    # --------------------------------------------------------

    if REPO_DIR.exists():
        subprocess.run(
            ["rm", "-rf", str(REPO_DIR)],
            check=True
        )

    execute([
        "git",
        "clone",
        "--depth", "1",
        "--branch", BRANCH,
        REPO_URL,
        str(REPO_DIR)
    ], env=clone_env)

    print("GitHub repository cloned successfully.")

    # --------------------------------------------------------
    # 3. Check GPU
    # --------------------------------------------------------

    execute([
        sys.executable,
        "-c",
        "import torch; "
        "print('PyTorch:', torch.__version__); "
        "print('CUDA available:', torch.cuda.is_available()); "
        "print('GPU:', torch.cuda.get_device_name(0) "
        "if torch.cuda.is_available() else 'Not available')"
    ])

    # --------------------------------------------------------
    # 4. Install project dependencies
    # --------------------------------------------------------

    requirements_path = REPO_DIR / REQUIREMENTS

    if requirements_path.exists():

        print("Installing project dependencies...")

        execute([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(requirements_path)
        ])

    else:
        print("No requirements.txt found. Skipping installation.")

    # --------------------------------------------------------
    # 5. Execute actual project main.py
    # --------------------------------------------------------

    entrypoint = REPO_DIR / ENTRYPOINT

    if not entrypoint.exists():
        raise FileNotFoundError(
            f"Project entrypoint not found: {entrypoint}"
        )

    os.chdir(REPO_DIR)

    sys.path.insert(0, str(REPO_DIR))

    print("\nStarting project execution...")
    print("Entry point:", entrypoint)

    runpy.run_path(
        str(entrypoint),
        run_name="__main__"
    )


if __name__ == "__main__":
    main()
