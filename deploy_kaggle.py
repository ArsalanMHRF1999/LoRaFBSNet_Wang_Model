
#!/usr/bin/env python3
"""
VS Code -> GitHub -> Kaggle GPU deployment

Workflow:
1. Check Git repository and committed changes.
2. Push the current branch to GitHub.
3. Generate Kaggle kernel metadata.
4. Generate a Kaggle bootstrap script.
5. Push the bootstrap kernel using Kaggle CLI.
6. Kaggle clones GitHub and executes the real main.py.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

KAGGLE_ID = "mohammadmehranfar/lorafb-snet-experiments"
KAGGLE_TITLE = "LoRaFB-SNet Experiments"

PROJECT_ENTRYPOINT = "main.py"
REQUIREMENTS_FILE = "requirements.txt"

DEPLOY_DIR = ROOT / ".kaggle_deploy"

KAGGLE_TIMEOUT = 7200

# GitHub private repository authentication:
# Add GITHUB_TOKEN as a Kaggle Secret if the repository is private.
GITHUB_SECRET_NAME = "GITHUB_TOKEN"


# ============================================================
# UTILITIES
# ============================================================

def run_command(command, cwd=None):
    """Run a shell command and stop if it fails."""

    print(f"\n>>> {' '.join(map(str, command))}")

    subprocess.run(
        command,
        cwd=cwd,
        check=True
    )


def git_output(*args):
    """Get output from a Git command."""

    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True
    )

    return result.stdout.strip()


def get_github_repository():
    """Get and normalize the origin GitHub URL."""

    url = git_output("remote", "get-url", "origin")

    if url.startswith("git@github.com:"):
        url = url.replace(
            "git@github.com:",
            "https://github.com/",
            1
        )

    elif url.startswith("ssh://git@github.com/"):
        url = url.replace(
            "ssh://git@github.com/",
            "https://github.com/",
            1
        )

    if not url.startswith("https://github.com/"):
        raise ValueError(
            f"Expected a GitHub origin URL, got: {url}"
        )

    return url.removesuffix(".git") + ".git"


def check_git():
    """Validate Git repository and committed changes."""

    print("\n[1/5] Checking Git repository...")

    branch = git_output("branch", "--show-current")

    if not branch:
        raise RuntimeError(
            "Git is in detached HEAD state. Checkout a branch first."
        )

    changes = git_output(
        "status",
        "--porcelain",
        "--untracked-files=normal"
    )

    if changes:
        print("\nUncommitted changes detected:")
        print(changes)

        raise RuntimeError(
            "Commit your changes before deployment.\n"
            "Example:\n"
            "git add .\n"
            'git commit -m "Update experiments"'
        )

    repo_url = get_github_repository()

    print(f"GitHub repository: {repo_url}")
    print(f"Current branch: {branch}")

    return repo_url, branch


# ============================================================
# GITHUB PUSH
# ============================================================

def push_to_github(branch):
    """Push current committed branch to GitHub."""

    print("\n[2/5] Pushing to GitHub...")

    run_command(
        ["git", "push", "-u", "origin", branch],
        cwd=ROOT
    )

    print("GitHub push completed.")


# ============================================================
# KAGGLE BOOTSTRAP SCRIPT
# ============================================================

def create_bootstrap(repo_url, branch):
    """
    Generate the main.py that Kaggle executes.

    This script clones the GitHub repository,
    installs requirements, and runs the actual project.
    """

    print("\n[3/5] Creating Kaggle bootstrap script...")

    bootstrap = f'''
import os
import sys
import runpy
import subprocess
import base64
from pathlib import Path

REPO_URL = {repo_url!r}
BRANCH = {branch!r}

ENTRYPOINT = {PROJECT_ENTRYPOINT!r}
REQUIREMENTS = {REQUIREMENTS_FILE!r}

WORK_DIR = Path("/kaggle/working")
REPO_DIR = WORK_DIR / "LoRaFBsnet_Wang"


def execute(command, env=None):
    print("\\n>>>", " ".join(command), flush=True)

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
            f"x-access-token:{{token}}".encode()
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
            f"Project entrypoint not found: {{entrypoint}}"
        )

    os.chdir(REPO_DIR)

    sys.path.insert(0, str(REPO_DIR))

    print("\\nStarting project execution...")
    print("Entry point:", entrypoint)

    runpy.run_path(
        str(entrypoint),
        run_name="__main__"
    )


if __name__ == "__main__":
    main()
'''

    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

    bootstrap_path = DEPLOY_DIR / "main.py"

    bootstrap_path.write_text(
        bootstrap,
        encoding="utf-8"
    )

    print(f"Bootstrap created: {bootstrap_path}")


# ============================================================
# KAGGLE METADATA
# ============================================================

def create_metadata():
    """Generate Kaggle kernel-metadata.json."""

    print("\n[4/5] Generating Kaggle metadata...")

    metadata = {
        "id": KAGGLE_ID,
        "title": KAGGLE_TITLE,
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True
    }

    metadata_path = DEPLOY_DIR / "kernel-metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=4
        )

    print(f"Metadata created: {metadata_path}")


# ============================================================
# KAGGLE DEPLOYMENT
# ============================================================

def deploy_to_kaggle(no_run=False):
    """Upload and execute Kaggle kernel."""

    print("\n[5/5] Deploying to Kaggle...")

    command = [
        "kaggle",
        "kernels",
        "push",
        "-p",
        str(DEPLOY_DIR),
        "--timeout",
        str(KAGGLE_TIMEOUT)
    ]

    if no_run:
        command.append("--no-run")

    run_command(command)

    print("\n" + "=" * 60)
    print("KAGGLE DEPLOYMENT COMPLETED")
    print("=" * 60)

    print(f"Kernel: https://www.kaggle.com/code/{KAGGLE_ID}")


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Deploy LoRaFB-SNet experiments to Kaggle."
    )

    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Upload a new Kaggle version without executing it."
    )

    parser.add_argument(
        "--skip-git-push",
        action="store_true",
        help="Skip pushing the current Git branch."
    )

    args = parser.parse_args()

    try:

        repo_url, branch = check_git()

        if not args.skip_git_push:
            push_to_github(branch)

        create_bootstrap(repo_url, branch)

        create_metadata()

        deploy_to_kaggle(
            no_run=args.no_run
        )

    except subprocess.CalledProcessError as error:

        print("\nDeployment failed.")
        print("Command exit code:", error.returncode)
        sys.exit(error.returncode)

    except Exception as error:

        print("\nERROR:", error)
        sys.exit(1)


if __name__ == "__main__":
    main()
