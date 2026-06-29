"""Tests for scripts/install-dev.sh — the developer setup script."""

import os
import re
import stat
import subprocess

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
SCRIPT_PATH = REPO_ROOT / 'scripts' / 'install-dev.sh'
SCRIPT_TEXT = (
    SCRIPT_PATH.read_text(encoding='utf-8') if SCRIPT_PATH.exists() else ''
)


# ---------------------------------------------------------------------------
# Existence and permissions
# ---------------------------------------------------------------------------


def test_script_file_exists() -> None:
    """install-dev.sh must exist in the scripts/ directory."""
    assert SCRIPT_PATH.exists(), f'Expected {SCRIPT_PATH} to exist'


def test_script_is_executable() -> None:
    """install-dev.sh must have the executable bit set."""
    mode = SCRIPT_PATH.stat().st_mode
    assert mode & stat.S_IXUSR, 'install-dev.sh must be executable by owner'


def test_script_has_bash_shebang() -> None:
    """The first line must be a valid bash shebang."""
    first_line = SCRIPT_TEXT.splitlines()[0]
    assert first_line.startswith('#!/'), (
        'Script must start with a shebang (#!)'
    )
    assert 'bash' in first_line, 'Shebang must reference bash'


# ---------------------------------------------------------------------------
# Bash syntax check (non-executing)
# ---------------------------------------------------------------------------


def test_script_passes_bash_syntax_check() -> None:
    """bash -n must report no syntax errors in install-dev.sh."""
    result = subprocess.run(
        ['bash', '-n', str(SCRIPT_PATH)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f'bash -n reported syntax errors:\n{result.stderr}'
    )


# ---------------------------------------------------------------------------
# Safety directives
# ---------------------------------------------------------------------------


def test_script_uses_strict_mode() -> None:
    """Script must use 'set -euo pipefail' for safe execution."""
    assert 'set -euo pipefail' in SCRIPT_TEXT, (
        "install-dev.sh should use 'set -euo pipefail' for safe "
        'shell execution'
    )


# ---------------------------------------------------------------------------
# UV_PYTHON environment variable
# ---------------------------------------------------------------------------


def test_script_exports_uv_python() -> None:
    """Script must export UV_PYTHON so uv installs into the interpreter."""
    assert 'UV_PYTHON' in SCRIPT_TEXT, (
        'install-dev.sh should set UV_PYTHON to target the active Python '
        'interpreter'
    )
    assert 'export UV_PYTHON' in SCRIPT_TEXT


# ---------------------------------------------------------------------------
# OS detection logic
# ---------------------------------------------------------------------------


def test_script_handles_linux() -> None:
    """Script must contain a Linux branch."""
    assert re.search(r'Linux\*', SCRIPT_TEXT), (
        'install-dev.sh must handle the Linux* OS case'
    )


def test_script_handles_macos() -> None:
    """Script must contain a macOS (Darwin) branch."""
    assert re.search(r'Darwin\*', SCRIPT_TEXT), (
        'install-dev.sh must handle the Darwin* (macOS) OS case'
    )


def test_script_handles_windows_mingw() -> None:
    """Script must contain a Windows/MINGW branch."""
    assert re.search(r'MINGW\*', SCRIPT_TEXT), (
        'install-dev.sh must handle the MINGW* (Windows Git Bash) case'
    )


def test_script_exits_on_unsupported_os() -> None:
    """Script must exit with an error for unrecognised OS values."""
    # The wildcard catch-all branch should contain 'exit 1'
    assert 'exit 1' in SCRIPT_TEXT, (
        "install-dev.sh must call 'exit 1' for unsupported operating systems"
    )


def test_script_reports_unsupported_os_to_stderr() -> None:
    """Script must print an unsupported OS message to stderr."""
    assert '>&2' in SCRIPT_TEXT, (
        'install-dev.sh should direct error messages to stderr (>&2)'
    )


# ---------------------------------------------------------------------------
# CI vs local-dev differentiation on Linux
# ---------------------------------------------------------------------------


def test_script_uses_cpu_only_wheels_on_linux_ci() -> None:
    """On Linux CI the script should use the CPU-only PyTorch index URL."""
    assert 'https://download.pytorch.org/whl/cpu' in SCRIPT_TEXT, (
        'install-dev.sh should reference the CPU-only PyTorch index for CI'
    )


def test_script_checks_ci_environment_variable() -> None:
    """Script must branch on the CI environment variable."""
    assert (
        '${CI' in SCRIPT_TEXT
        or '"$CI"' in SCRIPT_TEXT
        or 'CI:-' in SCRIPT_TEXT
    ), 'install-dev.sh should test the CI env variable to choose torch wheels'


def test_script_installs_dev_extras() -> None:
    """Script must install the project with its [dev] optional dependencies."""
    assert 'pip install -e' in SCRIPT_TEXT or 'pip install' in SCRIPT_TEXT, (
        "install-dev.sh must install the project's dev extras"
    )
    assert '.[dev]' in SCRIPT_TEXT, (
        'install-dev.sh must install using the [dev] extras group'
    )


# ---------------------------------------------------------------------------
# Repo-root navigation
# ---------------------------------------------------------------------------


def test_script_changes_to_repo_root() -> None:
    """Script must cd to the repository root before installing."""
    assert 'SCRIPT_DIR' in SCRIPT_TEXT or 'cd ' in SCRIPT_TEXT, (
        'install-dev.sh should navigate to the repo root so relative paths '
        'work'
    )


# ---------------------------------------------------------------------------
# Unsupported-OS behaviour — functional test via subprocess with fake uname
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.name == 'nt', reason='Bash subprocess test not supported on Windows'
)
def test_script_exits_nonzero_for_unknown_os(tmp_path: Path) -> None:
    """Running the script with an unsupported OS value must exit non-zero."""
    # We intercept 'uname' and 'uv' and 'pip' to avoid side-effects.
    # The script exits 1 for unsupported OS before reaching pip.
    fake_uname = tmp_path / 'uname'
    fake_uname.write_text('#!/bin/sh\necho UnknownOS\n')
    fake_uname.chmod(0o755)

    env = {**os.environ, 'PATH': f'{tmp_path}:{os.environ.get("PATH", "")}'}
    result = subprocess.run(
        ['bash', str(SCRIPT_PATH)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode != 0, (
        'Script should exit with a non-zero code for an unsupported OS'
    )
    assert 'Unsupported OS' in result.stderr, (
        "Script should print 'Unsupported OS' message to stderr"
    )
