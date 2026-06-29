"""Tests for pyproject.toml structure and configuration correctness."""

import re
import sys

from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

REPO_ROOT = Path(__file__).parent.parent
PYPROJECT_PATH = REPO_ROOT / 'pyproject.toml'


@pytest.fixture(scope='module')
def pyproject() -> dict:  # type: ignore[type-arg]
    """Load and return the parsed pyproject.toml as a dict."""
    with open(PYPROJECT_PATH, 'rb') as fh:
        return tomllib.load(fh)


@pytest.fixture(scope='module')
def project(pyproject: dict) -> dict:  # type: ignore[type-arg]
    return pyproject['project']


# ---------------------------------------------------------------------------
# [project] section
# ---------------------------------------------------------------------------


def test_pyproject_file_exists() -> None:
    assert PYPROJECT_PATH.exists(), 'pyproject.toml must exist at repo root'


def test_project_name(project: dict) -> None:  # type: ignore[type-arg]
    assert project['name'] == 'hph-medvision-channel'


def test_project_version_format(project: dict) -> None:  # type: ignore[type-arg]
    version = project['version']
    # Version must follow semver-like pattern: MAJOR.MINOR.PATCH
    assert re.match(r'^\d+\.\d+\.\d+', version), (
        f'version "{version}" does not look like a semver string'
    )


def test_project_description_non_empty(project: dict) -> None:  # type: ignore[type-arg]
    assert project.get('description', '').strip() != ''


def test_project_license(project: dict) -> None:  # type: ignore[type-arg]
    assert project['license']['text'] == 'BSD-3-Clause'


def test_project_requires_python(project: dict) -> None:  # type: ignore[type-arg]
    requires = project['requires-python']
    assert '3.10' in requires, (
        'requires-python should reference Python 3.10 as the minimum'
    )


def test_project_requires_python_upper_bound(project: dict) -> None:  # type: ignore[type-arg]
    requires = project['requires-python']
    assert '<4' in requires, (
        'requires-python should have an upper bound excluding Python 4'
    )


def test_project_has_authors(project: dict) -> None:  # type: ignore[type-arg]
    authors = project.get('authors', [])
    assert len(authors) >= 1
    assert 'name' in authors[0]


def test_project_urls_homepage(project: dict) -> None:  # type: ignore[type-arg]
    urls = project.get('urls', {})
    assert 'Homepage' in urls
    assert urls['Homepage'].startswith('https://')


def test_project_urls_issues(project: dict) -> None:  # type: ignore[type-arg]
    urls = project.get('urls', {})
    assert 'Issues' in urls
    assert 'issues' in urls['Issues'].lower()


def test_project_dependencies_present(project: dict) -> None:  # type: ignore[type-arg]
    deps = project.get('dependencies', [])
    assert len(deps) > 0, 'Project should declare at least one dependency'


def test_project_dependencies_include_torch(project: dict) -> None:  # type: ignore[type-arg]
    deps = project.get('dependencies', [])
    dep_names = [d.split()[0].split('>=')[0].split('>')[0] for d in deps]
    assert 'torch' in dep_names


def test_project_dependencies_include_monai(project: dict) -> None:  # type: ignore[type-arg]
    deps = project.get('dependencies', [])
    assert any('monai' in d.lower() for d in deps)


def test_project_dependencies_include_numpy(project: dict) -> None:  # type: ignore[type-arg]
    deps = project.get('dependencies', [])
    assert any('numpy' in d.lower() for d in deps)


# ---------------------------------------------------------------------------
# [project.optional-dependencies] section
# ---------------------------------------------------------------------------


def test_dev_optional_deps_present(pyproject: dict) -> None:  # type: ignore[type-arg]
    opt = pyproject['project'].get('optional-dependencies', {})
    assert 'dev' in opt, (
        '[project.optional-dependencies] must have a "dev" group'
    )


def test_dev_deps_include_pytest(pyproject: dict) -> None:  # type: ignore[type-arg]
    dev_deps = pyproject['project']['optional-dependencies']['dev']
    assert any('pytest' in d for d in dev_deps)


def test_dev_deps_include_ruff(pyproject: dict) -> None:  # type: ignore[type-arg]
    dev_deps = pyproject['project']['optional-dependencies']['dev']
    assert any('ruff' in d for d in dev_deps)


def test_dev_deps_include_mypy(pyproject: dict) -> None:  # type: ignore[type-arg]
    dev_deps = pyproject['project']['optional-dependencies']['dev']
    assert any('mypy' in d for d in dev_deps)


def test_dev_deps_include_bandit(pyproject: dict) -> None:  # type: ignore[type-arg]
    dev_deps = pyproject['project']['optional-dependencies']['dev']
    assert any('bandit' in d for d in dev_deps)


def test_dev_deps_include_vulture(pyproject: dict) -> None:  # type: ignore[type-arg]
    dev_deps = pyproject['project']['optional-dependencies']['dev']
    assert any('vulture' in d for d in dev_deps)


def test_dev_deps_include_makim(pyproject: dict) -> None:  # type: ignore[type-arg]
    dev_deps = pyproject['project']['optional-dependencies']['dev']
    assert any('makim' in d for d in dev_deps)


# ---------------------------------------------------------------------------
# [build-system] section
# ---------------------------------------------------------------------------


def test_build_system_requires(pyproject: dict) -> None:  # type: ignore[type-arg]
    build = pyproject['build-system']
    requires = build['requires']
    assert any('setuptools' in r for r in requires)


def test_build_system_backend(pyproject: dict) -> None:  # type: ignore[type-arg]
    backend = pyproject['build-system']['build-backend']
    assert backend == 'setuptools.build_meta'


# ---------------------------------------------------------------------------
# [tool.setuptools.packages.find] section
# ---------------------------------------------------------------------------


def test_setuptools_packages_include_shared(pyproject: dict) -> None:  # type: ignore[type-arg]
    include = pyproject['tool']['setuptools']['packages']['find']['include']
    assert any(p.startswith('shared') for p in include)


def test_setuptools_packages_include_skills(pyproject: dict) -> None:  # type: ignore[type-arg]
    include = pyproject['tool']['setuptools']['packages']['find']['include']
    assert any(p.startswith('skills') for p in include)


# ---------------------------------------------------------------------------
# [tool.pytest.ini_options] section
# ---------------------------------------------------------------------------


def test_pytest_testpaths(pyproject: dict) -> None:  # type: ignore[type-arg]
    opts = pyproject['tool']['pytest']['ini_options']
    assert 'tests' in opts['testpaths']


# ---------------------------------------------------------------------------
# [tool.mypy] section
# ---------------------------------------------------------------------------


def test_mypy_python_version(pyproject: dict) -> None:  # type: ignore[type-arg]
    mypy = pyproject['tool']['mypy']
    assert mypy['python_version'] == '3.10'


def test_mypy_strict_enabled(pyproject: dict) -> None:  # type: ignore[type-arg]
    mypy = pyproject['tool']['mypy']
    assert mypy.get('strict') is True


# ---------------------------------------------------------------------------
# [tool.ruff] section
# ---------------------------------------------------------------------------


def test_ruff_target_version(pyproject: dict) -> None:  # type: ignore[type-arg]
    ruff = pyproject['tool']['ruff']
    assert ruff['target-version'] == 'py310'


def test_ruff_line_length(pyproject: dict) -> None:  # type: ignore[type-arg]
    ruff = pyproject['tool']['ruff']
    assert ruff['line-length'] == 79


def test_ruff_lint_selectors_present(pyproject: dict) -> None:  # type: ignore[type-arg]
    lint = pyproject['tool']['ruff']['lint']
    select = lint.get('select', [])
    # Must include pycodestyle, pyflakes, isort at minimum
    assert 'E' in select
    assert 'F' in select
    assert 'I001' in select


def test_ruff_format_quote_style(pyproject: dict) -> None:  # type: ignore[type-arg]
    fmt = pyproject['tool']['ruff']['format']
    assert fmt['quote-style'] == 'single'


# ---------------------------------------------------------------------------
# [tool.vulture] section
# ---------------------------------------------------------------------------


def test_vulture_min_confidence(pyproject: dict) -> None:  # type: ignore[type-arg]
    vulture = pyproject['tool']['vulture']
    assert vulture['min_confidence'] == 80


def test_vulture_paths_include_whitelist(pyproject: dict) -> None:  # type: ignore[type-arg]
    vulture = pyproject['tool']['vulture']
    assert 'vulture_whitelist.py' in vulture['paths']


def test_vulture_excludes_tests(pyproject: dict) -> None:  # type: ignore[type-arg]
    vulture = pyproject['tool']['vulture']
    assert 'tests' in vulture.get('exclude', [])


# ---------------------------------------------------------------------------
# [tool.bandit] section
# ---------------------------------------------------------------------------


def test_bandit_excludes_tests(pyproject: dict) -> None:  # type: ignore[type-arg]
    bandit = pyproject['tool']['bandit']
    assert 'tests' in bandit.get('exclude_dirs', [])


# ---------------------------------------------------------------------------
# Runtime compatibility check
# ---------------------------------------------------------------------------


def test_running_python_meets_minimum_requirement() -> None:
    """The Python interpreter running the tests satisfies requires-python."""
    assert sys.version_info >= (3, 10), (
        'Python 3.10+ is required according to pyproject.toml'
    )


# ---------------------------------------------------------------------------
# Semantic-release marker in version line
# ---------------------------------------------------------------------------


def test_version_line_has_semantic_release_marker() -> None:
    """pyproject.toml version line carries the # semantic-release marker."""
    content = PYPROJECT_PATH.read_text()
    assert re.search(
        r'version\s*=\s*"[^"]*"\s*#\s*semantic-release', content
    ), (
        'pyproject.toml version line must have "# semantic-release" comment '
        'so the release automation can update it'
    )
