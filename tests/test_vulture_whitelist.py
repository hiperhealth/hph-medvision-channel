"""Tests for vulture_whitelist.py — the false-positive suppression file."""

import ast
import py_compile

from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
WHITELIST_PATH = REPO_ROOT / 'vulture_whitelist.py'


def test_whitelist_file_exists() -> None:
    """vulture_whitelist.py must exist at the repo root."""
    assert WHITELIST_PATH.exists(), (
        'vulture_whitelist.py not found at repo root'
    )


def test_whitelist_is_valid_python_syntax() -> None:
    """vulture_whitelist.py must be syntactically valid Python."""
    source = WHITELIST_PATH.read_text(encoding='utf-8')
    # ast.parse raises SyntaxError on invalid syntax
    tree = ast.parse(source, filename=str(WHITELIST_PATH))
    assert tree is not None


def test_whitelist_compiles_without_error() -> None:
    """vulture_whitelist.py must compile without errors using py_compile."""
    py_compile.compile(str(WHITELIST_PATH), doraise=True)


def test_whitelist_has_header_comment() -> None:
    """The whitelist file should contain a descriptive header comment."""
    content = WHITELIST_PATH.read_text(encoding='utf-8')
    # File must start with a comment explaining its purpose
    assert content.lstrip().startswith('#'), (
        'vulture_whitelist.py should begin with an explanatory comment'
    )


def test_whitelist_mentions_vulture() -> None:
    """The whitelist file comments should reference vulture."""
    content = WHITELIST_PATH.read_text(encoding='utf-8')
    assert 'vulture' in content.lower() or 'Vulture' in content, (
        'vulture_whitelist.py should mention vulture in its header'
    )


def test_whitelist_mentions_false_positives() -> None:
    """The whitelist header should explain it suppresses false positives."""
    content = WHITELIST_PATH.read_text(encoding='utf-8')
    assert (
        'false positive' in content.lower() or 'whitelist' in content.lower()
    ), 'vulture_whitelist.py should explain its purpose'


def test_whitelist_references_vulture_docs() -> None:
    """The file should contain a reference to the vulture documentation."""
    content = WHITELIST_PATH.read_text(encoding='utf-8')
    assert 'jendrikseipp/vulture' in content, (
        'vulture_whitelist.py should link to the vulture project for guidance'
    )


def test_whitelist_contains_no_syntax_errors_when_executed() -> None:
    """Executing vulture_whitelist.py in a clean namespace must not raise."""
    source = WHITELIST_PATH.read_text(encoding='utf-8')
    namespace: dict = {}  # type: ignore[type-arg]
    # exec on comment-only files should be completely safe
    exec(compile(source, str(WHITELIST_PATH), 'exec'), namespace)


def test_whitelist_ast_body_is_comment_only() -> None:
    """An empty/comment-only whitelist should have no executable statements."""
    source = WHITELIST_PATH.read_text(encoding='utf-8')
    tree = ast.parse(source)
    # A whitelist with only comments (no actual entries yet) has an empty body
    # or only contains Expr nodes with string constants (docstrings).
    non_trivial = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.stmt) and not isinstance(node, ast.Expr)
    ]
    assert non_trivial == [], (
        'vulture_whitelist.py should contain only comments or simple '
        'expressions at this stage'
    )


def test_whitelist_file_is_not_empty() -> None:
    """vulture_whitelist.py must not be a zero-byte file."""
    assert WHITELIST_PATH.stat().st_size > 0, (
        'vulture_whitelist.py should contain at least a header comment'
    )


def test_whitelist_encoding_is_utf8_compatible() -> None:
    """vulture_whitelist.py should be readable as UTF-8."""
    content = WHITELIST_PATH.read_bytes()
    # Must decode without errors
    decoded = content.decode('utf-8')
    assert len(decoded) > 0
