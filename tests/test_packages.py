"""Tests for package importability of the newly added Python packages."""

import importlib
import types


def test_shared_package_is_importable() -> None:
    """shared package can be imported without errors."""
    mod = importlib.import_module('shared')
    assert isinstance(mod, types.ModuleType)


def test_shared_models_package_is_importable() -> None:
    """shared.models sub-package can be imported without errors."""
    mod = importlib.import_module('shared.models')
    assert isinstance(mod, types.ModuleType)


def test_skills_package_is_importable() -> None:
    """skills package can be imported without errors."""
    mod = importlib.import_module('skills')
    assert isinstance(mod, types.ModuleType)


def test_shared_is_a_package() -> None:
    """shared module has a __path__ attribute, confirming it is a package."""
    import shared

    assert hasattr(shared, '__path__')


def test_shared_models_is_a_package() -> None:
    """shared.models has a __path__ attribute, confirming it is a package."""
    import shared.models

    assert hasattr(shared.models, '__path__')


def test_skills_is_a_package() -> None:
    """skills module has a __path__ attribute, confirming it is a package."""
    import skills

    assert hasattr(skills, '__path__')


def test_shared_models_is_subpackage_of_shared() -> None:
    """shared.models is a sub-package of shared."""
    import shared
    import shared.models

    assert shared.models.__name__ == 'shared.models'
    assert shared.models.__name__.startswith(shared.__name__)


def test_shared_package_has_no_unexpected_public_api() -> None:
    """Newly added empty __init__.py exposes no public names."""
    import shared

    public_names = [n for n in dir(shared) if not n.startswith('_')]
    # Empty __init__.py should only contain standard dunder attributes
    # filtered above; no user-defined names should be present.
    assert set(public_names).issubset({'models'})


def test_skills_package_has_no_unexpected_public_api() -> None:
    """Newly added empty __init__.py for skills exposes no public names."""
    import skills

    public_names = [n for n in dir(skills) if not n.startswith('_')]
    assert public_names == []


def test_shared_models_package_has_no_unexpected_public_api() -> None:
    """Newly added __init__.py for shared.models exposes no public names."""
    import shared.models

    public_names = [n for n in dir(shared.models) if not n.startswith('_')]
    assert public_names == []


def test_double_import_is_idempotent() -> None:
    """Importing the same package twice returns the same object."""
    mod1 = importlib.import_module('shared')
    mod2 = importlib.import_module('shared')
    assert mod1 is mod2


def test_skills_double_import_is_idempotent() -> None:
    """Importing skills twice returns the same module object."""
    mod1 = importlib.import_module('skills')
    mod2 = importlib.import_module('skills')
    assert mod1 is mod2
