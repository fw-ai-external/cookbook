"""Gate: every cookbook-registered renderer must be in the shared QA matrix.

A renderer joins CI by adding a ``RendererCase`` row to
:mod:`renderer_matrix`. Names registered before this gate are listed in
``UNMATRIXED_RENDERERS``; that set may only shrink. Set
``RENDERER_MATRIX_BASE_FILE`` to the base revision's ``renderer_matrix.py``
(the Renderer QA workflow does this on PRs) to also fail when a change adds
names to that legacy set.

Set ``RENDERER_CHANGED_FILES`` to a file listing the PR's changed paths, one per
line (the workflow does this too), to also require a row for every renderer
whose source module changed, including names in the legacy set.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import os
from collections.abc import Callable, Iterable
from pathlib import Path

import pytest

import training.renderer  # noqa: F401  (imports every renderer module so registrations run)
from training.renderer import _registry
from training.renderer.base import Renderer
from training.tests.unit.renderer_matrix import RENDERER_MATRIX, UNMATRIXED_RENDERERS

_COOKBOOK_RENDERER_PACKAGE = "training.renderer"
_RENDERER_DIR = Path(training.renderer.__file__).parent
# Cross-family base and mixin modules. Changing one is covered by the full
# matrix run rather than by requiring rows for every legacy renderer.
_SHARED_RENDERER_MODULES = frozenset(
    {
        "training.renderer.base",
        "training.renderer._disaggregate_mixin",
        "training.renderer._think_prefill",
    }
)


def _builtin_renderer_names() -> set[str]:
    """Names resolved by the ``name == "..."`` branches in ``get_renderer``."""
    tree = ast.parse(inspect.getsource(_registry.get_renderer))
    names: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Name)
            and node.left.id == "name"
            and len(node.ops) == 1
            and isinstance(node.ops[0], ast.Eq)
            and isinstance(node.comparators[0], ast.Constant)
            and isinstance(node.comparators[0].value, str)
        ):
            names.add(node.comparators[0].value)
    return names


def _cookbook_registered_renderer_names() -> set[str]:
    """Renderer names owned by this package (installed extension plugins excluded)."""
    custom = {
        name
        for name, factory in _registry._CUSTOM_RENDERER_REGISTRY.items()
        if getattr(factory, "__module__", "").startswith(_COOKBOOK_RENDERER_PACKAGE)
    }
    return custom | _builtin_renderer_names()


def _matrix_renderer_names() -> set[str]:
    return {case.renderer for case in RENDERER_MATRIX}


def test_builtin_renderer_name_extraction_finds_branches() -> None:
    # Guards the AST walk: if get_renderer stops using ``name == "..."``
    # branches, this gate must be updated rather than silently pass.
    assert {"llama3", "role_colon"} <= _builtin_renderer_names()


def test_every_registered_renderer_has_a_matrix_row() -> None:
    missing = _cookbook_registered_renderer_names() - _matrix_renderer_names() - UNMATRIXED_RENDERERS
    assert not missing, (
        f"Renderer(s) {sorted(missing)} are registered but have no RendererCase row. Add a row to "
        "RENDERER_MATRIX in training/tests/unit/renderer_matrix.py (record known HF divergences in "
        "renderer_expected_divergences.py). Do not add them to UNMATRIXED_RENDERERS."
    )


def test_matrix_rows_name_registered_renderers() -> None:
    unknown = _matrix_renderer_names() - _cookbook_registered_renderer_names()
    assert not unknown, f"RENDERER_MATRIX rows name unregistered renderers: {sorted(unknown)}"


def test_unmatrixed_renderers_have_no_matrix_row() -> None:
    covered = UNMATRIXED_RENDERERS & _matrix_renderer_names()
    assert not covered, f"Remove {sorted(covered)} from UNMATRIXED_RENDERERS; they now have matrix rows."


def test_unmatrixed_renderers_are_still_registered() -> None:
    stale = UNMATRIXED_RENDERERS - _cookbook_registered_renderer_names()
    assert not stale, f"Remove unregistered names {sorted(stale)} from UNMATRIXED_RENDERERS."


def _unmatrixed_names_in(path: Path) -> frozenset[str] | None:
    """Parse ``UNMATRIXED_RENDERERS`` from a ``renderer_matrix.py`` without importing it."""
    for node in ast.parse(path.read_text()).body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "UNMATRIXED_RENDERERS"
        ):
            literal = node.value.args[0]
            return frozenset(ast.literal_eval(literal))
    return None


@pytest.mark.skipif(
    not os.environ.get("RENDERER_MATRIX_BASE_FILE"),
    reason="set RENDERER_MATRIX_BASE_FILE to the base revision's renderer_matrix.py",
)
def test_unmatrixed_renderers_only_shrink() -> None:
    base = _unmatrixed_names_in(Path(os.environ["RENDERER_MATRIX_BASE_FILE"]))
    if base is None:
        pytest.skip("base revision predates UNMATRIXED_RENDERERS")
    added = UNMATRIXED_RENDERERS - base
    assert not added, (
        f"UNMATRIXED_RENDERERS is a shrink-only legacy baseline; do not add {sorted(added)}. "
        "Add RendererCase rows to RENDERER_MATRIX instead."
    )


def _builtin_renderer_classes() -> dict[str, set[type]]:
    """Map each built-in ``get_renderer`` branch name to the renderer classes it constructs."""
    func = ast.parse(inspect.getsource(_registry.get_renderer)).body[0]
    imported: dict[str, type] = {}
    for node in ast.walk(func):
        if isinstance(node, ast.ImportFrom) and node.module:
            module = importlib.import_module(node.module)
            for alias in node.names:
                imported[alias.asname or alias.name] = getattr(module, alias.name)
    classes: dict[str, set[type]] = {}
    for node in ast.walk(func):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "name"
            and isinstance(test.comparators[0], ast.Constant)
        ):
            called = {
                call.func.id
                for stmt in node.body
                for call in ast.walk(stmt)
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
            }
            classes[test.comparators[0].value] = {imported[n] for n in called if isinstance(imported.get(n), type)}
    return classes


def _factory_renderer_classes(factory: Callable[..., object]) -> set[type]:
    """Renderer classes a registered factory references by global name."""
    code = getattr(factory, "__code__", None)
    if code is None:
        return set()
    names = set(code.co_names)
    for const in code.co_consts:
        if inspect.iscode(const):
            names |= set(const.co_names)
    found = {getattr(factory, "__globals__", {}).get(n) for n in names}
    return {obj for obj in found if isinstance(obj, type) and issubclass(obj, Renderer)}


def _source_modules(classes: Iterable[type]) -> set[str]:
    return {
        klass.__module__
        for cls in classes
        for klass in cls.__mro__
        if klass.__module__.startswith(_COOKBOOK_RENDERER_PACKAGE) and klass.__module__ not in _SHARED_RENDERER_MODULES
    }


def _renderer_source_modules() -> dict[str, set[str]]:
    """Map each cookbook renderer name to the modules defining its factory and class hierarchy."""
    modules: dict[str, set[str]] = {
        name: _source_modules(classes) for name, classes in _builtin_renderer_classes().items()
    }
    for name, factory in _registry._CUSTOM_RENDERER_REGISTRY.items():
        factory_module = getattr(factory, "__module__", "")
        if factory_module.startswith(_COOKBOOK_RENDERER_PACKAGE):
            modules.setdefault(name, set()).update(
                {factory_module} | _source_modules(_factory_renderer_classes(factory))
            )
    return modules


def _module_for_path(path: str) -> str | None:
    """``public-repos/cookbook/training/renderer/foo.py`` -> ``training.renderer.foo``."""
    parts = Path(path).parts
    for i in range(len(parts) - 1):
        if parts[i : i + 2] == ("training", "renderer") and path.endswith(".py"):
            rel = Path(*parts[i:]).with_suffix("")
            return ".".join(rel.parts).removesuffix(".__init__")
    return None


def _renderers_touched_by(paths: Iterable[str]) -> set[str]:
    changed = {m for m in map(_module_for_path, paths) if m}
    return {name for name, modules in _renderer_source_modules().items() if modules & changed}


def test_renderer_source_mapping_covers_registered_names() -> None:
    sources = _renderer_source_modules()
    unmapped = sorted(name for name in _cookbook_registered_renderer_names() if not sources.get(name))
    assert not unmapped, f"Cannot map renderer(s) {unmapped} to source modules; update this gate."


def test_editing_a_legacy_renderer_module_touches_its_names() -> None:
    touched = _renderers_touched_by(["public-repos/cookbook/training/renderer/muse_spark.py"])
    assert "muse_spark" in touched
    assert "llama3" not in touched
    assert "llama3" in _renderers_touched_by(["training/renderer/llama3.py"])
    assert not _renderers_touched_by(
        [
            "training/renderer/base.py",
            "training/renderer/_disaggregate_mixin.py",
            "training/tests/unit/renderer_matrix.py",
        ]
    )


@pytest.mark.skipif(
    not os.environ.get("RENDERER_CHANGED_FILES"),
    reason="set RENDERER_CHANGED_FILES to a file listing the change's paths",
)
def test_changed_renderers_have_matrix_rows() -> None:
    paths = Path(os.environ["RENDERER_CHANGED_FILES"]).read_text().split()
    missing = _renderers_touched_by(paths) - _matrix_renderer_names()
    assert not missing, (
        f"This change edits the source of renderer(s) {sorted(missing)}, which have no RendererCase row. "
        "Add rows to RENDERER_MATRIX in training/tests/unit/renderer_matrix.py and remove the names "
        "from UNMATRIXED_RENDERERS."
    )
