"""Smoke tests for package scaffold."""

from __future__ import annotations

import nightwatch


def test_version_is_set() -> None:
    assert nightwatch.__version__
