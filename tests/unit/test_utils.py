"""Tests for confusius._utils."""

from confusius._utils import _one_level_deeper, find_stack_level


def test_find_stack_level():
    """Test find_stack_level."""
    assert find_stack_level() == 1
    assert _one_level_deeper() == 2
