import pytest

from Lab_01.src.calculator import add, addThree, prod, sub


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [(2, 3, 5.0), (-1, 2, 1.0), (1.5, 2.25, 3.75)],
)
def test_add(left, right, expected):
    assert add(left, right) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [(5, 3, 2.0), (2, 5, -3.0), (4.5, 1.25, 3.25)],
)
def test_sub(left, right, expected):
    assert sub(left, right) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [(2, 3, 6.0), (-2, 3, -6.0), (1.5, 2, 3.0)],
)
def test_prod(left, right, expected):
    assert prod(left, right) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [(2, 3, 10.0), (-2, 3, -10.0), (1.5, 2, 6.0)],
)
def test_addThree(left, right, expected):
    assert addThree(left, right) == pytest.approx(expected)