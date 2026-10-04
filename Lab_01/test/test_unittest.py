import unittest

from Lab_01.src.calculator import add, addThree, prod, sub


class TestCalculator(unittest.TestCase):
    def test_add(self):
        cases = [(2, 3, 5.0), (-1, 2, 1.0), (1.5, 2.25, 3.75)]
        for left, right, expected in cases:
            with self.subTest(left=left, right=right):
                self.assertAlmostEqual(add(left, right), expected)

    def test_sub(self):
        cases = [(5, 3, 2.0), (2, 5, -3.0), (4.5, 1.25, 3.25)]
        for left, right, expected in cases:
            with self.subTest(left=left, right=right):
                self.assertAlmostEqual(sub(left, right), expected)

    def test_prod(self):
        cases = [(2, 3, 6.0), (-2, 3, -6.0), (1.5, 2, 3.0)]
        for left, right, expected in cases:
            with self.subTest(left=left, right=right):
                self.assertAlmostEqual(prod(left, right), expected)

    def test_addThree(self):
        cases = [(2, 3, 10.0), (-2, 3, -10.0), (1.5, 2, 6.0)]
        for left, right, expected in cases:
            with self.subTest(left=left, right=right):
                self.assertAlmostEqual(addThree(left, right), expected)


if __name__ == "__main__":
    unittest.main()