"""
Unittests for python_repository_template.math_functions module.
"""
from python_repository_template import math_functions


def test_add():
    """
    Test for the math_functions.add() method.
    """
    z = math_functions.add(1, 2)
    assert  z == 3
