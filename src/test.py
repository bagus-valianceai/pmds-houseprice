import pytest

def some_funcuntion(x):
    x += 5
    return x

def test_function():
    assert some_funcuntion(5) == 15
    