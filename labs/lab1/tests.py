from task10 import is_balanced


def test_is_balanced():
    print("Testing `is_balanced`...")

    assert is_balanced(23441)
    assert is_balanced(1231)
    assert is_balanced(7)
    assert is_balanced(10)
    assert not is_balanced(123456)
    assert not is_balanced(100)

    print("All tests passed!")


if __name__ == "__main__":
    test_is_balanced()
