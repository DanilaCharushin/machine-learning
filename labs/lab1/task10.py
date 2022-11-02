"""
Задача N10.

Написать функцию, которая принимает положительное целое число n и определяющая является ли число n сбалансированным.
Число является сбалансированным, если сумма цифр до средних цифр равна сумме цифр после средней цифры.
Если число нечетное, то средняя цифра одна, если четное, то средних цифр две.
В расчете средние числа не участвуют.

Например:
· Число 23441 сбалансированное, так как 2+3 = 4+1
· Число 7 сбалансированное, так как 0 = 0
· Число 1231 сбалансированное, так как 1 = 1
· Число 123456 несбалансированное, так как 1+2 != 5+6
"""


def main():
    print("============================================")
    print("================ Задача 10. ================")
    print("============================================")
    print()
    number = int(input("Введите число для проверки на сбалансированность: "))
    print(f"Число {number} {'' if is_balanced(number) else 'не'}сбалансированное")


def is_balanced(number: int) -> bool:
    number_as_str = str(number)
    digits_count = len(number_as_str)
    middle_digits_count = 2 - digits_count % 2

    if middle_digits_count == 2:
        left_middle_index = digits_count // 2 - 1
        right_middle_index = left_middle_index + 2
    else:
        left_middle_index = digits_count // 2
        right_middle_index = left_middle_index + 1

    digits = [int(digit) for digit in number_as_str]

    sum_before = sum_after = 0

    for i in range(left_middle_index):
        sum_before += digits[i]

    for i in range(right_middle_index, digits_count):
        sum_after += digits[i]

    return sum_before == sum_after


if __name__ == "__main__":
    main()
