# Write a program that checks whether a number is divisible by 5 or 7

def is_divisible(number):
    if number % 5 == 0 and number % 7 == 0:
        return True

    else:
        return False


print(is_divisible(900))