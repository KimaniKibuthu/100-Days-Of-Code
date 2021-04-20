# Prime number checker

number = int(input('Input a number: \n'))

def prime_checker(number):
    if number < 2:
        print(f'{number} is not a prime number')
    elif number > 1:
        checker = [number % nums for nums in range(2, number)]
        if 0 in checker:
            print(f'{number} is not a prime number')
        else:
            print(f'{number} is a prime number')

prime_checker(number)