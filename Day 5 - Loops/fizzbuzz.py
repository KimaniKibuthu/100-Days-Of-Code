# Fiz buzz game
for number in range(1,100):
    if number % 3 == 0:
        if number % 5 == 0:
            print('FizzBuzz')
        else:
            print('Fizz')
    
    elif number % 5 == 0:
        if number % 3 == 0:
            print('FizzBuzz')
        else:
            print('Buzz')
    
    else:
        print(number)