# Calculator 2
logo = """
 _____________________
|  _________________  |
| | Pythonista   0. | |  .----------------.  .----------------.  .----------------.  .----------------. 
| |_________________| | | .--------------. || .--------------. || .--------------. || .--------------. |
|  ___ ___ ___   ___  | | |     ______   | || |      __      | || |   _____      | || |     ______   | |
| | 7 | 8 | 9 | | + | | | |   .' ___  |  | || |     /  \     | || |  |_   _|     | || |   .' ___  |  | |
| |___|___|___| |___| | | |  / .'   \_|  | || |    / /\ \    | || |    | |       | || |  / .'   \_|  | |
| | 4 | 5 | 6 | | - | | | |  | |         | || |   / ____ \   | || |    | |   _   | || |  | |         | |
| |___|___|___| |___| | | |  \ `.___.'\  | || | _/ /    \ \_ | || |   _| |__/ |  | || |  \ `.___.'\  | |
| | 1 | 2 | 3 | | x | | | |   `._____.'  | || ||____|  |____|| || |  |________|  | || |   `._____.'  | |
| |___|___|___| |___| | | |              | || |              | || |              | || |              | |
| | . | 0 | = | | / | | | '--------------' || '--------------' || '--------------' || '--------------' |
| |___|___|___| |___| |  '----------------'  '----------------'  '----------------'  '----------------' 
|_____________________|
"""

print(logo)
# Define a clear function
def clear():
    import os
    os.system('cls' if os.name == 'nt' else 'clear')

# Define operations
def add(n1, n2):
    return n1+n2

def subtract(n1, n2):
    return n1-n2

def multiply(n1, n2):
    return n1*n2

def divide(n1, n2):
    return n1/n2

# operations dictionary 
operations = {'+':add, '-':subtract, '*':multiply, '/':divide}

# Calculator function
def calculator():
    n1 = float(input('Enter first number: \n'))
    to_continue = True
    while to_continue:
        print([operator for operator in operations.keys()])
        operator = input('Choose an operation: \n')
        n2 = float(input('Enter another number: \n'))
        my_function = operations[operator]
        answer = my_function(n1, n2)
        print(f'{n1} {operator} {n2} = {answer}')

        choice = input(f'Type "y " to continue calculation with {answer} and "n" to start a new calculation:\n')
    
        if choice == 'y':
            to_continue = True
            n1 = answer
        else:
            clear()
            to_continue = False
            calculator()

calculator()

