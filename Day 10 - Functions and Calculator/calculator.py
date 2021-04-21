# Calculator program

def clear():
    import os
    os.system('cls' if os.name == 'nt' else 'clear')

def calculator(first_number, second_number, operator):
   

    if operator == '*':
        answer = first_number * second_number
    if operator == '+':
        answer = first_number + second_number
    if operator == '/':
        answer = first_number / second_number
    if operator == '-':
        answer = first_number - second_number

    return answer

first_number = float(input('Enter first number: \n'))
operator = input('Choose an operation: (*, +, /, -) \n')
second_number = float(input('Enter second number: \n'))
answer = calculator(first_number, second_number, operator)
print(f'{first_number} {operator} {second_number} = {answer}')

to_continue = input(f'Type "y " to continue calculation with {answer} and "n" to start a new calculation:\n')

while to_continue == 'y':
    first_number = answer
    operator = input('Choose an operation: (*, +, /, -) \n')
    second_number = float(input('Enter second number: \n'))
    answer = calculator(first_number, second_number, operator)
    print(f'{first_number} {operator} {second_number} = {answer}')

    to_continue = input(f'Type "y " to continue calculation with {answer} and "n" to start a new calculation:\n')

else:
    clear()
    first_number = float(input('Enter first number: \n'))
    operator = input('Choose an operation: (*, +, /, -) \n')
    second_number = float(input('Enter second number: \n'))
    answer = calculator(first_number, second_number, operator)
    print(f'{first_number} {operator} {second_number} = {answer}')

    to_continue = input(f'Type "y " to continue calculation with {answer} and "n" to start a new calculation:\n')

# Works. However, there is alot of repetition
    



