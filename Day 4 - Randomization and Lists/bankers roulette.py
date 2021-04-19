# You are going to write a program which will select a random name from a list of names. The person selected will have to pay for everybody's food bill

import random

list_of_names = ['Angela', 'Ben', 'Jenny', 'Michael', 'Chloe']

# Choose name randomly
name_chosen = random.choice(list_of_names)

print(f'{name_chosen} is paying the bill')