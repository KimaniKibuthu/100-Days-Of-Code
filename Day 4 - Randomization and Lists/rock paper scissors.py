# Game of rock paper scissors
import random 

rock = '''
    _______
---'   ____)
      (_____)
      (_____)
      (____)
---.__(___)
'''

paper = '''
    _______
---'   ____)____
          ______)
          _______)
         _______)
---.__________)
'''

scissors = '''
    _______
---'   ____)____
          ______)
       __________)
      (____)
---.__(___)
'''

choices = ['rock', 'paper', 'scissors']

# Your choice
my_choice = input('Rock, paper or scissors?\n')

# Computers choice
comps_choice = random.choice(choices)

if my_choice == 'rock':
    print(rock)

    if comps_choice == 'paper':
        print(paper)
        print('Computer wins')
    elif comps_choice == 'scissors':
        print(scissors)
        print('You wins')
    else:
        print(rock)
        print('Its a draw')

elif my_choice == 'scissors':
    print(scissors)

    if comps_choice == 'paper':
        print(paper)
        print('You wins')
    elif comps_choice == 'scissors':
        print(scissors)
        print('Its a draw')
    else:
        print(rock)
        print('Computer wins')

else:
    print(paper)

    if comps_choice == 'paper':
        print(paper)
        print('Its a draw')
    elif comps_choice == 'scissors':
        print(scissors)
        print('Computer wins')
    else:
        print(rock)
        print('You draw')

