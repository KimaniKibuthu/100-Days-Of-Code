# import the necessary modules
import random
from logo import logo

def clear():
    import os
    os.system('cls' if os.name == 'nt' else 'clear')

# Welcome message
print(logo)
play = (input('Welcome to BlackJack! Would you like to play?\n')).lower()
cards = [11, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
while play == 'yes':
    your_cards = [random.choice(cards), random.choice(cards)]
    computer_cards = [random.choice(cards), random.choice(cards)]
    print(f'Your cards are {your_cards}')
    print(f'One of the computer\'s cards is {random.choice(computer_cards)}')

    while input('Do you want to be dealt another hand?\n') == 'yes':
        your_cards.append(random.choice(cards))
        print(f'Your cards are {your_cards}')

    my_total = sum(your_cards)
    computer_total = sum(computer_cards)

    while computer_total < 17:
        computer_cards.append(random.choice(cards))
        computer_total = sum(computer_cards)

    print(f'Your final hand is {your_cards}')
    print(f'The computer\'s final hand {computer_cards}')

    if my_total <= 21 and computer_total <= 21:
        if my_total > computer_total:
            print('You win')
        else:
            print('Computer wins')

    elif my_total > 21:
        print('Computer wins. Your total sum is more than 21')

    elif computer_total > 21:
        print('You win. Computer total sum is more than 21')


    

    if input('Do you want to play again?') == 'yes':
        clear()
        play = 'yes'
    else:
        clear()
        play= 'no'
        print('Thank you for playing.')